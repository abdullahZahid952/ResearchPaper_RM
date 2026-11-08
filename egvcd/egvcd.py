"""
egvcd.py -- minimal Kaggle (single T4) pipeline for:
  (1) reproducing the VCD baseline (Leng et al., CVPR 2024) on POPE-COCO with LLaVA-1.5-7B
  (2) the proposed component: Entropy-Gated VCD (EG-VCD) -- run the distorted-image pass only
      when the original next-token distribution is uncertain (entropy > tau)
  (3) the same evaluation under synthetic domain shift (low-light / noise / blur applied to the
      same COCO images, so questions and labels are unchanged)
  (4) wall-clock latency and number of image prefills per method

Scope: POPE only. POPE answers are one Yes/No token, so a "decoding step" is a single forward
pass over image+question. Open-ended captioning (CHAIR) is NOT covered.

Sub-commands:
  python egvcd.py prepare --k_images 150
  python egvcd.py cache --cond clean            # also: lowlight noise blur
  python egvcd.py timing --taus 0.05 0.1 0.2 0.3 0.5
  python egvcd.py analyze
"""
import argparse, json, os, time, math, random
import numpy as np

ROOT = os.environ.get("EGVCD_ROOT", "/kaggle/working/egvcd_data")
SPLITS = ["random", "popular", "adversarial"]
CONDS = ["clean", "lowlight", "noise", "blur"]
POPE_URL = "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_{}.json"
COCO_URL = "http://images.cocodataset.org/val2014/{}"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\n{q} Answer the question using a single word or phrase. ASSISTANT:"
ALPHA, BETA, NOISE_STEP, TOPK = 1.0, 0.1, 999, 32   # VCD defaults for POPE (alpha=1, beta=0.1, T=999)
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.69, 1e9]  # nats; 0 => always VCD, 1e9 => never (= base)


def p(*a):
    return os.path.join(ROOT, *a)


# ----------------------------------------------------------------------------- data
def prepare(k_images):
    import requests
    os.makedirs(p("images"), exist_ok=True)
    rows = {}
    for s in SPLITS:
        txt = requests.get(POPE_URL.format(s), timeout=60).text
        rows[s] = [json.loads(l) for l in txt.splitlines() if l.strip()]
    imgs = []
    for r in rows["random"]:
        if r["image"] not in imgs:
            imgs.append(r["image"])
    imgs = imgs[:k_images]
    keep = set(imgs)
    for s in SPLITS:
        sub = [r for r in rows[s] if r["image"] in keep]
        with open(p(f"pope_{s}.jsonl"), "w") as f:
            for r in sub:
                f.write(json.dumps(r) + "\n")
        print(s, len(sub), "questions")
    for n in imgs:
        dst = p("images", n)
        if os.path.exists(dst):
            continue
        for attempt in range(3):
            try:
                open(dst, "wb").write(requests.get(COCO_URL.format(n), timeout=60).content)
                break
            except Exception as e:
                time.sleep(2)
    print("images:", len(os.listdir(p("images"))))


def corrupt(img, cond):
    """One fixed severity per shift. clean = identity."""
    from PIL import ImageFilter
    if cond == "clean":
        return img
    if cond == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=4))
    x = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    rng = np.random.RandomState(0)
    if cond == "noise":
        x = x + rng.normal(0, 0.12, x.shape)
    elif cond == "lowlight":
        x = (x ** 3.0) * 0.6 + rng.normal(0, 0.02, x.shape)
    x = np.clip(x, 0, 1)
    from PIL import Image
    return Image.fromarray((x * 255).astype(np.uint8))


# ----------------------------------------------------------------------------- model
def load_model(device="cuda:0"):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    # load straight onto the (only visible) GPU so two parallel jobs don't exhaust CPU RAM;
    # which physical T4 is used is chosen with CUDA_VISIBLE_DEVICES when launching
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    return proc, model


def diffusion_noise(x, step, seed):
    """Forward-diffusion noising of the preprocessed image tensor, as described for VCD
    (sigmoid beta schedule, 1000 steps). Re-implemented; compare with the official repo."""
    import torch
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000, device=x.device)) * (0.5e-2 - 1e-5) + 1e-5
    ab = torch.cumprod(1 - betas, dim=0)
    g = torch.Generator(device=x.device).manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32)
    return (ab[step].sqrt() * x.float() + (1 - ab[step]).sqrt() * noise).to(x.dtype)


def last_logits(model, inputs, pixel_values):
    import torch
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    pixel_values=pixel_values, use_cache=False)
    return out.logits[0, -1].float()


def build_inputs(proc, model, img, q):
    import torch
    inp = proc(images=img, text=PROMPT.format(q=q), return_tensors="pt")
    inp = {k: v.to(model.device) for k, v in inp.items()}
    inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
    return inp


def entropy_nats(logits):
    import torch
    lp = torch.log_softmax(logits, -1)
    return float(-(lp.exp() * lp).sum())


def cd_logits(lo, ld):
    """VCD contrast + adaptive plausibility constraint on full-vocab logits (torch)."""
    import torch
    cut = lo.max() + math.log(BETA)
    d = (1 + ALPHA) * lo - ALPHA * ld
    return d.masked_fill(lo < cut, float("-inf"))


# ----------------------------------------------------------------------------- cache
def run_cache(cond):
    """For every question: store top-K logits of the original pass, the distorted-pass logits at
    the same token ids, entropy and logsumexp of the original distribution. All methods
    (base / VCD / EG-VCD for any tau) are then evaluated offline from this cache."""
    import torch
    from PIL import Image
    proc, model = load_model()
    tok = proc.tokenizer
    os.makedirs(p("cache"), exist_ok=True)
    for s in SPLITS:
        out = p("cache", f"{cond}_{s}.npz")
        if os.path.exists(out):
            print("skip", out); continue
        rows = [json.loads(l) for l in open(p(f"pope_{s}.jsonl"))]
        ids_a, lo_a, ld_a, H_a, lse_a, lab_a, yes_a = [], [], [], [], [], [], []
        t0 = time.time()
        for i, r in enumerate(rows):
            img = corrupt(Image.open(p("images", r["image"])).convert("RGB"), cond)
            inp = build_inputs(proc, model, img, r["text"])
            lo = last_logits(model, inp, inp["pixel_values"])
            pv_d = diffusion_noise(inp["pixel_values"], NOISE_STEP, seed=r["question_id"])
            ld = last_logits(model, inp, pv_d)
            top = torch.topk(lo, TOPK)
            ids = top.indices
            ids_a.append(ids.cpu().numpy()); lo_a.append(top.values.cpu().numpy())
            ld_a.append(ld[ids].cpu().numpy())
            H_a.append(entropy_nats(lo)); lse_a.append(float(torch.logsumexp(lo, -1)))
            lab_a.append(r["label"] == "yes")
            yes_a.append([not tok.decode([int(t)]).strip().lower().startswith("no") for t in ids])
            if i % 50 == 0:
                print(f"{cond}/{s} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        np.savez(out, ids=np.array(ids_a), lo=np.array(lo_a), ld=np.array(ld_a), H=np.array(H_a),
                 lse=np.array(lse_a), label=np.array(lab_a), yes=np.array(yes_a))
        print("saved", out)


# ----------------------------------------------------------------------------- timing
def run_timing(taus, n=100, warm=5):
    import torch
    from PIL import Image
    proc, model = load_model()
    rows = [json.loads(l) for l in open(p("pope_random.jsonl"))][: n + warm]
    prepped = []
    for r in rows:
        img = Image.open(p("images", r["image"])).convert("RGB")
        prepped.append((build_inputs(proc, model, img, r["text"]), r["question_id"]))

    def method(inp, qid, tau):          # tau=None -> base ; tau=-1 -> always VCD ; else gated
        lo = last_logits(model, inp, inp["pixel_values"])
        passes = 1
        if tau is None:
            return int(lo.argmax()), passes
        if tau < 0 or entropy_nats(lo) > tau:
            pv_d = diffusion_noise(inp["pixel_values"], NOISE_STEP, seed=qid)
            ld = last_logits(model, inp, pv_d)
            passes = 2
            return int(cd_logits(lo, ld).argmax()), passes
        return int(lo.argmax()), passes

    res = {}
    for name, tau in [("base", None), ("vcd", -1.0)] + [(f"eg_{t}", t) for t in taus]:
        for inp, qid in prepped[:warm]:
            method(inp, qid, tau)
        torch.cuda.synchronize(); ts, ps = [], []
        for inp, qid in prepped[warm:]:
            t0 = time.perf_counter(); _, k = method(inp, qid, tau); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000); ps.append(k)
        res[name] = {"ms_mean": float(np.mean(ts)), "ms_std": float(np.std(ts)), "prefills": float(np.mean(ps))}
        print(name, res[name], flush=True)
    json.dump(res, open(p("timing.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- analysis
def _softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def decisions(c, tau, mode, seed):
    """Return predicted-yes boolean per question. tau: None=base, -1=VCD, >=0 gated."""
    lo, ld, H, yes = c["lo"], c["ld"], c["H"], c["yes"]
    cut = lo[:, :1] + math.log(BETA)
    cd = np.where(lo >= cut, (1 + ALPHA) * lo - ALPHA * ld, -np.inf)
    if tau is None:
        gate = np.zeros(len(H), bool)
    elif tau < 0:
        gate = np.ones(len(H), bool)
    else:
        gate = H > tau
    logits = np.where(gate[:, None], cd, lo)
    if mode == "greedy":
        j = logits.argmax(1)
    else:
        rng = np.random.RandomState(seed)
        pr = _softmax(logits); u = rng.rand(len(pr), 1)
        j = (pr.cumsum(1) < u).sum(1).clip(max=pr.shape[1] - 1)
    return yes[np.arange(len(j)), j], gate


def prf(pred, y):
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(acc=(pred == y).mean() * 100, prec=P * 100, rec=R * 100,
                f1=2 * P * R / max(P + R, 1e-9) * 100, yes=pred.mean() * 100)


def mcnemar_p(a_correct, b_correct):
    from scipy.stats import binomtest
    b = int((a_correct & ~b_correct).sum()); c = int((~a_correct & b_correct).sum())
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


def run_analyze(seeds=5):
    import csv
    timing = json.load(open(p("timing.json"))) if os.path.exists(p("timing.json")) else None
    taus = [("base", None)] + [("vcd", -1.0)] + [(f"eg_{t}", t) for t in TAUS[:-1]]
    rows = []
    for cond in CONDS:
        for s in SPLITS:
            f = p("cache", f"{cond}_{s}.npz")
            if not os.path.exists(f):
                continue
            c = dict(np.load(f)); y = c["label"]
            cov = float(np.exp(c["lo"] - c["lse"][:, None]).sum(1).mean())  # top-K mass coverage
            for mode in ["greedy", "sample"]:
                for name, tau in taus:
                    ms = [prf(decisions(c, tau, mode, sd)[0], y) for sd in (range(seeds) if mode == "sample" else [0])]
                    g = decisions(c, tau, mode, 0)[1]
                    row = dict(cond=cond, split=s, mode=mode, method=name, rho=g.mean(), topk_cov=cov,
                               prefills=1 + g.mean() if tau is not None else 1.0)
                    for k in ms[0]:
                        row[k] = float(np.mean([m[k] for m in ms])); row[k + "_sd"] = float(np.std([m[k] for m in ms]))
                    # paired McNemar vs base (seed 0)
                    pb = decisions(c, None, mode, 0)[0]; pm = decisions(c, tau, mode, 0)[0]
                    row["p_vs_base"] = mcnemar_p(pm == y, pb == y)
                    if timing:
                        key = {"base": "base", "vcd": "vcd"}.get(name, name)
                        if key in timing:
                            row["ms"] = timing[key]["ms_mean"]
                    rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with open(p("results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # compact console summary: sample-mode F1 averaged over splits
    print(f"{'cond':9}{'method':10}{'acc':>7}{'f1':>7}{'yes%':>7}{'rho':>6}{'prefills':>9}")
    for cond in CONDS:
        for name, _ in taus:
            sel = [r for r in rows if r["cond"] == cond and r["mode"] == "sample" and r["method"] == name]
            if sel:
                print(f"{cond:9}{name:10}{np.mean([r['acc'] for r in sel]):7.2f}{np.mean([r['f1'] for r in sel]):7.2f}"
                      f"{np.mean([r['yes'] for r in sel]):7.1f}{np.mean([r['rho'] for r in sel]):6.2f}"
                      f"{np.mean([r['prefills'] for r in sel]):9.2f}")
    print("wrote", p("results.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "cache", "timing", "analyze"])
    ap.add_argument("--k_images", type=int, default=150)
    ap.add_argument("--cond", default="clean", choices=CONDS)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if a.cmd == "prepare": prepare(a.k_images)
    elif a.cmd == "cache": run_cache(a.cond)
    elif a.cmd == "timing": run_timing(a.taus)
    else: run_analyze()
