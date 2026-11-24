# =====================================================================================
#  EG-VCD  --  MODERN-MODELS RUNNER   (Accelerator = GPU T4 x2, Internet = ON)  -- ONE cell, just run it
#
#  QUESTION: does the exact gate (skip the VCD negative pass when the plausibility head has one token) still
#  help on 2025-2026 vision-language models, and how much of a long generated answer could it save?
#
#  Stage 1 (both GPUs in parallel, ~30 min): Qwen3-VL-4B (Oct 2025) on GPU0, Qwen3.5-4B (Feb 2026) on GPU1
#  Stage 2 (both GPUs together, ~1 h):       Gemma-4-E4B-it (Apr 2026), InternVL3.5-8B-HF (Aug 2025)
#  Per model: smoke test -> POPE-COCO logit cache (100 images, 1,800 questions) -> free-form generation
#             step statistics (60 images).  A model that cannot run on a T4 is SKIPPED with a message; the rest continue.
#  Results are printed live and saved after each model, so you can stop whenever time runs out.
# =====================================================================================
MODELS_STAGE1 = "qwen3vl_4b,qwen35_4b"      # one model per T4, run in parallel
MODELS_STAGE2 = "gemma4_e4b,internvl35_8b"  # 8B-class: weights split across both T4s, run one after another
N_IMAGES = "100"                             # POPE images per model (6 questions each, x3 splits)
N_GEN = "60"                                 # images for the free-form generation statistic
TRANSFORMERS_MAIN = "0"                      # "1" = install transformers from GitHub main (only if a new model type is 'not recognized')

import os, sys, time, subprocess, selectors, glob, shutil

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
ROOT = os.path.join(WORK, "egvcd_data")
S1, S2, S3 = (os.path.join(WORK, n) for n in ("egvcd.py", "egvcd_solid.py", "egvcd_modern.py"))
os.makedirs(ROOT, exist_ok=True)
MODELS_STAGE1, MODELS_STAGE2, N_IMAGES, N_GEN, TRANSFORMERS_MAIN = [os.environ.get(k, v) for k, v in (
    ("MODELS_STAGE1", MODELS_STAGE1), ("MODELS_STAGE2", MODELS_STAGE2), ("N_IMAGES", N_IMAGES), ("N_GEN", N_GEN), ("TRANSFORMERS_MAIN", TRANSFORMERS_MAIN))]
ENV = {**os.environ, "EGVCD_ROOT": ROOT, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
T0 = time.time()

def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')} | +{(time.time()-T0)/60:5.1f} min] {msg}", flush=True)

def run(cmd, env=None, fatal=True):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env or ENV)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    if p.wait() != 0:
        if fatal:
            raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
        return False
    return True

EGVCD_SRC = r'''"""
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
'''
SOLID_SRC = r'''"""
egvcd_solid.py -- extra experiments that stress-test the exact-gate result (uses egvcd.py).

  prepare_ext   download 150 NEW COCO images (never used for any threshold) + their POPE questions
  cache2        cache logits for any subset / any LLaVA-1.5 checkpoint (7B on one GPU, 13B across two GPUs)
  e2e           REAL end-to-end run: VCD vs exact-gated VCD vs entropy-gated, same noise seeds and the same
                sampling randomness; asserts identical outputs and MEASURES latency (no model of latency)
  analyze       offline (CPU): held-out replication, 7B vs 13B, alpha/beta sweep, image-clustered bootstrap CIs
"""
import os, sys, json, math, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E

SPLITS, CONDS = E.SPLITS, E.CONDS
LOGB = math.log(E.BETA)


# =============================================================================== data
def prepare_ext(k_start, k_end):
    import requests
    os.makedirs(E.p("images"), exist_ok=True)
    rows = {s: [json.loads(l) for l in requests.get(E.POPE_URL.format(s), timeout=60).text.splitlines() if l.strip()] for s in SPLITS}
    imgs, seen = [], set()
    for r in rows["random"]:
        if r["image"] not in seen:
            seen.add(r["image"]); imgs.append(r["image"])
    keep = set(imgs[k_start:k_end])
    for s in SPLITS:
        sub = [r for r in rows[s] if r["image"] in keep]
        with open(E.p(f"pope_{s}_ext.jsonl"), "w") as f:
            for r in sub:
                f.write(json.dumps(r) + "\n")
        print(s, "ext questions:", len(sub))
    for n in sorted(keep):
        dst = E.p("images", n)
        if os.path.exists(dst):
            continue
        for _ in range(3):
            try:
                open(dst, "wb").write(requests.get(E.COCO_URL.format(n), timeout=60).content); break
            except Exception:
                time.sleep(2)
    print("images on disk:", len(os.listdir(E.p("images"))))


def rows_for(s, suffix="", n_images=None):
    rows = [json.loads(l) for l in open(E.p(f"pope_{s}{suffix}.jsonl"))]
    if n_images:
        order = []
        for l in open(E.p(f"pope_random{suffix}.jsonl")):
            im = json.loads(l)["image"]
            if im not in order:
                order.append(im)
        keep = set(order[:n_images]); rows = [r for r in rows if r["image"] in keep]
    return rows


# =============================================================================== model / caching
def load_model2(model_id, multigpu):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    kw = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if multigpu:                                        # 13B: weights split across both T4s
        kw.update(device_map="auto", max_memory={0: "13GiB", 1: "13GiB"})
    else:
        kw.update(device_map={"": 0})
    return proc, LlavaForConditionalGeneration.from_pretrained(model_id, **kw).eval()


def run_cache2(cond, suffix, outdir, model_id, multigpu, n_images):
    import torch
    from PIL import Image
    proc, model = load_model2(model_id, multigpu); tok = proc.tokenizer
    os.makedirs(E.p(outdir), exist_ok=True)
    for s in SPLITS:
        out = E.p(outdir, f"{cond}_{s}{suffix}.npz")
        if os.path.exists(out):
            print("skip", out); continue
        rows = rows_for(s, suffix, n_images)
        A = {k: [] for k in ["ids", "lo", "ld", "H", "lse", "label", "yes", "image"]}
        t0 = time.time()
        for i, r in enumerate(rows):
            img = E.corrupt(Image.open(E.p("images", r["image"])).convert("RGB"), cond)
            inp = E.build_inputs(proc, model, img, r["text"])
            lo = E.last_logits(model, inp, inp["pixel_values"])
            ld = E.last_logits(model, inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=r["question_id"]))
            top = torch.topk(lo, E.TOPK); ids = top.indices
            A["ids"].append(ids.cpu().numpy()); A["lo"].append(top.values.cpu().numpy()); A["ld"].append(ld[ids].cpu().numpy())
            A["H"].append(E.entropy_nats(lo)); A["lse"].append(float(torch.logsumexp(lo, -1)))
            A["label"].append(r["label"] == "yes"); A["image"].append(r["image"])
            A["yes"].append([not tok.decode([int(t)]).strip().lower().startswith("no") for t in ids])
            if i % 50 == 0:
                print(f"{model_id.split('/')[-1]}|{cond}/{s}{suffix} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        np.savez(out, **{k: np.array(v) for k, v in A.items()})
        print("saved", out, flush=True)


# =============================================================================== end-to-end verification + measured latency
def run_e2e(n_per_split, smoke):
    import torch
    from PIL import Image
    proc, model = E.load_model(); tok = proc.tokenizer
    stride = max(1, 900 // n_per_split)
    qs = []
    for s in SPLITS:
        qs += rows_for(s)[::stride][:n_per_split]
    yes_cache = {}
    def is_yes(t):
        if t not in yes_cache:
            yes_cache[t] = not tok.decode([int(t)]).strip().lower().startswith("no")
        return yes_cache[t]
    prep = [(E.build_inputs(proc, model, Image.open(E.p("images", r["image"])).convert("RGB"), r["text"]), r["question_id"],
             r["label"] == "yes", float(np.random.RandomState(r["question_id"]).rand())) for r in qs]
    print(f"e2e on {len(prep)} questions (stride {stride})", flush=True)

    def fin(l, mode, u):                                  # greedy, or inverse-CDF sample over the finite (head) tokens
        if mode == "greedy":
            return int(l.argmax())
        head = torch.nonzero(torch.isfinite(l)).squeeze(-1)
        c = torch.cumsum(torch.softmax(l[head], -1), 0)
        return int(head[min(int((c < u).sum()), len(head) - 1)])

    def second_pass(inp, qid, lo):
        ld = E.last_logits(model, inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=qid))
        return E.cd_logits(lo, ld)

    def m_base(inp, qid, u, mode):
        lo = E.last_logits(model, inp, inp["pixel_values"]); return fin(lo, mode, u), 1

    def m_vcd(inp, qid, u, mode):
        lo = E.last_logits(model, inp, inp["pixel_values"]); return fin(second_pass(inp, qid, lo), mode, u), 2

    def m_evcd(inp, qid, u, mode):                        # exact gate: |V_head| == 1  <=>  l_(2) < l_(1) + ln(beta)
        lo = E.last_logits(model, inp, inp["pixel_values"])
        t2 = torch.topk(lo, 2).values
        if bool(t2[1] < t2[0] + LOGB):
            return int(lo.argmax()), 1
        return fin(second_pass(inp, qid, lo), mode, u), 2

    def make_egp(tau):                                    # exact gate, then entropy gate with APC-truncated fallback
        def m(inp, qid, u, mode):
            lo = E.last_logits(model, inp, inp["pixel_values"])
            t2 = torch.topk(lo, 2).values; cut = t2[0] + LOGB
            if bool(t2[1] < cut):
                return int(lo.argmax()), 1
            if E.entropy_nats(lo) <= tau:
                return fin(lo.masked_fill(lo < cut, float("-inf")), mode, u), 1
            return fin(second_pass(inp, qid, lo), mode, u), 2
        return m

    def run(fn, mode):
        for inp, qid, y, u in prep[:3]:
            fn(inp, qid, u, mode)
        torch.cuda.synchronize(); toks, ts, ps = [], [], []
        for inp, qid, y, u in prep:
            t0 = time.perf_counter(); tk, k = fn(inp, qid, u, mode); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000); toks.append(tk); ps.append(k)
        return toks, ts, ps

    labels = [y for _, _, y, _ in prep]
    acc = lambda toks: float(np.mean([is_yes(t) == y for t, y in zip(toks, labels)]) * 100)
    res, T = {}, {}
    plan = [("base", m_base, ["greedy", "sample"]), ("vcd", m_vcd, ["greedy", "sample"]), ("e_vcd", m_evcd, ["greedy", "sample"]),
            ("eg+_0.4", make_egp(0.4), ["greedy"]), ("eg+_0.5", make_egp(0.5), ["greedy", "sample"])]
    for name, fn, modes in plan:
        for mode in modes:
            toks, ts, ps = run(fn, mode); T[(name, mode)] = toks
            res[f"{name}|{mode}"] = dict(ms_mean=float(np.mean(ts)), ms_std=float(np.std(ts)), passes=float(np.mean(ps)), acc=acc(toks), n=len(toks))
            print(f"{name:8}{mode:7} {res[f'{name}|{mode}']}", flush=True)
    for mode in ["greedy", "sample"]:
        mism = int(sum(a != b for a, b in zip(T[("vcd", mode)], T[("e_vcd", mode)])))
        res[f"mismatch_vcd_vs_evcd|{mode}"] = mism
        print(f"OUTPUT MISMATCHES  VCD vs exact-gated VCD ({mode}): {mism} / {len(prep)}", flush=True)
    json.dump(res, open(E.p("e2e.json" if not smoke else "e2e_smoke.json"), "w"), indent=1)
    if smoke and (res["mismatch_vcd_vs_evcd|greedy"] or res["mismatch_vcd_vs_evcd|sample"]):
        raise SystemExit("SMOKE TEST FAILED: exact gate changed an output - stop and send me this log")


# =============================================================================== offline analysis
def _sm(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def probs2(c, alpha=1.0, beta=0.1):
    lo, ld, yes = c["lo"], c["ld"], c["yes"]; N = len(lo); ar = np.arange(N)
    head = lo >= lo[:, :1] + math.log(beta)
    assert head.sum(1).max() < lo.shape[1], "V_head not contained in cached top-K"
    cd = np.where(head, (1 + alpha) * lo - alpha * ld, -np.inf)
    return dict(p_base=(_sm(lo) * yes).sum(1), p_vcd=(_sm(cd) * yes).sum(1), p_apc=(_sm(np.where(head, lo, -np.inf)) * yes).sum(1),
                top=yes[:, 0].astype(float), S=~head[:, 1], H=c["H"], gb=yes[:, 0], gv=yes[ar, cd.argmax(1)])


def load_set(cache_dir, cond, suffix="", n_images=None):
    cs, ims = [], []
    for s in SPLITS:
        f = E.p(cache_dir, f"{cond}_{s}{suffix}.npz")
        if not os.path.exists(f):
            return None
        c = dict(np.load(f))
        im = c["image"] if "image" in c else np.array([r["image"] for r in rows_for(s, suffix, n_images)])
        assert len(im) == len(c["H"]), (f, len(im), len(c["H"]))
        cs.append(c); ims.append(im)
    pooled = {k: np.concatenate([c[k] for c in cs]) for k in ["lo", "ld", "H", "yes", "label"]}
    return pooled, np.concatenate(ims)


def summarize(c, imgs, taus=(0.4, 0.5), B=1000, seed=0):
    """All headline quantities with IMAGE-CLUSTERED bootstrap CIs (questions of one image are correlated)."""
    y = c["label"]; P = probs2(c); N = len(y)
    uim, inv = np.unique(imgs, return_inverse=True); ni = len(uim)
    idx = np.random.RandomState(seed).randint(0, ni, (B, ni))
    isum = lambda x: np.bincount(inv, weights=np.asarray(x, float), minlength=ni)
    cnt = isum(np.ones(N)); neg = isum(~y)
    cb = np.where(y, P["p_base"], 1 - P["p_base"]); cv = np.where(y, P["p_vcd"], 1 - P["p_vcd"])
    pa = np.where(P["S"], P["top"], P["p_apc"]); ca = np.where(y, pa, 1 - pa)
    def ratio(num, den):                                  # point estimate and 95% CI of sum(num)/sum(den), resampling images
        pt = num.sum() / den.sum(); b = num[idx].sum(1) / den[idx].sum(1); return float(pt), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out = dict(n_q=int(N), n_img=int(ni), rho_exact=float((~P["S"]).mean()))
    out["exact_maxdiff"] = float(np.abs(np.where(~P["S"], P["p_vcd"], P["top"]) - P["p_vcd"]).max())
    out["greedy_exact_identical"] = bool((np.where(P["S"], P["gb"], P["gv"]) == P["gv"]).all())
    for nm, arr in [("base", cb), ("vcd", cv), ("apc", ca)]:
        out[f"acc_{nm}"] = float(arr.mean() * 100)
    for nm, d in [("sample_gain", cv - cb), ("apc_gain", ca - cb), ("greedy_gain", (P["gv"] == y).astype(float) - (P["gb"] == y).astype(float))]:
        pt, ci = ratio(isum(d), cnt); out[nm] = pt * 100; out[nm + "_ci"] = [ci[0] * 100, ci[1] * 100]
    for mode, pb_, pv_ in [("sample", P["p_base"], P["p_vcd"]), ("greedy", P["gb"].astype(float), P["gv"].astype(float))]:
        fb, _ = ratio(isum(pb_ * ~y), neg); fv, _ = ratio(isum(pv_ * ~y), neg)
        pt, ci = ratio(isum((pv_ - pb_) * ~y), neg)
        out[f"fyr_{mode}"] = dict(base=fb * 100, vcd=fv * 100, diff=pt * 100, diff_ci=[ci[0] * 100, ci[1] * 100])
    gv_ = isum(cv - cb)
    for t in taus:
        gate = (~P["S"]) & (P["H"] > t); p = np.where(gate, P["p_vcd"], np.where(P["S"], P["top"], P["p_apc"])); ce = np.where(y, p, 1 - p)
        ge, gs = isum(ce - cb), isum(gate)
        ret = ge[idx].sum(1) / gv_[idx].sum(1); rho_b = gs[idx].sum(1) / cnt[idx].sum(1); ex = ret - rho_b
        out[f"eg+_{t}"] = dict(rho=float(gate.mean()), retention=float(ge.sum() / gv_.sum()), excess=float(ge.sum() / gv_.sum() - gate.mean()),
                               excess_ci=[float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))])
    return out


def run_analyze():
    N, lines = {}, []
    def say(s=""):
        print(s); lines.append(s)
    sets = [("D1: 150 images (original, 7B)", "cache", "", None), ("D2: 150 NEW images (held-out, 7B)", "cache", "_ext", None),
            ("D3: 100 images, LLaVA-1.5-13B", "cache_13b", "", 100)]
    say("=== A. REPLICATION: clean images, image-clustered 95% CIs (thresholds tau=0.4/0.5 fixed beforehand)")
    for name, d, suf, nimg in sets:
        r = load_set(d, "clean", suf, nimg)
        if r is None:
            say(f"{name}: not available"); continue
        s = summarize(*r); N[name] = s
        say(f"\n{name}   [{s['n_q']} questions, {s['n_img']} images]")
        say(f"  exact gate: second pass needed for {s['rho_exact']*100:.1f}% of questions ({(1-s['rho_exact'])*100:.1f}% skippable); max|p_E-VCD - p_VCD| = {s['exact_maxdiff']:.1e}; greedy identical: {s['greedy_exact_identical']}")
        say(f"  accuracy (expected, sampled): base {s['acc_base']:.2f} | APC-only {s['acc_apc']:.2f} | VCD {s['acc_vcd']:.2f}")
        say(f"  VCD gain sampled {s['sample_gain']:+.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}] | APC-only gain {s['apc_gain']:+.2f} | greedy gain {s['greedy_gain']:+.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]")
        for m in ["sample", "greedy"]:
            f = s[f"fyr_{m}"]; say(f"  false-yes rate ({m}): base {f['base']:.1f}% -> VCD {f['vcd']:.1f}%  (diff {f['diff']:+.1f} [{f['diff_ci'][0]:+.1f},{f['diff_ci'][1]:+.1f}])")
        for t in (0.4, 0.5):
            g = s[f"eg+_{t}"]; say(f"  EG+ tau={t}: second pass for {g['rho']*100:.0f}% | keeps {g['retention']*100:.1f}% of VCD gain | excess over random gating {g['excess']:+.2f} [{g['excess_ci'][0]:+.2f},{g['excess_ci'][1]:+.2f}]")
    say("\n=== B. ROBUSTNESS to VCD hyper-parameters (D1, offline): exactness must hold for every alpha, beta")
    say("cond      alpha beta | skippable% | max|dp| | VCD acc sampled / greedy | base acc sampled / greedy | VCD false-yes sampled")
    sweep = {}
    for cond in CONDS:
        r = load_set("cache", cond)
        if r is None:
            continue
        c, _ = r; y = c["label"]
        for a in (0.5, 1.0, 2.0):
            for b in (0.05, 0.1, 0.2, 0.5):
                P = probs2(c, a, b); acc = lambda p: float(np.where(y, p, 1 - p).mean() * 100)
                row = dict(skip=float(P["S"].mean() * 100), maxdiff=float(np.abs(np.where(~P["S"], P["p_vcd"], P["top"]) - P["p_vcd"]).max()),
                           vcd_s=acc(P["p_vcd"]), vcd_g=float((P["gv"] == y).mean() * 100), base_s=acc(P["p_base"]), base_g=float((P["gb"] == y).mean() * 100),
                           fyr=float(P["p_vcd"][~y].mean() * 100))
                sweep[f"{cond}|{a}|{b}"] = row
                if cond in ("clean", "blur"):
                    say(f"{cond:9} {a:4} {b:5} | {row['skip']:9.1f} | {row['maxdiff']:.0e} | {row['vcd_s']:.2f} / {row['vcd_g']:.2f} | {row['base_s']:.2f} / {row['base_g']:.2f} | {row['fyr']:.1f}")
    N["sweep"] = sweep
    if sweep:
        say(f"max |p_E-VCD - p_VCD| over all {len(sweep)} (condition, alpha, beta) settings: {max(v['maxdiff'] for v in sweep.values()):.1e}")
    say("\n=== C. REAL END-TO-END RUN (LLaVA-1.5-7B, same noise seeds and sampling randomness, measured latency)")
    if os.path.exists(E.p("e2e.json")):
        e = json.load(open(E.p("e2e.json"))); N["e2e"] = e
        say("method    mode     latency ms (mean±sd)  passes  accuracy% (n)")
        for k, v in e.items():
            if "|" in k and not k.startswith("mismatch"):
                nm, mode = k.split("|"); say(f"{nm:9} {mode:7}  {v['ms_mean']:7.0f} ± {v['ms_std']:4.0f}        {v['passes']:.2f}    {v['acc']:.1f} ({v['n']})")
        for mode in ("greedy", "sample"):
            say(f"output mismatches, VCD vs exact-gated VCD ({mode}): {e[f'mismatch_vcd_vs_evcd|{mode}']} of {e['vcd|greedy']['n']}")
        say(f"MEASURED latency saving of the exact gate vs VCD (greedy run): {(1 - e['e_vcd|greedy']['ms_mean'] / e['vcd|greedy']['ms_mean'])*100:.1f}%  (base = {e['base|greedy']['ms_mean']:.0f} ms, VCD = {e['vcd|greedy']['ms_mean']:.0f} ms, exact gate = {e['e_vcd|greedy']['ms_mean']:.0f} ms)")
    else:
        say("e2e.json not found")
    json.dump(N, open(E.p("solid_numbers.json"), "w"), indent=1, default=float)
    open(E.p("solid_report.txt"), "w").write("\n".join(lines))
    say("\nwrote solid_numbers.json and solid_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare_ext", "cache2", "e2e", "analyze"])
    ap.add_argument("--k_start", type=int, default=150); ap.add_argument("--k_end", type=int, default=300)
    ap.add_argument("--cond", default="clean", choices=CONDS); ap.add_argument("--suffix", default="")
    ap.add_argument("--outdir", default="cache"); ap.add_argument("--model", default=E.MODEL_ID)
    ap.add_argument("--multigpu", action="store_true"); ap.add_argument("--n_images", type=int, default=0)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(E.ROOT, exist_ok=True)
    if a.cmd == "prepare_ext": prepare_ext(a.k_start, a.k_end)
    elif a.cmd == "cache2": run_cache2(a.cond, a.suffix, a.outdir, a.model, a.multigpu, a.n_images or None)
    elif a.cmd == "e2e": run_e2e(a.n, a.smoke)
    else: run_analyze()
'''
MODERN_SRC = r'''"""
egvcd_modern.py -- does the exact gate still help on MODERN (2025-2026) vision-language models?

For each model it measures, on POPE-COCO (same first-N images as the main experiments, clean images):
  (1) cache   : the answer-token logits with the original image and with the VCD-noised image, then, offline,
                the share of second passes that are provably redundant (exact gate), VCD's gain, entropy-gate
                retention, plus the mean top-1 probability (how confident the model is)
  (2) gen     : the share of DECODING STEPS in free-form generation (greedy, 'Describe this image in detail')
                at which the plausibility head has a single token, i.e. where a contrastive negative pass
                would be provably wasted.  One pass only (no negative branch is run): this measures the
                opportunity along the base model's own trajectory; a real VCD run would drift slightly.

  python egvcd_modern.py run --model qwen3vl_4b --n_images 100       # smoke test -> cache -> gen
  python egvcd_modern.py analyze_modern                              # table over every finished model

Everything is guarded: a model that cannot run in fp16/bf16 on a T4, or whose template/processor differs,
fails its own smoke test with a clear message and is skipped; the others continue.
"""
import os, sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E
import egvcd_solid as S

SPLITS = E.SPLITS
LOGB = math.log(E.BETA)
OUTROOT = "cache_modern"

MODELS = {
    # key: id, number of GPUs needed in fp16, release note
    "qwen3vl_4b":    dict(id="Qwen/Qwen3-VL-4B-Instruct", gpus=1, note="Qwen3-VL, Oct 2025"),
    "qwen35_4b":     dict(id="Qwen/Qwen3.5-4B", gpus=1, note="Qwen3.5 (native multimodal, early fusion), Feb 2026"),
    "gemma4_e4b":    dict(id="google/gemma-4-E4B-it", gpus=2, note="Gemma 4, Apr 2026"),
    "internvl35_8b": dict(id="OpenGVLab/InternVL3_5-8B-HF", gpus=2, note="InternVL3.5, Aug 2025"),
    "qwen3vl_8b":    dict(id="Qwen/Qwen3-VL-8B-Instruct", gpus=2, note="Qwen3-VL, Oct 2025"),
}


def p_out(key, *a):
    return E.p(OUTROOT, key, *a)


# ----------------------------------------------------------------------------------------------- model plumbing
def load(key, dtype):
    import torch, transformers
    from transformers import AutoProcessor
    cfg = MODELS[key]; proc = AutoProcessor.from_pretrained(cfg["id"])
    classes = [getattr(transformers, n) for n in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq")
               if hasattr(transformers, n)]
    ngpu = torch.cuda.device_count()
    if cfg["gpus"] >= 2 and ngpu >= 2:
        place = dict(device_map="auto", max_memory={0: "13GiB", 1: "13GiB"})
    else:
        place = dict(device_map={"": 0})
    dt = getattr(torch, dtype); last = None
    for AM in classes:
        for kwname in ("dtype", "torch_dtype"):
            try:
                m = AM.from_pretrained(cfg["id"], **{kwname: dt}, low_cpu_mem_usage=True, **place).eval()
                print(f"loaded {cfg['id']} with {AM.__name__}({kwname}={dtype})", flush=True)
                return proc, m
            except Exception as e:                                  # wrong auto-class or kw name: try the next combination
                last = e
    raise RuntimeError(f"could not load {cfg['id']}: {last}")


def _template(proc, msgs):
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build(proc, model, img, text_q):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_q}]}]
    text = _template(proc, msgs); inp = None; err = None
    for im in ([img], img, [[img]]):
        try:
            inp = proc(images=im, text=[text], return_tensors="pt"); break
        except Exception as e:
            err = e
    if inp is None:
        raise RuntimeError(f"processor call failed: {err}")
    out = {}
    for k, v in inp.items():
        v = v.to(model.device)
        out[k] = v.to(model.dtype) if (k == "pixel_values" and v.is_floating_point()) else v
    if "pixel_values" not in out:
        raise RuntimeError("processor returned no 'pixel_values' (keys: %s) - this architecture is not supported by the noising code" % list(out))
    return out


def pope_q(q):
    return f"{q} Answer the question using a single word or phrase."


def fwd(model, inp, pv):
    import torch
    kw = {k: v for k, v in inp.items() if k != "pixel_values"}
    with torch.no_grad():
        try:
            out = model(**kw, pixel_values=pv, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = model(**kw, pixel_values=pv, use_cache=False)
    return out.logits[0, -1].float()


def smoke(proc, model):
    import torch
    from PIL import Image
    tok = proc.tokenizer; ok = 0
    for r in S.rows_for("random")[:4]:
        inp = build(proc, model, Image.open(E.p("images", r["image"])).convert("RGB"), pope_q(r["text"]))
        lo = fwd(model, inp, inp["pixel_values"])
        ld = fwd(model, inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=r["question_id"]))
        if not (bool(torch.isfinite(lo).all()) and bool(torch.isfinite(ld).all())):
            return False, "non-finite logits"
        top = [tok.decode([int(i)]).strip() for i in torch.topk(lo, 3).indices]
        ok += int(top[0].lower() in ("yes", "no")); print(f"  smoke: Q='{r['text']}' label={r['label']} top3={top}", flush=True)
    return (ok >= 3), ("top token is not Yes/No (%d/4): template or thinking mode problem" % ok if ok < 3 else "ok")


# ----------------------------------------------------------------------------------------------- stage 1: POPE logit cache
def cache_model(key, proc, model, n_images):
    import torch
    from PIL import Image
    tok = proc.tokenizer; os.makedirs(p_out(key), exist_ok=True); yes_cache = {}
    def is_yes(t):
        if t not in yes_cache:
            yes_cache[t] = not tok.decode([int(t)]).strip().lower().startswith("no")
        return yes_cache[t]
    for s in SPLITS:
        out = p_out(key, f"clean_{s}.npz")
        if os.path.exists(out):
            print("skip", out, flush=True); continue
        rows = S.rows_for(s, "", n_images); t0 = time.time()
        A = {k: [] for k in ["ids", "lo", "ld", "H", "lse", "label", "yes", "image"]}
        for i, r in enumerate(rows):
            inp = build(proc, model, Image.open(E.p("images", r["image"])).convert("RGB"), pope_q(r["text"]))
            lo = fwd(model, inp, inp["pixel_values"])
            ld = fwd(model, inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=r["question_id"]))
            assert torch.isfinite(lo).all() and torch.isfinite(ld).all(), "non-finite logits"
            top = torch.topk(lo, E.TOPK); ids = top.indices
            A["ids"].append(ids.cpu().numpy()); A["lo"].append(top.values.cpu().numpy()); A["ld"].append(ld[ids].cpu().numpy())
            A["H"].append(E.entropy_nats(lo)); A["lse"].append(float(torch.logsumexp(lo, -1)))
            A["label"].append(r["label"] == "yes"); A["image"].append(r["image"]); A["yes"].append([is_yes(t) for t in ids.tolist()])
            if i % 100 == 0:
                el = time.time() - t0
                print(f"{key}|{s} {i}/{len(rows)}  {el:.0f}s  (~{el / max(i, 1) * (len(rows) - i) / 60:.0f} min left for this split)", flush=True)
        np.savez(out, **{k: np.array(v) for k, v in A.items()}); print("saved", out, flush=True)


# ----------------------------------------------------------------------------------------------- stage 2: free-form generation
def singleton_steps(logits):
    """logits: (T, V). True where the runner-up logit is below top + ln(beta) => VCD head has ONE token (exact gate fires)."""
    import torch
    v = torch.topk(logits.float(), 2, dim=-1).values
    return (v[:, 1] < v[:, 0] + LOGB)


def gen_gate(key, proc, model, n_img, max_new):
    import torch
    from PIL import Image
    seen, imgs = set(), []
    for r in S.rows_for("random"):
        if r["image"] not in seen:
            seen.add(r["image"]); imgs.append(r["image"])
    imgs = imgs[:n_img]; res = []; t0 = time.time()
    for i, n in enumerate(imgs):
        inp = build(proc, model, Image.open(E.p("images", n)).convert("RGB"), "Describe this image in detail.")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, output_logits=True, return_dict_in_generate=True)
        lg = torch.stack([l[0] for l in out.logits], 0)
        sk = singleton_steps(lg).cpu().numpy()
        res.append(dict(image=n, steps=int(len(sk)), single=int(sk.sum())))
        if i % 10 == 0:
            print(f"{key}|gen {i}/{len(imgs)}  {time.time() - t0:.0f}s  singleton-head steps so far: "
                  f"{sum(x['single'] for x in res) / max(1, sum(x['steps'] for x in res)) * 100:.1f}%", flush=True)
    json.dump(res, open(p_out(key, "gen.json"), "w"))
    print("saved", p_out(key, "gen.json"), flush=True)


def run(key, n_images, n_gen, max_new, skip_gen=False):
    import torch, gc
    last = "?"
    for dtype in ("float16", "bfloat16"):
        proc, model = load(key, dtype)
        okk, msg = smoke(proc, model)
        if okk:
            break
        last = msg
        print(f"{key}: smoke test failed in {dtype} ({msg})", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
        if "Yes/No" in msg:                     # a prompt problem is not fixed by another dtype
            raise SystemExit(f"SKIPPED {key}: {msg}")
    else:
        raise SystemExit(f"SKIPPED {key}: {last}")
    print(f"{key}: smoke test passed in {dtype}", flush=True)
    os.makedirs(p_out(key), exist_ok=True)
    json.dump(dict(dtype=dtype, id=MODELS[key]["id"], note=MODELS[key]["note"]), open(p_out(key, "meta.json"), "w"))
    cache_model(key, proc, model, n_images)
    if not skip_gen and n_gen:
        try:
            gen_gate(key, proc, model, n_gen, max_new)
        except Exception as e:
            print(f"{key}: generation stage failed ({type(e).__name__}: {e}) - POPE results are still saved", flush=True)


# ----------------------------------------------------------------------------------------------- analysis
REF = [("LLaVA-1.5-7B (2023)", 63.9), ("LLaVA-NeXT-Mistral-7B (2024)", 76.0), ("Qwen2-VL-7B (2024)", 57.7)]


def load_model_set(key):
    per = {}
    for s in SPLITS:
        f = p_out(key, f"clean_{s}.npz")
        if not os.path.exists(f):
            return None
        per[s] = dict(np.load(f))
    c = {k: np.concatenate([per[s][k] for s in SPLITS]) for k in ["lo", "ld", "H", "yes", "label", "lse"]}
    return c, np.concatenate([per[s]["image"] for s in SPLITS])


def gen_summary(key, B=2000, seed=0):
    f = p_out(key, "gen.json")
    if not os.path.exists(f):
        return None
    g = json.load(open(f)); st = np.array([x["steps"] for x in g], float); sg = np.array([x["single"] for x in g], float)
    idx = np.random.RandomState(seed).randint(0, len(g), (B, len(g)))
    b = sg[idx].sum(1) / st[idx].sum(1)
    return dict(rate=float(sg.sum() / st.sum() * 100), ci=[float(np.percentile(b, 2.5) * 100), float(np.percentile(b, 97.5) * 100)], n_img=len(g), steps=int(st.sum()))


def analyze_modern():
    lines, N = [], {}
    def say(t=""):
        print(t); lines.append(t)
    say("=== MODERN MODELS: exact gate on POPE-COCO (clean images; image-clustered 95% CIs)")
    say("model                       n_q   skippable%  [95% CI]        mean top-1 p | VCD sampled gain [CI]   | greedy gain [CI]      | EG+.5 rho/GR | passes saved%")
    for key, cfg in MODELS.items():
        r = load_model_set(key)
        if r is None:
            continue
        c, im = r; s = S.summarize(c, im, B=1000)
        P = S.probs2(c); skf = P["S"].astype(float)
        uim, inv = np.unique(im, return_inverse=True); ni = len(uim)
        num = np.bincount(inv, weights=skf, minlength=ni); den = np.bincount(inv, minlength=ni).astype(float)
        bi = np.random.RandomState(0).randint(0, ni, (2000, ni)); bb = num[bi].sum(1) / den[bi].sum(1)
        ci = [float(np.percentile(bb, 2.5) * 100), float(np.percentile(bb, 97.5) * 100)]
        top1 = float(np.exp(c["lo"][:, 0] - c["lse"]).mean())
        g = s["eg+_0.5"]
        N[key] = dict(summary=s, skip=float(skf.mean() * 100), skip_ci=ci, top1=top1, gen=gen_summary(key), meta=json.load(open(p_out(key, "meta.json"))) if os.path.exists(p_out(key, "meta.json")) else {})
        say(f"{key:22} {s['n_q']:6d}   {skf.mean() * 100:6.1f}   [{ci[0]:.1f},{ci[1]:.1f}]    {top1:.3f}      | {s['sample_gain']:+.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}] | "
            f"{s['greedy_gain']:+.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}] | {g['rho'] * 100:.0f}% / {g['retention'] * 100:.0f}%  | {(1 - s['rho_exact']) / 2 * 100:.1f}")
        say(f"   exact check: max|p_E-VCD - p_VCD| = {s['exact_maxdiff']:.1e}; greedy identical: {s['greedy_exact_identical']}; accuracy base {s['acc_base']:.1f} / VCD {s['acc_vcd']:.1f}; "
            f"false-yes {s['fyr_sample']['base']:.1f}% -> {s['fyr_sample']['vcd']:.1f}%   [{cfg['id']}, {N[key]['meta'].get('dtype', '?')}]")
    say("\nearlier models, same protocol (from the main paper):  " + "; ".join(f"{n}: {v:.1f}%" for n, v in REF))
    say("\n=== FREE-FORM GENERATION: share of decoding steps where the VCD head has a single token (a negative pass there is provably wasted)")
    say("  measured along the base model's own greedy trajectory, one pass only; 'Describe this image in detail.'")
    for key in MODELS:
        if key in N and N[key]["gen"]:
            g = N[key]["gen"]
            say(f"{key:22} {g['rate']:5.1f}% of {g['steps']} steps  [95% CI {g['ci'][0]:.1f}, {g['ci'][1]:.1f}]  ({g['n_img']} images)")
    json.dump(N, open(E.p("modern_numbers.json"), "w"), indent=1, default=float)
    open(E.p("modern_report.txt"), "w").write("\n".join(lines))
    say("\nwrote modern_numbers.json and modern_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze_modern"])
    ap.add_argument("--model", default="qwen3vl_4b", choices=list(MODELS))
    ap.add_argument("--n_images", type=int, default=100); ap.add_argument("--n_gen", type=int, default=60)
    ap.add_argument("--max_new", type=int, default=96); ap.add_argument("--skip_gen", action="store_true")
    a = ap.parse_args(); os.makedirs(E.ROOT, exist_ok=True)
    if a.cmd == "run":
        run(a.model, a.n_images, a.n_gen, a.max_new, a.skip_gen)
    else:
        analyze_modern()
'''

open(S1, "w").write(EGVCD_SRC); open(S2, "w").write(SOLID_SRC); open(S3, "w").write(MODERN_SRC)
PY = sys.executable

say("STEP 0  checking GPUs")
try:
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
except FileNotFoundError:
    gpus = ""
print(gpus or "no nvidia-smi output")
n_gpus = int(os.environ.get("N_GPUS", gpus.count("GPU ")))
if n_gpus == 0:
    raise SystemExit("No GPU found. Notebook settings -> Accelerator -> GPU T4 x2, then re-run.")
if os.environ.get("SKIP_INSTALL") != "1":
    say("STEP 0  installing the newest transformers (new model types need it), 1-3 min")
    if TRANSFORMERS_MAIN == "1":
        run([PY, "-m", "pip", "install", "-q", "-U", "git+https://github.com/huggingface/transformers", "accelerate", "requests", "scipy", "huggingface_hub", "pillow"])
    else:
        run([PY, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "requests", "scipy", "huggingface_hub", "pillow"])
    run([PY, "-c", "import transformers; print('transformers', transformers.__version__)"])

say(f"STEP 1  data: {N_IMAGES} COCO images and their POPE questions (~1 min)")
run([PY, "-u", S1, "prepare", "--k_images", N_IMAGES])

stage1 = [m for m in MODELS_STAGE1.split(",") if m]
stage2 = [m for m in MODELS_STAGE2.split(",") if m]
skipped = []

def job(model, gpu_env):
    return [PY, "-u", S3, "run", "--model", model, "--n_images", N_IMAGES, "--n_gen", N_GEN]

# ---- STAGE 1: one 4B-class model per GPU, in parallel ------------------------------------------------------
if stage1:
    say(f"STAGE 1  {stage1}: one model per T4, in parallel (~30 min)")
    sel = selectors.DefaultSelector(); procs = {}
    for k, m in enumerate(stage1[:max(1, n_gpus)]):
        p = subprocess.Popen(job(m, k), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**ENV, "CUDA_VISIBLE_DEVICES": str(k)})
        sel.register(p.stdout, selectors.EVENT_READ, (m, p)); procs[m] = p
        time.sleep(45)                                   # stagger loads (CPU RAM)
    alive = set(procs)
    while alive:
        for key, _ in sel.select(timeout=2):
            m, p = key.data; line = p.stdout.readline()
            if line:
                print(f"[{m}] {line.rstrip()}", flush=True); continue
            sel.unregister(p.stdout); rc = p.wait(); alive.discard(m)
            if rc != 0:
                skipped.append(m); say(f"[{m}] did NOT finish (exit {rc}) - see the message above; continuing with the others")
            else:
                say(f"[{m}] finished")
    for m in stage1[max(1, n_gpus):]:                    # only one GPU: run the rest one by one
        if not run(job(m, 0), fatal=False): skipped.append(m)
    run([PY, "-u", S3, "analyze_modern"], fatal=False)

# ---- STAGE 2: 8B-class models, weights split across both GPUs ----------------------------------------------
for m in stage2:
    if n_gpus < 2:
        say(f"STAGE 2  {m} needs both T4s - skipped (only {n_gpus} GPU)"); skipped.append(m); continue
    say(f"STAGE 2  {m}: weights split across both T4s (~30-40 min incl. download)")
    if not run(job(m, "0,1"), env={**ENV, "CUDA_VISIBLE_DEVICES": "0,1"}, fatal=False):
        skipped.append(m); say(f"[{m}] did NOT finish - see the message above; continuing")
    run([PY, "-u", S3, "analyze_modern"], fatal=False)

say("FINAL ANALYSIS")
run([PY, "-u", S3, "analyze_modern"], fatal=False)
if skipped:
    print("\nSKIPPED / FAILED MODELS:", skipped, "\n(send me the lines that start with SKIPPED or the error text - some new architectures need TRANSFORMERS_MAIN=\"1\")")
say("packaging")
run(["bash", "-c", f"cd {WORK} && zip -qr modern_results.zip egvcd_data/cache_modern egvcd_data/modern_numbers.json egvcd_data/modern_report.txt egvcd.py egvcd_solid.py egvcd_modern.py"], fatal=False)
say(f"ALL DONE.  Send me the full report printed above (or {ROOT}/modern_report.txt) + {WORK}/modern_results.zip  (Output tab)")
