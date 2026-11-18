# =====================================================================================
#  EG-VCD  --  "REAL-DOMAIN (OOD) + 2nd NEW BACKBONE" ONE-CELL KAGGLE RUNNER   (GPU T4 x2, Internet ON)
#
#  WHY: closes the two biggest gaps in the proposal: (1) a REAL out-of-distribution domain
#       (satellite / aerial photos instead of COCO), (2) a second new architecture (LLaVA-NeXT).
#
#  OPTIONAL (recommended, 1 click): Add Input -> attach your earlier egvcd_results.zip / solid_results.zip / more_results.zip
#  so the final table can compare the new rows with the earlier ones.  The run still works without them.
#
#  Stage A  (~5 min)    get a real-domain dataset automatically (tries Hugging Face DIOR copies, or any dataset you attach)
#  Stage B  (~40-50 min) POPE-style questions on it, LLaVA-1.5-7B, VCD vs exact gate vs entropy gate (offline analysis)
#  Stage C  (~50-70 min) LLaVA-NeXT-Mistral-7B on COCO POPE, 100 images (set RUN_NEXT = "0" to skip)
#  Everything is printed live and saved after every stage -> you can stop whenever time runs out.
# =====================================================================================
OOD_HF_ID = ""            # optional: a Hugging Face dataset id with images + object classes (tried first)
OOD_N_SCAN = "4000"       # how many dataset images to scan for ones with >=3 different object classes
OOD_N_IMAGES = "100"      # images used for the OOD test
RUN_NEXT = "1"            # "0" skips the LLaVA-NeXT stage
RUN_OOD = "0"             # "0" skips the real-domain stage (use RUN_OOD="0" RUN_NEXT="1" on a second Kaggle account)
OOD_COCO_JSON = ""        # optional manual override (COCO-format json) ...
OOD_IMAGES_DIR = ""       # ... and its image folder

import os, sys, time, subprocess, selectors, glob, shutil

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
ROOT = os.path.join(WORK, "egvcd_data")
NAMES = ("egvcd.py", "egvcd_solid.py", "egvcd_more.py", "egvcd_quality.py")
S1, S2, S3, S4 = [os.path.join(WORK, n) for n in NAMES]
S5 = os.path.join(WORK, "fetch_ood.py")
SCR = {"S": S2, "M": S3, "Q": S4}
os.makedirs(ROOT, exist_ok=True)
OOD_HF_ID, OOD_N_SCAN, OOD_N_IMAGES, RUN_NEXT, RUN_OOD, OOD_COCO_JSON, OOD_IMAGES_DIR = [os.environ.get(k, v) for k, v in (
    ("OOD_HF_ID", OOD_HF_ID), ("OOD_N_SCAN", OOD_N_SCAN), ("OOD_N_IMAGES", OOD_N_IMAGES), ("RUN_NEXT", RUN_NEXT), ("RUN_OOD", RUN_OOD), ("OOD_COCO_JSON", OOD_COCO_JSON), ("OOD_IMAGES_DIR", OOD_IMAGES_DIR))]
ENV = {**os.environ, "EGVCD_ROOT": ROOT, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
T0, PY = time.time(), sys.executable

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

SRC1 = r'''"""
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
SRC2 = r'''"""
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
SRC3 = r'''"""
egvcd_more.py -- experiments that test whether the exact gate holds beyond POPE / LLaVA-1.5.

  gen          OPEN-ENDED generation (multi-token captions, LLaVA-1.5-7B): full VCD vs exact-gated VCD with lazy
               catch-up of the negative branch; token agreement, steps skippable, measured wall-clock
  cache_arch   POPE logit cache for a DIFFERENT architecture (qwen2vl | llavanext), weights split over 2 GPUs
  smoke_arch   quick sanity check (finite logits, yes/no answers) before the long architecture run
  noise        re-compute the negative-branch logits with another noise seed / noise step (pass 2 only)
  analyze2     offline consistency report over every model / dataset / noise setting
The decoding bookkeeping (decode()) is written against a tiny interface so it is unit-tested on a toy model (test_more.py).
"""
import os, sys, json, math, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E
import egvcd_solid as S

SPLITS, CONDS = E.SPLITS, E.CONDS
LOGB = math.log(E.BETA)
ARCH = {
    "qwen2vl": dict(id="Qwen/Qwen2-VL-7B-Instruct", proc_kw=dict(min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28), maxmem="12GiB"),
    "llavanext": dict(id="llava-hf/llava-v1.6-mistral-7b-hf", proc_kw={}, maxmem="13GiB"),
}


# =============================================================================== decoding core (framework independent)
def decode(method, mode, ops, prompt_ids, pos, mk_neg, qid, max_new, eos):
    """method: base | vcd (eager negative branch, the reference) | evcd (exact gate, lazy negative branch).
    pos / neg objects offer prefill(ids)->logits and extend(ids)->logits for the LAST position (cache semantics).
    One uniform u is drawn per step for every method, so sampled runs share their randomness."""
    rng = np.random.RandomState(qid)
    gen, st = [], dict(steps=0, single=0, neg_calls=0, margins=[])
    lo = pos.prefill(prompt_ids)
    neg, ld, pending = None, None, []
    if method == "vcd":
        neg = mk_neg(); ld = neg.prefill(prompt_ids)
    for t in range(max_new):
        u = float(rng.rand())
        if method == "base":
            single = None; tok = ops.fin(lo, mode, u)
        else:
            single = ops.single(lo)
            if method == "vcd":
                cd = ops.cd(lo, ld); tok = ops.fin(cd, mode, u); st["margins"].append(ops.margin(cd))
            elif single:                                   # exact gate: V_head has one token -> contrast cannot change it
                tok = ops.argmax(lo)
            else:
                if neg is None:                            # first time the contrast is needed: prefill with prompt + tokens so far
                    neg = mk_neg(); ld = neg.prefill(list(prompt_ids) + gen)
                else:                                      # catch up on the tokens the negative branch has not seen yet
                    ld = neg.extend(pending)
                pending = []; st["neg_calls"] += 1
                tok = ops.fin(ops.cd(lo, ld), mode, u)
        st["steps"] += 1; st["single"] += int(bool(single)) if single is not None else 0
        gen.append(int(tok))
        if tok == eos:
            break
        lo = pos.extend([int(tok)])
        if method == "vcd":
            ld = neg.extend([int(tok)])
        elif method == "evcd":
            pending.append(int(tok))
    return gen, st


class TorchOps:
    def __init__(self):
        import torch; self.t = torch
    def single(self, lo):
        t2 = self.t.topk(lo, 2).values; return bool(t2[1] < t2[0] + LOGB)
    def cd(self, lo, ld): return E.cd_logits(lo, ld)
    def argmax(self, l): return int(l.argmax())
    def margin(self, l):
        t2 = self.t.topk(l, 2).values; return float(t2[0] - t2[1])
    def fin(self, l, mode, u):
        if mode == "greedy":
            return int(l.argmax())
        head = self.t.nonzero(self.t.isfinite(l)).squeeze(-1)
        c = self.t.cumsum(self.t.softmax(l[head], -1), 0)
        return int(head[min(int((c < u).sum()), len(head) - 1)])


class HFBranch:
    """One image branch with its own KV cache."""
    def __init__(self, model, pv):
        import torch; self.t = torch; self.model, self.pv, self.cache, self.n = model, pv, None, 0
    def _fwd(self, ids, pv):
        t = self.t; x = t.tensor([ids], device=self.model.device)
        with t.no_grad():
            out = self.model(input_ids=x, attention_mask=t.ones(1, self.n + len(ids), dtype=t.long, device=self.model.device),
                             pixel_values=pv, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values; self.n += len(ids)
        return out.logits[0, -1].float()
    def prefill(self, ids):
        self.cache, self.n = None, 0; return self._fwd(list(ids), self.pv)
    def extend(self, ids):
        return self._fwd(list(ids), None)


# =============================================================================== generation experiment
def run_gen(n_img, max_new, smoke):
    import torch
    from PIL import Image
    proc, model = E.load_model(); tok = proc.tokenizer; eos = tok.eos_token_id; ops = TorchOps()
    imgs, seen = [], set()
    for r in S.rows_for("random"):
        if r["image"] not in seen:
            seen.add(r["image"]); imgs.append(r["image"])
    imgs = imgs[:n_img]
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    prep = []
    for i, name in enumerate(imgs):
        inp = proc(images=Image.open(E.p("images", name)).convert("RGB"), text=prompt, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        prep.append((name, i + 1, inp))

    def run_one(method, mode, inp, qid):
        pv = inp["pixel_values"]; ids = inp["input_ids"][0].tolist()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gen, st = decode(method, mode, ops, ids, HFBranch(model, pv),
                         lambda: HFBranch(model, E.diffusion_noise(pv, E.NOISE_STEP, seed=qid)), qid, max_new, eos)
        torch.cuda.synchronize(); return gen, st, (time.perf_counter() - t0) * 1000

    if smoke:                                             # does my manual decoding loop reproduce HF generate()?
        name, qid, inp = prep[0]
        g, _, _ = run_one("base", "greedy", inp, qid)
        with torch.no_grad():
            ref = model.generate(**inp, max_new_tokens=len(g), do_sample=False)[0, inp["input_ids"].shape[1]:].tolist()
        k = min(len(g), len(ref)); agree = sum(a == b for a, b in zip(g[:k], ref[:k]))
        print(f"smoke: manual loop vs HF generate: {agree}/{k} tokens identical", flush=True)
        print("smoke caption:", tok.decode(g, skip_special_tokens=True)[:200], flush=True)
        if k == 0 or agree < 0.9 * k:
            raise SystemExit("SMOKE TEST FAILED: manual decoding disagrees with HF generate - stop and send me this log")
        prep = prep[:2]
    plan = [("base", "greedy"), ("vcd", "greedy"), ("evcd", "greedy"), ("vcd", "sample"), ("evcd", "sample")]
    R = {f"{m}|{mo}": [] for m, mo in plan}; texts = {}
    for name, qid, inp in prep[:3]:                       # warm-up (not recorded)
        run_one("vcd", "greedy", inp, qid); break
    for j, (name, qid, inp) in enumerate(prep):
        for m, mo in plan:
            gen, st, ms = run_one(m, mo, inp, qid)
            R[f"{m}|{mo}"].append(dict(gen=gen, ms=ms, **{k: v for k, v in st.items()}))
            if j < 3 and mo == "greedy":
                texts[f"{m}|{name}"] = tok.decode(gen, skip_special_tokens=True)
        print(f"caption {j+1}/{len(prep)} done  ({len(R['vcd|greedy'][-1]['gen'])} tokens)", flush=True)
    out = dict(n=len(prep), max_new=max_new, methods={}, pairs={}, texts=texts)
    for k, v in R.items():
        steps = sum(x["steps"] for x in v)
        out["methods"][k] = dict(ms_mean=float(np.mean([x["ms"] for x in v])), ms_std=float(np.std([x["ms"] for x in v])),
                                 tokens_mean=float(np.mean([x["steps"] for x in v])), ms_per_token=float(sum(x["ms"] for x in v) / steps),
                                 frac_single=float(sum(x["single"] for x in v) / steps) if k.split("|")[0] != "base" else None,
                                 frac_neg_calls=float(sum(x["neg_calls"] for x in v) / steps) if k.startswith("evcd") else None)
    for mo in ("greedy", "sample"):
        a, b = R[f"vcd|{mo}"], R[f"evcd|{mo}"]; ident, first_div, margins, agree_tok, tot_tok = 0, [], [], 0, 0
        for x, y in zip(a, b):
            k = min(len(x["gen"]), len(y["gen"])); d = next((i for i in range(k) if x["gen"][i] != y["gen"][i]), None)
            ident += int(d is None and len(x["gen"]) == len(y["gen"])); tot_tok += len(x["gen"])
            agree_tok += (d if d is not None else k)
            if d is not None:
                first_div.append(d); margins.append(x["margins"][d])
        out["pairs"][mo] = dict(identical_sequences=ident, n=len(a), tokens_before_first_divergence=agree_tok, total_tokens=tot_tok,
                                first_divergence_steps=first_div, vcd_margin_at_divergence=margins)
        print(f"VCD vs exact-gated VCD ({mo}): identical sequences {ident}/{len(a)}; divergences at near-tie steps: {margins}", flush=True)
    json.dump(out, open(E.p("gen.json" if not smoke else "gen_smoke.json"), "w"), indent=1)


# =============================================================================== second architecture
def load_arch(arch, dtype="float16"):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AM
    cfg = ARCH[arch]; proc = AutoProcessor.from_pretrained(cfg["id"], **cfg["proc_kw"])
    model = AM.from_pretrained(cfg["id"], torch_dtype=getattr(torch, dtype), device_map="auto",
                               max_memory={0: cfg["maxmem"], 1: cfg["maxmem"]}, low_cpu_mem_usage=True).eval()
    return proc, model


def build_arch(proc, model, img, q):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{q} Answer the question using a single word or phrase."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(images=[img], text=[text], return_tensors="pt")
    inp = {k: v.to(model.device) for k, v in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    return inp


def fwd_arch(model, inp, pv):
    import torch
    kw = {k: v for k, v in inp.items() if k != "pixel_values"}
    with torch.no_grad():
        return model(**kw, pixel_values=pv, use_cache=False).logits[0, -1].float()


def smoke_arch(arch):
    import torch
    from PIL import Image
    proc, model = load_arch(arch); tok = proc.tokenizer; ok = 0
    rows = S.rows_for("random")[:4]
    for r in rows:
        inp = build_arch(proc, model, Image.open(E.p("images", r["image"])).convert("RGB"), r["text"])
        lo = fwd_arch(model, inp, inp["pixel_values"])
        if not bool(torch.isfinite(lo).all()):
            raise SystemExit(f"{arch}: non-finite logits in fp16 - this architecture cannot run on a T4 in fp16; skip it")
        top = [tok.decode([int(i)]).strip() for i in torch.topk(lo, 3).indices]
        ok += int(top[0].lower() in ("yes", "no")); print(f"{arch} smoke: Q='{r['text']}' label={r['label']} top3={top}", flush=True)
    if ok < 3:
        raise SystemExit(f"{arch}: top token is not Yes/No for most questions ({ok}/4) - prompt/format problem, skip this architecture")


def run_cache_arch(arch, cond, n_images, outdir):
    import torch
    from PIL import Image
    proc, model = load_arch(arch); tok = proc.tokenizer; os.makedirs(E.p(outdir), exist_ok=True)
    for s in SPLITS:
        out = E.p(outdir, f"{cond}_{s}.npz")
        if os.path.exists(out):
            print("skip", out); continue
        rows = S.rows_for(s, "", n_images); A = {k: [] for k in ["ids", "lo", "ld", "H", "lse", "label", "yes", "image"]}; t0 = time.time()
        for i, r in enumerate(rows):
            inp = build_arch(proc, model, E.corrupt(Image.open(E.p("images", r["image"])).convert("RGB"), cond), r["text"])
            lo = fwd_arch(model, inp, inp["pixel_values"])
            ld = fwd_arch(model, inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=r["question_id"]))
            top = torch.topk(lo, E.TOPK); ids = top.indices
            A["ids"].append(ids.cpu().numpy()); A["lo"].append(top.values.cpu().numpy()); A["ld"].append(ld[ids].cpu().numpy())
            A["H"].append(E.entropy_nats(lo)); A["lse"].append(float(torch.logsumexp(lo, -1)))
            A["label"].append(r["label"] == "yes"); A["image"].append(r["image"])
            A["yes"].append([not tok.decode([int(t)]).strip().lower().startswith("no") for t in ids])
            if i % 50 == 0:
                print(f"{arch}|{cond}/{s} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        np.savez(out, **{k: np.array(v) for k, v in A.items()}); print("saved", out, flush=True)


# =============================================================================== noise seed / noise step robustness (pass 2 only)
def run_noise(T, seed_idx, cond="clean"):
    import torch
    from PIL import Image
    proc, model = E.load_model(); os.makedirs(E.p("cache_noise"), exist_ok=True)
    for s in SPLITS:
        out = E.p("cache_noise", f"{cond}_{s}_T{T}_s{seed_idx}.npz")
        if os.path.exists(out):
            print("skip", out); continue
        c = dict(np.load(E.p("cache", f"{cond}_{s}.npz"))); rows = S.rows_for(s); ld_new = []; t0 = time.time()
        assert len(rows) == len(c["H"])
        for i, r in enumerate(rows):
            inp = E.build_inputs(proc, model, E.corrupt(Image.open(E.p("images", r["image"])).convert("RGB"), cond), r["text"])
            ld = E.last_logits(model, inp, E.diffusion_noise(inp["pixel_values"], T, seed=r["question_id"] + 100000 * seed_idx))
            ld_new.append(ld[torch.from_numpy(c["ids"][i]).to(ld.device)].cpu().numpy())
            if i % 100 == 0:
                print(f"noise T={T} seed={seed_idx} {s} {i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        c["ld"] = np.array(ld_new); c["image"] = np.array([r["image"] for r in rows]); np.savez(out, **c); print("saved", out, flush=True)


# =============================================================================== offline consistency report
def _line(name, s):
    return (f"{name:34}{s['n_q']:>6}{s['rho_exact']*100:>10.1f}{100-s['rho_exact']*100:>7.1f}  {s['sample_gain']:+5.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}]"
            f"  {s['greedy_gain']:+5.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]"
            f"  {s['apc_gain']/s['sample_gain']*100:>4.0f}%  {s['fyr_sample']['base']:.1f}->{s['fyr_sample']['vcd']:.1f}  {s['fyr_greedy']['base']:.1f}->{s['fyr_greedy']['vcd']:.1f}"
            f"  {s['eg+_0.5']['rho']*100:>3.0f}% {s['eg+_0.5']['retention']*100:>5.1f}% {s['eg+_0.5']['excess']:+.2f}")


def run_analyze2():
    N, lines = {}, []
    def say(s=""):
        print(s); lines.append(s)
    sets = [("LLaVA-1.5-7B, 150 imgs (D1)", "cache", "", None), ("LLaVA-1.5-7B, 150 new imgs (D2)", "cache", "_ext", None),
            ("LLaVA-1.5-13B, 100 imgs", "cache_13b", "", 100), ("Qwen2-VL-7B, 100 imgs", "cache_qwen2vl", "", 100),
            ("LLaVA-NeXT-Mistral-7B, 100 imgs", "cache_llavanext", "", 100)]
    say("=== 1. CONSISTENCY ACROSS MODELS AND IMAGE SETS (clean, POPE, image-clustered 95% CIs)")
    say(f"{'setting':34}{'n_q':>6}{'needs2nd%':>10}{'skip%':>7}  {'VCD gain sampled':>22}  {'VCD gain greedy':>22}  {'APC%':>5}  FYR smp      FYR grd     EG+.5 rho  keep   excess")
    rows = []
    for name, d, suf, nimg in sets:
        r = S.load_set(d, "clean", suf, nimg)
        if r is None:
            continue
        s = S.summarize(*r); N[name] = s; rows.append(s); say(_line(name, s))
    if rows:
        sk = [100 - s["rho_exact"] * 100 for s in rows]
        say(f"skippable share across {len(rows)} settings: min {min(sk):.1f}%  max {max(sk):.1f}%  | exactness violated in any setting: {any(s['exact_maxdiff'] > 0 or not s['greedy_exact_identical'] for s in rows)}")
        say(f"EG+ (tau=0.5) retains {min(s['eg+_0.5']['retention'] for s in rows)*100:.1f}-{max(s['eg+_0.5']['retention'] for s in rows)*100:.1f}% of the VCD gain; lower CI bound of excess over random gating: {min(s['eg+_0.5']['excess_ci'][0] for s in rows):+.2f}")
    say("\n=== 2. ROBUSTNESS TO THE NOISE DRAW AND NOISE STEP (LLaVA-1.5-7B, clean, D1; pass 2 recomputed)")
    say(f"{'noise setting':22}{'skip%':>7}  {'VCD gain sampled [CI]':>26}  {'VCD gain greedy [CI]':>24}  FYR sampled")
    ref = S.load_set("cache", "clean")
    cfg = [("T=999 seed 0 (original)", ref)]
    for T, sd in [(999, 1), (999, 2), (500, 0), (250, 0)]:
        cs, ims, ok = [], [], True
        for s in SPLITS:
            f = E.p("cache_noise", f"clean_{s}_T{T}_s{sd}.npz")
            if not os.path.exists(f):
                ok = False; break
            c = dict(np.load(f)); cs.append(c); ims.append(c["image"])
        if ok:
            cfg.append((f"T={T} seed {sd}", ({k: np.concatenate([c[k] for c in cs]) for k in ["lo", "ld", "H", "yes", "label"]}, np.concatenate(ims))))
    gains = {}
    for name, r in cfg:
        if r is None:
            continue
        s = S.summarize(*r); N[name] = s; gains[name] = (s["sample_gain"], s["greedy_gain"])
        say(f"{name:22}{100-s['rho_exact']*100:>7.1f}  {s['sample_gain']:+6.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}]   {s['greedy_gain']:+6.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]   {s['fyr_sample']['base']:.1f}->{s['fyr_sample']['vcd']:.1f}")
    seeds = [v[0] for k, v in gains.items() if k.startswith("T=999")]
    if len(seeds) > 1:
        say(f"sampled VCD gain over {len(seeds)} noise draws at T=999: mean {np.mean(seeds):+.2f}, sd {np.std(seeds, ddof=1):.2f}")
    say("(the skippable share is identical in every row by construction: it depends on pass 1 only, never on the noise)")
    say("\n=== 3. OPEN-ENDED GENERATION (LLaVA-1.5-7B, captions, measured)")
    if os.path.exists(E.p("gen.json")):
        g = json.load(open(E.p("gen.json"))); N["gen"] = g
        say(f"{g['n']} captions, up to {g['max_new']} new tokens each")
        say(f"{'method|mode':16}{'ms/caption':>12}{'tokens':>8}{'ms/token':>10}{'steps w/ |V_head|=1':>21}{'neg-branch calls':>18}")
        pc = lambda x: "" if x is None else f"{x * 100:.1f}%"
        for k, v in g["methods"].items():
            say(f"{k:16}{v['ms_mean']:>9.0f}±{v['ms_std']:<4.0f}{v['tokens_mean']:>7.1f}{v['ms_per_token']:>10.1f}{pc(v['frac_single']):>21}{pc(v['frac_neg_calls']):>18}")
        for mo in ("greedy", "sample"):
            p = g["pairs"][mo]; say(f"VCD vs exact-gated VCD ({mo}): {p['identical_sequences']}/{p['n']} sequences identical; {p['tokens_before_first_divergence']}/{p['total_tokens']} tokens agree before the first divergence"
                                  + (f"; margins at divergence (top-2 contrast-logit gap): {[round(x, 3) for x in p['vcd_margin_at_divergence']]}" if p["vcd_margin_at_divergence"] else ""))
        m = g["methods"]; say(f"MEASURED caption latency saving of the exact gate vs VCD (greedy): {(1 - m['evcd|greedy']['ms_mean'] / m['vcd|greedy']['ms_mean'])*100:.1f}%   (base {m['base|greedy']['ms_mean']:.0f} ms, VCD {m['vcd|greedy']['ms_mean']:.0f} ms, exact gate {m['evcd|greedy']['ms_mean']:.0f} ms)")
    else:
        say("gen.json not found")
    json.dump(N, open(E.p("more_numbers.json"), "w"), indent=1, default=float)
    open(E.p("more_report.txt"), "w").write("\n".join(lines)); say("\nwrote more_numbers.json and more_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "cache_arch", "smoke_arch", "noise", "analyze2"])
    ap.add_argument("--n_img", type=int, default=60); ap.add_argument("--max_new", type=int, default=96); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arch", default="qwen2vl", choices=list(ARCH)); ap.add_argument("--cond", default="clean", choices=CONDS)
    ap.add_argument("--n_images", type=int, default=100); ap.add_argument("--outdir", default="cache_qwen2vl")
    ap.add_argument("--T", type=int, default=999); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); os.makedirs(E.ROOT, exist_ok=True)
    if a.cmd == "gen": run_gen(a.n_img, a.max_new, a.smoke)
    elif a.cmd == "cache_arch": run_cache_arch(a.arch, a.cond, a.n_images, a.outdir)
    elif a.cmd == "smoke_arch": smoke_arch(a.arch)
    elif a.cmd == "noise": run_noise(a.T, a.seed, a.cond)
    else: run_analyze2()
'''
SRC4 = r'''"""
egvcd_quality.py -- what is still unchecked after egvcd_solid.py / egvcd_more.py.

  prepare_chair  COCO annotations + official CHAIR synonym list + images for the caption-quality experiment
  chair_gen      generate captions (base / VCD / exact-gated VCD / entropy+exact gated) in greedy and sampled mode, one shard per GPU
  chair_eval     CHAIR_s / CHAIR_i / coverage with image-bootstrap CIs, paired comparisons (CPU)
  numerics       WHY do exact-gated and eager VCD occasionally diverge in captions? eager vs catch-up negative-branch logits
  make_ood       build POPE-style questions from ANY COCO-format detection dataset you bring (satellite, documents, ...)
  analyze3       offline report: consistency across all caches, alpha/beta sweep on every dataset, degradation per model

CHAIR (Rohrbach et al., EMNLP 2018) is ported from the authors' public chair.py: same synonym list, double-word and
animal/vehicle rules, ground truth = COCO instance categories + objects named in the 5 reference captions.
Deviation: the pattern.en singulariser is replaced by a rule-based, vocabulary-aware singulariser (no extra installs).
"""
import os, sys, json, math, time, re, argparse, shutil, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E
import egvcd_solid as S
import egvcd_more as M

LOGB = math.log(E.BETA)
SYN_URL = "https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
TAU = 0.5


# =============================================================================== decoding with the entropy gate (extends M.decode)
def decode2(method, mode, ops, prompt_ids, pos, mk_neg, qid, max_new, eos, tau=TAU):
    """method: base | vcd | evcd | egp.  egp = exact gate, then entropy gate (H <= tau -> APC-truncated base, no 2nd pass)."""
    if method in ("base", "vcd", "evcd"):
        return M.decode(method, mode, ops, prompt_ids, pos, mk_neg, qid, max_new, eos)
    rng = np.random.RandomState(qid)
    gen, st = [], dict(steps=0, single=0, neg_calls=0, entropy_skips=0, margins=[])
    lo = pos.prefill(prompt_ids); neg, ld, pending = None, None, []
    for t in range(max_new):
        u = float(rng.rand()); single = ops.single(lo)
        if single:
            tok = ops.argmax(lo)
        elif ops.entropy(lo) <= tau:
            tok = ops.fin(ops.apc(lo), mode, u); st["entropy_skips"] += 1
        else:
            if neg is None:
                neg = mk_neg(); ld = neg.prefill(list(prompt_ids) + gen)
            else:
                ld = neg.extend(pending)
            pending = []; st["neg_calls"] += 1
            tok = ops.fin(ops.cd(lo, ld), mode, u)
        st["steps"] += 1; st["single"] += int(single); gen.append(int(tok))
        if tok == eos:
            break
        lo = pos.extend([int(tok)]); pending.append(int(tok))
    return gen, st


class TorchOps2(M.TorchOps):
    def entropy(self, lo): return E.entropy_nats(lo)
    def apc(self, lo):
        t2 = self.t.topk(lo, 2).values; return lo.masked_fill(lo < t2[0] + LOGB, float("-inf"))


# =============================================================================== CHAIR (port)
DOUBLE = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign',
          'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
          'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear',
          'home plate', 'train track']
IRREG = {"men": "man", "women": "woman", "children": "child", "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth",
         "knives": "knife", "leaves": "leaf", "shelves": "shelf", "loaves": "loaf", "wolves": "wolf", "scarves": "scarf"}


class Chair:
    def __init__(self, syn_path):
        syn = [s.strip().split(', ') for s in open(syn_path).readlines()]      # exactly as in the original (double spaces stay un-matchable)
        self.objs, self.inv = [], {}
        for line in syn:
            self.objs.extend(line)
            for s in line:
                self.inv[s] = line[0]
        self.objs = set(self.objs)
        self.dd = {d: d for d in DOUBLE}
        for a in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']:
            self.dd[f'baby {a}'] = a; self.dd[f'adult {a}'] = a
        for v in ['jet', 'train']:
            self.dd[f'passenger {v}'] = v
        self.dd['bow tie'] = 'tie'; self.dd['toilet seat'] = 'toilet'; self.dd['wine glas'] = 'wine glass'
        self.vocab = set(self.objs) | {w for d in DOUBLE for w in d.split()}

    def singular(self, w):
        if w in IRREG:
            return IRREG[w]
        if len(w) < 4 or not w.endswith("s") or w.endswith(("ss", "us", "is")):
            return w
        cands = ([w[:-3] + "y"] if w.endswith("ies") else []) + ([w[:-2]] if w.endswith("es") else []) + [w[:-1]]
        for c in cands:
            if c in self.vocab:
                return c
        return w[:-2] if w.endswith(("ches", "shes", "xes", "zes", "oes")) else w[:-1]

    def words(self, caption):
        w = [self.singular(x) for x in re.findall(r"[a-z]+", caption.lower())]
        out, i = [], 0
        while i < len(w):
            pair = " ".join(w[i:i + 2])
            if pair in self.dd:
                out.append(self.dd[pair]); i += 2
            else:
                out.append(w[i]); i += 1
        if "toilet" in out and "seat" in out:
            out = [x for x in out if x != "seat"]
        keep = [x for x in out if x in self.objs]
        return keep, [self.inv[x] for x in keep]

    def gt_objects(self, instance_names, captions):
        s = {self.inv.get(n, n) for n in instance_names}
        for c in captions:
            s.update(self.words(c)[1])
        return s


# =============================================================================== data preparation
def prepare_chair(n):
    import requests
    os.makedirs(E.p("images"), exist_ok=True)
    names, seen = [], set()
    for l in requests.get(E.POPE_URL.format("random"), timeout=60).text.splitlines():
        if l.strip():
            im = json.loads(l)["image"]
            if im not in seen:
                seen.add(im); names.append(im)
    names = names[:n]
    for nm in names:
        dst = E.p("images", nm)
        if not os.path.exists(dst):
            for _ in range(3):
                try:
                    open(dst, "wb").write(requests.get(E.COCO_URL.format(nm), timeout=60).content); break
                except Exception:
                    time.sleep(2)
    open(E.p("synonyms.txt"), "w").write(requests.get(SYN_URL, timeout=60).text)
    if not os.path.exists(E.p("chair_gt.json")):
        zp = E.p("annotations_trainval2014.zip"); print("downloading COCO annotations (~240 MB) ...", flush=True)
        with requests.get(ANN_URL, stream=True, timeout=120) as r, open(zp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(zp) as z:
            ins = json.load(z.open("annotations/instances_val2014.json")); caps = json.load(z.open("annotations/captions_val2014.json"))
        ids = {int(re.search(r"(\d+)\.jpg", nm).group(1)): nm for nm in names}
        cat = {c["id"]: c["name"] for c in ins["categories"]}; gt = {nm: {"objects": [], "captions": []} for nm in names}
        for a in ins["annotations"]:
            if a["image_id"] in ids:
                gt[ids[a["image_id"]]]["objects"].append(cat[a["category_id"]])
        for a in caps["annotations"]:
            if a["image_id"] in ids:
                gt[ids[a["image_id"]]]["captions"].append(a["caption"])
        for v in gt.values():
            v["objects"] = sorted(set(v["objects"]))
        json.dump(gt, open(E.p("chair_gt.json"), "w")); os.remove(zp)
    json.dump(names, open(E.p("chair_images.json"), "w"))
    print(f"prepared {len(names)} images, synonyms, ground truth", flush=True)


# =============================================================================== caption generation
def chair_gen(shard, nshards, n, modes, max_new, smoke):
    import torch
    from PIL import Image
    proc, model = E.load_model(); tok = proc.tokenizer; eos = tok.eos_token_id; ops = TorchOps2()
    names = json.load(open(E.p("chair_images.json")))[:n]
    mine = names[shard::nshards][:2] if smoke else names[shard::nshards]
    out = E.p(f"chair_shard{shard}{'_smoke' if smoke else ''}.jsonl"); done = set()
    if os.path.exists(out):
        done = {json.loads(l)["image"] for l in open(out)}
    plan = [(m, mo) for mo in modes for m in ("base", "vcd", "evcd", "egp")]
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    t0 = time.time()
    for k, name in enumerate(mine):
        if name in done:
            continue
        qid = int(re.search(r"(\d+)\.jpg", name).group(1))
        inp = proc(images=Image.open(E.p("images", name)).convert("RGB"), text=prompt, return_tensors="pt")
        inp = {a: b.to(model.device) for a, b in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        pv = inp["pixel_values"]; ids = inp["input_ids"][0].tolist(); rows = []
        for m, mo in plan:
            torch.cuda.synchronize(); t1 = time.perf_counter()
            gen, st = decode2(m, mo, ops, ids, M.HFBranch(model, pv), lambda: M.HFBranch(model, E.diffusion_noise(pv, E.NOISE_STEP, seed=qid)), qid, max_new, eos)
            torch.cuda.synchronize()
            rows.append(dict(image=name, method=m, mode=mo, caption=tok.decode(gen, skip_special_tokens=True), tokens=gen, ms=(time.perf_counter() - t1) * 1000,
                             steps=st["steps"], single=st["single"], neg_calls=st["neg_calls"], entropy_skips=st.get("entropy_skips", 0)))
        with open(out, "a") as f:                                  # one line per (image, method, mode); resumable per image
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"shard {shard}: image {k+1}/{len(mine)} done  ({time.time()-t0:.0f}s)", flush=True)
    if smoke:
        for l in list(open(out))[:8]:
            r = json.loads(l); print(f"  smoke {r['method']:5}{r['mode']:7} {r['ms']:7.0f} ms  neg_calls {r['neg_calls']:3}/{r['steps']:3}  {r['caption'][:90]}", flush=True)


# =============================================================================== CHAIR evaluation
def chair_eval():
    import glob
    chair = Chair(E.p("synonyms.txt")); gt = json.load(open(E.p("chair_gt.json"))); recs = []
    for f in sorted(glob.glob(E.p("chair_shard[0-9].jsonl"))):
        recs += [json.loads(l) for l in open(f)]
    if not recs:
        print("no caption files found"); return {}
    lines, N = [], {}
    def say(s=""):
        print(s); lines.append(s)
    per = {}                                                   # (method, mode) -> image -> dict
    for r in recs:
        ws, nodes = chair.words(r["caption"]); g = chair.gt_objects(gt[r["image"]]["objects"], gt[r["image"]]["captions"])
        hall = [x for x in nodes if x not in g]
        per.setdefault((r["method"], r["mode"]), {})[r["image"]] = dict(m=len(nodes), h=len(hall), cov=(len(set(nodes) & g) / len(g)) if g else 0.0,
                                                                         toks=len(r["tokens"]), ms=r["ms"], cap=r["caption"], hw=hall, single=r["single"], neg=r["neg_calls"], steps=r["steps"])
    common = sorted(set.intersection(*[set(v) for v in per.values()])); n = len(common)
    rng = np.random.RandomState(0); B = 2000; idx = rng.randint(0, n, (B, n))
    def arr(k, key): return np.array([per[k][im][key] for im in common], float)
    say(f"=== CHAIR caption-quality experiment: {n} images with all methods, LLaVA-1.5-7B, prompt 'Describe this image in detail.'")
    say(f"{'method|mode':14}{'CHAIR_s %':>12}{'  95% CI':>16}{'CHAIR_i %':>12}{'  95% CI':>16}{'coverage %':>12}{'tokens':>8}{'ms/caption':>12}")
    stats = {}
    for k in sorted(per, key=lambda x: (x[1], ["base", "vcd", "evcd", "egp"].index(x[0]))):
        h, m, c = arr(k, "h"), arr(k, "m"), arr(k, "cov"); hs = (h > 0).astype(float)
        cs, ci_ = hs.mean() * 100, h.sum() / m.sum() * 100
        bs = hs[idx].mean(1) * 100; bi = h[idx].sum(1) / m[idx].sum(1) * 100
        stats[k] = dict(chair_s=cs, chair_i=ci_, cov=c.mean() * 100, tokens=arr(k, "toks").mean(), ms=arr(k, "ms").mean(), hs=hs, h=h, m=m)
        say(f"{k[0] + '|' + k[1]:14}{cs:>12.1f}{'[%.1f,%.1f]' % tuple(np.percentile(bs, [2.5, 97.5])):>16}{ci_:>12.1f}{'[%.1f,%.1f]' % tuple(np.percentile(bi, [2.5, 97.5])):>16}"
            f"{c.mean()*100:>12.1f}{arr(k, 'toks').mean():>8.1f}{arr(k, 'ms').mean():>12.0f}")
    say("\npaired differences (same images; 95% bootstrap CI over images; negative CHAIR = fewer hallucinations)")
    for mode in sorted({k[1] for k in per}):
        for a, b, lab in [("vcd", "base", "VCD - base"), ("evcd", "vcd", "exact-gated - VCD"), ("egp", "vcd", "entropy+exact gated - VCD"), ("egp", "base", "entropy+exact gated - base")]:
            if (a, mode) not in per or (b, mode) not in per:
                continue
            A, Bq = stats[(a, mode)], stats[(b, mode)]
            ds = (A["hs"] - Bq["hs"]) * 100; dci = (A["h"][idx].sum(1) / A["m"][idx].sum(1) - Bq["h"][idx].sum(1) / Bq["m"][idx].sum(1)) * 100
            say(f"  {mode:7}{lab:30} CHAIR_s {ds.mean():+6.1f} [{np.percentile(ds[idx].mean(1), 2.5):+.1f},{np.percentile(ds[idx].mean(1), 97.5):+.1f}]   CHAIR_i {(A['chair_i']-Bq['chair_i']):+6.2f} [{np.percentile(dci, 2.5):+.2f},{np.percentile(dci, 97.5):+.2f}]")
    for mode in sorted({k[1] for k in per}):
        if ("vcd", mode) in per and ("evcd", mode) in per:
            same = sum(per[("vcd", mode)][im]["cap"] == per[("evcd", mode)][im]["cap"] for im in common)
            say(f"identical captions, VCD vs exact-gated ({mode}): {same}/{n}")
    for mode in sorted({k[1] for k in per}):
        if ("egp", mode) in per:
            e = per[("egp", mode)]; say(f"entropy+exact gate ({mode}): second pass at {sum(e[i]['neg'] for i in common)/sum(e[i]['steps'] for i in common)*100:.1f}% of steps (VCD: 100%, exact gate: {sum(per[('evcd', mode)][i]['neg'] for i in common)/sum(per[('evcd', mode)][i]['steps'] for i in common)*100:.1f}%)")
    ex = [(im, per[("base", "greedy")][im]["hw"], per[("vcd", "greedy")][im]["hw"]) for im in common if ("base", "greedy") in per and per[("base", "greedy")][im]["h"] != per[("vcd", "greedy")][im]["h"]][:3]
    for im, hb, hv in ex:
        say(f"\nexample {im}: hallucinated objects  base {hb}  | VCD {hv}\n   base: {per[('base', 'greedy')][im]['cap'][:230]}\n   VCD : {per[('vcd', 'greedy')][im]['cap'][:230]}")
    N = {f"{k[0]}|{k[1]}": {a: float(b) for a, b in v.items() if a in ("chair_s", "chair_i", "cov", "tokens", "ms")} for k, v in stats.items()}; N["n_images"] = n
    json.dump(N, open(E.p("chair_numbers.json"), "w"), indent=1); open(E.p("chair_report.txt"), "w").write("\n".join(lines)); return N


# =============================================================================== numerics: why do eager and exact-gated VCD sometimes diverge?
def run_numerics(n_caps, max_new):
    import torch
    from PIL import Image
    proc, model = E.load_model(); tok = proc.tokenizer; eos = tok.eos_token_id; ops = TorchOps2()
    names, seen = [], set()
    for r in S.rows_for("random"):
        if r["image"] not in seen:
            seen.add(r["image"]); names.append(r["image"])
    names = names[:n_caps]; prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    maxabs, tv, margins_flip, steps_total, needed, flips = [], [], [], 0, 0, 0
    for k, name in enumerate(names):
        qid = k + 1
        inp = proc(images=Image.open(E.p("images", name)).convert("RGB"), text=prompt, return_tensors="pt")
        inp = {a: b.to(model.device) for a, b in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        pv = inp["pixel_values"]; ids = inp["input_ids"][0].tolist(); pvn = E.diffusion_noise(pv, E.NOISE_STEP, seed=qid)
        gen, _ = M.decode("vcd", "greedy", ops, ids, M.HFBranch(model, pv), lambda: M.HFBranch(model, pvn), qid, max_new, eos); T = len(gen)
        pos = M.HFBranch(model, pv); lo = [pos.prefill(ids)]                   # one KV cache alive at a time (3 at once ran out of memory on a T4)
        for t in range(T - 1):
            lo.append(pos.extend([gen[t]]))
        del pos; torch.cuda.empty_cache()
        eager = M.HFBranch(model, pvn); ld_e = [eager.prefill(ids)]            # token-by-token, as in standard VCD
        for t in range(T - 1):
            ld_e.append(eager.extend([gen[t]]))
        del eager; torch.cuda.empty_cache()
        lazy, pend, ld_l = None, [], {}                                         # lazy prefill + batched catch-up, as in the exact gate
        for t in range(T):
            if not ops.single(lo[t]):
                if lazy is None:
                    lazy = M.HFBranch(model, pvn); ld_l[t] = lazy.prefill(ids + gen[:t])
                else:
                    ld_l[t] = lazy.extend(pend)
                pend = []
            if t < T - 1:
                pend.append(gen[t])
        steps_total += T; needed += len(ld_l)
        for t, l_l in ld_l.items():
            head = lo[t] >= lo[t].max() + LOGB
            maxabs.append(float((l_l - ld_e[t]).abs()[head].max()))
            ce, cl = E.cd_logits(lo[t], ld_e[t]), E.cd_logits(lo[t], l_l)
            pe, pl = torch.softmax(ce[head], -1), torch.softmax(cl[head], -1); tv.append(float(0.5 * (pe - pl).abs().sum()))
            if int(ce.argmax()) != int(cl.argmax()):
                flips += 1; margins_flip.append(float(torch.topk(ce, 2).values.diff().abs()))
        print(f"numerics {k+1}/{len(names)}  steps {T}  contrast steps {len(ld_l)}", flush=True)
        del lazy, lo, ld_e, ld_l, pvn, pv, inp; torch.cuda.empty_cache()
    maxabs, tv = np.array(maxabs), np.array(tv)
    out = dict(captions=len(names), steps=steps_total, contrast_steps=needed, logit_diff_head_tokens=dict(mean=float(maxabs.mean()), median=float(np.median(maxabs)), p99=float(np.percentile(maxabs, 99)), max=float(maxabs.max())),
               tv_distance=dict(mean=float(tv.mean()), sum=float(tv.sum()), max=float(tv.max())), expected_sampled_flips_per_1000_tokens=float(tv.sum() / steps_total * 1000),
               greedy_flips=flips, greedy_flip_margins=margins_flip)
    json.dump(out, open(E.p("numerics.json"), "w"), indent=1); print(json.dumps(out, indent=1), flush=True)


# =============================================================================== bring-your-own real-domain dataset (COCO-format detection json)
def make_ood(coco_json, images_dir, n_images, seed=0, per_split=3):
    d = json.load(open(coco_json)); cat = {c["id"]: c["name"].replace("_", " ").replace("-", " ") for c in d["categories"]}
    img = {i["id"]: i["file_name"] for i in d["images"]}; objs = {}
    for a in d["annotations"]:
        objs.setdefault(a["image_id"], set()).add(cat[a["category_id"]])
    ids = sorted([i for i, s in objs.items() if len(s) >= 3 and i in img]); rng = np.random.RandomState(seed); rng.shuffle(ids); ids = ids[:n_images]
    cats = sorted(cat.values()); freq = {c: sum(c in s for s in objs.values()) for c in cats}
    co = {a: {b: sum(a in s and b in s for s in objs.values()) for b in cats} for a in cats}
    os.makedirs(E.p("images"), exist_ok=True); rows = {s: [] for s in E.SPLITS}; qid = 0
    for i in ids:
        present = sorted(objs[i]); absent = [c for c in cats if c not in objs[i]]
        if len(absent) < per_split:
            continue
        name = img[i]; src = os.path.join(images_dir, name)
        if not os.path.exists(src):
            continue
        shutil.copy(src, E.p("images", name))
        pos = list(rng.choice(present, size=min(per_split, len(present)), replace=False))
        negs = {"random": list(rng.choice(absent, size=per_split, replace=False)),
                "popular": sorted(absent, key=lambda c: -freq[c])[:per_split],
                "adversarial": sorted(absent, key=lambda c: -sum(co[c][p] for p in present))[:per_split]}
        for s in E.SPLITS:
            for label, items in (("yes", pos), ("no", negs[s])):
                for c in items:
                    qid += 1; rows[s].append(dict(question_id=qid, image=name, text=f"Is there a {c} in the image?", label=label))
    used = len({r["image"] for r in rows["random"]})
    print(f"used {used} of the requested {len(ids)} images (images need >=3 object classes, >=3 absent classes, and a file in images_dir)")
    if used < 20:
        print("WARNING: very few images -> the OOD result will be too noisy to report; check --images_dir and the category count")
    for s in E.SPLITS:
        with open(E.p(f"pope_{s}_ood.jsonl"), "w") as f:
            for r in rows[s]:
                f.write(json.dumps(r) + "\n")
        print(s, "OOD questions:", len(rows[s]))
    print("next: python egvcd_solid.py cache2 --cond clean --suffix _ood --outdir cache")


# =============================================================================== offline report
def run_analyze3():
    lines, N = [], {}
    def say(s=""):
        print(s); lines.append(s)
    sets = [("LLaVA-1.5-7B, 150 imgs (D1)", "cache", "", None), ("LLaVA-1.5-7B, 150 new imgs (D2)", "cache", "_ext", None), ("LLaVA-1.5-13B, 100 imgs", "cache_13b", "", 100),
            ("Qwen2-VL-7B, 100 imgs", "cache_qwen2vl", "", 100), ("LLaVA-NeXT-7B, 100 imgs", "cache_llavanext", "", 100), ("LLaVA-1.5-7B, bring-your-own OOD set", "cache", "_ood", None)]
    say("=== 1. CONSISTENCY ACROSS MODELS AND IMAGE SETS (clean, POPE, image-clustered 95% CIs)")
    say(f"{'setting':38}{'n_q':>6}{'needs2nd%':>10}{'skip%':>7}  {'VCD gain sampled':>22}  {'VCD gain greedy':>22}  {'APC%':>5}  FYR smp      FYR grd     EG+.5 rho  keep   excess")
    have, rows = [], []
    for name, d, suf, nimg in sets:
        r = S.load_set(d, "clean", suf, nimg)
        if r is None:
            continue
        s = S.summarize(*r); N[name] = s; rows.append(s); have.append((name, r)); say(M._line(name[:37], s))
    if rows:
        sk = [100 - s["rho_exact"] * 100 for s in rows]
        say(f"skippable share across {len(rows)} settings: min {min(sk):.1f}%  max {max(sk):.1f}% | exactness violated in any: {any(s['exact_maxdiff'] > 0 or not s['greedy_exact_identical'] for s in rows)}")
    say("\n=== 2. alpha / beta SWEEP ON EVERY DATASET (offline). exact = max|p_exact-gated - p_VCD|; accuracies in %, FYR = false-yes rate")
    say(f"{'setting':38}{'alpha':>6}{'beta':>6}{'skip%':>7}{'exact':>7}  {'VCD acc smp/grd':>16}  {'base acc smp/grd':>17}  {'VCD FYR smp':>12}  {'base FYR smp/grd':>17}")
    for name, (c, _) in have:
        y = c["label"]
        for a, b in [(1.0, 0.1), (1.0, 0.2), (1.0, 0.5), (0.5, 0.5)]:
            P = S.probs2(c, a, b); acc = lambda p: float(np.where(y, p, 1 - p).mean() * 100)
            ex = float(np.abs(np.where(~P["S"], P["p_vcd"], P["top"]) - P["p_vcd"]).max())
            say(f"{name[:37]:38}{a:>6}{b:>6}{P['S'].mean()*100:>7.1f}{ex:>7.0e}  {acc(P['p_vcd']):>7.2f}/{(P['gv'] == y).mean()*100:<7.2f}  {acc(P['p_base']):>8.2f}/{(P['gb'] == y).mean()*100:<7.2f}  {P['p_vcd'][~y].mean()*100:>12.1f}  {P['p_base'][~y].mean()*100:>9.1f}/{P['gb'][~y].mean()*100:<6.1f}")
    say("\n=== 3. VCD UNDER IMAGE DEGRADATION, PER MODEL (sampled gain over the sampled baseline, image-clustered 95% CI)")
    say(f"{'model':16}{'condition':10}{'base acc':>9}{'skip%':>7}  {'VCD gain sampled':>22}  {'VCD gain greedy':>22}  {'FYR base->VCD (smp)':>20}")
    for mname, d, nimg in [("LLaVA-1.5-7B", "cache", None), ("Qwen2-VL-7B", "cache_qwen2vl", 100), ("LLaVA-NeXT-7B", "cache_llavanext", 100)]:
        for cond in E.CONDS:
            r = S.load_set(d, cond, "", nimg)
            if r is None:
                continue
            s = S.summarize(*r); say(f"{mname:16}{cond:10}{s['acc_base']:>9.2f}{100-s['rho_exact']*100:>7.1f}  {s['sample_gain']:+6.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}]   {s['greedy_gain']:+6.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]   {s['fyr_sample']['base']:>7.1f} -> {s['fyr_sample']['vcd']:.1f}")
    if os.path.exists(E.p("numerics.json")):
        n = json.load(open(E.p("numerics.json"))); N["numerics"] = n
        say("\n=== 4. NUMERICS: eager token-by-token negative branch vs lazy catch-up negative branch (teacher-forced, same tokens)")
        say(f"{n['captions']} captions, {n['steps']} steps, {n['contrast_steps']} steps needing the contrast")
        d = n["logit_diff_head_tokens"]; say(f"|logit difference| on plausible tokens: mean {d['mean']:.4f}  median {d['median']:.4f}  p99 {d['p99']:.4f}  max {d['max']:.4f}  (fp16 resolution near logit 10 is about 0.008)")
        say(f"total-variation distance between the two contrast distributions: mean {n['tv_distance']['mean']:.5f} per contrast step  -> expected sampled flips {n['expected_sampled_flips_per_1000_tokens']:.2f} per 1000 tokens (observed earlier: 7 flips in ~5600 steps = 1.25 per 1000)")
        say(f"greedy argmax flips: {n['greedy_flips']}  at contrast-logit margins {[round(x, 4) for x in n['greedy_flip_margins']]}")
    json.dump(N, open(E.p("quality_numbers.json"), "w"), indent=1, default=float); open(E.p("quality_report.txt"), "w").write("\n".join(lines)); say("\nwrote quality_numbers.json and quality_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare_chair", "chair_gen", "chair_eval", "numerics", "make_ood", "analyze3"])
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--modes", default="greedy,sample"); ap.add_argument("--max_new", type=int, default=96); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_caps", type=int, default=20); ap.add_argument("--coco_json", default=""); ap.add_argument("--images_dir", default=""); ap.add_argument("--n_images", type=int, default=100)
    a = ap.parse_args(); os.makedirs(E.ROOT, exist_ok=True)
    if a.cmd == "prepare_chair": prepare_chair(a.n)
    elif a.cmd == "chair_gen": chair_gen(a.shard, a.nshards, a.n, a.modes.split(","), a.max_new, a.smoke)
    elif a.cmd == "chair_eval": chair_eval()
    elif a.cmd == "numerics": run_numerics(a.n_caps, a.max_new)
    elif a.cmd == "make_ood": make_ood(a.coco_json, a.images_dir, a.n_images)
    else: run_analyze3()
'''
SRC5 = r'''"""
fetch_ood.py -- get a REAL-domain object dataset and write it as COCO-style json + images.
Order: (0) manual override  (1) anything you attached under /kaggle/input (COCO json or Pascal-VOC xml)
       (2) Hugging Face dataset ids (OOD_HF_ID first, then known DIOR aerial copies).
Only the first N_SCAN images are streamed; we keep images that contain >=3 different object classes.
"""
import os, sys, json, glob, io
import xml.etree.ElementTree as ET

OUT = os.path.join(os.environ.get("WORKDIR", "/kaggle/working"), "ood_src")
IMG = os.path.join(OUT, "images"); os.makedirs(IMG, exist_ok=True)
N_SCAN = int(os.environ.get("OOD_N_SCAN", "4000")); N_KEEP = 300; MINC = 3

def finish(images, anns, cats, source):
    json.dump(dict(images=images, annotations=anns, categories=[dict(id=i, name=n) for i, n in cats.items()]), open(os.path.join(OUT, "coco.json"), "w"))
    print(f"FETCH OK from {source}: {len(images)} images, {len(cats)} classes: {sorted(cats.values())[:25]}", flush=True)
    sys.exit(0)

# ---------------- (1) attached datasets ----------------
def from_input():
    root = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
    if not os.path.isdir(root):
        return
    files = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            files.setdefault(f, os.path.join(dp, f))
    # COCO json
    for p in [v for k, v in files.items() if k.endswith(".json") and os.path.getsize(v) < 400e6]:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
            have = [i for i in d["images"] if i["file_name"].split("/")[-1] in files]
            if len(have) >= 50:
                keep = {}; images = []; anns = []
                for i in have:
                    shutil_copy(files[i["file_name"].split("/")[-1]], os.path.join(IMG, i["file_name"].split("/")[-1])); keep[i["id"]] = 1
                    images.append(dict(id=i["id"], file_name=i["file_name"].split("/")[-1]))
                anns = [dict(image_id=a["image_id"], category_id=a["category_id"]) for a in d["annotations"] if a["image_id"] in keep]
                finish(images, anns, {c["id"]: c["name"] for c in d["categories"]}, f"attached COCO json {p}")
    # Pascal VOC xml
    xmls = [v for k, v in files.items() if k.endswith(".xml")][:60000]
    if len(xmls) >= 50:
        cats = {}; images = []; anns = []; n = 0
        for p in xmls:
            try:
                r = ET.parse(p).getroot()
            except Exception:
                continue
            fn = (r.findtext("filename") or os.path.basename(p)[:-4] + ".jpg").split("/")[-1]
            src = files.get(fn) or next((files[fn[:-4] + e] for e in (".jpg", ".png", ".jpeg") if fn[:-4] + e in files), None)
            names = {o.findtext("name") for o in r.findall("object") if o.findtext("name")}
            if src is None or len(names) < MINC:
                continue
            n += 1; shutil_copy(src, os.path.join(IMG, os.path.basename(src)))
            images.append(dict(id=n, file_name=os.path.basename(src)))
            for nm in names:
                cid = cats.setdefault(nm, len(cats) + 1); anns.append(dict(image_id=n, category_id=cid))
            if n >= N_KEEP:
                break
        if len(images) >= 50:
            finish(images, anns, {v: k for k, v in cats.items()}, "attached Pascal-VOC xml")

def shutil_copy(a, b):
    import shutil
    if not os.path.exists(b):
        shutil.copy(a, b)

# ---------------- (2) Hugging Face ----------------
def from_hf(ds_id):
    from datasets import load_dataset
    ds = None
    for sp in ("train", "test", "validation", "val"):
        try:
            ds = load_dataset(ds_id, split=sp, streaming=True); print(f"  loaded {ds_id} split={sp}", flush=True); break
        except Exception as e:
            print(f"  split {sp}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    if ds is None:
        return False
    feats = getattr(ds, "features", None)
    names = None; img_col = None; obj_col = None
    if feats:
        for k, v in feats.items():
            if v.__class__.__name__ == "Image" and img_col is None:
                img_col = k
        for k in ("objects", "annotations", "labels", "label"):
            if k in feats:
                obj_col = k
        try:
            sub = feats[obj_col]
            for key in ("category", "label", "category_id", "class"):
                f = sub.feature[key] if hasattr(sub, "feature") and isinstance(sub.feature, dict) else None
                if f is not None and hasattr(f, "names"):
                    names = f.names; break
        except Exception:
            pass
    if names is None and ds_id == "Marcel031/dior":   # class ids 0..19 listed in that dataset's README YAML header (alphabetical DIOR order)
        names = ["Airplane", "Airport", "Baseball field", "Basketball court", "Bridge", "Chimney", "Dam", "Expressway service area",
                 "Expressway toll station", "Golf course", "Ground track field", "Harbor", "Overpass", "Ship", "Stadium", "Storage tank",
                 "Tennis court", "Train station", "Vehicle", "Wind mill"]
        print("  using the 20 DIOR class names from the dataset README", flush=True)
    images, anns, cats = [], [], {}
    seen = 0
    for ex in ds:
        seen += 1
        if seen == 1:
            print("  example keys:", {k: type(v).__name__ for k, v in ex.items()}, flush=True)
            if isinstance(ex.get(obj_col or "objects"), dict):
                print("  objects keys / first values:", {k: (v[:3] if isinstance(v, list) else v) for k, v in ex[obj_col or "objects"].items()}, "| class names known:", names is not None, flush=True)
            if img_col is None:
                img_col = next((k for k, v in ex.items() if hasattr(v, "convert")), None)
            if obj_col is None:
                obj_col = next((k for k in ("objects", "annotations", "labels", "label") if k in ex), None)
            if img_col is None or obj_col is None:
                print("  cannot find image/object columns", flush=True); return False
        o = ex[obj_col]; cl = []
        if isinstance(o, dict):
            for key in ("category", "label", "category_id", "class", "categories", "labels"):
                if key in o:
                    cl = o[key]; break
        elif isinstance(o, list):
            for a in o:
                if isinstance(a, dict):
                    for key in ("category_id", "category", "label", "name"):
                        if key in a:
                            cl.append(a[key]); break
        elif isinstance(o, (int, str)):
            cl = [o]
        def nm(c):
            if isinstance(c, str):
                return c
            return names[int(c)] if names is not None and 0 <= int(c) < len(names) else None
        cs = {nm(c) for c in cl if nm(c)}
        if seen % 500 == 0:
            print(f"  scanned {seen}, kept {len(images)}", flush=True)
        if len(cs) >= MINC:
            fn = f"ood_{len(images):04d}.jpg"
            ex[img_col].convert("RGB").save(os.path.join(IMG, fn), quality=92)
            images.append(dict(id=len(images) + 1, file_name=fn))
            for c in cs:
                cid = cats.setdefault(c, len(cats) + 1); anns.append(dict(image_id=len(images), category_id=cid))
        if len(images) >= N_KEEP or seen >= N_SCAN:
            break
    print(f"  {ds_id}: scanned {seen}, kept {len(images)} images with >={MINC} classes", flush=True)
    if len(images) >= 40:
        finish(images, anns, {v: k for k, v in cats.items()}, f"Hugging Face {ds_id}")
    return False

if os.environ.get("OOD_HF_ID") != "SKIPALL":
    print("trying attached Kaggle inputs ...", flush=True)
    from_input()
    cand = [c for c in [os.environ.get("OOD_HF_ID", "")] if c] + ["Marcel031/dior", "HOHKUIK/DIOR"]
    for c in cand:
        print("trying Hugging Face dataset", c, flush=True)
        try:
            from_hf(c)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("FETCH FAILED: no real-domain dataset could be loaded automatically.")
sys.exit(3)
'''

for path, src in zip((S1, S2, S3, S4, S5), (SRC1, SRC2, SRC3, SRC4, SRC5)):
    open(path, "w").write(src)

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
    say("STEP 0  installing libraries")
    run([PY, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "requests", "scipy", "huggingface_hub", "datasets", "pillow"], fatal=False)

def copy_set(pattern, dest, only_parent=None, name_filter=None):
    hits = [h for h in glob.glob(pattern, recursive=True) if (only_parent is None or os.path.basename(os.path.dirname(h)) == only_parent)]
    if not hits:
        return 0
    src = os.path.dirname(hits[0]); os.makedirs(dest, exist_ok=True); n = 0
    for f in glob.glob(os.path.join(src, "*.npz")):
        if name_filter is None or name_filter(os.path.basename(f)):
            shutil.copy(f, os.path.join(dest, os.path.basename(f))); n += 1
    return n
n1 = copy_set("/kaggle/input/**/clean_random.npz", os.path.join(ROOT, "cache"), only_parent="cache", name_filter=lambda b: "_ext" not in b)
n2 = copy_set("/kaggle/input/**/clean_random_ext.npz", os.path.join(ROOT, "cache"), only_parent="cache", name_filter=lambda b: "_ext" in b)
n3 = copy_set("/kaggle/input/**/cache_13b/clean_random.npz", os.path.join(ROOT, "cache_13b"))
n4 = copy_set("/kaggle/input/**/cache_qwen2vl/clean_random.npz", os.path.join(ROOT, "cache_qwen2vl"))
print(f"found earlier results: LLaVA-7B original {n1}/12, held-out {n2}/3, 13B {n3}/3, Qwen2-VL {n4}/3  (optional; only used for the comparison table)")

say("STEP 1  POPE images (150) - needed so the earlier caches can be read")
run([PY, "-u", S1, "prepare", "--k_images", "150"], fatal=False)
if os.environ.get("SKIP_DL") != "1" and RUN_OOD == "1":
    say("STEP 1  downloading llava-1.5-7b-hf (~14 GB)")
    run([PY, "-c", "from huggingface_hub import snapshot_download as s; print('7B at', s('llava-hf/llava-1.5-7b-hf'))"])

# ---- STAGE A: real-domain dataset -------------------------------------------------------------------
say("STAGE A  fetching a real-domain (out-of-distribution) dataset")
ood_ok = False
if RUN_OOD != "1":
    ood_ok = None
    print("real-domain stage skipped (RUN_OOD=0)")
elif OOD_COCO_JSON and OOD_IMAGES_DIR:
    coco_json, images_dir, ood_ok = OOD_COCO_JSON, OOD_IMAGES_DIR, True
else:
    ood_ok = run([PY, "-u", S5], env={**ENV, "WORKDIR": WORK}, fatal=False)
    coco_json, images_dir = os.path.join(WORK, "ood_src", "coco.json"), os.path.join(WORK, "ood_src", "images")
if ood_ok:
    say(f"STAGE A  building POPE-style questions from it ({OOD_N_IMAGES} images, 3 present + 3 absent objects per image per split)")
    run([PY, "-u", S4, "make_ood", "--coco_json", coco_json, "--images_dir", images_dir, "--n_images", OOD_N_IMAGES])
    say("STAGE B  LLaVA-1.5-7B on the real-domain questions (VCD pass + noised pass for each question; ~40 min on one T4)")
    run([PY, "-u", S2, "cache2", "--cond", "clean", "--suffix", "_ood", "--outdir", "cache"], env={**ENV, "CUDA_VISIBLE_DEVICES": "0"})
    say("STAGE B  report (the row 'LLaVA-1.5-7B, bring-your-own OOD set' is the new real-domain result)")
    run([PY, "-u", S4, "analyze3"], fatal=False)
elif ood_ok is False:
    print("\n*** No real-domain dataset could be loaded automatically. ***")
    print("Fix (2 min): in the notebook sidebar click Add Input, search Kaggle for 'DIOR' (or 'DOTA', 'xView', 'aerial object detection'),")
    print("add one with images + annotations (COCO json or Pascal-VOC xml), then re-run this cell. It is auto-detected.")
    print("Meanwhile the LLaVA-NeXT stage below still runs so the session is not wasted.")

def package(tag):
    run(["bash", "-c", f"cd {WORK} && zip -qr ood_results.zip egvcd_data/pope_*_ood.jsonl egvcd_data/quality_report.txt egvcd_data/quality_numbers.json "
                       f"$(ls egvcd_data/cache/*_ood.npz 2>/dev/null) $(ls -d egvcd_data/cache_llavanext 2>/dev/null) ood_src/coco.json egvcd.py egvcd_solid.py egvcd_more.py egvcd_quality.py fetch_ood.py"], fatal=False)
    say(f"saved {WORK}/ood_results.zip  ({tag})")
package("after real-domain stage")

# ---- STAGE C: LLaVA-NeXT ------------------------------------------------------------------------------
if RUN_NEXT == "1" and n_gpus >= 2:
    both = {**ENV, "CUDA_VISIBLE_DEVICES": "0,1"}
    say("STAGE C  LLaVA-NeXT-Mistral-7B on COCO POPE, 100 images, weights split across both T4s (download ~15 GB, then ~50 min)")
    run([PY, "-u", S3, "cache_arch", "--arch", "llavanext", "--cond", "clean", "--n_images", "100", "--outdir", "cache_llavanext"], env=both, fatal=False)
    say("STAGE C  final report")
    run([PY, "-u", S4, "analyze3"], fatal=False)
    package("final")
elif RUN_NEXT == "1":
    print("Stage C skipped: needs 2 GPUs (Accelerator = GPU T4 x2).")

say(f"ALL DONE.  Send me the full report printed above (look for the OOD row and the LLaVA-NeXT row) + {WORK}/ood_results.zip")
