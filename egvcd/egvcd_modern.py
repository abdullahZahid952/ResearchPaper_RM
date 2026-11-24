"""
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
