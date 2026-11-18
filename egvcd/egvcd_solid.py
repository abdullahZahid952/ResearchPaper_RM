"""
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
