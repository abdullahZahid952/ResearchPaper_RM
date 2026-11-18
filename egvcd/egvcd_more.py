"""
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
