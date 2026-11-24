"""
egvcd_review.py -- GPU experiments that answer reviewer questions 5, 7, 8 and 10.

  bench     (Q7, Q10) LLaVA-1.5-7B, POPE: latency (mean/p50/p95), throughput (questions/s) and peak GPU memory for
            plain model | VCD sequential | VCD BATCHED (both images in one batch = the optimized baseline) | exact gate,
            at batch sizes 1, 8, 16 (left padding).  Also counts answers that differ between VCD and the gate.
  exactgen  (Q5)  free-form captions, full recompute (no KV cache, so no catch-up numerics): VCD vs exact gate under
            GREEDY, SAMPLING (coupled: one shared uniform per step) and BEAM SEARCH (3 beams). Same sequences expected.
  cache     (Q8)  POPE logit cache for a model on a dataset: coco (AoiDragon POPE), aokvqa or gqa (VCD repo POPE files)
  analyze         one report over everything that finished

  python egvcd_review.py prepare --ds coco --n_images 300
  python egvcd_review.py bench --n_q 200
  python egvcd_review.py exactgen --n_img 20 --max_new 24
  python egvcd_review.py cache --model llava7b --ds aokvqa --n_images 200
  python egvcd_review.py analyze
"""
import os, sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E

SPLITS = E.SPLITS
LOGB = math.log(E.BETA)
DS_URL = {
    "coco": "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_{}.json",
    "aokvqa": "https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/experiments/data/POPE/aokvqa/aokvqa_pope_{}.json",
    "gqa": "https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/experiments/data/POPE/gqa/gqa_pope_{}.json",
}
IMG_URLS = {
    "coco": ["http://images.cocodataset.org/val2014/{}"],
    "aokvqa": ["http://images.cocodataset.org/val2014/{}"],
    "gqa": ["https://cs.stanford.edu/people/rak248/VG_100K_2/{}", "https://cs.stanford.edu/people/rak248/VG_100K/{}"],
}
OUT = "review"


def rp(*a):
    return E.p(OUT, *a)


# ============================================================================================== data
def prepare(ds, n_images):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(E.p("images"), exist_ok=True); os.makedirs(rp(), exist_ok=True)
    rows = {}
    for s in SPLITS:
        txt = requests.get(DS_URL[ds].format(s), timeout=60).text
        rows[s] = [json.loads(l) for l in txt.splitlines() if l.strip()]
    order = list(dict.fromkeys(r["image"] for r in rows["random"]))

    def get(n):
        dst = E.p("images", n)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return True
        for u in IMG_URLS[ds]:
            for _ in range(3):
                try:
                    b = requests.get(u.format(n), timeout=60).content
                    if len(b) > 1000 and b[:2] == b"\xff\xd8":
                        open(dst, "wb").write(b); return True
                except Exception:
                    time.sleep(2)
        return False
    with ThreadPoolExecutor(8) as ex:
        ok = dict(zip(order[:n_images + 40], ex.map(get, order[:n_images + 40])))
    keep = [n for n in order if ok.get(n)][:n_images]
    if len(keep) < max(20, n_images // 2):
        raise SystemExit(f"{ds}: only {len(keep)} images could be downloaded")
    ks = set(keep)
    for s in SPLITS:
        with open(rp(f"pope_{s}_{ds}.jsonl"), "w") as f:
            for r in rows[s]:
                if r["image"] in ks:
                    f.write(json.dumps(r) + "\n")
    json.dump(keep, open(rp(f"images_{ds}.json"), "w"))
    print(f"{ds}: {len(keep)} images, {sum(1 for r in rows['random'] if r['image'] in ks)} questions per split")


def rows(ds, s, n_images=None):
    keep = json.load(open(rp(f"images_{ds}.json")))
    ks = set(keep[:n_images] if n_images else keep)
    return [r for r in map(json.loads, open(rp(f"pope_{s}_{ds}.jsonl"))) if r["image"] in ks]


# ============================================================================================== model wrappers
def get_model(key):
    """returns (proc, model, build(img, question)->inputs, fwd(inputs, pixel_values)->last logits)"""
    if key == "llava7b":
        import egvcd_solid as S
        proc, model = S.load_model2(E.MODEL_ID, False)
        return proc, model, (lambda img, q: E.build_inputs(proc, model, img, q)), (lambda inp, pv: E.last_logits(model, inp, pv))
    import egvcd_modern as M
    proc, model = M.load(key, "float16")
    return proc, model, (lambda img, q: M.build(proc, model, img, M.pope_q(q))), (lambda inp, pv: M.fwd(model, inp, pv))


# ============================================================================================== Q8: caches
def cache(key, ds, n_images):
    import torch
    from PIL import Image
    od = rp(f"cache_{key}_{ds}"); os.makedirs(od, exist_ok=True)
    if all(os.path.exists(os.path.join(od, f"clean_{s}.npz")) for s in SPLITS):
        print("already done", od); return
    proc, model, build, fwd = get_model(key); tok = proc.tokenizer; yc = {}
    def is_yes(t):
        if t not in yc:
            yc[t] = not tok.decode([int(t)]).strip().lower().startswith("no")
        return yc[t]
    for s in SPLITS:
        out = os.path.join(od, f"clean_{s}.npz")
        if os.path.exists(out):
            continue
        R = rows(ds, s, n_images); A = {k: [] for k in ["ids", "lo", "ld", "H", "lse", "label", "yes", "image"]}; t0 = time.time()
        for i, r in enumerate(R):
            inp = build(Image.open(E.p("images", r["image"])).convert("RGB"), r["text"])
            lo = fwd(inp, inp["pixel_values"]); ld = fwd(inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=r["question_id"]))
            assert torch.isfinite(lo).all() and torch.isfinite(ld).all(), "non-finite logits"
            top = torch.topk(lo, E.TOPK); ids = top.indices
            A["ids"].append(ids.cpu().numpy()); A["lo"].append(top.values.cpu().numpy()); A["ld"].append(ld[ids].cpu().numpy())
            A["H"].append(E.entropy_nats(lo)); A["lse"].append(float(torch.logsumexp(lo, -1)))
            A["label"].append(r["label"] == "yes"); A["image"].append(r["image"]); A["yes"].append([is_yes(t) for t in ids.tolist()])
            if i % 200 == 0:
                el = time.time() - t0; print(f"{key}|{ds}|{s} {i}/{len(R)}  ~{el / max(i, 1) * (len(R) - i) / 60:.0f} min left", flush=True)
        np.savez(out, **{k: np.array(v) for k, v in A.items()}); print("saved", out, flush=True)


# ============================================================================================== Q7, Q10: deployment benchmark
def bench(n_q, batch_sizes=(1, 8, 16), warm=5):
    prev = json.load(open(rp("bench.json"))) if os.path.exists(rp("bench.json")) else {}
    import torch
    from PIL import Image
    import egvcd_solid as S
    proc, model = S.load_model2(E.MODEL_ID, False)
    proc.tokenizer.padding_side = "left"
    R = rows("coco", "random")[:n_q]
    imgs = {r["image"]: Image.open(E.p("images", r["image"])).convert("RGB") for r in R}
    dev = model.device
    def fwd_batch(ids, am, pv):
        with torch.no_grad():
            try:
                o = model(input_ids=ids, attention_mask=am, pixel_values=pv, use_cache=False, logits_to_keep=1)
            except TypeError:
                o = model(input_ids=ids, attention_mask=am, pixel_values=pv, use_cache=False)
        return o.logits[:, -1].float()
    def make(rs):
        inp = proc(images=[imgs[r["image"]] for r in rs], text=[E.PROMPT.format(q=r["text"]) for r in rs], padding=True, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(torch.float16); return inp
    def single(lo):
        v = torch.topk(lo, 2, -1).values; return v[:, 1] < v[:, 0] + LOGB
    def noised(pv, rs):
        return torch.cat([E.diffusion_noise(pv[i:i + 1], E.NOISE_STEP, seed=r["question_id"]) for i, r in enumerate(rs)])
    def cdarg(lo, ld):
        d = 2 * lo - ld; d = d.masked_fill(lo < lo.max(-1, keepdim=True).values + LOGB, float("-inf")); return d.argmax(-1)

    def run(method, rs, inp):
        pv, ids, am = inp["pixel_values"], inp["input_ids"], inp["attention_mask"]
        if method == "plain model (1 pass)":
            return fwd_batch(ids, am, pv).argmax(-1), len(rs), 0
        if method == "VCD sequential (2 passes)":
            lo = fwd_batch(ids, am, pv); ld = fwd_batch(ids, am, noised(pv, rs)); return cdarg(lo, ld), len(rs), len(rs)
        if method == "VCD batched (both images in one batch)":
            lg = fwd_batch(torch.cat([ids, ids]), torch.cat([am, am]), torch.cat([pv, noised(pv, rs)]))
            return cdarg(lg[:len(rs)], lg[len(rs):]), len(rs), len(rs)
        if method == "exact gate (skip 2nd pass when provably redundant)":
            lo = fwd_batch(ids, am, pv); sg = single(lo); out = lo.argmax(-1); need = (~sg).nonzero().flatten()
            if len(need):
                sub = [rs[i] for i in need.tolist()]
                ld = fwd_batch(ids[need], am[need], noised(pv[need], sub)); out[need] = cdarg(lo[need], ld)
            return out, len(rs), len(need)
    methods = ["plain model (1 pass)", "VCD sequential (2 passes)", "VCD batched (both images in one batch)",
               "exact gate (skip 2nd pass when provably redundant)"]
    res = {}
    for B in batch_sizes:
        batches = [R[i:i + B] for i in range(0, len(R), B)]
        inputs = [make(b) for b in batches]                     # preprocessing excluded from timing (same for all methods)
        preds = {}
        for m in methods:
            try:
                for b, inp in list(zip(batches, inputs))[:max(1, warm // B)]:
                    run(m, b, inp)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base_mem = torch.cuda.memory_allocated()
                ts, P, n2 = [], [], 0
                for b, inp in zip(batches, inputs):
                    torch.cuda.synchronize(); t = time.perf_counter()
                    pr, n, k = run(m, b, inp); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
                    P.append(pr.cpu()); n2 += k
                preds[m] = torch.cat(P)
                tot = sum(ts) / 1000; per_q = np.array(ts) / np.array([len(b) for b in batches])
                res[f"B={B} | {m}"] = dict(batch=B, method=m, ms_per_q=float(per_q.mean()), p50=float(np.percentile(per_q, 50)),
                                           p95=float(np.percentile(per_q, 95)), q_per_s=float(len(R) / tot),
                                           peak_mem_gb=float(torch.cuda.max_memory_allocated() / 2 ** 30),
                                           extra_mem_gb=float((torch.cuda.max_memory_allocated() - base_mem) / 2 ** 30),
                                           second_pass_share=float(n2 / len(R)))
                r = res[f"B={B} | {m}"]
                print(f"B={B:2d} {m:52} {r['ms_per_q']:7.1f} ms/q  p95 {r['p95']:7.1f}  {r['q_per_s']:5.2f} q/s  peak {r['peak_mem_gb']:.2f} GB "
                      f"(+{r['extra_mem_gb']:.2f})  2nd pass {r['second_pass_share'] * 100:.0f}%", flush=True)
            except torch.cuda.OutOfMemoryError:
                res[f"B={B} | {m}"] = dict(batch=B, method=m, oom=True); print(f"B={B} {m}: OUT OF MEMORY", flush=True)
                torch.cuda.empty_cache()
        ref = "VCD batched (both images in one batch)"
        for m in methods[1:]:
            if m in preds and ref in preds and m != ref:
                res[f"B={B} | {m}"]["answers_differing_from_batched_VCD"] = int((preds[m] != preds[ref]).sum())
    prev.update(res); prev = dict(sorted(prev.items(), key=lambda kv: kv[1]["batch"]))
    json.dump(prev, open(rp("bench.json"), "w"), indent=1)


# ============================================================================================== Q5: exactness in generation
def exactgen(n_img, max_new, beams=3, seed=0, modes=("greedy", "sample", "beam")):
    import torch
    from PIL import Image
    import egvcd_solid as S
    proc, model = S.load_model2(E.MODEL_ID, False); tok = proc.tokenizer; eos = tok.eos_token_id
    imgs = json.load(open(rp("images_coco.json")))[:n_img]
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"

    def last(ids, pv):                                       # full recompute, no KV cache
        with torch.no_grad():
            try:
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv.expand(ids.shape[0], -1, -1, -1),
                          use_cache=False, logits_to_keep=1)
            except TypeError:
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv.expand(ids.shape[0], -1, -1, -1), use_cache=False)
        return o.logits[:, -1].float()

    def step_logp(ids, pv, pvn, gate, stats):
        lo = last(ids, pv); keep = lo >= lo.max(-1, keepdim=True).values + LOGB
        out = torch.full_like(lo, float("-inf"))
        for i in range(ids.shape[0]):                        # negative branch row by row (batch 1) in BOTH methods: identical numerics
            stats["steps"] += 1
            if gate and int(keep[i].sum()) == 1:
                out[i, lo[i].argmax()] = 0.0; stats["skipped"] += 1; continue
            ld = last(ids[i:i + 1], pvn)[0]; stats["neg"] += 1
            d = torch.where(keep[i], 2 * lo[i] - ld, torch.full_like(ld, float("-inf")))
            out[i] = torch.log_softmax(d, -1)
        return out

    def decode(mode, inp, pv, pvn, gate, stats, qseed):
        ids = inp["input_ids"]; n0 = ids.shape[1]
        if mode in ("greedy", "sample"):
            g = torch.Generator().manual_seed(qseed)
            for _ in range(max_new):
                lp = step_logp(ids, pv, pvn, gate, stats)[0]
                if mode == "greedy":
                    t = int(lp.argmax())
                else:                                        # coupled sampling: one uniform per step for both methods
                    u = float(torch.rand(1, generator=g)); cdf = torch.cumsum(lp.exp().double(), 0)
                    t = int(min(int(torch.searchsorted(cdf, torch.tensor([u * float(cdf[-1])], dtype=cdf.dtype, device=cdf.device))), cdf.numel() - 1))
                ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], 1)
                if t == eos:
                    break
            return ids[0, n0:].tolist(), 0.0
        alive = [(ids, 0.0)]; done = []                     # beam search, sum of log-probs, no length penalty
        for _ in range(max_new):
            batch = torch.cat([a[0] for a in alive]); sc = torch.tensor([a[1] for a in alive], device=batch.device, dtype=torch.float64)
            lp = step_logp(batch, pv, pvn, gate, stats).double()
            cand = (sc[:, None] + lp).flatten(); top = torch.topk(cand, min(2 * beams, int(torch.isfinite(cand).sum())))
            nxt = []
            for v, j in zip(top.values.tolist(), top.indices.tolist()):
                b, t = divmod(j, lp.shape[1]); seq = torch.cat([alive[b][0], torch.tensor([[t]], device=batch.device)], 1)
                (done if t == eos else nxt).append((seq, v))
                if len(nxt) == beams:
                    break
            alive = nxt
            if len(done) >= beams or not alive:
                break
        best = max(done + alive, key=lambda x: x[1]); return best[0][0, n0:].tolist(), best[1]

    res = {}
    for mode in modes:
        same, sdiff = 0, 0.0; st = {False: dict(steps=0, skipped=0, neg=0), True: dict(steps=0, skipped=0, neg=0)}; t0 = time.time()
        for k, n in enumerate(imgs):
            inp = proc(images=Image.open(E.p("images", n)).convert("RGB"), text=prompt, return_tensors="pt")
            inp = {kk: v.to(model.device) for kk, v in inp.items()}; pv = inp["pixel_values"].to(torch.float16)
            pvn = E.diffusion_noise(pv, E.NOISE_STEP, seed=1000 + k)
            a, sa = decode(mode, inp, pv, pvn, False, st[False], seed + k)
            b, sb = decode(mode, inp, pv, pvn, True, st[True], seed + k)
            same += int(a == b); sdiff = max(sdiff, abs(sa - sb))
            if k % 5 == 0:
                print(f"exactgen|{mode} {k + 1}/{len(imgs)} identical so far {same}/{k + 1}  {time.time() - t0:.0f}s", flush=True)
        res[mode] = dict(n=len(imgs), identical=same, max_score_diff=sdiff, steps=st[True]["steps"],
                         singleton_steps=st[True]["skipped"], neg_vcd=st[False]["neg"], neg_gate=st[True]["neg"])
        r = res[mode]
        print(f"{mode}: identical {r['identical']}/{r['n']} | negative passes VCD {r['neg_vcd']} vs gate {r['neg_gate']} "
              f"({(1 - r['neg_gate'] / max(1, r['neg_vcd'])) * 100:.1f}% saved)", flush=True)
        json.dump(r, open(rp(f"exactgen_{mode}.json"), "w"), indent=1)          # saved per mode: a later failure loses nothing


# ============================================================================================== analysis
def analyze():
    import glob
    import egvcd_solid as S
    lines, N = [], {}
    def say(t=""):
        print(t); lines.append(t)
    if os.path.exists(rp("bench.json")):
        say("=== Q7/Q10  DEPLOYMENT BENCHMARK  (LLaVA-1.5-7B, POPE, one T4, fp16; preprocessing excluded, identical for all methods)")
        say(f"{'batch':5} {'method':52} {'ms/question':>11} {'p95':>8} {'q/s':>6} {'peak GB':>8} {'2nd pass':>8}  answers != batched VCD")
        for k, r in json.load(open(rp("bench.json"))).items():
            if r.get("oom"):
                say(f"{r['batch']:<5} {r['method']:52} OUT OF MEMORY"); continue
            say(f"{r['batch']:<5} {r['method']:52} {r['ms_per_q']:11.1f} {r['p95']:8.1f} {r['q_per_s']:6.2f} {r['peak_mem_gb']:8.2f} "
                f"{r['second_pass_share'] * 100:7.0f}%  {r.get('answers_differing_from_batched_VCD', '')}")
        N["bench"] = json.load(open(rp("bench.json")))
    g = {m: json.load(open(rp(f"exactgen_{m}.json"))) for m in ("greedy", "sample", "beam") if os.path.exists(rp(f"exactgen_{m}.json"))}
    if g:
        N["exactgen"] = g
        say("\n=== Q5  EXACTNESS IN FREE-FORM GENERATION  (captions, full recompute, negative branch computed identically in both)")
        for m, r in g.items():
            say(f"{m:7} identical captions {r['identical']}/{r['n']} | max beam-score diff {r['max_score_diff']:.1e} | "
                f"negative passes: VCD {r['neg_vcd']} vs gate {r['neg_gate']} ({(1 - r['neg_gate'] / max(1, r['neg_vcd'])) * 100:.1f}% fewer)")
    say("\n=== Q8  MORE DATA / NEW DATASETS  (exact gate on POPE; image-clustered 95% CIs)")
    for d in sorted(glob.glob(rp("cache_*"))):
        fs = [os.path.join(d, f"clean_{s}.npz") for s in SPLITS]
        if not all(os.path.exists(f) for f in fs):
            continue
        cs = [dict(np.load(f)) for f in fs]
        c = {k: np.concatenate([x[k] for x in cs]) for k in ["lo", "ld", "H", "yes", "label"]}; im = np.concatenate([x["image"] for x in cs])
        s = S.summarize(c, im, B=1000); N[os.path.basename(d)] = s
        say(f"{os.path.basename(d)[6:]:22} {s['n_q']:5d} q / {s['n_img']:3d} img | skippable {(1 - s['rho_exact']) * 100:5.1f}% | exact {s['exact_maxdiff']:.0e} "
            f"| base {s['acc_base']:.1f} VCD {s['acc_vcd']:.1f} | sampled gain {s['sample_gain']:+.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}] "
            f"(APC-only {s['apc_gain']:+.2f}) | greedy gain {s['greedy_gain']:+.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]")
    json.dump(N, open(rp("review_gpu_numbers.json"), "w"), indent=1, default=float)
    open(rp("review_gpu_report.txt"), "w").write("\n".join(lines))
    say("\nwrote review/review_gpu_numbers.json and review/review_gpu_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "bench", "exactgen", "cache", "analyze"])
    ap.add_argument("--ds", default="coco"); ap.add_argument("--n_images", type=int, default=0)
    ap.add_argument("--model", default="llava7b"); ap.add_argument("--n_q", type=int, default=200)
    ap.add_argument("--n_img", type=int, default=20); ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--modes", default="greedy,sample,beam"); ap.add_argument("--batch_sizes", default="1,8,16")
    a = ap.parse_args(); os.makedirs(rp(), exist_ok=True)
    if a.cmd == "prepare": prepare(a.ds, a.n_images or 300)
    elif a.cmd == "bench": bench(a.n_q, tuple(int(x) for x in a.batch_sizes.split(",")))
    elif a.cmd == "exactgen": exactgen(a.n_img, a.max_new, modes=tuple(a.modes.split(",")))
    elif a.cmd == "cache": cache(a.model, a.ds, a.n_images or None)
    else: analyze()
