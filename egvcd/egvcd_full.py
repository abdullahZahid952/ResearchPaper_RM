"""
egvcd_full.py -- run the exact gate on the COMPLETE POPE-COCO benchmark (500 images, 3 x 3000 = 9000 questions)
with LLaVA-1.5-7B, clean images, and analyse it with image-clustered bootstrap CIs.

  python egvcd_full.py prepare_full                       # all 500 images + all questions
  python egvcd_full.py cache_shard --shard 0 --nshards 2  # one process per GPU (CUDA_VISIBLE_DEVICES picks the GPU)
  python egvcd_full.py analyze_full                       # pools the shards, writes full_numbers.json / full_report.txt

Each shard is resumable: finished (split, shard) files are skipped on a re-run.
Reuses egvcd.py (model, noise, logits) and egvcd_solid.py (offline probabilities, bootstrap) unchanged.
"""
import os, sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E
import egvcd_solid as S

SPLITS = E.SPLITS
OUT = "cache_full"


def image_order():
    seen, order = set(), []
    for l in open(E.p("pope_random_full.jsonl")):
        im = json.loads(l)["image"]
        if im not in seen:
            seen.add(im); order.append(im)
    return order


# ---------------------------------------------------------------------------------------------- data
def prepare_full():
    import requests
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(E.p("images"), exist_ok=True)
    rows = {s: [json.loads(l) for l in requests.get(E.POPE_URL.format(s), timeout=60).text.splitlines() if l.strip()] for s in SPLITS}
    for s in SPLITS:
        with open(E.p(f"pope_{s}_full.jsonl"), "w") as f:
            for r in rows[s]:
                f.write(json.dumps(r) + "\n")
        print(s, "questions:", len(rows[s]))
    imgs = []
    seen = set()
    for r in rows["random"]:
        if r["image"] not in seen:
            seen.add(r["image"]); imgs.append(r["image"])
    print("distinct images:", len(imgs))

    def get(n):
        dst = E.p("images", n)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return True
        for _ in range(4):
            try:
                b = requests.get(E.COCO_URL.format(n), timeout=60).content
                if len(b) > 1000:
                    open(dst, "wb").write(b); return True
            except Exception:
                time.sleep(2)
        return False
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(get, imgs))
    print(f"images on disk: {sum(ok)}/{len(imgs)}")
    if not all(ok):
        raise SystemExit("some COCO images could not be downloaded -- re-run this step")


# ---------------------------------------------------------------------------------------------- caching
def cache_shard(shard, nshards, limit=0, outdir=OUT):
    import torch
    from PIL import Image
    order = image_order()
    mine = set(order[shard::nshards])
    proc, model = S.load_model2(E.MODEL_ID, False); tok = proc.tokenizer
    os.makedirs(E.p(outdir), exist_ok=True)
    yes_cache = {}
    def is_yes(t):
        if t not in yes_cache:
            yes_cache[t] = not tok.decode([int(t)]).strip().lower().startswith("no")
        return yes_cache[t]
    for s in SPLITS:
        out = E.p(outdir, f"clean_{s}_full_s{shard}.npz")
        if os.path.exists(out):
            print("skip", out, flush=True); continue
        rows = [r for r in map(json.loads, open(E.p(f"pope_{s}_full.jsonl"))) if r["image"] in mine]
        if limit:
            rows = rows[:limit]
        A = {k: [] for k in ["ids", "lo", "ld", "H", "lse", "label", "yes", "image"]}
        t0 = time.time()
        for i, r in enumerate(rows):
            img = Image.open(E.p("images", r["image"])).convert("RGB")
            inp = E.build_inputs(proc, model, img, r["text"])
            lo = E.last_logits(model, inp, inp["pixel_values"])
            ld = E.last_logits(model, inp, E.diffusion_noise(inp["pixel_values"], E.NOISE_STEP, seed=r["question_id"]))
            assert torch.isfinite(lo).all() and torch.isfinite(ld).all(), "non-finite logits"
            top = torch.topk(lo, E.TOPK); ids = top.indices
            A["ids"].append(ids.cpu().numpy()); A["lo"].append(top.values.cpu().numpy()); A["ld"].append(ld[ids].cpu().numpy())
            A["H"].append(E.entropy_nats(lo)); A["lse"].append(float(torch.logsumexp(lo, -1)))
            A["label"].append(r["label"] == "yes"); A["image"].append(r["image"])
            A["yes"].append([is_yes(t) for t in ids.tolist()])
            if i % 100 == 0:
                el = time.time() - t0
                print(f"shard{shard}|{s} {i}/{len(rows)}  {el:.0f}s  (~{el / max(i, 1) * (len(rows) - i) / 60:.0f} min left for this split)", flush=True)
        np.savez(out, **{k: np.array(v) for k, v in A.items()})
        print("saved", out, len(rows), "questions", flush=True)


# ---------------------------------------------------------------------------------------------- analysis
def load_full(cache_dir=OUT, suffix="_full"):
    per, order = {}, None
    for s in SPLITS:
        import glob
        files = sorted(glob.glob(E.p(cache_dir, f"clean_{s}{suffix}_s*.npz")))
        if not files:
            return None
        cs = [dict(np.load(f)) for f in files]
        per[s] = {k: np.concatenate([c[k] for c in cs]) for k in ["lo", "ld", "H", "yes", "label", "image"]}
    return per


def cat(per, splits=SPLITS, keep=None):
    c = {k: np.concatenate([per[s][k] for s in splits]) for k in ["lo", "ld", "H", "yes", "label"]}
    im = np.concatenate([per[s]["image"] for s in splits])
    if keep is not None:
        m = np.isin(im, list(keep)); c = {k: v[m] for k, v in c.items()}; im = im[m]
    return c, im


def skip_ci(c, im, B=2000, seed=0):
    P = S.probs2(c); sk = P["S"].astype(float)
    uim, inv = np.unique(im, return_inverse=True); ni = len(uim)
    num = np.bincount(inv, weights=sk, minlength=ni); den = np.bincount(inv, minlength=ni).astype(float)
    idx = np.random.RandomState(seed).randint(0, ni, (B, ni))
    b = num[idx].sum(1) / den[idx].sum(1)
    return float(sk.mean() * 100), [float(np.percentile(b, 2.5) * 100), float(np.percentile(b, 97.5) * 100)]


def analyze_full():
    per = load_full()
    if per is None:
        raise SystemExit("no shards found in " + E.p(OUT))
    lines, N = [], {}
    def say(t=""):
        print(t); lines.append(t)
    order = image_order() if os.path.exists(E.p("pope_random_full.jsonl")) else list(dict.fromkeys(per["random"]["image"].tolist()))
    c, im = cat(per); s = S.summarize(c, im, B=2000); N["all"] = s
    sk, skci = skip_ci(c, im); N["skip_ci"] = skci
    say(f"=== FULL POPE-COCO, LLaVA-1.5-7B, clean images: {s['n_q']} questions, {s['n_img']} images (image-clustered 95% bootstrap CIs, 2000 resamples)")
    say(f"EXACT GATE: the second (noised-image) pass is skippable for {sk:.1f}% of questions  [95% CI {skci[0]:.1f}, {skci[1]:.1f}]")
    say(f"   forward passes per question: VCD 2.00 -> exact gate {1 + s['rho_exact']:.2f}  (-{(1 - s['rho_exact']) / 2 * 100:.1f}% of all forward passes, no change in output)")
    say(f"   max |p_E-VCD - p_VCD| = {s['exact_maxdiff']:.1e};  greedy outputs identical on every question: {s['greedy_exact_identical']}")
    say(f"accuracy (expected, sampled): base {s['acc_base']:.2f} | APC-only {s['acc_apc']:.2f} | VCD {s['acc_vcd']:.2f}")
    say(f"VCD gain sampled {s['sample_gain']:+.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}] | APC-only gain {s['apc_gain']:+.2f} [{s['apc_gain_ci'][0]:+.2f},{s['apc_gain_ci'][1]:+.2f}] | greedy gain {s['greedy_gain']:+.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]")
    for m in ("sample", "greedy"):
        f = s[f"fyr_{m}"]; say(f"false-yes rate ({m}): base {f['base']:.1f}% -> VCD {f['vcd']:.1f}%  (diff {f['diff']:+.1f} [{f['diff_ci'][0]:+.1f},{f['diff_ci'][1]:+.1f}])")
    for t in (0.4, 0.5):
        g = s[f"eg+_{t}"]
        say(f"EG+ tau={t}: second pass for {g['rho'] * 100:.0f}% | keeps {g['retention'] * 100:.1f}% of VCD gain | excess over random gating {g['excess']:+.3f} [{g['excess_ci'][0]:+.3f},{g['excess_ci'][1]:+.3f}]")
    say("\n=== per POPE split")
    N["splits"] = {}
    for sp in SPLITS:
        cs, ims = cat(per, [sp]); ss = S.summarize(cs, ims, B=1000); N["splits"][sp] = ss
        say(f"{sp:12} n={ss['n_q']:5d}  skippable {(1 - ss['rho_exact']) * 100:5.1f}% | acc base {ss['acc_base']:.2f} VCD {ss['acc_vcd']:.2f} | sampled gain {ss['sample_gain']:+.2f} "
            f"[{ss['sample_gain_ci'][0]:+.2f},{ss['sample_gain_ci'][1]:+.2f}] | greedy gain {ss['greedy_gain']:+.2f} | EG+.5 {ss['eg+_0.5']['rho'] * 100:.0f}% / keeps {ss['eg+_0.5']['retention'] * 100:.0f}%")
    say("\n=== does the result depend on sample size?  (first n images; same questions as the smaller experiments are nested inside)")
    N["growth"] = {}
    for n in (150, 300, 500):
        if n > len(order):
            continue
        keep = order[:n]
        cn, imn = cat(per, keep=keep)
        if len(np.unique(imn)) < min(n, len(order)):
            continue
        sn = S.summarize(cn, imn, B=500); a, b = skip_ci(cn, imn, B=500)[1]
        N["growth"][n] = dict(skip=(1 - sn["rho_exact"]) * 100, skip_ci=[a, b], gain=sn["sample_gain"], gain_ci=sn["sample_gain_ci"],
                              greedy=sn["greedy_gain"], ret=sn["eg+_0.5"]["retention"] * 100, rho=sn["eg+_0.5"]["rho"] * 100)
        say(f"n={n:3d} images ({sn['n_q']} q): skippable {(1 - sn['rho_exact']) * 100:.1f}% [{a:.1f},{b:.1f}] | sampled gain {sn['sample_gain']:+.2f} "
            f"[{sn['sample_gain_ci'][0]:+.2f},{sn['sample_gain_ci'][1]:+.2f}] | greedy gain {sn['greedy_gain']:+.2f} | EG+.5 {sn['eg+_0.5']['rho'] * 100:.0f}% keeps {sn['eg+_0.5']['retention'] * 100:.0f}%")
    json.dump(N, open(E.p("full_numbers.json"), "w"), indent=1, default=float)
    open(E.p("full_report.txt"), "w").write("\n".join(lines))
    say("\nwrote full_numbers.json and full_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare_full", "cache_shard", "analyze_full"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args(); os.makedirs(E.ROOT, exist_ok=True)
    if a.cmd == "prepare_full": prepare_full()
    elif a.cmd == "cache_shard": cache_shard(a.shard, a.nshards, a.limit, a.outdir)
    else: analyze_full()
