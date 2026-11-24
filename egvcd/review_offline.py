"""
review_offline.py -- answers to reviewer questions that need NO GPU (pure numpy on the cached top-32 logits).

  Q4  sensitivity to the plausibility threshold beta: skip rate, modelled latency, accuracy, VCD gain, false-yes rate
  Q6  which skipping criterion is best at an EQUAL budget (share of second passes actually run):
      entropy  vs  max-probability (confidence)  vs  top-1/top-2 logit margin  vs  random  vs  oracle,
      each with two fallbacks for skipped items: plain model (as in confidence-gated CD) or APC-truncated (ours)
  Q3  decomposition of VCD's gain into  (a) APC truncation alone (needs no second pass)  and  (b) the contrast itself

  python review_offline.py DATASET_DIR [DATASET_DIR ...]      (each dir holds clean_{random,popular,adversarial}*.npz)
Writes review_numbers.json and review_report.txt next to this file.
"""
import sys, os, glob, json, math
import numpy as np

SPLITS = ("random", "popular", "adversarial")
BETAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
BUDGETS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
T1_MS, T2_MS = 661.3, 1294.8          # measured single-pass and two-pass (VCD) latency, LLaVA-1.5-7B, T4 (timing.json)


def load(d):
    out = []
    for s in SPLITS:
        fs = sorted(glob.glob(os.path.join(d, f"clean_{s}*.npz")))
        if not fs:
            raise SystemExit(f"no clean_{s}*.npz in {d}")
        out += [dict(np.load(f)) for f in fs]
    return {k: np.concatenate([o[k] for o in out]) for k in ("lo", "ld", "H", "lse", "yes", "label", "image")}


def sm(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def dists(c, beta, alpha=1.0):
    lo, ld, yes = c["lo"], c["ld"], c["yes"]
    head = lo >= lo[:, :1] + math.log(beta)
    full = head.all(1)                                 # head reaches the end of the cached top-32: cannot evaluate exactly
    cd = np.where(head, (1 + alpha) * lo - alpha * ld, -np.inf)
    return dict(head=head, single=~head[:, 1], bad=full,
                p_base=(sm(lo) * yes).sum(1), p_vcd=(sm(cd) * yes).sum(1), p_apc=(sm(np.where(head, lo, -np.inf)) * yes).sum(1),
                g_base=yes[:, 0], g_vcd=yes[np.arange(len(lo)), cd.argmax(1)])


def acc(p, y):
    return np.where(y, p, 1 - p)


def boot(per_q, im, B=1000, seed=0):
    u, inv = np.unique(im, return_inverse=True); n = len(u)
    s = np.bincount(inv, weights=per_q, minlength=n); c = np.bincount(inv, minlength=n)
    i = np.random.RandomState(seed).randint(0, n, (B, n)); b = s[i].sum(1) / c[i].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def beta_sweep(c):
    y = c["label"]; rows = []
    for b in BETAS:
        D = dists(c, b)
        if D["bad"].any():
            rows.append(dict(beta=b, valid=False)); continue
        ab, av = acc(D["p_base"], y), acc(D["p_vcd"], y)
        rho = float((~D["single"]).mean())
        fy_b = float((D["p_base"] * ~y).sum() / (~y).sum() * 100); fy_v = float((D["p_vcd"] * ~y).sum() / (~y).sum() * 100)
        pa = np.where(D["single"], D["g_base"].astype(float), D["p_apc"])
        rows.append(dict(beta=b, valid=True, skip=(1 - rho) * 100, passes=1 + rho, latency_ms=T1_MS + rho * (T2_MS - T1_MS),
                         acc_base=float(ab.mean() * 100), acc_vcd=float(av.mean() * 100), gain=float((av - ab).mean() * 100),
                         gain_ci=[x * 100 for x in boot(av - ab, c["image"])],
                         apc_gain=float((acc(pa, y) - ab).mean() * 100),
                         greedy_gain=float(((D["g_vcd"] == y).astype(float) - (D["g_base"] == y)).mean() * 100),
                         fyr_base=fy_b, fyr_vcd=fy_v,
                         exact_maxdiff=float(np.abs(np.where(D["single"], D["g_base"].astype(float), D["p_vcd"]) - D["p_vcd"]).max())))
    return rows


def criteria(c, beta=0.1, seed=0):
    """retention of VCD's sampled gain vs share of second passes run (budget), for several skip criteria"""
    y = c["label"]; D = dists(c, beta); N = len(y)
    pmax = np.exp(c["lo"][:, 0] - c["lse"]); margin = c["lo"][:, 0] - c["lo"][:, 1]
    ab, av = acc(D["p_base"], y), acc(D["p_vcd"], y); G = (av - ab).sum()
    fb = {"plain model": D["p_base"], "APC-truncated (ours)": np.where(D["single"], D["g_base"].astype(float), D["p_apc"])}
    rng = np.random.RandomState(seed)
    base_scores = {"entropy": c["H"], "1 - max prob (confidence)": 1 - pmax, "small top-2 margin": -margin, "random": rng.rand(N)}
    res = {}
    for fname, pf in fb.items():
        af = acc(pf, y)
        scores = dict(base_scores, **{"oracle (true gain, upper bound)": av - af})
        for sname, s in scores.items():
            s = np.where(D["single"], -np.inf, s)           # the exact gate is applied first for every criterion (free and lossless)
            order = np.argsort(-s); curve = []
            for bud in BUDGETS:
                k = min(int(round(bud * N)), int((~D["single"]).sum()))
                run = np.zeros(N, bool); run[order[:k]] = True
                curve.append(float((np.where(run, av, af) - ab).sum() / G * 100))
            res[f"{sname} | fallback: {fname}"] = curve
    return res, float((~D["single"]).mean() * 100)


def main(dirs):
    lines, N = [], {}
    def say(t=""):
        print(t); lines.append(t)
    for d in dirs:
        name = d.rstrip("/").split("/")[-1]; c = load(d)
        say(f"\n######## {d}   ({len(c['label'])} questions, {len(np.unique(c['image']))} images)")
        sw = beta_sweep(c)
        say("Q4  beta | skip% | passes/q | modelled ms/q | acc base | acc VCD | VCD gain [95% CI] | of which APC-only | greedy gain | false-yes base->VCD | exact")
        for r in sw:
            if not r["valid"]:
                say(f"    {r['beta']:<5} | plausible set exceeds the cached top-32 for some question: not evaluable"); continue
            say(f"    {r['beta']:<5} | {r['skip']:5.1f} | {r['passes']:.2f} | {r['latency_ms']:6.0f} | {r['acc_base']:.2f} | {r['acc_vcd']:.2f} | "
                f"{r['gain']:+.2f} [{r['gain_ci'][0]:+.2f},{r['gain_ci'][1]:+.2f}] | {r['apc_gain']:+.2f} | {r['greedy_gain']:+.2f} | "
                f"{r['fyr_base']:.1f}->{r['fyr_vcd']:.1f} | {r['exact_maxdiff']:.0e}")
        cr, rho = criteria(c)
        say(f"Q6  share of VCD's sampled gain kept (%) at a given share of questions that get the 2nd pass (exact gate first; it needs {rho:.1f}%)")
        say("    budget ->                                              " + "  ".join(f"{b * 100:5.0f}%" for b in BUDGETS))
        for k, v in cr.items():
            say(f"    {k:55} " + "  ".join(f"{x:6.1f}" for x in v))
        y = c["label"]; D1 = dists(c, 0.1); av = acc(D1["p_vcd"], y); tv = {}
        say("Q3  truncation only (plain model restricted to the plausible set, NO second pass) minus default VCD (beta=0.1), sampled accuracy:")
        for b in (0.3, 0.5):
            Db = dists(c, b); aa = acc(np.where(Db["single"], Db["g_base"].astype(float), Db["p_apc"]), y); ci = boot(aa - av, c["image"], B=2000)
            tv[b] = dict(diff=float((aa - av).mean() * 100), ci=[ci[0] * 100, ci[1] * 100])
            say(f"    beta={b}: {tv[b]['diff']:+.2f} [{tv[b]['ci'][0]:+.2f},{tv[b]['ci'][1]:+.2f}]  (beta chosen after seeing the data: exploratory)")
        N[name] = dict(beta_sweep=sw, criteria=cr, budgets=BUDGETS, rho_exact=rho, trunc_minus_vcd=tv)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(N, open(os.path.join(here, "review_numbers.json"), "w"), indent=1)
    open(os.path.join(here, "review_report.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
