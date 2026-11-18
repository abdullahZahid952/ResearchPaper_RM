#!/usr/bin/env python3
"""
audit_eval.py v2 - offline audit of the evaluation issues raised in the external review.

What it checks (CPU-only, needs only the egvcd caches + HF tokenizers):

  P1  ANSWER PARSING.  Compares four answer-parsing conventions on the cached tokens:
        orig   - the notebook's rule: yes iff decoded token does not start with "no"
        pope   - official POPE evaluate.py (RUCAIBox): negative iff a word of the
                 first sentence is exactly 'No', 'not' or 'no'; otherwise affirmative
        vcd    - the VCD repo's own eval_pope.py (DAMO-NLP-SG): label-conditional
                 substring check ('yes' in answer / 'no' in answer); junk = wrong
        strict - 3-class: token starts with "yes" -> Y, with "no" -> N, else JUNK
                 (junk counts wrong on both label sides)
      Reports how often each parser disagrees and what the argmax tokens actually are.

  P2  PROBABILITY NORMALISATION.  The notebook's offline baseline arm renormalised the
      cached top-32 logits; the full-vocab logsumexp (lse) was stored but unused.
      All corrected numbers use p_i = exp(lo_i - lse).  The yes/no mass outside the
      top-32 is not recoverable, so baseline-arm numbers are [lo, hi] bounds
      (hi attributes the whole tail to the favourable class).  VCD/APC arms are
      point values: their support (APC head) provably fits inside the cache
      (sorted-prefix + assert argument), so they were already exact.

  P3  CORRECTED ACCURACY MATHS.  Accuracy uses label-specific probabilities,
        acc = mean( y * P(yes-answer) + (1-y) * P(no-answer) ),
      never "1 - P(yes)".  Junk probability is reported for base, APC and VCD arms.

  P4  SHARED-UNIFORM COUPLING.  utility coupling_disagreement(p, q) gives the exact
      P(disagree) of the notebook's inverse-CDF sampler under a shared uniform
      (TV(p,q) is only a lower bound; they are equal only when the CDF difference
      is single-crossing).  Use it to replace the TV-sum in the caption diagnostic.

The "old" columns reproduce the notebook's numbers exactly (lenient parser +
top-32 renormalisation + 1-p formula), so every printed delta is attributable.

Run where the caches live:
    !python audit_eval.py --root /kaggle/working/egvcd_data --json audit_out.json
    %run audit_eval.py --root /kaggle/working/egvcd_data
    from audit_eval import run; run(root="/kaggle/working/egvcd_data")
"""
import argparse, glob, json, math, os
import numpy as np

DEF_TOK = {
    "cache":            "llava-hf/llava-1.5-7b-hf",
    "cache_13b":        "llava-hf/llava-1.5-13b-hf",
    "cache_qwen2vl":    "Qwen/Qwen2-VL-7B-Instruct",
    "cache_llavanext":  "llava-hf/llava-v1.6-mistral-7b-hf",
}
REQUIRED = ("ids", "lo", "ld", "lse", "label", "yes")
PARSERS = ("orig", "pope", "vcd", "strict")


def lse_rows(x):
    m = x.max(1)
    return m + np.log(np.exp(x - m[:, None]).sum(1))


def coupling_disagreement(p, q):
    """Exact P(X != Y) when X~p and Y~q are drawn by the notebook's sampler
    (inverse-CDF over tokens in index order with ONE shared uniform u):
        token = head[ #(cumsum(softmax) < u) ].
    P(X=Y) = sum_i max(0, min(Fp_i, Fq_i) - max(Fp_{i-1}, Fq_{i-1})).
    NOTE: TV(p,q) is a lower bound on this, NOT an equality in general; equality
    holds when the CDF difference Fp-Fq crosses zero at most once.
    p, q: (N, V) row-stochastic (or unnormalised nonneg; rows are normalised here).
    """
    p = p / p.sum(1, keepdims=True); q = q / q.sum(1, keepdims=True)
    cp, cq = np.cumsum(p, 1), np.cumsum(q, 1)
    z = np.zeros((p.shape[0], 1))
    cpp, cqp = np.hstack([z, cp[:, :-1]]), np.hstack([z, cq[:, :-1]])
    overlap = np.clip(np.minimum(cp, cq) - np.maximum(cpp, cqp), 0, None)
    return 1.0 - overlap.sum(1)


def token_info(decode, ids):
    """Classify every cached token under the four parsers. Returns dict of
    (N, K) boolean arrays: {parser}_yes, {parser}_no (may be neither or both)."""
    info = {}
    for t in np.unique(ids):
        s = decode([int(t)])
        sl = s.strip().lower()
        first = s.split(".")[0] if "." in s else s
        words = first.replace(",", "").split()
        info[int(t)] = {
            "orig_yes":   not sl.startswith("no"),
            "orig_no":    sl.startswith("no"),
            "pope_yes":   not any(w in ("No", "not", "no") for w in words),
            "pope_no":    any(w in ("No", "not", "no") for w in words),
            "vcd_yes":    "yes" in sl,
            "vcd_no":     "no" in sl,
            "strict_yes": sl.startswith("yes"),
            "strict_no":  sl.startswith("no"),
            "text":       s,
        }
    return info


def audit_file(path, decode, alpha=1.0, beta=0.1, topk_list=10):
    d = dict(np.load(path))
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        return {"skipped": f"missing keys {missing}"}
    ids, lo, ld = d["ids"], d["lo"].astype(np.float64), d["ld"].astype(np.float64)
    lse, label, yes_len = d["lse"].astype(np.float64), d["label"], d["yes"].astype(bool)
    H = d.get("H")
    N, y = len(ids), label.astype(bool)
    logb = math.log(beta)

    ti = token_info(decode, ids)
    M = {f"{p}_{s}": np.array([[ti[int(t)][f"{p}_{s}"] for t in row] for row in ids], bool)
         for p in PARSERS for s in ("yes", "no")}

    # probabilities over the cached tokens: exact full-vocab (lse) and old renormalised (lse32)
    lse32 = lse_rows(lo)
    p_full = np.exp(lo - lse[:, None])
    p32 = np.exp(lo - lse32[:, None])
    tail = np.clip(1.0 - np.exp(lse32 - lse), 0.0, 1.0)

    # APC head / VCD combined distribution (exact; support = head)
    head = lo >= lo[:, :1] + logb
    viol = int((head.sum(1) >= lo.shape[1]).sum())
    cd = np.where(head, (1 + alpha) * lo - alpha * ld, -np.inf)
    p_cd = np.exp(cd - lse_rows(np.where(head, cd, -np.inf))[:, None])
    p_h = np.exp(np.where(head, lo, -np.inf) - lse_rows(np.where(head, lo, -np.inf))[:, None])
    S = ~head[:, 1]

    def acc(p_yes, p_no):
        return float((y * p_yes + (1 - y) * p_no).mean() * 100)

    def arm(masks, probs, bounded):
        """acc per parser; bounds if the tail composition is unknown."""
        out = {}
        for p in PARSERS:
            py = (probs * masks[f"{p}_yes"]).sum(1)
            pn = (probs * masks[f"{p}_no"]).sum(1)
            if bounded:
                out[p] = [acc(py, pn), acc(py + tail, pn + tail)]
            else:
                out[p] = [acc(py, pn)] * 2
        return out

    # old numbers: lenient(orig) parser + top-32 renormalisation + 1-p formula, exactly as the notebook
    pb_old = (p32 * yes_len).sum(1)
    pv_old = (p_cd * yes_len).sum(1)
    old = {
        "acc_base": acc(pb_old, 1 - pb_old), "acc_vcd": acc(pv_old, 1 - pv_old),
        "gain": acc(pv_old, 1 - pv_old) - acc(pb_old, 1 - pb_old),
        "fyr_base": float(pb_old[~y].mean() * 100), "fyr_vcd": float(pv_old[~y].mean() * 100),
        "gb": float((yes_len[:, 0] == y).mean() * 100),
        "gv": float((yes_len[np.arange(N), cd.argmax(1)] == y).mean() * 100),
    }

    base_new = arm(M, p_full, bounded=True)
    vcd_new = arm(M, p_cd, bounded=False)
    apc_new = arm(M, p_h, bounded=False)
    gain_new = {p: [vcd_new[p][0] - base_new[p][1], vcd_new[p][0] - base_new[p][0]] for p in PARSERS}
    fyr_new = {}
    for p in PARSERS:
        pyb = (p_full * M[f"{p}_yes"]).sum(1); pyv = (p_cd * M[f"{p}_yes"]).sum(1)
        fyr_new[p] = [[float(pyb[~y].mean() * 100), float((pyb[~y] + tail[~y]).mean() * 100)],
                      [float(pyv[~y].mean() * 100)] * 2]

    # greedy under each parser (argmax token of base / vcd)
    i0 = np.arange(N)            # FIX: was np.zeros -> indexed row 0 for every question
    iv = cd.argmax(1)
    greedy = {}
    for p in PARSERS:
        greedy[p] = {
            "base": [acc(M[f"{p}_yes"][i0, 0].astype(float), M[f"{p}_no"][i0, 0].astype(float))] * 2,
            "vcd":  [acc(M[f"{p}_yes"][np.arange(N), iv].astype(float),
                         M[f"{p}_no"][np.arange(N), iv].astype(float))] * 2,
        }

    # junk diagnostics
    junk_arg = int((~M["strict_yes"][:, 0] & ~M["strict_no"][:, 0]).sum())
    pj_head = (p_full * (~M["strict_yes"] & ~M["strict_no"])).sum(1)
    junk_vcd = float(((p_cd * (~M["strict_yes"] & ~M["strict_no"])).sum(1)).mean() * 100)
    junk_apc = float(((p_h * (~M["strict_yes"] & ~M["strict_no"])).sum(1)).mean() * 100)
    parser_disagree = {p: int((M["orig_yes"][:, 0] != M[f"{p}_yes"][:, 0]).sum()) for p in PARSERS if p != "orig"}

    # argmax token census
    vals, cnts = np.unique([ti[int(t)]["text"] for t in ids[:, 0]], return_counts=True)
    order = np.argsort(-cnts)[:topk_list]
    census = []
    for v, c in zip(vals[order], cnts[order]):
        s = str(v).strip().lower()
        census.append([str(v), int(c), "Y" if s.startswith("yes") else ("N" if s.startswith("no") else "J")])

    # entropy-gate retention (tau=0.4/0.5): old pipeline + corrected strict pipeline
    ret = {}
    if H is not None and np.isfinite(np.asarray(H, float)).all():
        Hs = np.asarray(H, float)
        for tau in (0.4, 0.5):
            gate = (~S) & (Hs > tau)
            # old (lenient, renorm32 base)
            pa_old = (p_h * yes_len).sum(1)
            pg_old = np.where(gate, pv_old, np.where(S, yes_len[:, 0].astype(float), pa_old))
            cb, cv = np.where(y, pb_old, 1 - pb_old), np.where(y, pv_old, 1 - pv_old)
            r_old = float((np.where(y, pg_old, 1 - pg_old) - cb).sum() / (cv - cb).sum() * 100) if abs((cv - cb).sum()) > 1e-12 else float("nan")
            # strict, bounded
            gY = np.where(gate, (p_cd * M["strict_yes"]).sum(1),
                          np.where(S, M["strict_yes"][:, 0].astype(float), (p_h * M["strict_yes"]).sum(1)))
            gN = np.where(gate, (p_cd * M["strict_no"]).sum(1),
                          np.where(S, M["strict_no"][:, 0].astype(float), (p_h * M["strict_no"]).sum(1)))
            bY, bN = (p_full * M["strict_yes"]).sum(1), (p_full * M["strict_no"]).sum(1)
            vY, vN = (p_cd * M["strict_yes"]).sum(1), (p_cd * M["strict_no"]).sum(1)
            num_hi = (y * gY + (1 - y) * gN - (y * bY + (1 - y) * bN)).sum()
            num_lo = (y * gY + (1 - y) * gN - (y * (bY + tail) + (1 - y) * (bN + tail))).sum()
            den = (y * vY + (1 - y) * vN - (y * bY + (1 - y) * bN)).sum()
            den_lo = (y * vY + (1 - y) * vN - (y * (bY + tail) + (1 - y) * (bN + tail))).sum()
            ret[f"tau{tau}"] = {"rho": float(gate.mean() * 100),
                                "old": r_old,
                                "strict": [100*float(num_lo / den) if abs(den) > 1e-12 else float("nan"),
                                           100*float(num_hi / den_lo) if abs(den_lo) > 1e-12 else float("nan")],
                                "strict_retention_lo_den": 100*float(num_lo / den_lo) if abs(den_lo) > 1e-12 else float("nan"),
                                "strict_retention_hi_den": 100*float(num_hi / den) if abs(den) > 1e-12 else float("nan")}

    return {
        "n": N, "yes_labels": float(y.mean() * 100), "alpha": alpha, "beta": beta,
        "junk_argmax": junk_arg, "junk_argmax_pct": junk_arg / N * 100,
        "junk_argmax_by_label": [int((~M["strict_yes"][:, 0] & ~M["strict_no"][:, 0] & y).sum()),
                                 int((~M["strict_yes"][:, 0] & ~M["strict_no"][:, 0] & ~y).sum())],
        "E_junk_base_head_pct": float(pj_head.mean() * 100),
        "E_junk_vcd_pct": junk_vcd, "E_junk_apc_pct": junk_apc,
        "tail_pct": {"p50": float(np.percentile(tail, 50) * 100), "p90": float(np.percentile(tail, 90) * 100),
                     "p99": float(np.percentile(tail, 99) * 100), "max": float(tail.max() * 100)},
        "head_containment_violations": viol,
        "parser_argmax_disagreements_vs_orig": parser_disagree,
        "argmax_census": census,
        "old": old,
        "acc_base_smp": base_new, "acc_vcd_smp": vcd_new, "gain_smp": gain_new,
        "fyr_smp": fyr_new, "greedy": greedy, "retention": ret,
    }


def fmt_block(name, r):
    if "skipped" in r:
        return f"[{name}]  SKIPPED: {r['skipped']}"
    def rng(v, sign=False):
        if isinstance(v, list):
            return (f"{v[0]:+.2f}..{v[1]:+.2f}" if sign else f"{v[0]:.2f}-{v[1]:.2f}")
        return f"{v:+.2f}" if sign else f"{v:.2f}"
    L = [f"[{name}]  n={r['n']}  (yes-labels {r['yes_labels']:.1f}%)  alpha={r['alpha']} beta={r['beta']}"]
    L.append(f"  junk argmax (strict): {r['junk_argmax']}/{r['n']} = {r['junk_argmax_pct']:.2f}%  "
             f"[yes-lab {r['junk_argmax_by_label'][0]} / no-lab {r['junk_argmax_by_label'][1]}]   "
             f"argmax-parser disagreements vs orig: {r['parser_argmax_disagreements_vs_orig']}")
    L.append(f"  E[junk|sampled]: base-head {r['E_junk_base_head_pct']:.2f}% (+tail) | APC {r['E_junk_apc_pct']:.2f}% | VCD {r['E_junk_vcd_pct']:.2f}%")
    t = r["tail_pct"]
    L.append(f"  tail mass beyond top-32:  p50 {t['p50']:.3f}%  p90 {t['p90']:.3f}%  p99 {t['p99']:.3f}%  max {t['max']:.3f}%")
    if r["head_containment_violations"]:
        L.append(f"  !! APC head NOT contained in top-32 for {r['head_containment_violations']} rows -> vcd/arm numbers unreliable here")
    L.append(f"  top argmax tokens: " + "  ".join(f"'{w}'x{c}({k})" for w, c, k in r["argmax_census"][:8]))
    o = r["old"]
    L.append(f"  OLD (as published): base {o['acc_base']:.2f}  vcd {o['acc_vcd']:.2f}  gain {o['gain']:+.2f}  "
             f"fyr {o['fyr_base']:.1f}->{o['fyr_vcd']:.1f}  greedy {o['gb']:.2f}/{o['gv']:.2f}")
    L.append(f"  {'':26}" + "".join(f"{p:>13}" for p in ("orig", "pope", "vcd", "strict")))
    def row4(title, key, sub=None, sign=False):
        vals = []
        for p in PARSERS:
            v = r[key][p] if sub is None else r[key][p][sub]
            vals.append(rng(v, sign))
        L.append(f"  {title:26}" + "".join(f"{v:>13}" for v in vals))
    row4("acc base (smp)", "acc_base_smp")
    row4("acc vcd (smp)", "acc_vcd_smp")
    row4("vcd gain (smp)", "gain_smp", sign=True)
    row4("false-yes base (smp)", "fyr_smp", 0)
    row4("false-yes vcd (smp)", "fyr_smp", 1)
    row4("acc base (greedy)", "greedy", "base")
    row4("acc vcd (greedy)", "greedy", "vcd")
    for tau, r_ in r["retention"].items():
        L.append(f"  EG+ {tau}: 2nd pass for {r_['rho']:.0f}% | retention old {r_['old']:.1f}% | "
                 f"strict {r_['strict'][0]:+.1f}..{r_['strict'][1]:+.1f}%  "
                 f"(worst-case mix {r_['strict_retention_lo_den']:+.1f}..{r_['strict_retention_hi_den']:+.1f}%)")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/working/egvcd_data")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--json", default=None, help="also dump all numbers to this file")
    args, _unknown = ap.parse_known_args(argv)  # tolerate Jupyter's -f kernel flag

    files = sorted(glob.glob(os.path.join(args.root, "cache*", "*.npz")))
    if not files:
        raise SystemExit(f"no .npz caches found under {args.root}/cache*/")
    toks, res = {}, {}
    for f in files:
        d = os.path.basename(os.path.dirname(f))
        tok_id = DEF_TOK.get(d)
        if tok_id is None:
            print(f"[{f}]  SKIPPED: unknown cache dir '{d}' (add its tokenizer to DEF_TOK)")
            continue
        if tok_id not in toks:
            from transformers import AutoTokenizer
            print(f"loading tokenizer {tok_id} ...", flush=True)
            toks[tok_id] = AutoTokenizer.from_pretrained(tok_id)
        name = os.path.relpath(f, args.root)
        try:
            r = audit_file(f, toks[tok_id].decode, args.alpha, args.beta)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"skipped": f"{type(e).__name__}: {e}"}
        res[name] = r
        print(fmt_block(name, r), flush=True)

    print("\n=== GLOBAL ===")
    ok = [r for r in res.values() if "skipped" not in r]
    for p in PARSERS:
        db = max(abs(r["acc_base_smp"][p][0] - r["old"]["acc_base"]) for r in ok) if ok else 0
        w = max(r["acc_base_smp"][p][1] - r["acc_base_smp"][p][0] for r in ok) if ok else 0
        print(f"  base acc: {p:7} max|old-corrected| {db:5.2f} pts   max bound width {w:5.2f} pts")
    if ok:
        print(f"  worst junk-argmax rate: {max(r['junk_argmax_pct'] for r in ok):.2f}%")
        print(f"  worst head-containment violations: {max(r['head_containment_violations'] for r in ok)}")
    print("\nShare the full --json output, not just this verdict.")

    if args.json:
        json.dump(res, open(args.json, "w"), indent=1, default=float)
        print(f"wrote {args.json}")


def run(root="/kaggle/working/egvcd_data", alpha=1.0, beta=0.1, json_out=None):
    """Notebook-friendly entry point."""
    argv = ["--root", root, "--alpha", str(alpha), "--beta", str(beta)]
    if json_out:
        argv += ["--json", json_out]
    return main(argv)


if __name__ == "__main__":
    main()
