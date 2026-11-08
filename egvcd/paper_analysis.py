"""
paper_analysis.py -- regenerates every number and figure in the paper from the Kaggle outputs.
Needs only numpy, scipy, pandas, matplotlib (CPU, ~1 minute).   Usage:
    python paper_analysis.py  /path/to/egvcd_data  /path/to/outdir
Reads  <data>/cache/*.npz (cached top-32 logits),  <data>/timing.json,  <data>/results.csv
Writes <out>/paper_numbers.json, <out>/fig_pareto.png, <out>/fig_domain.png, <out>/analysis_output.txt

Method variants (all evaluated offline from the same cached logits, one POPE question = one decoding step):
  base      : original next-token distribution                                         (1 image pass)
  apc_only  : base distribution truncated by VCD's adaptive plausibility set V_head      (1 pass)
  vcd       : full visual contrastive decoding (alpha=1, beta=0.1)                       (2 passes)
  e_vcd     : EXACT gate - run the 2nd pass only if |V_head|>1.  Identical output to vcd. (1+rho_exact)
  eg_raw    : entropy gate, falls back to the *untruncated* base distribution            (1+rho)   [first GPU run]
  eg_plus   : exact gate + entropy gate on the rest, falls back to APC-truncated base     (1+rho)
"""
import sys, os, json, math
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "egvcd_data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_out"
os.makedirs(OUT, exist_ok=True)
ALPHA, BETA = 1.0, 0.1; LOGB = math.log(BETA)
CONDS = ["clean", "lowlight", "noise", "blur"]; SPLITS = ["random", "popular", "adversarial"]
B = 1000; rng = np.random.RandomState(0)
T = json.load(open(f"{DATA}/timing.json")); TB, TV = T["base"]["ms_mean"], T["vcd"]["ms_mean"]
lat = lambda rho: TB + rho * (TV - TB)               # linear latency model: validated against measured gated runs below


def softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def load(c, s): return dict(np.load(f"{DATA}/cache/{c}_{s}.npz"))


def pool(c): 
    cs = [load(c, s) for s in SPLITS]; return {k: np.concatenate([x[k] for x in cs]) for k in cs[0]}


def probs(c):
    """P(answer = yes) under each variant (analytic, exact for sampling) + greedy yes-decisions."""
    lo, ld, yes, H = c["lo"], c["ld"], c["yes"], c["H"]; N = len(H); ar = np.arange(N)
    head = lo >= lo[:, :1] + LOGB                      # V_head (APC support); lo is sorted descending
    S = ~head[:, 1]                                    # |V_head| == 1  -> contrast cannot change the output
    cd = np.where(head, (1 + ALPHA) * lo - ALPHA * ld, -np.inf)
    p_base = (softmax(lo) * yes).sum(1); p_vcd = (softmax(cd) * yes).sum(1)
    p_apc = (softmax(np.where(head, lo, -np.inf)) * yes).sum(1)
    top = yes[:, 0].astype(float)                      # answer if the top-1 token is output
    g = dict(base=yes[:, 0], vcd=yes[ar, cd.argmax(1)])
    return dict(p_base=p_base, p_vcd=p_vcd, p_apc=p_apc, top=top, S=S, H=H, g=g)


def variant_p(P, name, tau=None):
    S, H = P["S"], P["H"]
    if name == "base": return P["p_base"], np.zeros(len(H), bool)
    if name == "apc_only": return np.where(S, P["top"], P["p_apc"]), np.zeros(len(H), bool)
    if name == "vcd": return P["p_vcd"], np.ones(len(H), bool)
    if name == "e_vcd": gate = ~S; return np.where(gate, P["p_vcd"], P["top"]), gate
    if name == "eg_raw": gate = H > tau; return np.where(gate, P["p_vcd"], P["p_base"]), gate
    if name == "eg_plus": gate = (~S) & (H > tau); return np.where(gate, P["p_vcd"], np.where(S, P["top"], P["p_apc"])), gate


def variant_greedy(P, name, tau=None):
    S, H = P["S"], P["H"]; gb, gv = P["g"]["base"], P["g"]["vcd"]
    if name in ("base", "apc_only"): return gb
    if name == "vcd": return gv
    if name == "e_vcd": return np.where(S, gb, gv)
    if name == "eg_raw": return np.where(H > tau, gv, gb)
    if name == "eg_plus": return np.where((~S) & (H > tau), gv, gb)


def metrics(p, y):
    p = p.astype(float); acc = np.where(y, p, 1 - p).mean() * 100; fpr = p[~y].mean() * 100; rec = p[y].mean() * 100
    tp = p[y].sum(); fp = p[~y].sum(); prec = tp / max(tp + fp, 1e-9) * 100
    return dict(acc=acc, fpr=fpr, rec=rec, prec=prec, f1=2 * prec * rec / max(prec + rec, 1e-9), yes=p.mean() * 100)


def boot_ci(x, lo=2.5, hi=97.5):
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)


N = {}                                                  # everything that goes into the paper
# ---------------------------------------------------------------- 0. sanity: top-32 mass, exactness of the skip rule
mass = min(np.exp(load(c, s)["lo"] - load(c, s)["lse"][:, None]).sum(1).min() for c in CONDS for s in SPLITS)
N["min_top32_mass"] = float(mass)
exact_ok, greedy_ok = 0.0, True
for c in CONDS:
    for s in SPLITS:
        P = probs(load(c, s))
        exact_ok = max(exact_ok, float(np.abs(variant_p(P, "e_vcd")[0] - P["p_vcd"]).max()))
        greedy_ok &= bool((variant_greedy(P, "e_vcd") == P["g"]["vcd"]).all())
N["exact_max_abs_diff"] = exact_ok; N["exact_greedy_identical"] = greedy_ok
say(f"min top-32 mass {mass:.5f} | exact-skip max|p_e_vcd - p_vcd| = {exact_ok:.2e} | greedy identical in all 12 caches: {greedy_ok}")

# ---------------------------------------------------------------- 1. reproduction table (5-seed sampled, clean) from results.csv
d = pd.read_csv(f"{DATA}/results.csv"); rep = {}
for mode in ["sample", "greedy"]:
    for s in SPLITS:
        for m in ["base", "vcd"]:
            r = d[(d.cond == "clean") & (d["mode"] == mode) & (d.split == s) & (d.method == m)].iloc[0]
            rep[f"{mode}|{s}|{m}"] = {k: float(r[k]) for k in ["acc", "acc_sd", "prec", "prec_sd", "rec", "rec_sd", "f1", "f1_sd", "yes"]}
N["repro"] = rep

# ---------------------------------------------------------------- 2. main per-condition table (pooled over the 3 POPE splits)
TAU_PLUS = [0.4, 0.5, 0.69]; TAU_RAW = [0.1, 0.2, 0.3, 0.5]
variants = [("base", None), ("apc_only", None), ("vcd", None), ("e_vcd", None)] + [("eg_plus", t) for t in TAU_PLUS] + [("eg_raw", t) for t in TAU_RAW]
main = {}
for mode in ["sample", "greedy"]:
    for cond in CONDS:
        c = pool(cond); y = c["label"]; P = probs(c); Nq = len(y)
        for name, tau in variants:
            key = f"{mode}|{cond}|{name}|{tau}"
            if mode == "sample": p, gate = variant_p(P, name, tau)
            else: p, gate = variant_greedy(P, name, tau).astype(float), variant_p(P, name, tau)[1]
            m = metrics(p, y); m["rho"] = float(gate.mean()); m["lat_ms"] = float(lat(gate.mean())); main[key] = m
N["main"] = main

# ---------------------------------------------------------------- 3. gain retention vs the random-gating expectation (sample mode)
ret = {}
for cond in CONDS:
    c = pool(cond); y = c["label"]; P = probs(c); Nq = len(y)
    cb = np.where(y, P["p_base"], 1 - P["p_base"]); cv = np.where(y, P["p_vcd"], 1 - P["p_vcd"])
    idx = rng.randint(0, Nq, (B, Nq)); gv_b = (cv[idx] - cb[idx]).mean(1)
    for name, tau in variants:
        if name in ("base", "vcd"): continue
        p, gate = variant_p(P, name, tau); ce = np.where(y, p, 1 - p)
        gain_e = (ce - cb).mean() * 100; gain_v = (cv - cb).mean() * 100
        ge_b = (ce[idx] - cb[idx]).mean(1); rho_b = gate[idx].mean(1)
        ret[f"{cond}|{name}|{tau}"] = dict(rho=float(gate.mean()), retention=float(gain_e / gain_v), gain=float(gain_e), gain_vcd=float(gain_v),
                                          excess=float(gain_e / gain_v - gate.mean()), excess_ci=boot_ci(ge_b / gv_b - rho_b),
                                          ei=float(gain_e / gain_v / gate.mean()) if gate.mean() > 0 else None)
N["ret"] = ret

# ---------------------------------------------------------------- 4. effect of VCD (sample & greedy) per condition, with CIs and McNemar
eff = {}
for cond in CONDS:
    c = pool(cond); y = c["label"]; P = probs(c); Nq = len(y); idx = rng.randint(0, Nq, (B, Nq))
    cb = np.where(y, P["p_base"], 1 - P["p_base"]); cv = np.where(y, P["p_vcd"], 1 - P["p_vcd"]); ca = np.where(y, P["p_apc"], 1 - P["p_apc"])
    ok_b = P["g"]["base"] == y; ok_v = P["g"]["vcd"] == y; dg = ok_v.astype(float) - ok_b.astype(float)
    w, l = int((ok_v & ~ok_b).sum()), int((~ok_v & ok_b).sum())
    e = dict(sample_gain=float((cv - cb).mean() * 100), sample_ci=boot_ci((cv[idx] - cb[idx]).mean(1) * 100),
             apc_gain=float((np.where(y, variant_p(P, "apc_only")[0], 1 - variant_p(P, "apc_only")[0]) - cb).mean() * 100),
             greedy_gain=float(dg.mean() * 100), greedy_ci=boot_ci(dg[idx].mean(1) * 100), greedy_p=float(binomtest(w, w + l, .5).pvalue) if w + l else 1.0,
             mean_entropy=float(c["H"].mean()), rho_exact=float(1 - P["S"].mean()))
    for t in [0.5]:                                    # greedy: eg_plus vs base / vcd
        gg = variant_greedy(P, "eg_plus", t); a = (gg == y).astype(float)
        for ref, nm in [(P["g"]["base"], "base"), (P["g"]["vcd"], "vcd")]:
            b = (ref == y).astype(float); dd = a - b; n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
            e[f"greedy_egplus{t}_vs_{nm}"] = dict(diff=float(dd.mean() * 100), ci=boot_ci(dd[idx].mean(1) * 100), p=float(binomtest(n10, n10 + n01, .5).pvalue) if n10 + n01 else 1.0)
    eff[cond] = e
for s in SPLITS:                                       # greedy VCD vs base per split (clean)
    c = load("clean", s); y = c["label"]; P = probs(c); ok_b = P["g"]["base"] == y; ok_v = P["g"]["vcd"] == y
    w, l = int((ok_v & ~ok_b).sum()), int((~ok_v & ok_b).sum())
    eff[f"greedy_clean_{s}"] = dict(base=float(ok_b.mean() * 100), vcd=float(ok_v.mean() * 100), wins=w, losses=l, p=float(binomtest(w, w + l, .5).pvalue),
                                    fpr_base=float(P["g"]["base"][~y].mean() * 100), fpr_vcd=float(P["g"]["vcd"][~y].mean() * 100))
N["eff"] = eff

# ---------------------------------------------------------------- 5. latency
tm = {}
for k, v in T.items():
    if k.startswith("eg"):
        pred = TB + (v["prefills"] - 1) * (TV - TB); tm[k] = dict(meas=v["ms_mean"], std=v["ms_std"], prefills=v["prefills"], pred=pred, err=v["ms_mean"] - pred)
N["timing"] = dict(base=T["base"], vcd=T["vcd"], eg=tm, ratio=TV / TB, delta_ms=TV - TB)
json.dump(N, open(f"{OUT}/paper_numbers.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- 6. figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Serif"})
col = {"clean": "#1f3b73", "lowlight": "#b4531c", "noise": "#2e7d5b", "blur": "#7a3b8f"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax[0].plot([0, 1], [0, 1], "k--", lw=.8, label="random gating (expected)")
for cond in CONDS:
    pts = sorted([(ret[f"{cond}|{n}|{t}"]["rho"], ret[f"{cond}|{n}|{t}"]["retention"]) for n, t in variants if n == "eg_raw"] + [(0, 0)])
    ax[0].plot(*zip(*pts), "o-", ms=3, lw=1, color=col[cond], label=f"entropy gate, {cond}")
pts = sorted([(ret[f"clean|eg_plus|{t}"]["rho"], ret[f"clean|eg_plus|{t}"]["retention"]) for t in TAU_PLUS] + [(ret["clean|e_vcd|None"]["rho"], 1.0)])
ax[0].plot(*zip(*pts), "s-", ms=3.5, lw=1.4, color="k", label="exact + entropy (clean)")
ax[0].set_xlabel("fraction of questions given a 2nd image pass (ρ)"); ax[0].set_ylabel("fraction of VCD accuracy gain retained")
ax[0].set_title("(a) gain retained vs. second-pass rate", fontsize=8); ax[0].legend(fontsize=5.5, frameon=False, loc="lower right"); ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.05)
for cond in ["clean", "blur"]:
    ax[1].scatter([TB, TV], [main[f"sample|{cond}|base|None"]["acc"], main[f"sample|{cond}|vcd|None"]["acc"]], color=col[cond], s=22, zorder=3, marker="D")
    xs = [main[f"sample|{cond}|e_vcd|None"]["lat_ms"]] + [main[f"sample|{cond}|eg_plus|{t}"]["lat_ms"] for t in TAU_PLUS]
    ys = [main[f"sample|{cond}|e_vcd|None"]["acc"]] + [main[f"sample|{cond}|eg_plus|{t}"]["acc"] for t in TAU_PLUS]
    ax[1].plot(xs, ys, "s-", ms=3.5, color=col[cond], lw=1.2, label=f"{cond}: exact + entropy")
    xr = [main[f"sample|{cond}|eg_raw|{t}"]["lat_ms"] for t in TAU_RAW]; yr = [main[f"sample|{cond}|eg_raw|{t}"]["acc"] for t in TAU_RAW]
    ax[1].plot(xr, yr, "o:", ms=2.5, color=col[cond], lw=.9, label=f"{cond}: entropy only")
ax[1].set_xlabel("latency per question (ms, T4)"); ax[1].set_ylabel("expected POPE accuracy (%)"); ax[1].set_title("(b) accuracy vs. latency (◆ = base, VCD)", fontsize=8)
ax[1].legend(fontsize=5.5, frameon=False, loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_pareto.png", dpi=220); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5)); w = .2; xs = np.arange(4)
for j, (mode, ttl) in enumerate([("sample", "(a) accuracy, sampled decoding"), ("greedy", "(b) accuracy, greedy decoding")]):
    for i, (nm, tau, lab, cc) in enumerate([("base", None, "base", "#9aa7b8"), ("vcd", None, "VCD", "#1f3b73"), ("e_vcd", None, "exact-gated", "#2e7d5b"), ("eg_plus", 0.5, "exact+entropy τ=.5", "#d98e32")]):
        ax[j].bar(xs + (i - 1.5) * w, [main[f"{mode}|{c}|{nm}|{tau}"]["acc"] for c in CONDS], w, color=cc, label=lab)
    ax[j].set_xticks(xs); ax[j].set_xticklabels(CONDS, fontsize=6.5); ax[j].set_ylim(70, 90); ax[j].set_title(ttl, fontsize=8)
ax[0].set_ylabel("expected accuracy (%)"); ax[0].legend(fontsize=5.5, frameon=False, ncol=2, loc="upper right")
for i, (mode, lab, cc) in enumerate([("sample", "sampled", "#1f3b73"), ("greedy", "greedy", "#d98e32")]):
    ax[2].bar(xs + (i - .5) * .3, [main[f"{mode}|{c}|vcd|None"]["fpr"] - main[f"{mode}|{c}|base|None"]["fpr"] for c in CONDS], .3, color=cc, label=lab)
ax[2].set_xticks(xs); ax[2].set_xticklabels(CONDS, fontsize=6.5); ax[2].set_title("(c) VCD: Δ false-yes rate (pp)", fontsize=8); ax[2].legend(fontsize=6, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_domain.png", dpi=220); plt.close()

# ---------------------------------------------------------------- console summary
say("\nVCD effect (pooled 2700 q per condition)")
for cond in CONDS:
    e = eff[cond]; say(f"{cond:9} sampled gain {e['sample_gain']:+.2f} {e['sample_ci']} | APC-only gain {e['apc_gain']:+.2f} | greedy gain {e['greedy_gain']:+.2f} {e['greedy_ci']} p={e['greedy_p']:.3f} | rho_exact {e['rho_exact']:.3f}")
say("\nVariants, sampled (acc / F1 / false-yes / rho / est. latency ms)")
for cond in CONDS:
    for name, tau in variants:
        m = main[f"sample|{cond}|{name}|{tau}"]; say(f"{cond:9}{name:9}{str(tau):5} {m['acc']:.2f} {m['f1']:.2f} {m['fpr']:.1f} rho={m['rho']:.2f} {m['lat_ms']:.0f}ms")
say("\nTiming model check:", {k: round(v['err'], 1) for k, v in tm.items()})
open(f"{OUT}/analysis_output.txt", "w").write("\n".join(log))


# ---------------------------------------------------------------- 7. workflow diagram (Fig. 1)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(7.0, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(-2, 48); ax.axis("off")
def box(x, y, w, h, t, fc="#e8eef6", ec="#1f3b73", fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(p, t="", tx=0, ty=0):
    for a, b in zip(p[:-2], p[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color="#333", lw=.9)
    ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle="-|>", mutation_scale=7, lw=.9, color="#333"))
    if t: ax.text(tx, ty, t, fontsize=5.8, color="#a33", fontweight="bold")
ax.text(1, 46, "Gated visual contrastive decoding: where the second image pass is skipped", fontsize=7.2, fontweight="bold")
box(1, 33, 14, 9, "image v +\nquestion x\n\"Is there a ...?\"")
box(20, 33, 17, 9, "LLaVA-1.5-7B\npass 1 (original v)\n→ logits l_o", fc="#dfeadf", ec="#2e7d5b")
box(42, 33, 21, 9, "from l_o compute\nV_head = {y : p ≥ β·p_max}\nentropy H", fc="#f6ecd9", ec="#b4531c")
box(68, 34, 31, 7, "GATE 1 (exact):  |V_head| = 1 ?", fc="#fbe3e3", ec="#a33", bold=True)
arrow([(15, 37.5), (20, 37.5)]); arrow([(37, 37.5), (42, 37.5)]); arrow([(63, 37.5), (68, 37.5)])
box(36, 20, 27, 8, "output top-1 token\n(distribution identical to VCD)", fc="#e8f5e8", ec="#2e7d5b")
arrow([(70, 34), (63, 28)], "yes", 68.2, 30.8)
box(68, 20, 31, 7, "GATE 2 (optional, lossy):  H > τ ?", fc="#fbe3e3", ec="#a33", bold=True)
arrow([(83.5, 34), (83.5, 27)], "no", 85, 30.3)
box(36, 5, 27, 8, "sample from APC-truncated l_o\n(no second pass)", fc="#e8f5e8", ec="#2e7d5b")
arrow([(72, 20), (63, 11)], "H ≤ τ", 64, 16.5)
box(68, 2, 31, 13, "pass 2 on noised image v′ (T = 999)\n→ logits l_d\ncontrast on V_head:\n(1+α)·l_o − α·l_d", fc="#dfeadf", ec="#2e7d5b")
arrow([(92, 20), (92, 15)], "H > τ", 93, 17)
box(1, 9, 26, 15, "answer token\n(sample or argmax)\n→ Yes / No\n→ POPE accuracy,\nprecision, recall, F1", fc="#e8eef6", ec="#1f3b73")
arrow([(36, 24), (27, 19)]); arrow([(36, 9), (27, 13)])
arrow([(83, 2), (83, -1), (14, -1), (14, 9)])
plt.savefig(f"{OUT}/fig_workflow.png", dpi=220, bbox_inches="tight"); plt.close()
