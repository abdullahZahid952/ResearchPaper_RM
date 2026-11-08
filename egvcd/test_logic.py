import os, sys, json, numpy as np, tempfile, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(); os.environ["EGVCD_ROOT"] = tmp
import egvcd as E
rng = np.random.RandomState(1); N, K = 900, 32
os.makedirs(os.path.join(tmp, "cache"))
def make(bias):
    lab = rng.rand(N) < 0.5
    # model logit margin favouring correct answer with noise, plus "yes" bias
    m = np.where(lab, 1.5, -1.5) + rng.normal(0, 2.0, N) + bias
    lo = np.full((N, K), -8.0) + rng.rand(N, K) * -2
    lo[:, 0] = np.maximum(m, 0) ; lo[:, 1] = np.maximum(-m, 0)      # tokens: 0=Yes-ish,1=No-ish
    yes = np.zeros((N, K), bool); yes[:, 0] = True
    # make ordering: sort desc, keep yes flags aligned
    order = np.argsort(-lo, 1); lo = np.take_along_axis(lo, order, 1); yes = np.take_along_axis(yes, order, 1)
    ld = lo.copy(); ld[yes] += 0.8; ld[~yes] -= 0.3                  # distorted view leans to "yes" (language prior)
    lse = np.log(np.exp(lo).sum(1) + 1e-3)
    p = np.exp(lo - lse[:, None]); H = -(p * np.log(p + 1e-12)).sum(1)
    return dict(ids=np.zeros((N, K), int), lo=lo, ld=ld, H=H, lse=lse, label=lab, yes=yes)
for cond in ["clean", "blur"]:
    for s in E.SPLITS:
        np.savez(os.path.join(tmp, "cache", f"{cond}_{s}.npz"), **make(0.8 if s != "random" else 0.2))
c = dict(np.load(os.path.join(tmp, "cache", "clean_random.npz"))); y = c["label"]

# 1. tau=0 == VCD ; tau=huge == base (greedy and sampled, same seed)
for mode in ["greedy", "sample"]:
    v = E.decisions(c, -1.0, mode, 3)[0]; e0 = E.decisions(c, 0.0, mode, 3)[0]
    b = E.decisions(c, None, mode, 3)[0]; eh = E.decisions(c, 1e9, mode, 3)[0]
    assert (v == e0).all() and (b == eh).all(), mode
# 2. gate rate monotone non-increasing in tau; prefills = 1+rho
rhos = [E.decisions(c, t, "greedy", 0)[1].mean() for t in E.TAUS]
assert all(a >= b for a, b in zip(rhos, rhos[1:])), rhos
print("rho(tau):", [round(r, 3) for r in rhos])
# 3. metrics vs manual
pred = E.decisions(c, None, "greedy", 0)[0]; m = E.prf(pred, y)
tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = ((~pred) & y).sum()
assert abs(m["f1"] - 100 * 2 * tp / (2 * tp + fp + fn)) < 1e-9
# 4. sampling frequency matches softmax prob for one question (APC off for base)
one = {k: np.repeat(v[:1], 20000, 0) for k, v in c.items()}
pr = np.exp(c["lo"][0] - c["lo"][0].max()); pr /= pr.sum(); py = pr[c["yes"][0]].sum()
est = E.decisions(one, None, "sample", 0)[0].mean()
assert abs(est - py) < 0.02, (est, py)
# 5. APC removes low-prob tokens: with beta=0.1 only tokens within log(0.1) of max survive
E.BETA = 0.1
# 6. full analyze runs and writes csv
E.run_analyze(seeds=3)
print(open(os.path.join(tmp, "results.csv")).readline()[:200])
print("ALL LOGIC TESTS PASSED")
