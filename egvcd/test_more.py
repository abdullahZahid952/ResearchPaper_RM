"""Unit test of the multi-token decoding bookkeeping (decode()) on a toy language model with KV-cache semantics.
No GPU needed.  The toy 'logits' are an exact function of (branch, token prefix), so eager VCD and lazy exact-gated VCD
MUST produce identical tokens if the lazy prefill / catch-up logic is right."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd_more as M

V, BETA, ALPHA = 40, 0.1, 1.0
LOGB = math.log(BETA)

def toy_logits(branch, prefix):
    seed = (hash((branch, tuple(prefix))) & 0x7FFFFFFF)
    r = np.random.RandomState(seed); l = r.randn(V) * 1.5
    if branch == "pos":                       # sometimes a very confident step, sometimes not
        l[int(r.randint(V))] += float(r.choice([0.0, 0.0, 9.0]))
    else:                                     # negative view: different but correlated
        l = 0.6 * toy_logits("pos", prefix) + 0.8 * l
    return l

class ToyBranch:
    def __init__(self, name): self.name, self.cache, self.calls, self.fed = name, [], 0, 0
    def prefill(self, ids): self.cache = list(ids); self.calls += 1; self.fed += len(ids); return toy_logits(self.name, self.cache)
    def extend(self, ids):
        assert len(ids) >= 1; self.cache += list(ids); self.calls += 1; self.fed += len(ids); return toy_logits(self.name, self.cache)

class NpOps:
    def single(self, lo): t = np.sort(lo)[::-1]; return bool(t[1] < t[0] + LOGB)
    def cd(self, lo, ld):
        cut = lo.max() + LOGB; d = (1 + ALPHA) * lo - ALPHA * ld; return np.where(lo < cut, -np.inf, d)
    def argmax(self, l): return int(np.argmax(l))
    def margin(self, l): t = np.sort(l)[::-1]; return float(t[0] - t[1])
    def fin(self, l, mode, u):
        if mode == "greedy": return int(np.argmax(l))
        head = np.nonzero(np.isfinite(l))[0]; p = np.exp(l[head] - l[head].max()); c = np.cumsum(p / p.sum())
        return int(head[min(int((c < u).sum()), len(head) - 1)])

ops = NpOps(); EOS = 0; prompt = [1, 2, 3, 4, 5]
n_cmp = n_single = n_steps = n_negcalls = 0
negs = {}
for qid in range(1, 301):
    for mode in ("greedy", "sample"):
        negs.clear()
        def mk(kind):
            b = ToyBranch("neg"); negs[kind] = b; return b
        g_ref, st_ref = M.decode("vcd", mode, ops, prompt, ToyBranch("pos"), lambda: mk("vcd"), qid, 40, EOS)
        g_new, st_new = M.decode("evcd", mode, ops, prompt, ToyBranch("pos"), lambda: mk("evcd"), qid, 40, EOS)
        assert g_ref == g_new, (qid, mode, g_ref, g_new)
        assert st_ref["single"] == st_new["single"] and st_ref["steps"] == st_new["steps"]
        assert st_new["neg_calls"] == st_new["steps"] - st_new["single"], "negative-branch calls must equal non-singleton steps"
        base, _ = M.decode("base", mode, ops, prompt, ToyBranch("pos"), None, qid, 40, EOS)   # base must run without a negative branch
        n_cmp += 1; n_single += st_new["single"]; n_steps += st_new["steps"]; n_negcalls += st_new["neg_calls"]
print(f"{n_cmp} sequences (greedy+sampled): eager VCD and lazy exact-gated VCD identical in all of them")
print(f"steps with |V_head|=1: {n_single}/{n_steps} ({100*n_single/n_steps:.0f}%); negative-branch calls {n_negcalls} (= non-singleton steps)")
# edge case: confident from the first step to the last -> the negative branch must never be created
class Never:
    def single(self, lo): return True
    def cd(self, *a): raise AssertionError("contrast must not be computed")
    def argmax(self, l): return 7
    def margin(self, l): return 0.0
    def fin(self, *a): raise AssertionError
created = []
g, st = M.decode("evcd", "greedy", Never(), prompt, ToyBranch("pos"), lambda: created.append(1) or ToyBranch("neg"), 1, 5, EOS)
assert not created and st["neg_calls"] == 0 and g == [7] * 5
print("edge case OK: all steps confident -> negative branch (and its image pass) never created")
print("ALL DECODING TESTS PASSED")
