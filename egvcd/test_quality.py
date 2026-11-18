"""Offline tests for egvcd_quality.py (no GPU): entropy-gated decoding, CHAIR port, OOD builder, CHAIR evaluation, report.
Needs the official CHAIR synonyms file (downloaded automatically)."""
import os, sys, json, math, tempfile, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(); os.environ["EGVCD_ROOT"] = TMP
import egvcd as E, egvcd_more as M, egvcd_quality as Q
syn = os.path.join(TMP, "synonyms.txt")
try: urllib.request.urlretrieve(Q.SYN_URL, syn)
except Exception: import shutil; shutil.copy("/home/claude/synonyms_test.txt", syn)

# ---------------- 1. decode2: entropy gate properties on the toy model of test_more.py
import test_more as T   # re-runs its own tests too (fast)
ops = T.NpOps()
class Ops2(T.NpOps):
    def entropy(self, lo): p = np.exp(lo - lo.max()); p /= p.sum(); return float(-(p * np.log(p + 1e-12)).sum())
    def apc(self, lo): return np.where(lo < lo.max() + T.LOGB, -np.inf, lo)
ops = Ops2(); EOS, prompt = 0, [1, 2, 3, 4, 5]
for qid in range(1, 150):
    for mode in ("greedy", "sample"):
        mk = lambda: T.ToyBranch("neg")
        ev, _ = M.decode("evcd", mode, ops, prompt, T.ToyBranch("pos"), mk, qid, 40, EOS)
        eg0, st0 = Q.decode2("egp", mode, ops, prompt, T.ToyBranch("pos"), mk, qid, 40, EOS, tau=-1.0)     # entropy gate never skips -> == exact gate
        assert ev == eg0, (qid, mode)
        made = []
        eginf, stinf = Q.decode2("egp", mode, ops, prompt, T.ToyBranch("pos"), lambda: made.append(1) or T.ToyBranch("neg"), qid, 40, EOS, tau=1e9)  # always skips
        assert not made and stinf["neg_calls"] == 0, "tau=inf must never build the negative branch"
        if mode == "greedy":
            base, _ = M.decode("base", "greedy", ops, prompt, T.ToyBranch("pos"), None, qid, 40, EOS); assert base == eginf, "greedy + tau=inf must equal base"
        mid, stm = Q.decode2("egp", mode, ops, prompt, T.ToyBranch("pos"), mk, qid, 40, EOS, tau=1.0)
        assert stm["neg_calls"] + stm["entropy_skips"] == stm["steps"] - stm["single"]
print("decode2 OK: tau=-1 == exact gate, tau=inf == APC-only without a negative branch (greedy == base), call accounting consistent")

# ---------------- 2. CHAIR port: official synonyms, rules, singulariser
ch = Q.Chair(syn); assert len(ch.objs) > 300 and ch.inv["person"] == "person" and ch.inv["man"] == "person"
assert ch.words("Two men and a woman ride a motorbike near a hot dog stand")[1] == ["person", "person", "motorcycle", "hot dog"], ch.words("Two men and a woman ride a motorbike near a hot dog stand")
assert ch.inv.get(ch.singular("skis")) == ch.inv.get("ski"), "skis and ski must map to the same CHAIR category"
cases = {"buses": "bus", "horses": "horse", "glasses": "glass", "sandwiches": "sandwich", "tomatoes": "tomato", "knives": "knife", "women": "woman",
         "cars": "car", "bus": "bus", "glass": "glass", "dishes": "dish", "benches": "bench", "teddy": "teddy", "tennis": "tennis", "ties": "tie", "zebras": "zebra"}
bad = {w: ch.singular(w) for w, s in cases.items() if ch.singular(w) != s}; print("singulariser mismatches:", bad)
assert not bad, bad
w, n = ch.words("A teddy bear and baby elephant by a toilet seat and a stop sign with traffic lights and cell phones")
assert n == ["teddy bear", "elephant", "toilet", "stop sign", "traffic light", "cell phone"], n   # canonical names per the official list
g = ch.gt_objects(["person", "dog"], ["A man walks a dog", "A person with a leash"]); assert g == {"person", "dog"}, g
print("CHAIR port OK (official synonym list: %d surface forms)" % len(ch.objs), "->", n)

# ---------------- 3. OOD builder on a tiny COCO-format dataset
d = TMP + "/ds"; os.makedirs(d + "/imgs"); cats = [{"id": i, "name": nm} for i, nm in enumerate(["plane", "ship", "storage_tank", "bridge", "harbor", "vehicle", "court", "field", "pool", "roundabout", "chimney", "dam"], 1)]
rng = np.random.RandomState(0); imgs, anns = [], []
for i in range(1, 31):
    imgs.append({"id": i, "file_name": f"a{i}.png"}); open(f"{d}/imgs/a{i}.png", "wb").write(b"x")
    for c in rng.choice(12, size=int(rng.randint(3, 6)), replace=False): anns.append({"image_id": i, "category_id": int(c) + 1})
json.dump({"categories": cats, "images": imgs, "annotations": anns}, open(d + "/c.json", "w"))
Q.make_ood(d + "/c.json", d + "/imgs", 20)
for s in E.SPLITS:
    rows = [json.loads(l) for l in open(E.p(f"pope_{s}_ood.jsonl"))]; assert len(rows) == 20 * 6 and sum(r["label"] == "yes" for r in rows) == 60
    for im in {r["image"] for r in rows}:
        present = {r["text"] for r in rows if r["image"] == im and r["label"] == "yes"}; absent = {r["text"] for r in rows if r["image"] == im and r["label"] == "no"}; assert not (present & absent)
print("make_ood OK: 20 images -> 120 questions per split, 50% yes, no object both present and absent")

# ---------------- 4. chair_eval on synthetic caption files + report
gt = {f"COCO_val2014_{i:012d}.jpg": {"objects": ["person", "dog"], "captions": ["a man with a dog", "a person and a dog in a park"]} for i in range(1, 41)}
json.dump(gt, open(E.p("chair_gt.json"), "w"))   # synonyms.txt is already in place (same path as `syn`)
assert os.path.getsize(E.p("synonyms.txt")) > 3000
r = np.random.RandomState(1); lines = []
for sh in (0, 1):
    for im in list(gt)[sh::2]:
        for mode in ("greedy", "sample"):
            for m in ("base", "vcd", "evcd", "egp"):
                hallu = r.rand() < {"base": .5, "vcd": .35, "evcd": .35, "egp": .38}[m]
                cap = "A man walks a dog" + (" next to a bicycle and a car" if hallu else "") if not (m == "evcd") else None
                lines.append(dict(image=im, method=m, mode=mode, caption=cap, tokens=[1] * 30, ms=float(r.rand() * 1000 + 5000), steps=30, single=10, neg_calls=0 if m == "base" else 15, entropy_skips=0))
    # make evcd == vcd captions
    byk = {(l["image"], l["mode"], l["method"]): l for l in lines}
    for (im, mode, m), l in byk.items():
        if m == "evcd": l["caption"] = byk[(im, mode, "vcd")]["caption"]
    open(E.p(f"chair_shard{sh}.jsonl"), "w").write("".join(json.dumps(l) + "\n" for l in lines if l["image"] in list(gt)[sh::2]))
# deterministic check: a caption naming bicycle+car for an image whose ground truth is {person, dog}
w_, n_ = ch.words("A man walks a dog next to a bicycle and a car"); g_ = ch.gt_objects(["person", "dog"], ["a man with a dog"])
assert [x for x in n_ if x not in g_] == ["bicycle", "car"], n_
N = Q.chair_eval(); assert N["n_images"] == 40 and abs(N["evcd|greedy"]["chair_s"] - N["vcd|greedy"]["chair_s"]) < 1e-9
assert all(np.isfinite(v["chair_i"]) for k, v in N.items() if k != "n_images"), "CHAIR_i must be finite"
assert 25 < N["base|greedy"]["chair_s"] < 75 and N["base|greedy"]["chair_s"] > N["vcd|greedy"]["chair_s"] - 15, N["base|greedy"]   # base is built to hallucinate more than VCD
assert N["base|greedy"]["chair_i"] > 5, "hallucinated mentions must be counted"
print("chair_eval OK on synthetic captions (evcd == vcd by construction)")
print("ALL QUALITY TESTS PASSED")
