"""
egvcd_quality.py -- what is still unchecked after egvcd_solid.py / egvcd_more.py.

  prepare_chair  COCO annotations + official CHAIR synonym list + images for the caption-quality experiment
  chair_gen      generate captions (base / VCD / exact-gated VCD / entropy+exact gated) in greedy and sampled mode, one shard per GPU
  chair_eval     CHAIR_s / CHAIR_i / coverage with image-bootstrap CIs, paired comparisons (CPU)
  numerics       WHY do exact-gated and eager VCD occasionally diverge in captions? eager vs catch-up negative-branch logits
  make_ood       build POPE-style questions from ANY COCO-format detection dataset you bring (satellite, documents, ...)
  analyze3       offline report: consistency across all caches, alpha/beta sweep on every dataset, degradation per model

CHAIR (Rohrbach et al., EMNLP 2018) is ported from the authors' public chair.py: same synonym list, double-word and
animal/vehicle rules, ground truth = COCO instance categories + objects named in the 5 reference captions.
Deviation: the pattern.en singulariser is replaced by a rule-based, vocabulary-aware singulariser (no extra installs).
"""
import os, sys, json, math, time, re, argparse, shutil, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egvcd as E
import egvcd_solid as S
import egvcd_more as M

LOGB = math.log(E.BETA)
SYN_URL = "https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
TAU = 0.5


# =============================================================================== decoding with the entropy gate (extends M.decode)
def decode2(method, mode, ops, prompt_ids, pos, mk_neg, qid, max_new, eos, tau=TAU):
    """method: base | vcd | evcd | egp.  egp = exact gate, then entropy gate (H <= tau -> APC-truncated base, no 2nd pass)."""
    if method in ("base", "vcd", "evcd"):
        return M.decode(method, mode, ops, prompt_ids, pos, mk_neg, qid, max_new, eos)
    rng = np.random.RandomState(qid)
    gen, st = [], dict(steps=0, single=0, neg_calls=0, entropy_skips=0, margins=[])
    lo = pos.prefill(prompt_ids); neg, ld, pending = None, None, []
    for t in range(max_new):
        u = float(rng.rand()); single = ops.single(lo)
        if single:
            tok = ops.argmax(lo)
        elif ops.entropy(lo) <= tau:
            tok = ops.fin(ops.apc(lo), mode, u); st["entropy_skips"] += 1
        else:
            if neg is None:
                neg = mk_neg(); ld = neg.prefill(list(prompt_ids) + gen)
            else:
                ld = neg.extend(pending)
            pending = []; st["neg_calls"] += 1
            tok = ops.fin(ops.cd(lo, ld), mode, u)
        st["steps"] += 1; st["single"] += int(single); gen.append(int(tok))
        if tok == eos:
            break
        lo = pos.extend([int(tok)]); pending.append(int(tok))
    return gen, st


class TorchOps2(M.TorchOps):
    def entropy(self, lo): return E.entropy_nats(lo)
    def apc(self, lo):
        t2 = self.t.topk(lo, 2).values; return lo.masked_fill(lo < t2[0] + LOGB, float("-inf"))


# =============================================================================== CHAIR (port)
DOUBLE = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign',
          'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
          'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear',
          'home plate', 'train track']
IRREG = {"men": "man", "women": "woman", "children": "child", "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth",
         "knives": "knife", "leaves": "leaf", "shelves": "shelf", "loaves": "loaf", "wolves": "wolf", "scarves": "scarf"}


class Chair:
    def __init__(self, syn_path):
        syn = [s.strip().split(', ') for s in open(syn_path).readlines()]      # exactly as in the original (double spaces stay un-matchable)
        self.objs, self.inv = [], {}
        for line in syn:
            self.objs.extend(line)
            for s in line:
                self.inv[s] = line[0]
        self.objs = set(self.objs)
        self.dd = {d: d for d in DOUBLE}
        for a in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']:
            self.dd[f'baby {a}'] = a; self.dd[f'adult {a}'] = a
        for v in ['jet', 'train']:
            self.dd[f'passenger {v}'] = v
        self.dd['bow tie'] = 'tie'; self.dd['toilet seat'] = 'toilet'; self.dd['wine glas'] = 'wine glass'
        self.vocab = set(self.objs) | {w for d in DOUBLE for w in d.split()}

    def singular(self, w):
        if w in IRREG:
            return IRREG[w]
        if len(w) < 4 or not w.endswith("s") or w.endswith(("ss", "us", "is")):
            return w
        cands = ([w[:-3] + "y"] if w.endswith("ies") else []) + ([w[:-2]] if w.endswith("es") else []) + [w[:-1]]
        for c in cands:
            if c in self.vocab:
                return c
        return w[:-2] if w.endswith(("ches", "shes", "xes", "zes", "oes")) else w[:-1]

    def words(self, caption):
        w = [self.singular(x) for x in re.findall(r"[a-z]+", caption.lower())]
        out, i = [], 0
        while i < len(w):
            pair = " ".join(w[i:i + 2])
            if pair in self.dd:
                out.append(self.dd[pair]); i += 2
            else:
                out.append(w[i]); i += 1
        if "toilet" in out and "seat" in out:
            out = [x for x in out if x != "seat"]
        keep = [x for x in out if x in self.objs]
        return keep, [self.inv[x] for x in keep]

    def gt_objects(self, instance_names, captions):
        s = {self.inv.get(n, n) for n in instance_names}
        for c in captions:
            s.update(self.words(c)[1])
        return s


# =============================================================================== data preparation
def prepare_chair(n):
    import requests
    os.makedirs(E.p("images"), exist_ok=True)
    names, seen = [], set()
    for l in requests.get(E.POPE_URL.format("random"), timeout=60).text.splitlines():
        if l.strip():
            im = json.loads(l)["image"]
            if im not in seen:
                seen.add(im); names.append(im)
    names = names[:n]
    for nm in names:
        dst = E.p("images", nm)
        if not os.path.exists(dst):
            for _ in range(3):
                try:
                    open(dst, "wb").write(requests.get(E.COCO_URL.format(nm), timeout=60).content); break
                except Exception:
                    time.sleep(2)
    open(E.p("synonyms.txt"), "w").write(requests.get(SYN_URL, timeout=60).text)
    if not os.path.exists(E.p("chair_gt.json")):
        zp = E.p("annotations_trainval2014.zip"); print("downloading COCO annotations (~240 MB) ...", flush=True)
        with requests.get(ANN_URL, stream=True, timeout=120) as r, open(zp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(zp) as z:
            ins = json.load(z.open("annotations/instances_val2014.json")); caps = json.load(z.open("annotations/captions_val2014.json"))
        ids = {int(re.search(r"(\d+)\.jpg", nm).group(1)): nm for nm in names}
        cat = {c["id"]: c["name"] for c in ins["categories"]}; gt = {nm: {"objects": [], "captions": []} for nm in names}
        for a in ins["annotations"]:
            if a["image_id"] in ids:
                gt[ids[a["image_id"]]]["objects"].append(cat[a["category_id"]])
        for a in caps["annotations"]:
            if a["image_id"] in ids:
                gt[ids[a["image_id"]]]["captions"].append(a["caption"])
        for v in gt.values():
            v["objects"] = sorted(set(v["objects"]))
        json.dump(gt, open(E.p("chair_gt.json"), "w")); os.remove(zp)
    json.dump(names, open(E.p("chair_images.json"), "w"))
    print(f"prepared {len(names)} images, synonyms, ground truth", flush=True)


# =============================================================================== caption generation
def chair_gen(shard, nshards, n, modes, max_new, smoke):
    import torch
    from PIL import Image
    proc, model = E.load_model(); tok = proc.tokenizer; eos = tok.eos_token_id; ops = TorchOps2()
    names = json.load(open(E.p("chair_images.json")))[:n]
    mine = names[shard::nshards][:2] if smoke else names[shard::nshards]
    out = E.p(f"chair_shard{shard}{'_smoke' if smoke else ''}.jsonl"); done = set()
    if os.path.exists(out):
        done = {json.loads(l)["image"] for l in open(out)}
    plan = [(m, mo) for mo in modes for m in ("base", "vcd", "evcd", "egp")]
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    t0 = time.time()
    for k, name in enumerate(mine):
        if name in done:
            continue
        qid = int(re.search(r"(\d+)\.jpg", name).group(1))
        inp = proc(images=Image.open(E.p("images", name)).convert("RGB"), text=prompt, return_tensors="pt")
        inp = {a: b.to(model.device) for a, b in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        pv = inp["pixel_values"]; ids = inp["input_ids"][0].tolist(); rows = []
        for m, mo in plan:
            torch.cuda.synchronize(); t1 = time.perf_counter()
            gen, st = decode2(m, mo, ops, ids, M.HFBranch(model, pv), lambda: M.HFBranch(model, E.diffusion_noise(pv, E.NOISE_STEP, seed=qid)), qid, max_new, eos)
            torch.cuda.synchronize()
            rows.append(dict(image=name, method=m, mode=mo, caption=tok.decode(gen, skip_special_tokens=True), tokens=gen, ms=(time.perf_counter() - t1) * 1000,
                             steps=st["steps"], single=st["single"], neg_calls=st["neg_calls"], entropy_skips=st.get("entropy_skips", 0)))
        with open(out, "a") as f:                                  # one line per (image, method, mode); resumable per image
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"shard {shard}: image {k+1}/{len(mine)} done  ({time.time()-t0:.0f}s)", flush=True)
    if smoke:
        for l in list(open(out))[:8]:
            r = json.loads(l); print(f"  smoke {r['method']:5}{r['mode']:7} {r['ms']:7.0f} ms  neg_calls {r['neg_calls']:3}/{r['steps']:3}  {r['caption'][:90]}", flush=True)


# =============================================================================== CHAIR evaluation
def chair_eval():
    import glob
    chair = Chair(E.p("synonyms.txt")); gt = json.load(open(E.p("chair_gt.json"))); recs = []
    for f in sorted(glob.glob(E.p("chair_shard[0-9].jsonl"))):
        recs += [json.loads(l) for l in open(f)]
    if not recs:
        print("no caption files found"); return {}
    lines, N = [], {}
    def say(s=""):
        print(s); lines.append(s)
    per = {}                                                   # (method, mode) -> image -> dict
    for r in recs:
        ws, nodes = chair.words(r["caption"]); g = chair.gt_objects(gt[r["image"]]["objects"], gt[r["image"]]["captions"])
        hall = [x for x in nodes if x not in g]
        per.setdefault((r["method"], r["mode"]), {})[r["image"]] = dict(m=len(nodes), h=len(hall), cov=(len(set(nodes) & g) / len(g)) if g else 0.0,
                                                                         toks=len(r["tokens"]), ms=r["ms"], cap=r["caption"], hw=hall, single=r["single"], neg=r["neg_calls"], steps=r["steps"])
    common = sorted(set.intersection(*[set(v) for v in per.values()])); n = len(common)
    rng = np.random.RandomState(0); B = 2000; idx = rng.randint(0, n, (B, n))
    def arr(k, key): return np.array([per[k][im][key] for im in common], float)
    say(f"=== CHAIR caption-quality experiment: {n} images with all methods, LLaVA-1.5-7B, prompt 'Describe this image in detail.'")
    say(f"{'method|mode':14}{'CHAIR_s %':>12}{'  95% CI':>16}{'CHAIR_i %':>12}{'  95% CI':>16}{'coverage %':>12}{'tokens':>8}{'ms/caption':>12}")
    stats = {}
    for k in sorted(per, key=lambda x: (x[1], ["base", "vcd", "evcd", "egp"].index(x[0]))):
        h, m, c = arr(k, "h"), arr(k, "m"), arr(k, "cov"); hs = (h > 0).astype(float)
        cs, ci_ = hs.mean() * 100, h.sum() / m.sum() * 100
        bs = hs[idx].mean(1) * 100; bi = h[idx].sum(1) / m[idx].sum(1) * 100
        stats[k] = dict(chair_s=cs, chair_i=ci_, cov=c.mean() * 100, tokens=arr(k, "toks").mean(), ms=arr(k, "ms").mean(), hs=hs, h=h, m=m)
        say(f"{k[0] + '|' + k[1]:14}{cs:>12.1f}{'[%.1f,%.1f]' % tuple(np.percentile(bs, [2.5, 97.5])):>16}{ci_:>12.1f}{'[%.1f,%.1f]' % tuple(np.percentile(bi, [2.5, 97.5])):>16}"
            f"{c.mean()*100:>12.1f}{arr(k, 'toks').mean():>8.1f}{arr(k, 'ms').mean():>12.0f}")
    say("\npaired differences (same images; 95% bootstrap CI over images; negative CHAIR = fewer hallucinations)")
    for mode in sorted({k[1] for k in per}):
        for a, b, lab in [("vcd", "base", "VCD - base"), ("evcd", "vcd", "exact-gated - VCD"), ("egp", "vcd", "entropy+exact gated - VCD"), ("egp", "base", "entropy+exact gated - base")]:
            if (a, mode) not in per or (b, mode) not in per:
                continue
            A, Bq = stats[(a, mode)], stats[(b, mode)]
            ds = (A["hs"] - Bq["hs"]) * 100; dci = (A["h"][idx].sum(1) / A["m"][idx].sum(1) - Bq["h"][idx].sum(1) / Bq["m"][idx].sum(1)) * 100
            say(f"  {mode:7}{lab:30} CHAIR_s {ds.mean():+6.1f} [{np.percentile(ds[idx].mean(1), 2.5):+.1f},{np.percentile(ds[idx].mean(1), 97.5):+.1f}]   CHAIR_i {(A['chair_i']-Bq['chair_i']):+6.2f} [{np.percentile(dci, 2.5):+.2f},{np.percentile(dci, 97.5):+.2f}]")
    for mode in sorted({k[1] for k in per}):
        if ("vcd", mode) in per and ("evcd", mode) in per:
            same = sum(per[("vcd", mode)][im]["cap"] == per[("evcd", mode)][im]["cap"] for im in common)
            say(f"identical captions, VCD vs exact-gated ({mode}): {same}/{n}")
    for mode in sorted({k[1] for k in per}):
        if ("egp", mode) in per:
            e = per[("egp", mode)]; say(f"entropy+exact gate ({mode}): second pass at {sum(e[i]['neg'] for i in common)/sum(e[i]['steps'] for i in common)*100:.1f}% of steps (VCD: 100%, exact gate: {sum(per[('evcd', mode)][i]['neg'] for i in common)/sum(per[('evcd', mode)][i]['steps'] for i in common)*100:.1f}%)")
    ex = [(im, per[("base", "greedy")][im]["hw"], per[("vcd", "greedy")][im]["hw"]) for im in common if ("base", "greedy") in per and per[("base", "greedy")][im]["h"] != per[("vcd", "greedy")][im]["h"]][:3]
    for im, hb, hv in ex:
        say(f"\nexample {im}: hallucinated objects  base {hb}  | VCD {hv}\n   base: {per[('base', 'greedy')][im]['cap'][:230]}\n   VCD : {per[('vcd', 'greedy')][im]['cap'][:230]}")
    N = {f"{k[0]}|{k[1]}": {a: float(b) for a, b in v.items() if a in ("chair_s", "chair_i", "cov", "tokens", "ms")} for k, v in stats.items()}; N["n_images"] = n
    json.dump(N, open(E.p("chair_numbers.json"), "w"), indent=1); open(E.p("chair_report.txt"), "w").write("\n".join(lines)); return N


# =============================================================================== numerics: why do eager and exact-gated VCD sometimes diverge?
def run_numerics(n_caps, max_new):
    import torch
    from PIL import Image
    proc, model = E.load_model(); tok = proc.tokenizer; eos = tok.eos_token_id; ops = TorchOps2()
    names, seen = [], set()
    for r in S.rows_for("random"):
        if r["image"] not in seen:
            seen.add(r["image"]); names.append(r["image"])
    names = names[:n_caps]; prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    maxabs, tv, margins_flip, steps_total, needed, flips = [], [], [], 0, 0, 0
    for k, name in enumerate(names):
        qid = k + 1
        inp = proc(images=Image.open(E.p("images", name)).convert("RGB"), text=prompt, return_tensors="pt")
        inp = {a: b.to(model.device) for a, b in inp.items()}; inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        pv = inp["pixel_values"]; ids = inp["input_ids"][0].tolist(); pvn = E.diffusion_noise(pv, E.NOISE_STEP, seed=qid)
        gen, _ = M.decode("vcd", "greedy", ops, ids, M.HFBranch(model, pv), lambda: M.HFBranch(model, pvn), qid, max_new, eos); T = len(gen)
        pos = M.HFBranch(model, pv); lo = [pos.prefill(ids)]                   # one KV cache alive at a time (3 at once ran out of memory on a T4)
        for t in range(T - 1):
            lo.append(pos.extend([gen[t]]))
        del pos; torch.cuda.empty_cache()
        eager = M.HFBranch(model, pvn); ld_e = [eager.prefill(ids)]            # token-by-token, as in standard VCD
        for t in range(T - 1):
            ld_e.append(eager.extend([gen[t]]))
        del eager; torch.cuda.empty_cache()
        lazy, pend, ld_l = None, [], {}                                         # lazy prefill + batched catch-up, as in the exact gate
        for t in range(T):
            if not ops.single(lo[t]):
                if lazy is None:
                    lazy = M.HFBranch(model, pvn); ld_l[t] = lazy.prefill(ids + gen[:t])
                else:
                    ld_l[t] = lazy.extend(pend)
                pend = []
            if t < T - 1:
                pend.append(gen[t])
        steps_total += T; needed += len(ld_l)
        for t, l_l in ld_l.items():
            head = lo[t] >= lo[t].max() + LOGB
            maxabs.append(float((l_l - ld_e[t]).abs()[head].max()))
            ce, cl = E.cd_logits(lo[t], ld_e[t]), E.cd_logits(lo[t], l_l)
            pe, pl = torch.softmax(ce[head], -1), torch.softmax(cl[head], -1); tv.append(float(0.5 * (pe - pl).abs().sum()))
            if int(ce.argmax()) != int(cl.argmax()):
                flips += 1; margins_flip.append(float(torch.topk(ce, 2).values.diff().abs()))
        print(f"numerics {k+1}/{len(names)}  steps {T}  contrast steps {len(ld_l)}", flush=True)
        del lazy, lo, ld_e, ld_l, pvn, pv, inp; torch.cuda.empty_cache()
    maxabs, tv = np.array(maxabs), np.array(tv)
    out = dict(captions=len(names), steps=steps_total, contrast_steps=needed, logit_diff_head_tokens=dict(mean=float(maxabs.mean()), median=float(np.median(maxabs)), p99=float(np.percentile(maxabs, 99)), max=float(maxabs.max())),
               tv_distance=dict(mean=float(tv.mean()), sum=float(tv.sum()), max=float(tv.max())), expected_sampled_flips_per_1000_tokens=float(tv.sum() / steps_total * 1000),
               greedy_flips=flips, greedy_flip_margins=margins_flip)
    json.dump(out, open(E.p("numerics.json"), "w"), indent=1); print(json.dumps(out, indent=1), flush=True)


# =============================================================================== bring-your-own real-domain dataset (COCO-format detection json)
def make_ood(coco_json, images_dir, n_images, seed=0, per_split=3):
    d = json.load(open(coco_json)); cat = {c["id"]: c["name"].replace("_", " ").replace("-", " ") for c in d["categories"]}
    img = {i["id"]: i["file_name"] for i in d["images"]}; objs = {}
    for a in d["annotations"]:
        objs.setdefault(a["image_id"], set()).add(cat[a["category_id"]])
    ids = sorted([i for i, s in objs.items() if len(s) >= 3 and i in img]); rng = np.random.RandomState(seed); rng.shuffle(ids); ids = ids[:n_images]
    cats = sorted(cat.values()); freq = {c: sum(c in s for s in objs.values()) for c in cats}
    co = {a: {b: sum(a in s and b in s for s in objs.values()) for b in cats} for a in cats}
    os.makedirs(E.p("images"), exist_ok=True); rows = {s: [] for s in E.SPLITS}; qid = 0
    for i in ids:
        present = sorted(objs[i]); absent = [c for c in cats if c not in objs[i]]
        if len(absent) < per_split:
            continue
        name = img[i]; src = os.path.join(images_dir, name)
        if not os.path.exists(src):
            continue
        shutil.copy(src, E.p("images", name))
        pos = list(rng.choice(present, size=min(per_split, len(present)), replace=False))
        negs = {"random": list(rng.choice(absent, size=per_split, replace=False)),
                "popular": sorted(absent, key=lambda c: -freq[c])[:per_split],
                "adversarial": sorted(absent, key=lambda c: -sum(co[c][p] for p in present))[:per_split]}
        for s in E.SPLITS:
            for label, items in (("yes", pos), ("no", negs[s])):
                for c in items:
                    qid += 1; rows[s].append(dict(question_id=qid, image=name, text=f"Is there a {c} in the image?", label=label))
    used = len({r["image"] for r in rows["random"]})
    print(f"used {used} of the requested {len(ids)} images (images need >=3 object classes, >=3 absent classes, and a file in images_dir)")
    if used < 20:
        print("WARNING: very few images -> the OOD result will be too noisy to report; check --images_dir and the category count")
    for s in E.SPLITS:
        with open(E.p(f"pope_{s}_ood.jsonl"), "w") as f:
            for r in rows[s]:
                f.write(json.dumps(r) + "\n")
        print(s, "OOD questions:", len(rows[s]))
    print("next: python egvcd_solid.py cache2 --cond clean --suffix _ood --outdir cache")


# =============================================================================== offline report
def run_analyze3():
    lines, N = [], {}
    def say(s=""):
        print(s); lines.append(s)
    sets = [("LLaVA-1.5-7B, 150 imgs (D1)", "cache", "", None), ("LLaVA-1.5-7B, 150 new imgs (D2)", "cache", "_ext", None), ("LLaVA-1.5-13B, 100 imgs", "cache_13b", "", 100),
            ("Qwen2-VL-7B, 100 imgs", "cache_qwen2vl", "", 100), ("LLaVA-NeXT-7B, 100 imgs", "cache_llavanext", "", 100), ("LLaVA-1.5-7B, bring-your-own OOD set", "cache", "_ood", None)]
    say("=== 1. CONSISTENCY ACROSS MODELS AND IMAGE SETS (clean, POPE, image-clustered 95% CIs)")
    say(f"{'setting':38}{'n_q':>6}{'needs2nd%':>10}{'skip%':>7}  {'VCD gain sampled':>22}  {'VCD gain greedy':>22}  {'APC%':>5}  FYR smp      FYR grd     EG+.5 rho  keep   excess")
    have, rows = [], []
    for name, d, suf, nimg in sets:
        r = S.load_set(d, "clean", suf, nimg)
        if r is None:
            continue
        s = S.summarize(*r); N[name] = s; rows.append(s); have.append((name, r)); say(M._line(name[:37], s))
    if rows:
        sk = [100 - s["rho_exact"] * 100 for s in rows]
        say(f"skippable share across {len(rows)} settings: min {min(sk):.1f}%  max {max(sk):.1f}% | exactness violated in any: {any(s['exact_maxdiff'] > 0 or not s['greedy_exact_identical'] for s in rows)}")
    say("\n=== 2. alpha / beta SWEEP ON EVERY DATASET (offline). exact = max|p_exact-gated - p_VCD|; accuracies in %, FYR = false-yes rate")
    say(f"{'setting':38}{'alpha':>6}{'beta':>6}{'skip%':>7}{'exact':>7}  {'VCD acc smp/grd':>16}  {'base acc smp/grd':>17}  {'VCD FYR smp':>12}  {'base FYR smp/grd':>17}")
    for name, (c, _) in have:
        y = c["label"]
        for a, b in [(1.0, 0.1), (1.0, 0.2), (1.0, 0.5), (0.5, 0.5)]:
            P = S.probs2(c, a, b); acc = lambda p: float(np.where(y, p, 1 - p).mean() * 100)
            ex = float(np.abs(np.where(~P["S"], P["p_vcd"], P["top"]) - P["p_vcd"]).max())
            say(f"{name[:37]:38}{a:>6}{b:>6}{P['S'].mean()*100:>7.1f}{ex:>7.0e}  {acc(P['p_vcd']):>7.2f}/{(P['gv'] == y).mean()*100:<7.2f}  {acc(P['p_base']):>8.2f}/{(P['gb'] == y).mean()*100:<7.2f}  {P['p_vcd'][~y].mean()*100:>12.1f}  {P['p_base'][~y].mean()*100:>9.1f}/{P['gb'][~y].mean()*100:<6.1f}")
    say("\n=== 3. VCD UNDER IMAGE DEGRADATION, PER MODEL (sampled gain over the sampled baseline, image-clustered 95% CI)")
    say(f"{'model':16}{'condition':10}{'base acc':>9}{'skip%':>7}  {'VCD gain sampled':>22}  {'VCD gain greedy':>22}  {'FYR base->VCD (smp)':>20}")
    for mname, d, nimg in [("LLaVA-1.5-7B", "cache", None), ("Qwen2-VL-7B", "cache_qwen2vl", 100), ("LLaVA-NeXT-7B", "cache_llavanext", 100)]:
        for cond in E.CONDS:
            r = S.load_set(d, cond, "", nimg)
            if r is None:
                continue
            s = S.summarize(*r); say(f"{mname:16}{cond:10}{s['acc_base']:>9.2f}{100-s['rho_exact']*100:>7.1f}  {s['sample_gain']:+6.2f} [{s['sample_gain_ci'][0]:+.2f},{s['sample_gain_ci'][1]:+.2f}]   {s['greedy_gain']:+6.2f} [{s['greedy_gain_ci'][0]:+.2f},{s['greedy_gain_ci'][1]:+.2f}]   {s['fyr_sample']['base']:>7.1f} -> {s['fyr_sample']['vcd']:.1f}")
    if os.path.exists(E.p("numerics.json")):
        n = json.load(open(E.p("numerics.json"))); N["numerics"] = n
        say("\n=== 4. NUMERICS: eager token-by-token negative branch vs lazy catch-up negative branch (teacher-forced, same tokens)")
        say(f"{n['captions']} captions, {n['steps']} steps, {n['contrast_steps']} steps needing the contrast")
        d = n["logit_diff_head_tokens"]; say(f"|logit difference| on plausible tokens: mean {d['mean']:.4f}  median {d['median']:.4f}  p99 {d['p99']:.4f}  max {d['max']:.4f}  (fp16 resolution near logit 10 is about 0.008)")
        say(f"total-variation distance between the two contrast distributions: mean {n['tv_distance']['mean']:.5f} per contrast step  -> expected sampled flips {n['expected_sampled_flips_per_1000_tokens']:.2f} per 1000 tokens (observed earlier: 7 flips in ~5600 steps = 1.25 per 1000)")
        say(f"greedy argmax flips: {n['greedy_flips']}  at contrast-logit margins {[round(x, 4) for x in n['greedy_flip_margins']]}")
    json.dump(N, open(E.p("quality_numbers.json"), "w"), indent=1, default=float); open(E.p("quality_report.txt"), "w").write("\n".join(lines)); say("\nwrote quality_numbers.json and quality_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare_chair", "chair_gen", "chair_eval", "numerics", "make_ood", "analyze3"])
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--modes", default="greedy,sample"); ap.add_argument("--max_new", type=int, default=96); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_caps", type=int, default=20); ap.add_argument("--coco_json", default=""); ap.add_argument("--images_dir", default=""); ap.add_argument("--n_images", type=int, default=100)
    a = ap.parse_args(); os.makedirs(E.ROOT, exist_ok=True)
    if a.cmd == "prepare_chair": prepare_chair(a.n)
    elif a.cmd == "chair_gen": chair_gen(a.shard, a.nshards, a.n, a.modes.split(","), a.max_new, a.smoke)
    elif a.cmd == "chair_eval": chair_eval()
    elif a.cmd == "numerics": run_numerics(a.n_caps, a.max_new)
    elif a.cmd == "make_ood": make_ood(a.coco_json, a.images_dir, a.n_images)
    else: run_analyze3()
