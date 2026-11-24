"""GPU-free mock test of egvcd_review.py: fake LLaVA (vocab 60) -> bench, exactgen (greedy / sampling / beam), cache, analyze.
Needs torch + pillow:  python test_review.py"""
import os, sys, json, tempfile
R = tempfile.mkdtemp(); os.environ["EGVCD_ROOT"] = R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import torch, egvcd as E, egvcd_solid as S, egvcd_review as V
os.makedirs(V.rp(), exist_ok=True); os.makedirs(E.p("images"), exist_ok=True)
imgs = [f"im{i}.jpg" for i in range(8)]
for i, n in enumerate(imgs): Image.new("RGB", (32, 32), (i * 30, 10, 200 - i * 20)).save(E.p("images", n))
json.dump(imgs, open(V.rp("images_coco.json"), "w"))
for s in V.SPLITS:
    with open(V.rp(f"pope_{s}_coco.jsonl"), "w") as f:
        for i, n in enumerate(imgs):
            for j in range(3): f.write(json.dumps({"image": n, "text": f"Is there a thing {j}?", "label": "yes" if (i + j) % 2 else "no", "question_id": i * 10 + j}) + "\n")
V_ = 60; EOS = 2
class Tok:
    eos_token_id = EOS; padding_side = "right"
    def decode(self, ids): return ["yes", "no"][ids[0] % 2]
class Proc:
    tokenizer = Tok()
    def __call__(self, images, text, return_tensors, padding=False):
        ims = images if isinstance(images, list) else [images]; txt = text if isinstance(text, list) else [text]
        L = max(len(t) for t in txt) % 7 + 5
        ids = torch.stack([torch.tensor([(len(t) * 7 + k) % 50 + 3 for k in range(L)]) for t in txt])
        pv = torch.stack([torch.tensor(list(im.getpixel((0, 0))), dtype=torch.float32)[:, None, None].expand(3, 4, 4) / 255 for im in ims])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": pv}
W = torch.randn(V_, V_, generator=torch.Generator().manual_seed(0)) * 2.5
class Out: pass
class Model:
    device = torch.device("cpu"); dtype = torch.float16
    def __call__(self, input_ids, attention_mask, pixel_values, use_cache, logits_to_keep=None):
        last = input_ids[:, -1]; L = input_ids.shape[1]
        img = pixel_values.float().reshape(pixel_values.shape[0], -1).mean(1)
        lg = W[last] + 4 * torch.sin(img[:, None] * 30 + torch.arange(V_)[None] * 0.7 + L * 0.3)
        o = Out(); o.logits = lg[:, None, :].expand(-1, L, -1).clone(); return o
S.load_model2 = lambda *a, **k: (Proc(), Model())
E.build_inputs = lambda proc, model, img, q: {k: v for k, v in proc(images=img, text=E.PROMPT.format(q=q), return_tensors="pt").items()}
_ts = torch.cuda.synchronize; torch.cuda.synchronize = lambda *a: None
torch.cuda.reset_peak_memory_stats = lambda *a: None; torch.cuda.memory_allocated = lambda *a: 0; torch.cuda.max_memory_allocated = lambda *a: 0
torch.cuda.empty_cache = lambda *a: None
V.bench(16, batch_sizes=(1, 4), warm=2)
b = json.load(open(V.rp("bench.json")))
assert all(r.get("answers_differing_from_batched_VCD", 0) == 0 for r in b.values() if "exact gate" in r["method"]), b
V.exactgen(4, 10)
g = {m: json.load(open(V.rp(f"exactgen_{m}.json"))) for m in ("greedy", "sample", "beam")}
for m, r in g.items():
    assert r["identical"] == r["n"], (m, r); assert r["neg_gate"] <= r["neg_vcd"]
    assert r["singleton_steps"] > 0 and r["neg_gate"] > 0, (m, r)     # both branches of the gate exercised
V.get_model = lambda key: (Proc(), Model(), (lambda img, q: E.build_inputs(Proc(), None, img, q)), (lambda inp, pv: Model()(inp["input_ids"], None, pv, False).logits[0, -1].float()))
V.cache("llava7b", "coco", None)
V.analyze()
print("test_review OK", {m: (r["identical"], r["neg_vcd"], r["neg_gate"]) for m, r in g.items()})
