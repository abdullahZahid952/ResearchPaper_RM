"""Mock test of egvcd_full.py (no GPU/model needed): fake model, 10 images, 2 shards -> pooling + analysis run end to end.
Needs torch + pillow:  python test_full.py"""
import os, sys, json, tempfile
R = tempfile.mkdtemp(); os.environ["EGVCD_ROOT"] = R; os.makedirs(R + "/images")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import torch, egvcd as E, egvcd_solid as S, egvcd_full as F
imgs = [f"im{i}.jpg" for i in range(10)]
for n in imgs: Image.new("RGB", (64, 64)).save(f"{R}/images/{n}")
for s in F.SPLITS:
    with open(f"{R}/pope_{s}_full.jsonl", "w") as f:
        for i, n in enumerate(imgs):
            for j in range(2): f.write(json.dumps({"image": n, "text": "Is there a cat?", "label": "yes" if (i + j) % 2 else "no", "question_id": i * 10 + j}) + "\n")
class Tok:
    def decode(self, ids): return "yes" if ids[0] % 2 else "No"
class Proc: tokenizer = Tok()
class M: device = "cpu"
S.load_model2 = lambda *a, **k: (Proc(), M())
E.build_inputs = lambda proc, model, img, q: {"pixel_values": torch.zeros(1, 3, 8, 8, dtype=torch.float16)}
g = torch.Generator().manual_seed(0)
E.last_logits = lambda model, inp, pv: torch.randn(100, generator=g) * 3
E.diffusion_noise = lambda x, step, seed: x
for k in (0, 1): F.cache_shard(k, 2)
F.analyze_full()
N = json.load(open(R + "/full_numbers.json"))
assert N["all"]["n_q"] == 60 and N["all"]["exact_maxdiff"] == 0.0 and N["all"]["greedy_exact_identical"]
print("test_full OK")
