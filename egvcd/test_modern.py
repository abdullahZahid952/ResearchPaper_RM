"""GPU-free mock test of egvcd_modern.py (fake processor/model): smoke test -> cache -> generation stats -> analysis.
Needs torch + pillow:  python test_modern.py"""
import os, sys, json, tempfile
R = tempfile.mkdtemp(); os.environ["EGVCD_ROOT"] = R; os.makedirs(R + "/images")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import torch, egvcd as E, egvcd_modern as M
imgs = [f"im{i}.jpg" for i in range(6)]
for n in imgs: Image.new("RGB", (64, 64)).save(f"{R}/images/{n}")
for s in M.SPLITS:
    with open(f"{R}/pope_{s}.jsonl", "w") as f:
        for i, n in enumerate(imgs):
            for j in range(2): f.write(json.dumps({"image": n, "text": "Is there a cat?", "label": "yes" if (i + j) % 2 else "no", "question_id": i * 10 + j}) + "\n")
class Tok:
    def decode(self, ids): return ["Yes", "No", " yes"][ids[0] % 3]
class Proc:
    tokenizer = Tok()
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
        if "enable_thinking" in kw: raise TypeError("older template")      # exercises the fallback
        return "TEXT"
    def __call__(self, images, text, return_tensors):
        return {"input_ids": torch.ones(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long), "pixel_values": torch.randn(1, 3, 8, 8)}
class Out: pass
class Model:
    device = "cpu"; dtype = torch.float16
    def __call__(self, input_ids, attention_mask, pixel_values, use_cache, **kw):
        if "logits_to_keep" in kw: raise TypeError("unsupported")
        o = Out(); lg = torch.randn(1, 5, 50) * 3; lg[0, -1, 0] += 6; o.logits = lg; return o
    def generate(self, max_new_tokens, **kw):
        o = Out(); o.logits = [torch.randn(1, 50) * 3 for _ in range(7)]; return o
M.load = lambda key, dtype: (Proc(), Model())
assert M.singleton_steps(torch.tensor([[5., 2., 0.], [5., 2.8, 0.]])).tolist() == [True, False]
M.run("qwen3vl_4b", n_images=6, n_gen=3, max_new=7)
M.analyze_modern()
assert json.load(open(R + "/modern_numbers.json"))["qwen3vl_4b"]["summary"]["exact_maxdiff"] == 0.0
print("test_modern OK")
