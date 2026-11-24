# Exact Early Exit for Visual Contrastive Decoding (EG-VCD)

Code, results and logs for the paper *"Exact Early Exit for Visual Contrastive Decoding in Vision-Language Models: A Lossless Gate for the Second Image Pass and a Critical Re-evaluation"* (Abdullah Bin Zahid, 26I-7833, FAST-NUCES, Research Methodology, Dr. Akhtar Jamil).

## Idea in three lines
* VCD (Leng et al., CVPR 2024) needs two image forward passes per token. Its adaptive plausibility constraint keeps only tokens with p >= beta * p_max.
* If the runner-up logit is below the top logit + ln(beta), exactly one token survives, so the second (noised-image) pass **cannot** change VCD's output. Skipping it is lossless (Proposition 1; tested: 0 mismatches).
* An entropy gate (lossy) skips more passes; we measure how much of VCD's gain it keeps, against a random-gating control.

## Headline results (all from the runs in `results/` and `logs/`)
| Setting | Skippable 2nd passes | VCD sampled gain (image-clustered 95% CI) | Greedy gain |
|---|---|---|---|
| LLaVA-1.5-7B, 150 POPE-COCO images | 63.9% | +3.38 [+2.52, +4.34] | -0.85 [-2.37, +0.81] |
| LLaVA-1.5-7B, 150 held-out images | 62.4% | +3.15 [+2.36, +3.94] | -1.15 [-2.67, +0.41] |
| LLaVA-1.5-13B, 100 images | 66.6% | +3.17 [+2.41, +3.87] | +1.33 [-0.33, +3.06] |
| Qwen2-VL-7B, 100 images | 57.7% | +1.82 [+1.19, +2.45] | -0.44 [-1.67, +0.83] |
| LLaVA-NeXT-Mistral-7B, 100 images | 76.0% | +3.43 [+2.44, +4.40] | +1.06 [-0.67, +2.83] |
| LLaVA-1.5-7B, **real aerial domain (DIOR)**, 100 images | 34.1% | +2.75 [+1.53, +4.03] | -0.67 [-2.94, +1.72] |

* End-to-end (300 questions, real decoders): 0 output differences between VCD and the exact gate; measured latency 1009 ms -> 712 ms (-29.5%).
* Captions: latency -17..22%, outputs near-identical (141/150 greedy, 136/150 sampled identical; fp16-scale differences), CHAIR within noise.
* Skippable share drops out of domain (34% on aerial photos) so the saving shrinks there.

## Files
| File | Purpose |
|---|---|
| `egvcd.py` | Core: data prep, corruptions, VCD, gates, caching, timing, analysis |
| `egvcd_solid.py` | End-to-end verification and measured latency, held-out images, 13B, alpha/beta sweep, image-clustered CIs |
| `egvcd_more.py` | Multi-token generation with lazy negative-branch catch-up, noise-seed robustness, Qwen2-VL / LLaVA-NeXT caching |
| `egvcd_quality.py` | CHAIR port, numerics diagnostic, real-domain (OOD) question builder, per-model report |
| `fetch_ood.py` (inside `KAGGLE_OOD.py`) | Fetches the aerial DIOR images (Hugging Face `Marcel031/dior`) |
| `paper_analysis.py` | Regenerates the original tables and figures from `results/` |
| `audit_eval.py`, `AUDIT.md`, `results/audit_*` | CPU audit of answer parsing, top-32 normalisation, accuracy formula and coupling (answers to peer review) |
| `test_logic.py`, `test_more.py`, `test_quality.py` | Offline unit tests, no GPU |
| `KAGGLE_RUN_ALL`, `KAGGLE_SOLID`, `KAGGLE_MORE`, `KAGGLE_QUALITY`, `KAGGLE_NUMERICS_FIX`, `KAGGLE_OOD`, `KAGGLE_NEXT` (`.ipynb`/`.py`) | One-cell Kaggle notebooks (2x T4, Internet ON), in the order they were run |
| `results/` | Original run (`results.csv`, `timing.json`, `cache/`), real-domain + LLaVA-NeXT run (`results/ood/`), and every raw run as a zip in `results/raw_runs/` |
| `logs/` | Console output of every Kaggle run |
| `figures/` | Figures used in the paper |

## Reproduce
1. Run order on Kaggle (GPU T4 x2, Internet ON, one cell each): `KAGGLE_RUN_ALL` -> `KAGGLE_SOLID` -> `KAGGLE_MORE` -> `KAGGLE_QUALITY` (+ `KAGGLE_NUMERICS_FIX`) -> `KAGGLE_OOD` (set `RUN_NEXT="0"` to skip LLaVA-NeXT, ~4 h). Later notebooks optionally attach earlier result zips as Kaggle datasets.
2. CPU only: `python paper_analysis.py results paper_out`, `python egvcd_quality.py analyze3`, `python test_logic.py`.
3. Audit: `python audit_eval.py --root <folder containing cache*/>`.

## Limits (see paper Section IV-H)
Four LVLMs from three families, 100-150 images per setting; POPE (and CHAIR on LLaVA-1.5-7B); one real domain (aerial photos, labels from DIOR's 20 annotated classes so unannotated objects count as absent); synthetic shifts have one severity; only VCD-style contrastive decoding was implemented (no DAMRO/MaskCD/AGLA comparison); noise schedule independently implemented; sampled POPE numbers are analytic expectations from cached top-32 logits (audit bounds the effect at <=0.04 points); the exact gate is a direct consequence of VCD's plausibility constraint, so the contribution is its characterisation and measured saving, not a new decoding method.


## Full POPE-COCO run (500 images, 9,000 questions)
`KAGGLE_FULL.py` / `.ipynb` (one cell, T4 x2, ~1.5 h) runs `egvcd_full.py` on the complete benchmark and writes `full_report.txt` / `full_numbers.json`; `test_full.py` is a GPU-free mock test of the pipeline.

## Modern models (2025-2026)
`KAGGLE_MODERN.py` / `.ipynb` (one cell, T4 x2, ~1.5-2 h) runs `egvcd_modern.py` on Qwen3-VL-4B, Qwen3.5-4B, Gemma-4-E4B and InternVL3.5-8B: POPE skip rate / VCD gain / gate retention, plus the share of free-form decoding steps where the VCD head has a single token. `test_modern.py` is a GPU-free mock test.

## Reviewer questions (Nov 2026)
- `review_offline.py` (CPU only): threshold (beta) sweep, skip-criterion comparison at equal budget, truncation-only vs VCD. Output: `results/review/review_report.txt`.
- `KAGGLE_REVIEW.py` / `.ipynb` (one cell, T4 x2, ~1.5-2 h) runs `egvcd_review.py`: deployment benchmark against batched VCD (latency, throughput, memory), exactness under greedy / sampling / beam search, Qwen3-VL-4B on 300 COCO images, POPE-A-OKVQA. `KAGGLE_REVIEW_FIX.py` reruns sampling / beam search and adds batch size 4. `test_review.py` is a GPU-free mock test.

### Results added after the first version
| Folder / file | What it holds |
|---|---|
| `results/full/` | Full POPE-COCO run, LLaVA-1.5-7B, 500 images / 9,000 questions (`full_report.txt`, caches) |
| `results/modern/modern_report.txt` | Qwen3-VL-4B, Qwen3.5-4B, Gemma-4-E4B (100 images each) + free-form generation step statistics |
| `results/review/review_report.txt` | Threshold sweep, skip-criterion comparison, truncation-only vs VCD on the earlier caches |
| `results/review_gpu/` | Deployment benchmark (`bench.json`), generation exactness (`exactgen_*.json`), Qwen3-VL-4B on 300 COCO images, POPE-A-OKVQA for LLaVA-1.5-7B and Qwen3-VL-4B (`review_gpu_report.txt`, `review_report_new_data.txt`) |
| `logs/kaggle_modern_console.txt`, `logs/kaggle_review_console.txt`, `logs/kaggle_review_fix_console.txt` | Console output of those runs |
| `figures/fig1_idea.png`, `fig2_skip_rate_by_model.png`, `fig3_speed_and_accuracy.png` | Summary figures |

### Key numbers from the review experiments
- Exact gate: 0 output differences in all 11 model/dataset settings, at every threshold 0.01-0.5, and in captions under greedy (20/20), sampling (20/20) and beam search (12/12).
- Against VCD with both images in one batch (LLaVA-1.5-7B, T4): 758 vs 982 ms per question at batch 1 (-23%), 677 vs 969 ms at batch 4 (-30%); batched VCD runs out of memory at batch 8.
- Skip rate at beta=0.1: 34% (aerial) to 94% (Gemma-4-E4B); 10% to 90% for LLaVA-1.5-7B as beta goes from 0.01 to 0.5.
- 23-92% of VCD's sampled gain comes from the plausibility truncation alone; under greedy decoding VCD gives no significant gain in any setting and significantly hurts LLaVA-1.5-7B on POPE-A-OKVQA (-2.47 points).
