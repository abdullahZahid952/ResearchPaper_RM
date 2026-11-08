# Exact and Entropy-Gated Visual Contrastive Decoding (EG-VCD)

Code for the paper *"Exact and Entropy-Gated Visual Contrastive Decoding: Cost-Efficient Object Hallucination Mitigation in LLaVA-1.5 Under Image Degradation"* (Abdullah Bin Zahid, 26I-7833, FAST-NUCES, Research Methodology).

## What this repository does
1. Reproduces the VCD baseline (Leng et al., CVPR 2024) on LLaVA-1.5-7B with POPE-COCO (150-image subset).
2. Implements the proposed component: an **exact gate** (skip the second image pass when VCD's plausibility set has one token - provably lossless) and an **entropy gate** (lossy).
3. Evaluates all methods on clean images and three synthetic shifts (low-light, noise, blur), plus latency on a Kaggle T4.

## Files
| File | Purpose |
|---|---|
| `KAGGLE_RUN_ALL.ipynb` / `.py` | One cell for Kaggle (2x T4, Internet ON): installs, downloads data/model, caches logits on both GPUs, times, analyses |
| `egvcd.py` | Core implementation (data prep, corruptions, VCD, gates, caching, timing, analysis) |
| `paper_analysis.py` | Regenerates every table number and figure in the paper from `results/` |
| `test_logic.py` | Offline unit checks (gate equivalences, metrics, sampling), no GPU needed |
| `results/` | `results.csv`, `timing.json`, `paper_numbers.json`, cached logits (`cache/*.npz`) |
| `figures/` | Figures used in the paper |
| `logs/kaggle_console_output.txt` | Condensed console output of the Kaggle run |

## Reproduce
- Full run: upload `KAGGLE_RUN_ALL.ipynb` to Kaggle (GPU T4 x2, Internet ON) and run the single cell (~2.2 h).
- Only the analysis/figures (CPU, ~1 min): `python paper_analysis.py results paper_out`
- Offline test: `python test_logic.py`

## Notes / limitations (see paper, Section IV-G)
One backbone, POPE single-token answers, synthetic shifts, thresholds swept on the evaluation data, latency of the exact gate is modelled (not measured).
