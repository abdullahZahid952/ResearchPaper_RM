# Evaluation audit (CPU-only, on the saved caches)

Peer reviewers raised: (1) lenient yes/no parser, (2) top-32 renormalisation of the baseline arm, (3) accuracy computed as 1-P(yes),
(4) shared-uniform sampling is not a maximal coupling, (5) caption savings are deferred, not eliminated.

Run: `python audit_eval.py --root <dir containing cache*/>` (needs sentencepiece or HF tokenizers). Output: results/audit_out.json, results/audit_report.txt.

Findings (LLaVA-1.5-7B x 12 + 3 held-out, 13B x 3, 12 noise-draw caches; Qwen2-VL checked tokenizer-free):
* argmax junk tokens: 0 in every cache; APC head contains only Yes/No spellings; all four parsers agree to <=0.01 pt.
* tail mass beyond the cached top-32: <=0.011% (7B), 0.021% (13B), 0.24% (Qwen2-VL). Using the stored full-vocab lse changes every sampled gain by <=0.01 pt (LLaVA) / <=0.04 pt (Qwen).
* VCD/APC arms were already exact (APC head always inside the cache; head_containment_violations = 0).
* Reviewer point (4) is CORRECT: shared-uniform inverse-CDF disagreement >= TV. The 0.59/1000 caption figure is a lower bound; observed 1.25/1000 is consistent.
* Reviewer point (5) is CORRECT: caption saving = deferred/batched negative branch; report latency only.
* Bug fixed in the supplied audit script: greedy baseline indexed row 0 (np.zeros) instead of each row; strict retention printed as a fraction.

Update (final runs): the same checks on the real-domain (aerial DIOR, 3 x 600 questions) and LLaVA-NeXT caches:
* aerial: 0 junk argmax tokens, tail mass <=0.014%, 0 head violations, sampled gains change by <=0.005 pt after the exact normalisation.
* LLaVA-NeXT (tokenizer unavailable offline): the top-1 token is one of two ids (Yes=5592, No=1770) in all 1,800 questions, the APC head contains only those two ids, tail mass <=0.028%.
