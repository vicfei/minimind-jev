# Distillation notes — why the teacher's signal didn't survive (v0.2 arc)

The v0.2 question was: bge-small-zh's frozen embeddings carry enough signal for a 660K head
to reach 89% of hosted Jev's nDCG@5 with plain binary labels — would a frontier teacher's
4-level relevance judgments push it further? The measured answer is **no, it pulls it back**,
and this note documents the arc so the tutorial can teach it with real numbers.

## Setup

- **Teacher**: `deepseek-v4-flash` grading every (query, passage) pair of the T2Reranking
  hard-negative pools into the pattern library's 4-level rubric (irrelevant / related /
  partial / answers), one pooled call per query (10 passages), `thinking` disabled after a
  cost/quality probe, temperature 0.2. 17,390 judgments, zero parse failures, **~¥8**
  off-peak (the naive per-pair shape priced at ~¥56 — see `distill_generate.py`).
- **Teacher quality**: expected-score separation **AUC ≈ 0.716** — the same band as hosted
  Jev on the same kind of pools (0.714–0.718). The signal to distill exists.
- **Student**: the same frozen-encoder + 660K MLP head as v0.1, output widened to 4 logits.
  Ranking score = expected level weight (0, 1/3, 2/3, 1). Three configs: pure soft
  cross-entropy; soft CE + auxiliary BCE on the expected score vs the binary label at
  weight 0.5 and 1.0.

## What happened (s1, n=100, paired 95% CI)

| config | nDCG@5 | vs v0.1 (paired) | vs BM25 (paired) |
|---|---|---|---|
| v0.1 binary BCE | 0.488 | — | +0.075 [+0.002, +0.154] ✓ |
| v0.2 pure distillation | 0.428 | −0.060 [−0.106, −0.018] ✗ | +0.015 [−0.059, +0.096] (n.s.) |
| v0.2.1 mixed w=0.5 | 0.430 | −0.057 [−0.102, −0.019] ✗ | +0.018 [−0.055, +0.092] (n.s.) |
| v0.2.1 mixed w=1.0 | 0.428 | −0.060 [−0.102, −0.021] ✗ | +0.015 [−0.056, +0.088] (n.s.) |

Soft labels did decompress the probabilities — positive-mean expected score rose from 0.114
to ~0.35–0.39, exactly the intended fix for v0.1's prior-compressed outputs — but separation
AUC *dropped* (0.630 → 0.608–0.613). The trade went the wrong way for ranking.

## Why (mechanism, as far as the evidence goes)

1. **The metric is gold ranking, and binary labels ARE the metric.** nDCG against the
   dataset's own relevance labels is directly optimized by BCE on those labels. The
   teacher's rubric agrees with those labels at 81.8% argmax — the 18% disagreement is
   (mostly) rubric granularity, but for this metric it acts as label noise.
2. **The supervision path is weaker, not just softer.** v0.1's BCE pushes the score logit
   directly. The soft config's gradient flows through a 4-way distribution; the mixed
   config's auxiliary BCE sits on the *expected score* — a scalar derived from the softmax —
   so the direct signal reaches the logits only through the distribution's coupling.
   Weight 0.5 → 1.0 barely moved the result, which is why we attribute the gap to the path
   rather than the weight.
3. **Capacity ceiling on frozen features.** The teacher's AUC 0.716 lives in a frontier
   model reading the text; the student reads bge-small-zh's frozen embedding. What the head
   can express is bounded by those features — v0.1 already extracts most of the available
   ranking signal from them.

## The recipe we'd recommend

- If binary gold labels exist for your target metric, **start with them**; treat distillation
  as a controlled contrast, not a default upgrade.
- Distillation earns its keep when labels for the *target* behavior don't exist — grading
  quality, rubric levels, question-form coverage — or when you need decompressed, interpretable
  scores and can pay for ranking.
- Teacher signal quality (AUC 0.716) does not imply distillability; the student's feature
  ceiling and the supervision path decide.

## Artifacts

- Teacher judgments (17,390, slim format): `data/distilled.jsonl` ·
  [HF datasets cnfeichen/t2reranking-4level-judgments](https://huggingface.co/datasets/cnfeichen/t2reranking-4level-judgments)
- Checkpoints & curves: `artifacts/v0.2-run1` (pure) · `artifacts/v0.2-run2` (w=0.5 / w=1.0)
- Full analyzer output: `results/s1-5way_report.md` in the benchmark repo
- Total arc cost: **~¥8 teacher API + $0 inference** (all local)
