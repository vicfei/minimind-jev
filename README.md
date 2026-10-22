# minimind-jev

Train your own tiny **Jev** — a System One typed-decision model, in the spirit of the [minimind](https://github.com/jingyaogong/minimind) series (open the black box, not chase SOTA), applied to decision models instead of LLMs.

> Status: **v0.1 trained, evaluated, and published** — [HF cnfeichen/minimind-jev-v0.1](https://huggingface.co/cnfeichen/minimind-jev-v0.1) · [ModelScope mirror](https://modelscope.cn/models/cnfeichen/minimind-jev-v0.1). The **decision head is trained from scratch**; the text encoder is frozen ([BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)). The v0.2 distillation arc is complete — as a documented negative result ([DISTILLATION-NOTES](docs/DISTILLATION-NOTES.md)); v0.3 (from-scratch transformer + Chinese tutorial) is in build.

## Results (2026-10-21)

Same protocol as [jev-rag-zh-benchmark](https://github.com/vicfei/jev-rag-zh-benchmark): 100 queries over T2Reranking hard-negative pools, 95% bootstrap CI over queries, model pinned `typesafe/jev-1.13-20260917` for hosted rows.

| system | nDCG@5 | nDCG@10 | MRR@10 | R@1 | separation AUC |
|---|---|---|---|---|---|
| hosted Jev score-zh (1.13) | 0.546 [.481,.611] | 0.602 | 0.477 | 0.26 | 0.714 |
| hosted Jev noul-zh (1.13) | 0.531 [.468,.596] | 0.588 | 0.458 | 0.24 | 0.718 |
| **minimind-jev v0.1 (binary labels)** | **0.488 [.417,.559]** | 0.576 | 0.445 | 0.23 | 0.630 |
| BM25 (jieba + rank_bm25) | 0.413 [.353,.477] | 0.513 | 0.363 | 0.11 | — |
| v0.2 pure distillation | 0.428 [.359,.495] | 0.539 | 0.398 | 0.19 | 0.608 |
| v0.2.1 mixed w=0.5 | 0.430 [.360,.498] | 0.542 | 0.401 | 0.19 | 0.610 |
| v0.2.1 mixed w=1.0 | 0.428 [.355,.499] | 0.547 | 0.409 | 0.21 | 0.613 |
| v0.3-lite (from scratch, 1.5M, CPU 3.5h) | 0.440 [.372,.509] | 0.535 | 0.391 | 0.16 | 0.647 |

**v0.1 is the headline**: paired bootstrap gives minimind-jev − BM25 nDCG@5 **+0.075 [+0.002, +0.154]** (CI excludes 0) and hosted Jev − minimind-jev +0.059 [−0.040, +0.157] (CI includes 0 — not separable at n=100). Three minutes of training on a laptop RTX 4060 (8GB), zero API calls.

**v0.3-lite is the "how much is pretraining worth" datapoint**: a 1.5M-param transformer with *no* pretrained encoder — own 8K BBPE tokenizer, 2 short MLM epochs (loss 7.71→7.51, barely above the ln(8192)≈9.0 random baseline), binary-BCE fine-tune — trained 3.5 hours on a 2-core CPU, $0. It lands at 0.440: paired deltas vs BM25 +0.027 [−0.050, +0.102] and vs v0.1 −0.048 [−0.134, +0.037] — **statistically tied with both**. Its separation AUC (0.647) actually exceeds v0.1's (0.630) while ranking lower — different error structure, a teaching point in its own right. Honest curve note: val_loss bottomed at epoch 2 (0.3175, vs the 0.325 constant-prior baseline) and the checkpoint saved at epoch 5 is past its overfitting onset. The full 14M `--preset full` run is queued for the GPU node.

**v0.2 is the lesson**: distilling a frontier teacher's 4-level relevance judgments into the same head *hurts* ranking — all three soft-label configs land at BM25 level and significantly below v0.1 (each paired delta vs v0.1 has a CI excluding 0, −0.057 to −0.060). The teacher's own expected-score separation is AUC 0.716 — same band as hosted Jev — so the signal exists in the teacher but does not survive distillation into a 660K head over frozen encoder features. Note that v0.3-lite (from scratch, direct supervision) also outscores the distilled heads — the supervision-choice lesson is consistent across both arcs. Full analysis, mechanism, and the recipe we'd recommend: [docs/DISTILLATION-NOTES.md](docs/DISTILLATION-NOTES.md).

**Calibration notes (the honest part):** the split is 1:9 (1 positive + 9 negatives per query), so val_acc 0.9000 is exactly the majority-class baseline — accuracy is uninformative here and the ranking metrics above are the real signal. v0.1's fitted temperature is 1.031 and its probabilities stay compressed near the class prior (mean 0.114 on positives vs 0.096 on negatives): v0.1 is a **ranker, not a calibrated probability source** — thresholds need per-decision fitting, mirroring the benchmark's finding on fixed 0.5 cutoffs. Soft labels did decompress the probabilities as hoped (positive mean 0.114 → 0.386) — but at the cost of the ranking signal, which is the trade the notes document.

## Tutorial (中文教程)

Eight chapters from tokenizer to browser deployment, every number from this repo's real runs — including the failures: [docs/tutorial/zh/](docs/tutorial/zh/README.md)（决策模型心智模型 → 数据 → BBPE → 白盒 transformer → 训练与校准 → 评测读表 → 蒸馏负结果 → ONNX 部署，全程 ≤ ¥8）.

## Roadmap

- **v0.1 (done)** — frozen Chinese encoder + from-scratch MLP head, trained on [C-MTEB/T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking) hard negatives (binary labels, the dataset's own; question wording frozen from [awesome-jev-prompts](https://github.com/vicfei/awesome-jev-prompts)). Temperature-scaled on a held-out split. Runs on a laptop CPU; measured at ~3 min on GPU. Published as a checkpoint.
- **v0.2 (done — negative result, documented)** — teacher distillation into the 4-level rubric. Data generated (17,390 judgments, [released separately](https://huggingface.co/datasets/cnfeichen/t2reranking-4level-judgments)); training arc (pure + two mixed-supervision weights) regressed ranking across the board. The analysis is part of the tutorial's value — see [DISTILLATION-NOTES](docs/DISTILLATION-NOTES.md).
- **v0.3 (next)** — the minimind-pure form: from-scratch tiny transformer, single GPU, ~2h, full Chinese tutorial. Loss design starts from v0.1's binary BCE; distillation enters as a controlled contrast, not the default path.

## Evaluation

Same protocol as [jev-rag-zh-benchmark](https://github.com/vicfei/jev-rag-zh-benchmark): the eval scripts export logs compatible with the benchmark's analyzer, producing the table above — hosted Jev (`jev-1.13`) vs minimind-jev variants vs BM25 — on the same 100 queries over hard negatives.

## Why this exists

The Jev ecosystem has replicas (Kev, Bosun, …) and research-grade trainers (Valen), but no Chinese-language, open-the-box educational path. This is that path, built on the pattern library's question-design rules: the model is only half the artifact; the other half is the questions.

## Layout

```
scripts/prepare_data.py      # T2Reranking -> typed-question train/val splits (seed 42, eval queries excluded)
scripts/train.py             # v0.1: frozen encoder + from-scratch head, temperature scaling
scripts/distill_generate.py  # v0.2: teacher judgments over 4-level rubric (pooled calls, breakers)
scripts/train_v2.py          # v0.2: 4-level head on soft labels (+ optional aux BCE, --aux-hard-weight)
scripts/eval_export*.py      # benchmark-compatible eval logs (v0.1 / v0.2 heads)
data/eval_sample.jsonl       # in-repo copy of the benchmark eval sample (exclusion guard)
data/distilled.jsonl         # 17,390 teacher judgments (also mirrored on HF datasets)
```

Independent community project; not affiliated with TypeSafe AI, the minimind series, or their authors. MIT.
