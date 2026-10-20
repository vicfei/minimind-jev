# minimind-jev

Train your own tiny **Jev** — a System One typed-decision model, in the spirit of the [minimind](https://github.com/jingyaogong/minimind) series (open the black box, not chase SOTA), applied to decision models instead of LLMs.

> Status: **v0.1 trained, evaluated, and published** — [HF cnfeichen/minimind-jev-v0.1](https://huggingface.co/cnfeichen/minimind-jev-v0.1) · [ModelScope mirror](https://modelscope.cn/models/cnfeichen/minimind-jev-v0.1). The **decision head is trained from scratch**; the text encoder is frozen ([BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)). Repo stays private while v0.2 (distillation) and v0.3 (from-scratch transformer + Chinese tutorial) are in build.

## Results — v0.1 (2026-10-21)

Same protocol as [jev-rag-zh-benchmark](https://github.com/vicfei/jev-rag-zh-benchmark): 100 queries over T2Reranking hard-negative pools, 95% bootstrap CI over queries, model pinned `typesafe/jev-1.13-20260917` for hosted rows.

| system | nDCG@5 | nDCG@10 | MRR@10 | R@1 | separation AUC |
|---|---|---|---|---|---|
| hosted Jev score-zh (1.13) | 0.546 [.481,.611] | 0.602 | 0.477 | 0.26 | 0.714 |
| hosted Jev noul-zh (1.13) | 0.531 [.468,.596] | 0.588 | 0.458 | 0.24 | 0.718 |
| **minimind-jev v0.1** | **0.488 [.417,.559]** | 0.576 | 0.445 | 0.23 | 0.630 |
| BM25 (jieba + rank_bm25) | 0.413 [.353,.477] | 0.513 | 0.363 | 0.11 | — |

Paired bootstrap (n=1000): minimind-jev − BM25 nDCG@5 **+0.075 [+0.002, +0.154]** (CI excludes 0); hosted Jev − minimind-jev +0.059 [−0.040, +0.157] (CI includes 0 — not separable at n=100). Training cost: **~3 minutes on a laptop RTX 4060 (8GB), zero API calls**.

**Calibration notes (the honest part):** the split is 1:9 (1 positive + 9 negatives per query), so val_acc 0.9000 is exactly the majority-class baseline — accuracy is uninformative here and the ranking metrics above are the real signal. Fitted temperature is 1.031 (barely moves): output probabilities stay compressed near the class prior (mean 0.114 on positives vs 0.096 on negatives), so v0.1 is a **ranker, not a calibrated probability source** — thresholds need per-decision fitting, mirroring the benchmark's finding on fixed 0.5 cutoffs. Decompressing the signal with 4-level soft labels is exactly what v0.2 distillation is for.

## Roadmap

- **v0.1 (done)** — frozen Chinese encoder + from-scratch MLP head, trained on [C-MTEB/T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking) hard negatives (binary labels, the dataset's own; question wording frozen from [awesome-jev-prompts](https://github.com/vicfei/awesome-jev-prompts)). Temperature-scaled on a held-out split. Runs on a laptop CPU; measured at ~3 min on GPU.
- **v0.2 (next)** — multi-task distillation: a frontier chat model grades pairs into the pattern library's 4-level rubric; soft targets; more question forms (Choice/Score/Noul).
- **v0.3** — the minimind-pure form: from-scratch tiny transformer, single GPU, ~2h, full Chinese tutorial.

## Evaluation

Same protocol as [jev-rag-zh-benchmark](https://github.com/vicfei/jev-rag-zh-benchmark): the eval script exports logs compatible with the benchmark's analyzer, producing the three-way table above — hosted Jev (`jev-1.13`) vs minimind-jev vs BM25 — on the same 100 queries over hard negatives.

## Why this exists

The Jev ecosystem has replicas (Kev, Bosun, …) and research-grade trainers (Valen), but no Chinese-language, open-the-box educational path. This is that path, built on the pattern library's question-design rules: the model is only half the artifact; the other half is the questions.

## Layout

```
scripts/prepare_data.py   # T2Reranking -> typed-question train/val splits (seed 42, eval queries excluded)
scripts/train.py          # frozen encoder + from-scratch head, CPU, temperature scaling
scripts/eval_export.py    # benchmark-compatible eval logs
model/head.pt             # trained head (gitignored; released via HF/ModelScope, artifacts/* branches keep run copies)
data/eval_sample.jsonl    # in-repo copy of the benchmark eval sample (exclusion guard, ships with repo)
```

Independent community project; not affiliated with TypeSafe AI, the minimind series, or their authors. MIT.
