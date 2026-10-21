# RUN REPORT — v0.2-run1（WSL2 训练节点, 2026-10-21）

## 环境

- 与 v0.1 同一 venv: torch 2.6.0+cu124 | GPU RTX 4060 Laptop 8GB | Python 3.12.3（uv venv）
- 数据零改动: 直接使用 main 分支 `data/distilled.jsonl`（17,390 条教师软标签, PR #11）

## E2b 训练（v0.2 冻结 bge-small-zh + 4 级软标签头, 5 epochs）

- train 15660 pairs / val 1730 pairs（4-level soft targets, 与 1:9 hard 标签同查询不同目标形式）

| epoch | train_loss | val_loss | val_acc@0.5 |
|---|---|---|---|
| 1 | 1.2970 | 1.2634 | 0.9000 |
| 2 | 1.2563 | 1.2381 | 0.8647 |
| 3 | 1.2384 | 1.2282 | 0.8428 |
| 4 | 1.2281 | 1.2256 | 0.7988 |
| 5 | 1.2220 | 1.2210 | 0.8179 |

- temperature (fitted on val): **1.000**（v0.1 为 1.031）
- 产物: model/head_v2.pt（795 KB, 权重 + dim 512 + 4 levels + 温度）
- 编码在 GPU 完成, 总耗时 **52 秒**（含编码 + 5 epochs）

## 通过判据核对（按 Phase 2 任务书）

- val_loss 单调下降: 1.2634 → 1.2210 ✓
- 温度拟出: T=1.000 ✓
- head_v2.pt 落盘 ✓
- val_acc@0.5 = 0.8179, **低于 0.9000 多数类基线——按任务书不作为门槛**（4 级软 CE 用 0.5 硬判据
  本就吃亏, v0.1 教训已写进 RUNBOOK）。真实裁决在 SG 侧四方表, 数字如实带回。

## 磁盘

- 训练前 776 GB avail → 训练后 776 GB avail（净增 <50 MB, 符合预估）

## 异常与观察

1. 首次启动卡在 huggingface.co HEAD 重试循环 ~12 分钟（本节点 huggingface.co 直连不通,
   sentence-transformers 逐文件 5 次退避重试, 模型本地缓存本不需要联网）。
   杀掉后加 `HF_HUB_OFFLINE=1` 重跑, 52 秒干净完成。**建议训练机一律带此环境变量**（缓存已就位时）。
2. temperature 恰为 1.000: 拟合值落在 1.0, 是否为优化下界截断留解读, 本节点不调参重跑。
3. val_loss 五轮降幅收窄（-0.027/-0.010/-0.003/-0.005）, 5 epochs 内未见回升, 无过拟合迹象。

## 结论

E2b 通过（val_loss 下降、温度拟出、head_v2.pt 落盘）。本会话产物: head_v2.pt + head.pt 副本
（取自 artifacts/v0.1-run1）+ 两份训练日志 + 本报告。等 SG 四方表（Jev-1.13 / minimind-v0.1 /
minimind-v0.2 / BM25）做最终裁决。
