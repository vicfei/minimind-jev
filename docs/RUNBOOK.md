# RUNBOOK — 训练机交接手册（v0.1）

训练机只需做三件事：克隆、准备数据、训练。评测和发布回本机做。

## 机器需求

| 阶段 | 最低配置 | 预期耗时 |
|------|---------|---------|
| v0.1（决策头） | **纯 CPU 即可**：4 核 + 8GB RAM（无 GPU 也能跑） | 30–60 分钟（含编码） |
| v0.3（从零 transformer） | 单张 3090/4090（16–24GB 显存） | 目标 ≤ 2 小时/epoch（复现 minimind 的"两小时"锚点） |

## 环境

```bash
git clone https://github.com/vicfei/minimind-jev.git && cd minimind-jev
python3 -m venv .venv && source .venv/bin/activate
pip install torch sentence-transformers pyarrow   # GPU 机装 CUDA 版 torch
```

## 三条命令

```bash
python scripts/prepare_data.py          # 自动下载数据（或用 --parquet 指定本地路径）
python scripts/train.py --epochs 5      # 输出 model/head.pt + 温度校准值
```

预期输出样例（本机已验证数据阶段）：

```
train: 1566 queries / 15660 pairs -> data/train.jsonl（基准评测集的 100 个查询已排除，零泄漏）
val:   173 queries / 1730 pairs   -> data/val.jsonl
epoch 1: train_loss=... val_loss=... val_acc=...
temperature (fitted on val): ...
saved model/head.pt
```

## 产物回传清单

- `model/head.pt`（含 head 权重 + 维度 + 温度）
- 训练控制台日志（epoch 曲线贴回 issue #1 即可）
- v0.3 阶段追加：`model/transformer.pt`、tokenizer 文件、`config.json`

## 回本机后的流程

```bash
python scripts/eval_export.py           # 在基准 s1 样本上出兼容日志
cd ../jev-rag-zh-benchmark && uv run python scripts/analyze.py \
  --sample data/s1.jsonl \
  --logs logs/s1.score-zh.jsonl logs/s1.noul-zh.jsonl ../minimind-jev/logs/s1.minimind-jev.jsonl \
  --tag s1-3way
```

→ 直接产出三方表（Jev-1.13 / minimind-jev / BM25），同协议同样本同置信区间。
