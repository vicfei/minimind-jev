# RUNBOOK — 训练机交接手册（v2，2026-10-21 按训练节点实测校准）

训练机只需做四件事：克隆、环境、准备数据、训练。评测和发布回 SG 节点做。

## 机器需求（v2 修正）

| 阶段 | 实测配置 | 实测耗时 |
|------|---------|---------|
| v0.1（决策头） | RTX 4060 Laptop **8GB** 即可（纯 CPU 也能跑，30–60 分钟） | GPU **~3 分钟**（含编码 17,390 条） |
| v0.3（从零 transformer，~15M） | 同上，8GB 显存余量充足（优化器态 ~0.2GB + 激活 <3GB） | 目标 ≤ 2 小时/epoch（minimind 锚点） |

> v1 写的"3090/4090 16–24GB"是过时保守估计，已按 2026-10-21 实测（sm_89, torch 2.6.0+cu124）修正。

## 环境

```bash
git clone https://github.com/vicfei/minimind-jev.git && cd minimind-jev
python3 -m venv .venv || uv venv .venv          # 系统 venv 缺 ensurepip 时用 uv 兜底
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers pyarrow
```

大陆网络下载 parquet/编码器受阻时：`export HF_ENDPOINT=https://hf-mirror.com`（prepare 脚本已支持），
或从镜像手动下载后 `--parquet <path>` 喂入。

## 数据准备（必做；新克隆里没有 data/*.jsonl）

`data/*.jsonl` 不进 git。基准评测样本已入仓（`data/eval_sample.jsonl`），排除逻辑默认指向它、
缺文件会**硬报错**（宁可停下，不静默产出泄漏版——训练节点首跑事故的教训）：

```bash
python scripts/prepare_data.py
# 预期: excluded 100 / train: 1566 queries (15660 pairs) / val: 173 queries (1730 pairs)
```

## 训练

```bash
python scripts/train.py --epochs 5 2>&1 | tee logs-train-v01.txt
# 预期: model/head.pt + 每轮 val 曲线 + temperature
```

注意 val_acc 的读法：本数据 1:9 类不平衡，**多数类基线 = 0.90**——val_acc 恒 0.90 只是"没输给
先验"，模型真实排序能力以 SG 侧三方表的 nDCG 为准（v0.1-run1 即如此：val_acc 0.90 但 nDCG@5
见三方表）。

## 产物回传（双通路）

- **通路 A（默认）**：直连 push 到 artifacts 分支
  ```bash
  git checkout -b artifacts/v0.1-runN
  git add -f model/head.pt logs-train-v01.txt RUN-REPORT.md
  git commit -m "artifacts: v0.1-runN [skip ci]" && git push -u origin artifacts/v0.1-runN
  ```
- **通路 B（GitHub 443 抖动时）**：经 SG 节点中转 push（训练节点已验证可行），本机 commit 不丢。

回传清单：`model/head.pt`、训练日志、RUN-REPORT.md（环境/曲线/温度/磁盘水位/异常）。
v0.3 阶段追加：`model/transformer.pt`、tokenizer 文件、`config.json`。

## 回 SG 节点后的评测

```bash
python scripts/eval_export.py           # 基准 s1 样本 → logs/s1.minimind-jev.jsonl
cd ../jev-rag-zh-benchmark && uv run python scripts/analyze.py \
  --sample data/s1.jsonl \
  --logs logs/s1.score-zh.jev-113.jsonl logs/s1.noul-zh.jsonl ../minimind-jev/logs/s1.minimind-jev.jsonl \
  --tag s1-3way
```

→ 三方表（Jev-1.13 / minimind-jev / BM25），同协议同样本同置信区间。
（注意日志文件名带版本标签；eval_export 的 answer 信封必须与基准日志同构 `answer.answers.*`，
多套一层会被 analyze 静默当成池原序——已修，见 #4。）
