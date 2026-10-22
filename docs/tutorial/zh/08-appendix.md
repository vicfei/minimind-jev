# 附录 · 复现手册与常见问题

## 一键复现（CPU 即可）

```bash
git clone https://github.com/vicfei/minimind-jev.git && cd minimind-jev
python3 -m venv .venv && source .venv/bin/activate
pip install torch sentence-transformers pyarrow tokenizers

python scripts/prepare_data.py                 # 第 1 章：1566/173，零泄漏
python scripts/train.py --epochs 5             # v0.1：冻结 bge + 从零头
python scripts/train_tokenizer.py              # 第 2 章：8192 词表
python scripts/train_v3.py --preset lite --mlm-epochs 2 --epochs 5   # 第 4 章：~3.5h CPU
python scripts/export_onnx.py --verify         # 第 7 章：数值对齐
```

GPU（4060 级）把第五行换成 `--preset full`，全程约 1 小时。

## 评测对齐

```bash
git clone https://github.com/vicfei/jev-rag-zh-benchmark.git ../jev-rag-zh-benchmark
python scripts/eval_export.py                  # v0.1
python scripts/eval_export_v3.py --checkpoint model/v3-lite.pt
cd ../jev-rag-zh-benchmark && python scripts/analyze.py \
  --sample data/s1.jsonl --logs logs/s1.score-zh.jev-113.jsonl logs/s1.noul-zh.jsonl \
  ../minimind-jev/logs/s1.minimind-jev*.jsonl --tag s1-mine
```

## 常见问题

**val_acc 一直是 0.9000，坏了？** 没坏。1:9 不平衡下多数类基线就是 0.90——看 val_loss（跟 0.325 的先验基线比）和基准上的 nDCG（第 4、5 章）。

**训练时 huggingface.co 连不上卡住？** 编码器缓存已就位时带 `HF_HUB_OFFLINE=1`；大陆网络下数据下载走 `HF_ENDPOINT=https://hf-mirror.com` 或 `prepare_data.py --parquet <本地路径>`（两个都是实测过的路径）。

**ONNX 加载报 INVALID_GRAPH（Split num_outputs）？** opset 用 18（第 7 章）。

**蒸馏能不能救一下我的模型？** 先读第 6 章——gold 标签存在时大概率会退步；要试，先小规模。

**权重没有随包分发？** v0.1 检查点在 [HF](https://huggingface.co/cnfeichen/minimind-jev-v0.1)；v0.3 按 CLI 自训即为复现。教师四级判断数据集在 [HF datasets](https://huggingface.co/datasets/cnfeichen/t2reranking-4level-judgments)。

## 时间线（全部实测记录）

- v0.1：2026-10-21，WSL2/4060 训练 3 分钟，评测同日
- v0.2 蒸馏：2026-10-21，教师调用 ~5 分钟（池化+关推理），~¥8
- v0.3-lite：2026-10-22，2 核 CPU 3.5 小时（WSL2 故障期间 SG 节点承接）
- v0.3 full：等 GPU 节点，代码与流程已就绪

## 三个留给读者的实验

1. ctx 256 → 512，lite 能涨多少？（第 3 章的假设待检验）
2. best-checkpoint 而非 last（第 4 章的瑕疵待修复，五行以内）
3. int8 量化 + 浏览器端到端（第 7 章的未竟事项）
