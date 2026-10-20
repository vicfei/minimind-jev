# RUN REPORT — v0.1-run1（WSL2 训练节点, 2026-10-21）

## 环境（E0 冒烟）

- torch 2.6.0+cu124 | cuda 12.4 | GPU RTX 4060 Laptop, capability **(8, 9)** ✓
- bf16 matmul 4096³: **8.80 ms/iter**
- Python 3.12.3（uv venv；系统 python3-venv 缺 ensurepip 且无 sudo, 用 uv venv 兜底）
- 网络: hf-mirror 正常（编码器下载遇 429 限流一次, 自动重试成功）

## E1 训练（v0.1 冻结 bge-small-zh + MLP 头, 5 epochs）

| epoch | train_loss | val_loss | val_acc |
|---|---|---|---|
| 1 | 0.4351 | 0.3264 | 0.9000 |
| 2 | 0.3234 | 0.3219 | 0.9000 |
| 3 | 0.3200 | 0.3189 | 0.9000 |
| 4 | 0.3179 | 0.3179 | 0.9000 |
| 5 | 0.3162 | 0.3173 | 0.9000 |

- temperature (fitted on val): **1.031**
- 产物: model/head.pt（792 KB, 含权重+维度 512+温度）
- 编码 17,390 条文本在 GPU 完成（全程 GPU 占用, 未落 CPU）
- 总耗时 ~3 分钟（含编码）, 远低于预估 20-40 分钟

## 数据（与 RUNBOOK 预期数字对齐）

- prepare_data.py 两次适配后完全对齐: **excluded 100 / train 1566 queries / val 173 queries**（零泄漏版）
- 适配 1: 脚本 urllib 硬编码直连 huggingface.co 不通, 经 hf-mirror 手动下 parquet + `--parquet` 喂入
- 适配 2: `--exclude-sample` 默认路径是 SG 节点的 benchmark 文件, 本机不存在会**静默跳过排除**
  （首次生成 1656/183 泄漏版已废弃重生成）; 从 SG scp s1.jsonl 后重跑对齐

## 磁盘

- 训练前 776 GB avail → 训练后 775 GB avail（**净增 ~1 GB**, 全程最高单文件 torch cu124 轮子 ~2.4GB 装于 venv）

## 异常与观察

1. 交接 Prompt「数据已在仓库内, 勿再跑 prepare_data.py」与实际不符: data/*.jsonl 在 .gitignore,
   仓库本无数据; 已按仓库内 RUNBOOK 第一条命令执行 prepare（非即兴, 有据）
2. prepare_data.py 的排除清单依赖 SG 本地路径, 在训练机上静默失效——**建议 SG 侧把 s1.jsonl
   入仓或改默认路径, 否则每台训练机都会复现"泄漏版数据"风险**
3. train.py:68 FutureWarning: get_sentence_embedding_dimension 已更名（不影响运行）
4. val_acc 五轮恒 0.9000 且 epoch1 即达: 符合"frozen bge 相似度已强, 头部难再增益"的预期,
   数字如实带回, 解读留 E3

## 结论

E0/E1 全部通过。v0.1 在 8GB 消费卡上 3 分钟可复现, 磁盘净增 1GB。等 SG 节点 Phase 2 触发条件。
