# RUN REPORT — v0.2-run2（WSL2 训练节点, 2026-10-21, E2c 混合监督）

## 任务

E2c（PR #13, commit 46f2a6b）：`--aux-hard-weight` 软 CE + 硬标签 BCE 混合损失，两个权重各一跑。
前置已知：v0.2 纯蒸馏在四方表退步（nDCG@5 0.428 vs v0.1 0.488），本实验探混合监督能否找回直接排序信号。

## 口径说明（跨 run 可比性）

`val_loss` 始终是**纯软 CE 对教师分布**（脚本 L133），辅助 BCE 只加在训练损失（L127）——
三个 run 的 val_loss 同口径可直接对比。checkpoint 内含 `aux_hard_weight` 字段留档。

## E2c-w05（aux-hard-weight 0.5）

| epoch | train_loss | val_loss | val_acc@0.5 |
|---|---|---|---|
| 1 | 1.5567 | 1.2699 | 0.9000 |
| 5 | 1.4534 | 1.2318 | 0.8884 |

- temperature (fitted on val): **1.215**；产物 model/head_v21_w05.pt（795 KB）

## E2c-w10（aux-hard-weight 1.0）

| epoch | train_loss | val_loss | val_acc@0.5 |
|---|---|---|---|
| 1 | 1.8042 | 1.2790 | 0.9000 |
| 2 | 1.6972 | 1.2857 | 0.9000 |
| 5 | 1.6622 | 1.2486 | 0.9000 |

- epoch2 微升 0.0067 后回落，终点仍低于起点；temperature: **1.344**；产物 model/head_v21_w10.pt（795 KB）
- val_acc@0.5 五轮恒 0.9000：w=1.0 下打分从未越过 0.5，回到保守形态（w=0.5 时 0.8884，会做正例判断）

## 三 run 对照（val 同口径软 CE）

| 配置 | final val_loss | temperature | val_acc@0.5 |
|---|---|---|---|
| 纯蒸馏（run1） | 1.2210 | 1.000 | 0.8179 |
| 混合 w=0.5 | 1.2318 | 1.215 | 0.8884 |
| 混合 w=1.0 | 1.2486 | 1.344 | 0.9000 |

观察：硬权重越高，对教师分布的拟合让位越多（val_loss 单调抬升）、logits 越锐（温度 1.0→1.215→1.344）。
这符合"用教师拟合度换直接监督"的设计意图；**排序裁决在 SG 侧 nDCG，本节点不预判**。

## 环境与磁盘

- 同一 venv（torch 2.6.0+cu124, RTX 4060 Laptop），`HF_HUB_OFFLINE=1`（RUNBOOK 新规），两跑各 ~50 秒
- 磁盘 775 GB avail → 775 GB avail（净增 ~1.6 MB，两个 head 文件）
- head_v2.pt 纯蒸馏基线保留在 artifacts/v0.2-run1（bbec255），工作区副本已恢复

## 异常与观察

1. 无环境异常（HF_HUB_OFFLINE=1 生效，零联网探测，首跑即 50 秒完成）。
2. git checkout main 会清掉 artifacts 分支跟踪的 model/ 文件（.gitignore 下被 -f 加入的路径），
   head_v2.pt/head.pt 工作区副本需 `git show artifacts/v0.2-run1:…` 恢复——已恢复，非丢失。

## 结论

E2c 两跑完成：`E2c-w05: PASS（val_loss=1.2318）`、`E2c-w10: PASS（val_loss=1.2486）`。
产物推 artifacts/v0.2-run2：两个 head + 两份日志 + 本报告。等 SG 侧五方表（含两个混合权重）裁决。
