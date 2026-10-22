# 第 4 章 · 训练循环：损失、温度校准、早停

`scripts/train_v3.py`，两个阶段加一个收尾。这一章的数字全部来自 2026-10-22 在 2 核 CPU 上的 lite 实跑。

## 阶段一：短 MLM 预训练——"从零"的诚实形态

从零不等于直接上监督任务。1,730 万参数的模型（含 MLM 头）从随机初始化直接学"能否回答"，等于让一个不识字的人做阅读理解。所以先做**掩码语言模型**（15% token 挖掉让模型猜）：

```python
masked, targets = mask_mlm(ids, mask_id)          # 15% → [MASK]，其余 target 置 -100
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-100)
```

两轮之后的诚实数字：**loss 7.71 → 7.51**。随机基线是 ln(8192) ≈ 9.01——下降了 1.5，但离"学会了中文"差得远（作为参照，正经预训练要降到 3 以下，需要大几个数量级的语料和算力）。这不是失败，是**标定**：我们的 lite 模型带着"只学了两小时字母表"的语言水平进入下一阶段，第 5 章会量化这个差距值多少个 nDCG 点。

## 阶段二：二元 BCE 微调（v0.2 教训的直接应用）

```python
logits = model(tr_ids[idx], tr_keep[idx])
loss = nn.BCEWithLogitsLoss()(logits, ytr[idx])
```

没有蒸馏、没有软标签、没有花活——因为 v0.2 弧线（第 6 章）已经用三组对照证明：**当指标对应的 gold 标签存在时，直接监督它们就是最好的损失**。实测曲线：

```
epoch 1: train_loss=0.3289 val_loss=0.3187 val_acc=0.9000
epoch 2: train_loss=0.3198 val_loss=0.3175 val_acc=0.9000   ← val 最优
epoch 3: train_loss=0.3142 val_loss=0.3215 val_acc=0.9000
epoch 4: train_loss=0.3031 val_loss=0.3253 val_acc=0.9000
epoch 5: train_loss=0.2807 val_loss=0.3477 val_acc=0.8965   ← 存盘的是它
```

三个读数，每一个都是教学点：

**① accuracy 在撒谎。** val_acc 恒 0.9000 不是"训练得不错"，是**多数类基线**——1:9 不平衡下全预测"负"就是 0.90。同理 v0.1 的 0.90、v0.2 混合版的 0.8884 都要在同一条基线下读。

**② 损失要跟先验比。** 常量先验（永远输出 0.1 概率）的 BCE ≈ 0.325。val_loss 0.3175 只比"只会输出先验"好 0.008——模型学到的排序信号极其微薄，但第 5 章会看到，这么薄的信号已经足够在基准上和 BM25 打平。

**③ 我们踩了过拟合还存了错的检查点。** val_loss 第 2 轮最优，之后单调回升——教科书级的过拟合起点；而脚本存的是第 5 轮（`torch.save` 在循环外，永远是最后一轮）。**本教程如实保留这个瑕疵**，修复方式留给读者当练习（提示：跟踪 best val_loss，存 best 而非 last；改动不超过五行）。诚实地说，如果存的是 epoch 2，六方表里 v0.3-lite 的 0.440 可能还会好看一点。

## 收尾：温度校准

```python
T = nn.Parameter(torch.ones(1))
topt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
def closure(): ... loss = lossf(logits / T, yva) ...
topt.step(closure)      # 实测拟合出 T = 1.157
```

温度只动一个数：logits 除以 T 再过 sigmoid。它在验证集上让输出概率与真实正确率对齐。本模型 T=1.157（v0.1 是 1.031，v0.2 混合版 1.215-1.344）——不同模型的 logits 锐度不同，这正是"校准要每个模型单独做"的证据。

## 让 CPU 跑得动的工程：长度分桶

固定宽度补位在混合长度的批里浪费严重（短句跟着长句补到 512）。修复（`bucketed_batches` + `trim_batch`）：按长度排序切批、每批只补到**本批**最大长度、批序每轮打乱。实测在 CPU 上省 2-3 倍算力——同样的代码 GPU 上也白赚。

## 动手

```bash
# 全量 lite（本教程实测配置）：
python scripts/train_v3.py --preset lite --mlm-epochs 2 --epochs 5 --out model/v3-lite.pt
# 有 GPU 时换旗舰规格（约 1 小时）：
python scripts/train_v3.py --preset full --out model/v3.pt
```

## 本章成本

纯 CPU 3.5 小时 / $0（电费）；GPU 约 1 小时。下一章：[评测——六方表怎么读，以及一次差点发出去的假数字](05-eval.md)。
