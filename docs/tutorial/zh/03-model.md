# 第 3 章 · 白盒实现一个最小的选项打分 transformer

全部模型代码在 `src/minimind_jev/model.py`，**不到 120 行**。这一章逐块读。

## 先定形：Encoder-only，不是 LLM

决策模型不生成 token，所以**没有解码器、没有因果掩码**——这是一个纯编码器（BERT 形态）：整段输入互相看见，池化成一个向量，投到一个 logit 上。

```
ids (B, L) → 嵌入 + 位置 → N × Block → LayerNorm → 掩码均值池化 → Linear → logit (B,)
```

## 一个 Block：pre-LN 注意力 + FFN

```python
def forward(self, x, keep_mask):
    q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)     # ① 预归一化后一次算出 QKV
    ...分头...
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep_mask[:, None, None, :])  # ②
    x = x + self.proj(a)                                  # ③ 残差
    x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))       # ④ FFN + 残差
    return x
```

四个值得注意的白盒选择：

1. **pre-LN**（先归一化再进注意力/FFN）：训练稳定，不需要学习率热身——小模型小数据的现实约束。
2. **手写 QKV 但用官方 sdpa 算注意力**：QKV 投影、分头、拼回来的全过程可见；注意力核心交给 `F.scaled_dot_product_attention`，因为那是一个融合算子——CPU/GPU/未来后端都拿优化好的实现，而且 `attn_mask` 直接吃我们的补位掩码（True=有效）。想看注意力怎么算的数学，它就是 softmax(QKᵀ/√d)·V，教科书原样。
3. **补位掩码贯穿始终**：`keep_mask` (B, L) 布尔，True 是真实 token。注意力看不见补位，池化也不算补位。
4. **FFN 4 倍扩展 + GELU**：标准配置，没有花活。

## 池化与头：均值，不是 CLS

```python
pooled = (h * keep_mask.unsqueeze(-1)).sum(1) / keep_mask.sum(1, keepdim=True).clamp(min=1)
return self.head(pooled).squeeze(-1)
```

掩码均值池化。选它而不是 CLS token 有一个教学理由：**补位不变性可以直接测试**——同一句话不管补多少个 PAD，logit 必须一动不动：

```python
# tests/test_v3.py::test_padding_invariance（实测通过，差异 <1e-5）
logit([5,6,7])          == logit([5,6,7,0,0] with keep=[T,T,T,F,F])
```

这是白盒模型的好处：**不变性不是祈祷出来的，是断言出来的**。

## 参数账本（设计即预算）

| 部件 | lite（2L/d128/ctx256） | full（6L/d384/ctx512） |
|---|---|---|
| 词嵌入 8192×d | 1,048,576 | 3,145,728 |
| 位置嵌入 ctx×d | 32,768 | 196,608 |
| 每层（QKV+proj+FFN） | 82,432 | 851,968 |
| 层数 | 2 | 6 |
| **合计** | **1,477,249** | **13,981,057** |

两个规格都有明确用途：**full 是旗舰规格**（对齐 minimind 的"单卡两小时"叙事，4060 级 GPU 约 1 小时）；**lite 是"任何机器都能跑"规格**（2 核 CPU 3.5 小时，本教程第 4 章实测用的就是它）。注意嵌入层占了 lite 的七成参数——这是第 2 章说"词表直接吃参数预算"的账面证据。

## 为什么上下文取 256/512 而不是更长

第 1 章的段落上限 3000 字符，BBPE 下约 1500-2000 token——任何现实窗口都要截断。取 512 与 bge-small-zh 对齐（v0.1 与 v0.3 才是同任务对比），lite 取 256 再省一半算力。截断的代价是长段落只看前半——对"能否回答"这个判断，前 256 token 通常够用，但这是一个可以挑战的假设（留给你的第一个实验：把 ctx 提到 512，lite 还能涨多少？）。

## 本章成本

$0。下一章：[训练循环——包括 accuracy 是怎么骗我们的](04-training.md)。
