# 第 2 章 · 分词器与词表：从零训练 BBPE

## 为什么从零训词表

v0.1 借用 bge-small-zh 的词表，那是"别人的预训练"的一部分。v0.3 要做纯粹的从零，词表也得是自己的——它是整条流水线的第一块自有资产：**语料 → 词表 → 一切后续训练都建立在这串 id 上**。

## 为什么选 Byte-Level BPE（BBPE）

三种常见选择，权衡如下：

| 方案 | 中文表现 | 权衡 |
|---|---|---|
| 字级 | 每字一个 token | 词表小但序列长，模型学不到"词"的概念 |
| 词级（jieba 分词） | 词内 OOV 直接变 UNK | 依赖外部分词器，OOO 处理丑陋 |
| **BBPE（本教程）** | 字节回退 + 子词合并 | 无 UNK、**编码可逆**、与 minimind 系列同路线 |

BBPE 先把文本映射到字节（UTF-8），再在高频字节对上迭代合并。任何合法字符串都能编码——没有 UNK，且**解码严格还原原文**（后面验证）。

## 实现（`src/minimind_jev/tokenizer.py`，完整不到 40 行）

```python
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["[PAD]", "[MASK]"],      # PAD 补位，MASK 给第 4 章的 MLM 用
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(corpus, trainer)
tok.decoder = decoders.ByteLevel()           # 字节级解码，保证可逆
```

语料来自第 1 章的切分：全部查询 + 段落去重后约 **17,570 条文本、两千万字符**——够一个 8K 词表学到高频中文子词。

## 实测（本仓库 2026-10-22 跑的）

```bash
python scripts/train_tokenizer.py
# corpus: 17570 unique texts, 20,354,798 chars
# vocab size: 8192 -> model/tokenizer.json        # 约 1 分钟，纯 CPU
```

验证三件事（对应 `tests/test_v3.py::test_tokenizer_roundtrip_and_encode`）：

```python
tok = load_tokenizer(Path("model/tokenizer.json"))
s = "中文分词测试句子。"
assert tok.decode(tok.encode(s).ids) == s          # ① 可逆：字节级保证
assert len(tok.encode("仅凭该段落能回答该查询吗？量子计算的基本原理").ids) == 18
                                                   # ② 密度：8K 词表下约 1 字/token 出头
# ③ 特殊 token 在位：[PAD]=0，[MASK]=1，训练与掩码逻辑依赖它们
```

## 词表大小怎么选

8K 是本教程语料规模下的甜点：两千万字符 ÷ 8192 合并 → 高频子词都进得来；再大（16K+）会有大量只出现一两次的低频 token，白占嵌入参数（嵌入层 8192×128 = 105 万参数，是 lite 模型 1.48M 的七成——**词表大小直接吃参数预算**）。语料再大一个量级，词表再翻倍不迟。

## 本章成本

$0，约 1 分钟 CPU。产物 `model/tokenizer.json`（约 600KB）被第 4 章的训练和第 7 章的部署共同引用——部署检查点里存的是它的相对路径。下一章：[白盒实现最小的选项打分 transformer](03-model.md)。
