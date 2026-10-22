# 第 1 章 · 从 T2Reranking 到类型化问答：数据构造

## 数据集：T2Reranking 与难负例

[T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking)（C-MTEB 系列，DuReader 衍生）的每条记录是：一个查询、一个**正例段落**（能回答查询）、若干**难负例**（检索系统捞回来的相似但不相关的段落）。"难"是关键——随机负例谁都能排对，难负例才考验判断。

我们把它组织成**池**：每查询 1 正 + 9 负，共 10 个候选：

```
{"qid": "q00123", "query": "……", "positive": "……", "negatives": ["……", …×9]}
```

切分结果（seed 42）：**train 1566 查询（15,660 对）/ val 173 查询（1,730 对）**。

## 一次真实的泄漏事故（值得先读）

评测集 s1 的 100 个查询来自同一个 T2Reranking dev 集。第一次在训练机上生成数据时，产物是 **1656/183**——比正确数字正好多出 100**。原因：`prepare_data.py` 的排除清单默认指向一个本机路径，文件不存在时**静默跳过排除**。训练机自然没有那个路径，于是评测查询混进了训练集。

多出来的 100 个查询就是被泄漏的评测集。用这份数据训出来的模型，评测分数会虚高——它见过考题。

修复分两层（都已进仓库）：

1. **评测样本入仓**（`data/eval_sample.jsonl`，1.3MB），排除逻辑默认指向它，任何机器上都可用；
2. **缺文件硬报错**，不再静默跳过：

```python
if not exclude_path.exists():
    raise SystemExit(
        f"error: --exclude-sample '{args.exclude_sample}' not found — refusing to build a "
        "leaky split. ...")
```

教训一句话：**可复现性要求关键依赖显式存在，而不是"找不到就算了"**。静默降级在数据管道里等于埋雷。

## 问题设计：措辞是契约的一部分

每条训练文本的构造（`src/minimind_jev/data.py` 与 `scripts/train.py` 共用）：

```python
QUESTION = "仅凭该段落能回答该查询吗？"
text = f"{query}\n{QUESTION}\n{passage}"
```

这个措辞**冻结**自 [awesome-jev-prompts](https://github.com/vicfei/awesome-jev-prompts) 模式库的 noul 中文变体，v0.1 到 v0.3 全程不改。为什么这么较真：

- 决策模型对问题措辞敏感（措辞变化会移动概率），**问题文本是输入分布的一部分**——训练和评测必须用同一份措辞，否则你在测两个不同的模型；
- 模式库里还有 51 条不同场景的措辞与反模式（比如"不要问'相关吗'，要问'能回答吗'"），换任务时去那里挑，不要现场发明。

## 1:9 的类不平衡，先打个预防针

每个池 1 正 9 负，二元标签天然 1:9。这带来两个后果，第 4 章会展开：

1. **准确率会说谎**：全预测"负"就有 90% 准确率——这正是 v0.1 训练时 val_acc 恒等于 0.9000 的原因；
2. 常量先验的交叉熵约 0.325，训练曲线要跟这个数比，而不是跟零比。

## 动手

```bash
python scripts/prepare_data.py
# 预期输出:
# excluded 100 queries (present in the benchmark eval sample — no leakage)
# train: 1566 queries / 15660 pairs -> data/train.jsonl
# val:   173 queries / 1730 pairs   -> data/val.jsonl
```

数据文件不进 git（`.gitignore` 里的 `data/*.jsonl`，评测样本除外）——任何人 clone 后跑上面这一行即可复现同样的切分（seed 42 写死在默认参数里）。

## 本章成本

$0，CPU 分钟级。下一章：[从零训练你自己的 BBPE 分词器](02-tokenizer.md)。
