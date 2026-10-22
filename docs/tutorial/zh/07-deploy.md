# 第 7 章 · 部署：ONNX 导出，浏览器里跑

最后一站：把训练好的检查点变成一个**不依赖 PyTorch、不依赖 Python** 的推理工件。1.5M 参数的模型导出后约 6MB——比一张缩略图还小，浏览器里一次前向几毫秒。这正是生态里 [jev-omni.js](https://github.com/ai-ecoverse/jev-omni.js)（WebGPU 决策模型）指出的形态：**类型化决策模型小到可以住进页面**。

## 导出（`scripts/export_onnx.py`）

```bash
python scripts/export_onnx.py --checkpoint model/v3-lite.pt --verify
# exported model/v3-lite.onnx (graph 127 KB; 权重走外部数据 v3-lite.onnx.data 5.9 MB)
# verify: torch [-4.936, -4.755] vs onnx [-4.936, -4.755] | max|Δ|=1.91e-06
# verify: PASS (<1e-4)
```

三件事值得知道：

1. **外部数据格式**：torch 2.14 的导出器默认把权重放在伴生的 `.onnx.data` 文件里（图结构 127KB + 权重 5.9MB）。部署时两个文件要一起带走。
2. **opset 的坑**：动态 shape 导出时新导出器会生成 `Split(num_outputs)` 算子——onnxruntime 只认 **opset ≥ 18** 的这个属性，钉 17 会在加载时报 `INVALID_GRAPH`。报错信息不会告诉你原因，本章替你踩过了。
3. **数值验证是必做项**：`--verify` 用真实分词输入对比 torch 与 ORT 的 logits（实测最大偏差 1.9e-06）。第 5 章的教训在这里同样适用——跨运行时的转换错误也是静默的。

## 在浏览器里跑（onnxruntime-web）

数值等价已在 CPU 版 onnxruntime 验证；浏览器侧是标准加载路径：

```javascript
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";

const session = await ort.InferenceSession.create(
  "./v3-lite.onnx", { executionProviders: ["wasm"] });   // 有 WebGPU 就换 "webgpu"

// ids/keep 由分词产生：把 tokenizer.json 一起带上，
// 用 @xenova/transformers 的 PreTrainedTokenizer 加载即可（BBPE 是标准格式）
const ids   = new ort.Tensor("int64", BigInt64Array.from([...]), [1, L]);
const keep  = new ort.Tensor("bool",  Uint8Array.from([...]),    [1, L]);
const { logit } = await session.run({ ids, keep });
const p = 1 / (1 + Math.exp(-logit.data[0] / T));       // T = 1.157，第 4 章拟合的温度
```

一个诚实的边界：**分词器在本章没有做 WASM 端到端验证**（BBPE 格式是标准的，`@xenova/transformers` 可直接加载 `tokenizer.json`，但"在浏览器里从原始文本到 logit"的完整链路尚未实测——留给读者的第二个实验，也是本教程下一版的待办）。

## 温度要跟着走

检查点里存了 `"temp": 1.157`。部署侧的 sigmoid 必须用它——否则概率的绝对值不对（排序不受影响，但第 5 章讲过：做阈值判断时概率必须是校准过的）。

## 往哪里去

- **全量 14M**：`--preset full` 在 GPU 上约 1 小时，导出流程完全相同（约 56MB 权重）。
- **量化**：1.5M 模型 int8 量化后预计 <2MB，浏览器加载更快（`onnxruntime.quantization`，未实测，练习三）。
- **接入真实检索**：把第 1 章的问题措辞原样搬进你的召回管线，这个 logit 就是"要不要把这段送进大模型"的门闸。

## 全教程结账

| 项 | 成本 |
|---|---|
| 数据/词表/训练/评测/导出 | $0（CPU 3.5h 或 GPU ~1h，电费） |
| 教师蒸馏（第 6 章，可选且已证明不该做） | ~¥8 |
| **合计** | **≤ ¥8 + 电费** |

你手里现在有：一个自有词表、一个白盒 transformer、一次校准、一套基准协议下的六方对比、一次蒸馏失败的完整档案、一个 6MB 的部署工件——以及读完这八章之后，对"决策模型到底是什么"的答案。
