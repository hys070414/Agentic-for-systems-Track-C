# C3 公开测试集

本目录描述 C3 算子调度的评测模型和测试配置。

## 评测模型

| 模型 | 任务 | 输入形状 | 输出形状 | 准确率阈值 |
|------|------|----------|----------|-----------|
| MLP | MNIST 手写数字分类 | `[N, 1, 28, 28]` | `[N, 10]` | >= 98% |
| ResNet-18（简化） | CIFAR-10 图像分类 | `[N, 3, 32, 32]` | `[N, 10]` | >= 85% |
| Transformer（decoder-only） | 合成序列任务 | `[N, 18]`（int64） | `[N, 18, 14]` | -- |

## MLP 架构

```text
输入 [N, 1, 28, 28]
  |
  v
Flatten -> [N, 784]
  |
  v
Gemm (fc1: 784 -> 256) + Relu
  |
  v
Gemm (fc2: 256 -> 128) + Relu
  |
  v
Gemm (fc3: 128 -> 10)
  |
  v
输出 [N, 10]
```

**ONNX 算子**：`Flatten`、`Gemm`、`Relu`

## ResNet-18（简化）架构

标准 ResNet-18 适配 CIFAR-10（3x32x32 输入）：

- 初始 Conv（3x3，stride 1）
- 4 个残差 stage，block 数为 [2, 2, 2, 2]
- 全局平均池化
- 最终 Gemm（512 -> 10）

**ONNX 算子**：`Conv`、`Relu`、`Add`、`GlobalAveragePool`、`Flatten`、`Gemm`

**注**：ONNX 导出时 BatchNorm 已折叠到 Conv 权重中。图中无 BN 节点。

## Transformer（Decoder-only）架构

简化的 decoder-only transformer，用于合成序列任务：

- 通过 Gather 进行 Token 嵌入
- 多个 transformer block，包含：
  - 多头自注意力（MatMul、Softmax 等）
  - LayerNorm
  - 前馈网络（Gemm）
  - 残差连接（Add）
- 输出投影

**ONNX 算子**：`Gather`、`Add`、`LayerNormalization`、`MatMul`、`Constant`、`Split`、`Reshape`、`Transpose`、`Div`、`Softmax`、`Erf`、`Mul`

## 完整 ONNX 算子集（17 种）

`Add`、`Constant`、`Conv`、`Div`、`Erf`、`Flatten`、`Gather`、`Gemm`、`GlobalAveragePool`、`LayerNormalization`、`MatMul`、`Mul`、`Relu`、`Reshape`、`Softmax`、`Split`、`Transpose`

## 运行评测

```bash
# C3.1 计算图解析
python export_dag.py --onnx models/mlp.onnx --output dag.json

# C3.5 端到端推理
python infer.py \
  --onnx models/mlp.onnx \
  --input data/mnist_test/ \
  --output results/ \
  --batch-size 256
```

## 精度要求

- 所有比对使用 `rtol = atol = 1e-3`
- 参考输出由 CPU（numpy/PyTorch）以 FP32 精度生成
- 参赛者内部可使用更低精度，但必须满足准确率门禁
