# C3 算子调度与模型部署

## 项目概述

本项目实现了一个完整的 GPU 推理引擎，支持 ONNX 模型的计算图解析、算子分解、算子融合、内存调度和推理执行。

## 环境要求

- Python 3.8+
- CUDA 12.0+
- CuPy
- ONNX

## 安装依赖

```bash
pip install -r requirements.txt
```

## 子任务说明

### C3.1 计算图解析与表示

命令行模板：
```bash
python cli/export_dag.py --onnx {onnx} --output {output}
```

功能：读取 ONNX 模型，解析计算图，导出为 DAG JSON 文件。

### C3.5 典型模型部署

命令行模板：
```bash
python cli/infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256
```

功能：加载 ONNX 模型，执行推理，输出结果。

## 目录结构

```
C3-scheduler/
├── cli/
│   ├── export_dag.py      # C3.1 计算图解析与导出
│   └── infer.py           # C3.5 推理程序入口
├── scheduler/
│   ├── core/              # 计算图解析与表示 (C3.1)
│   │   ├── onnx_parser.py # ONNX 模型解析器
│   │   └── graph.py       # GraphDAG 数据结构
│   ├── strategies/        # 算子分解与内核选择 (C3.2)
│   │   ├── decomposition_strategy.py
│   │   └── precision_strategy.py
│   ├── graph_passes/      # 算子融合与图优化 (C3.3)
│   │   └── fusion.py
│   ├── memory/            # 内存规划与调度 (C3.4)
│   │   ├── memory_pool.py
│   │   └── stream_scheduler.py
│   └── runtime/           # 推理引擎 (C3.5)
│       ├── cupy_engine.py
│       └── inference_engine.py
├── kernels/
│   ├── cupy_kernels.py    # CuPy 算子实现
│   └── cuda_kernels.py    # CUDA 算子实现
├── benchmarks/
│   └── c32_c33/           # C3.2/C3.3 评测脚本
├── requirements.txt       # Python 依赖
└── README.md              # 构建与运行说明
```

## 支持的算子

共支持 18 种 ONNX 算子：
- Add, Constant, Conv, Div, Erf, Flatten, Identity
- Gather, Gemm, GlobalAveragePool, LayerNormalization
- MatMul, Mul, Relu, Reshape, Softmax, Split, Transpose

## 运行测试

```bash
# C3.1 测试
python cli/export_dag.py --onnx testcases/release_to_competitors/models/mlp_v1.onnx --output mlp_dag.json

# C3.5 测试
python cli/infer.py --onnx testcases/release_to_competitors/models/mlp_v1.onnx \
    --input testcases/release_to_competitors/testdata/c35/mlp_v1/input/ \
    --output test_output/mlp_v1/ \
    --batch-size 256

# C3.2/C3.3 评测
python benchmarks/c32_c33/bench_c32_c33.py --models mlp_v1 resnet_v1 --output-dir benchmarks/c32_c33/results
```
