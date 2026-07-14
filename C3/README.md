# C3 算子调度与模型部署

## 目录结构

```
C3/
├── framework/            # 框架源码
│   ├── benchmarks/
│   ├── cli/
│   ├── kernels/
│   ├── scheduler/
│   ├── scripts/
│   ├── testcases/
│   └── ...
└── README.md
```

## 命令模板

### C3.1 计算图解析与表示

```bash
python framework/cli/export_dag.py --onnx <model.onnx> --output <dag.json>
```

### C3.5 模型推理

```bash
python framework/cli/infer_worker.py
```