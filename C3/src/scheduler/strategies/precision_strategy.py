from typing import Dict, Set, List, Optional
from scheduler.core.graph import NodeInfo


class PrecisionProfile:
    def __init__(self, precision: str):
        self.precision = precision


class HardwareInfo:
    def __init__(self):
        self.max_threads_per_block = 1024
        self.smem_bytes = 163 * 1024
        self._supported_precisions = {"fp32", "fp16", "fp8", "fp4"}

    def supported_precisions(self) -> Set[str]:
        return self._supported_precisions


class PrecisionStrategy:
    SENSITIVE_OPS = {
        "Softmax", "LayerNormalization", "BatchNormalization",
        "ReduceMax", "ReduceSum", "ReduceMean"
    }

    HIGH_THROUGHPUT_OPS = {
        "MatMul", "Gemm", "Conv", "Conv2d"
    }

    def __init__(self, hardware: HardwareInfo = None):
        self.hardware = hardware or HardwareInfo()

    def select_precision(self, node: NodeInfo, graph) -> PrecisionProfile:
        if node.op_type in self.SENSITIVE_OPS:
            return PrecisionProfile("fp32")
        
        if node.op_type in self.HIGH_THROUGHPUT_OPS:
            supported = self.hardware.supported_precisions()
            if "fp8" in supported:
                return PrecisionProfile("fp8")
            elif "fp16" in supported:
                return PrecisionProfile("fp16")
        
        return PrecisionProfile("fp32")
