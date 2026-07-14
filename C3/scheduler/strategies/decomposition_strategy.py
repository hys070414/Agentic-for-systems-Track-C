from typing import List, Dict, Tuple
from scheduler.core.graph import NodeInfo
from .precision_strategy import PrecisionProfile


class KernelSpecRef:
    def __init__(self, name: str, inputs: List[str], outputs: List[str]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs


class KernelTuningParams:
    def __init__(self, block_x: int = 256, grid_x: int = 1, smem_bytes: int = 0):
        self.block_x = block_x
        self.grid_x = grid_x
        self.smem_bytes = smem_bytes


class DecompositionStrategy:
    def __init__(self):
        self.intermediate_counter = 0

    def _generate_intermediate_name(self) -> str:
        self.intermediate_counter += 1
        return f"__c3_inter_{self.intermediate_counter}__"

    def decompose(self, node: NodeInfo, graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
        op_type = node.op_type
        
        if op_type in ["MatMul", "Gemm"]:
            return self._decompose_matmul(node, precision)
        elif op_type == "Softmax":
            return self._decompose_softmax(node, precision)
        elif op_type == "LayerNormalization":
            return self._decompose_layernorm(node, precision)
        elif op_type == "Conv":
            return self._decompose_conv(node, precision)
        elif op_type in ["Relu", "Add", "Mul", "Div"]:
            return self._decompose_elementwise(node, precision)
        else:
            return [KernelSpecRef(f"{op_type.lower()}_{precision.precision}", node.inputs, node.outputs)]

    def _decompose_matmul(self, node: NodeInfo, precision: PrecisionProfile) -> List[KernelSpecRef]:
        kernel_name = f"matmul_{precision.precision}"
        return [KernelSpecRef(kernel_name, node.inputs, node.outputs)]

    def _decompose_softmax(self, node: NodeInfo, precision: PrecisionProfile) -> List[KernelSpecRef]:
        inter1 = self._generate_intermediate_name()
        inter2 = self._generate_intermediate_name()
        inter3 = self._generate_intermediate_name()
        
        return [
            KernelSpecRef(f"reduce_max_{precision.precision}", node.inputs, [inter1]),
            KernelSpecRef(f"sub_{precision.precision}", [node.inputs[0], inter1], [inter2]),
            KernelSpecRef(f"exp_{precision.precision}", [inter2], [inter3]),
            KernelSpecRef(f"reduce_sum_{precision.precision}", [inter3], [inter2]),
            KernelSpecRef(f"div_{precision.precision}", [inter3, inter2], node.outputs),
        ]

    def _decompose_layernorm(self, node: NodeInfo, precision: PrecisionProfile) -> List[KernelSpecRef]:
        inter1 = self._generate_intermediate_name()
        inter2 = self._generate_intermediate_name()
        inter3 = self._generate_intermediate_name()
        inter4 = self._generate_intermediate_name()
        
        return [
            KernelSpecRef(f"reduce_mean_{precision.precision}", node.inputs[:1], [inter1]),
            KernelSpecRef(f"sub_{precision.precision}", [node.inputs[0], inter1], [inter2]),
            KernelSpecRef(f"mul_{precision.precision}", [inter2, node.inputs[1]], [inter3]),
            KernelSpecRef(f"sqrt_{precision.precision}", [inter3], [inter4]),
            KernelSpecRef(f"div_{precision.precision}", [inter2, inter4], node.outputs),
        ]

    def _decompose_conv(self, node: NodeInfo, precision: PrecisionProfile) -> List[KernelSpecRef]:
        kernel_name = f"winograd_forward_{precision.precision}"
        return [KernelSpecRef(kernel_name, node.inputs, node.outputs)]

    def _decompose_elementwise(self, node: NodeInfo, precision: PrecisionProfile) -> List[KernelSpecRef]:
        kernel_name = f"{node.op_type.lower()}_{precision.precision}"
        return [KernelSpecRef(kernel_name, node.inputs, node.outputs)]

    def tune_kernel(self, ref: KernelSpecRef, precision: PrecisionProfile, problem_size: Tuple[int, ...]) -> KernelTuningParams:
        block_x = 256
        grid_x = 1
        smem_bytes = 0
        
        if problem_size:
            if "matmul" in ref.name:
                block_x = 256
                total_elements = problem_size[0] * problem_size[1] if len(problem_size) >= 2 else problem_size[0]
                grid_x = max(1, (total_elements + block_x - 1) // block_x)
                smem_bytes = min(block_x * 64 * 4, 163 * 1024)
            elif "winograd" in ref.name or "conv" in ref.name.lower():
                block_x = 256
                grid_x = max(1, problem_size[0] * problem_size[1] // block_x) if len(problem_size) >= 2 else 1
                smem_bytes = min(block_x * 128 * 4, 163 * 1024)
            elif "reduce" in ref.name:
                block_x = 256
                grid_x = max(1, problem_size[0] // block_x) if problem_size else 1
                smem_bytes = 0
            else:
                block_x = min(256, problem_size[0])
                grid_x = max(1, (problem_size[0] + block_x - 1) // block_x)
        
        return KernelTuningParams(block_x=block_x, grid_x=grid_x, smem_bytes=smem_bytes)
