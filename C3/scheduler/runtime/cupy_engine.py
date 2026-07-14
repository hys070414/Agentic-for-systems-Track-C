import os
import json
import numpy as np
import cupy as cp

import onnx
from onnx import TensorProto
from typing import Dict, List, Tuple, Optional
from kernels.cupy_kernels import OPERATOR_REGISTRY, get_operator
from scheduler.memory.memory_pool import (
    DeviceMemoryPool, 
    LifetimeAnalyzer, 
    MemoryReuseScheduler,
    WeightPrefetcher,
    DynamicMemoryManager
)


class CuPyInferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.graph = None
        self.tensors: Dict[str, cp.ndarray] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.execution_order: List = []
        self._node_name_map: Dict[str, onnx.NodeProto] = {}
        
        self.memory_manager = DynamicMemoryManager()
        self.lifetime_analyzer = LifetimeAnalyzer()
        self.memory_reuse_scheduler = MemoryReuseScheduler(
            self.memory_manager.memory_pool
        )
        self.prefetcher = self.memory_manager.prefetcher
        
        self._initialize()

    def _initialize(self):
        self.model = onnx.load(self.model_path, load_external_data=True)
        self.graph = self.model.graph
        
        self._load_initializers_lazy()
        self._load_constants()
        self._extract_io_names()
        self._build_execution_order()
        self._analyze_lifetime()

    def _load_initializers_lazy(self):
        for init in self.graph.initializer:
            data = self._tensor_proto_to_numpy(init)
            self.prefetcher.register_weight(init.name, data)

    def _load_constants(self):
        for node in self.graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value':
                        self.tensors[node.output[0]] = self._tensor_proto_to_cupy(attr.t)

    def _tensor_proto_to_numpy(self, tensor_proto) -> np.ndarray:
        if tensor_proto.data_type == TensorProto.FLOAT:
            if tensor_proto.float_data:
                data = np.array(tensor_proto.float_data, dtype=np.float32)
            else:
                data = np.frombuffer(tensor_proto.raw_data, dtype=np.float32)
        elif tensor_proto.data_type == TensorProto.FLOAT16:
            data = np.frombuffer(tensor_proto.raw_data, dtype=np.float16)
        elif tensor_proto.data_type == TensorProto.DOUBLE:
            if tensor_proto.double_data:
                data = np.array(tensor_proto.double_data, dtype=np.float64)
            else:
                data = np.frombuffer(tensor_proto.raw_data, dtype=np.float64)
        elif tensor_proto.data_type == TensorProto.INT32:
            if tensor_proto.int32_data:
                data = np.array(tensor_proto.int32_data, dtype=np.int32)
            else:
                data = np.frombuffer(tensor_proto.raw_data, dtype=np.int32)
        elif tensor_proto.data_type == TensorProto.INT64:
            if tensor_proto.int64_data:
                data = np.array(tensor_proto.int64_data, dtype=np.int64)
            else:
                data = np.frombuffer(tensor_proto.raw_data, dtype=np.int64)
        elif tensor_proto.data_type == TensorProto.UINT8:
            if tensor_proto.int32_data:
                data = np.array(tensor_proto.int32_data, dtype=np.uint8)
            else:
                data = np.frombuffer(tensor_proto.raw_data, dtype=np.uint8)
        else:
            data = np.frombuffer(tensor_proto.raw_data, dtype=np.float32)
        
        shape = []
        for d in tensor_proto.dims:
            if hasattr(d, 'dim_value'):
                shape.append(d.dim_value)
            else:
                shape.append(int(d))
        if shape:
            data = data.reshape(shape)
        
        return data

    def _tensor_proto_to_cupy(self, tensor_proto) -> cp.ndarray:
        return cp.array(self._tensor_proto_to_numpy(tensor_proto))

    def _analyze_lifetime(self):
        nodes_info = []
        edges = []
        
        for idx, node in enumerate(self.graph.node):
            node_name = node.name if node.name else f"{node.op_type}_{idx}"
            nodes_info.append({
                'name': node_name,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
            
            for inp in node.input:
                for prev_idx, prev_node in enumerate(self.graph.node[:idx]):
                    prev_name = prev_node.name if prev_node.name else f"{prev_node.op_type}_{prev_idx}"
                    if inp in prev_node.output:
                        edges.append({
                            'src_node': prev_name,
                            'dst_node': node_name,
                            'tensor': inp
                        })
        
        self.lifetimes = self.lifetime_analyzer.analyze(nodes_info, edges)
        self.tensor_slot_map = self.memory_reuse_scheduler.assign_slots(self.lifetimes)

    def _prefetch_weight(self, name: str):
        device_weight = self.prefetcher.get_device_weight(name)
        if device_weight is not None:
            return device_weight
        
        return self.memory_manager.get_weight(name)

    def _clear_unused_weights(self, current_node_name: str):
        current_idx = self.execution_order.index(current_node_name) if current_node_name in self.execution_order else -1
        if current_idx < 0:
            return
        
        used_in_future = set()
        for node_name in self.execution_order[current_idx + 1:]:
            node = self._node_name_map.get(node_name)
            if node:
                for inp in node.input:
                    if inp in self.prefetcher.host_weights:
                        used_in_future.add(inp)
        
        to_remove = []
        for name in list(self.tensors.keys()):
            if name in self.prefetcher.host_weights and name not in used_in_future:
                to_remove.append(name)
        
        for name in to_remove:
            del self.tensors[name]
            cp.cuda.runtime.deviceSynchronize()

    def _extract_io_names(self):
        initializer_names = {init.name for init in self.graph.initializer}
        self.input_names = [i.name for i in self.graph.input if i.name not in initializer_names]
        self.output_names = [o.name for o in self.graph.output]

    def _build_execution_order(self):
        node_output_map = {}
        dependencies = {}
        
        for node in self.graph.node:
            node_name = node.name if node.name else f"{node.op_type}_{len(node_output_map)}"
            dependencies[node_name] = set()
            
            for inp in node.input:
                if inp in node_output_map:
                    dependencies[node_name].add(node_output_map[inp])
            
            for out in node.output:
                node_output_map[out] = node_name
        
        in_degree = {node_name: len(deps) for node_name, deps in dependencies.items()}
        queue = [n for n in in_degree if in_degree[n] == 0]
        order = []
        
        while queue:
            node_name = queue.pop(0)
            order.append(node_name)
            
            for target_node, deps in dependencies.items():
                if node_name in deps:
                    in_degree[target_node] -= 1
                    if in_degree[target_node] == 0:
                        queue.append(target_node)
        
        self.execution_order = order
        self._node_name_map = {node.name if node.name else f"{node.op_type}_{i}": node 
                               for i, node in enumerate(self.graph.node)}

    def _get_attr(self, node, attr_name, default=None):
        for attr in node.attribute:
            if attr.name == attr_name:
                if attr.type == onnx.AttributeProto.FLOAT:
                    return attr.f
                elif attr.type == onnx.AttributeProto.INT:
                    return attr.i
                elif attr.type == onnx.AttributeProto.STRING:
                    return attr.s.decode()
                elif attr.type == onnx.AttributeProto.INTS:
                    return list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    return list(attr.floats)
                elif attr.type == onnx.AttributeProto.TENSOR:
                    return self._tensor_proto_to_cupy(attr.t)
        return default

    def _execute_node(self, node):
        if node.op_type == 'Constant':
            return
        
        operator = get_operator(node.op_type)
        if operator is None:
            raise NotImplementedError(f"Operator {node.op_type} not implemented")
        
        inputs = []
        for inp in node.input:
            if inp in self.tensors:
                inputs.append(self.tensors[inp])
            elif inp in self.prefetcher.host_weights:
                device_weight = self._prefetch_weight(inp)
                inputs.append(device_weight)
                self.tensors[inp] = device_weight
            elif inp == '':
                inputs.append(None)
        
        kwargs = {}
        
        if node.op_type == 'Gemm':
            kwargs['alpha'] = self._get_attr(node, 'alpha', 1.0)
            kwargs['beta'] = self._get_attr(node, 'beta', 1.0)
            kwargs['transA'] = self._get_attr(node, 'transA', 0)
            kwargs['transB'] = self._get_attr(node, 'transB', 0)
        elif node.op_type == 'Flatten':
            kwargs['axis'] = self._get_attr(node, 'axis', 1)
        elif node.op_type == 'Reshape':
            kwargs['shape'] = self._get_attr(node, 'shape', [])
        elif node.op_type == 'Transpose':
            kwargs['perm'] = self._get_attr(node, 'perm', [])
        elif node.op_type == 'Split':
            kwargs['split_sizes'] = self._get_attr(node, 'split', [])
            kwargs['axis'] = self._get_attr(node, 'axis', 0)
            kwargs['num_outputs'] = len(node.output)
        elif node.op_type == 'Conv':
            kwargs['pads'] = self._get_attr(node, 'pads', [0, 0, 0, 0])
            kwargs['strides'] = self._get_attr(node, 'strides', [1, 1])
            kwargs['dilations'] = self._get_attr(node, 'dilations', [1, 1])
        elif node.op_type == 'Softmax':
            kwargs['axis'] = self._get_attr(node, 'axis', -1)
        elif node.op_type == 'LayerNormalization':
            kwargs['epsilon'] = self._get_attr(node, 'epsilon', 1e-5)
        elif node.op_type == 'Gather':
            kwargs['axis'] = self._get_attr(node, 'axis', 0)
        
        result = operator(inputs, **kwargs)
        
        if isinstance(result, tuple) or isinstance(result, list):
            for i, out in enumerate(node.output):
                if i < len(result):
                    self.tensors[out] = result[i]
        else:
            for out in node.output:
                self.tensors[out] = result

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name, data in inputs.items():
            if isinstance(data, np.ndarray):
                self.tensors[name] = cp.array(data)
            else:
                self.tensors[name] = data
        
        for node_name in self.execution_order:
            node = self._node_name_map[node_name]
            self._execute_node(node)
            self._clear_unused_weights(node_name)
        
        outputs = {}
        for name in self.output_names:
            if name in self.tensors:
                outputs[name] = cp.asnumpy(self.tensors[name])
        
        return outputs

    def run_batch(self, inputs: Dict[str, np.ndarray], batch_size: int = 256) -> Dict[str, np.ndarray]:
        results = {}
        
        for name in self.output_names:
            output_shape = self._get_output_shape(inputs, name)
            results[name] = np.zeros(output_shape, dtype=np.float32)
        
        total_samples = inputs[list(inputs.keys())[0]].shape[0]
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_inputs = {k: v[start:end] for k, v in inputs.items()}
            batch_outputs = self.run(batch_inputs)
            
            for name, output in batch_outputs.items():
                results[name][start:end] = output
        
        return results

    def _get_output_shape(self, inputs: Dict[str, np.ndarray], output_name: str) -> List[int]:
        sample_input = {k: v[:1] for k, v in inputs.items()}
        sample_output = self.run(sample_input)
        
        shape = list(sample_output[output_name].shape)
        shape[0] = inputs[list(inputs.keys())[0]].shape[0]
        
        return shape

    def cleanup(self):
        self.memory_manager.cleanup()


def load_input_from_dir(input_dir: str) -> Dict[str, np.ndarray]:
    manifest_path = os.path.join(input_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    inputs = {}
    for tensor_info in manifest['tensors']:
        name = tensor_info['name']
        file_path = os.path.join(input_dir, tensor_info['file'])
        inputs[name] = np.load(file_path)
    
    return inputs


def write_output_to_dir(output_dir: str, outputs: Dict[str, np.ndarray]):
    os.makedirs(output_dir, exist_ok=True)
    
    tensors = []
    for name, data in outputs.items():
        file_name = f"{name}.npy"
        file_path = os.path.join(output_dir, file_name)
        np.save(file_path, data)
        
        tensors.append({
            'name': name,
            'file': file_name,
            'dtype': str(data.dtype),
            'shape': list(data.shape)
        })
    
    manifest = {'tensors': tensors}
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)