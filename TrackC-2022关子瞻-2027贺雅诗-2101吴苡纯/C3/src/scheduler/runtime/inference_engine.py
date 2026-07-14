import os
import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional


class InferenceEngine:
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None
        self.input_names = []
        self.output_names = []
        self._initialize()

    def _initialize(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
            sess_options=self._create_session_options()
        )
        
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _create_session_options(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        return options

    def get_input_shapes(self) -> Dict[str, List[int]]:
        return {
            input.name: list(input.shape)
            for input in self.session.get_inputs()
        }

    def get_output_shapes(self) -> Dict[str, List[int]]:
        return {
            output.name: list(output.shape)
            for output in self.session.get_outputs()
        }

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        outputs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outputs))

    def run_batch(self, inputs: Dict[str, np.ndarray], batch_size: int = 256) -> Dict[str, np.ndarray]:
        results = {}
        
        for name in self.output_names:
            output_shape = self.get_output_shapes()[name]
            output_shape[0] = inputs[list(inputs.keys())[0]].shape[0]
            results[name] = np.zeros(output_shape, dtype=np.float32)
        
        total_samples = inputs[list(inputs.keys())[0]].shape[0]
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_inputs = {k: v[start:end] for k, v in inputs.items()}
            batch_outputs = self.run(batch_inputs)
            
            for name, output in batch_outputs.items():
                results[name][start:end] = output
        
        return results


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
