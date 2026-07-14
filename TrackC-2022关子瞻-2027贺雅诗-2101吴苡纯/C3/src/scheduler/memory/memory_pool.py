import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Tuple


class DeviceMemoryPool:
    def __init__(self, total_size: int = 2 * 1024 * 1024 * 1024):
        self.total_size = total_size
        self._cupy_pool = cp.cuda.MemoryPool()
        self._allocated_blocks: Dict[int, cp.ndarray] = {}
        self._free_list: List[Tuple[int, cp.ndarray]] = []
        self._cuda_stream = cp.cuda.Stream()

    def malloc(self, size: int) -> cp.ndarray:
        for cached_size, cached_array in self._free_list:
            if cached_size >= size:
                self._free_list.remove((cached_size, cached_array))
                ptr = id(cached_array)
                self._allocated_blocks[ptr] = cached_array
                return cached_array
        
        array = self._cupy_pool.malloc(size).reshape(1)
        ptr = id(array)
        self._allocated_blocks[ptr] = array
        return array

    def free(self, array: cp.ndarray):
        ptr = id(array)
        if ptr in self._allocated_blocks:
            del self._allocated_blocks[ptr]
            self._free_list.append((array.nbytes, array))
            if len(self._free_list) > 100:
                self._coalesce()

    def _coalesce(self):
        sorted_free = sorted(self._free_list, key=lambda x: x[0])
        self._free_list = []
        for size, array in sorted_free:
            merged = False
            for i, (existing_size, existing_array) in enumerate(self._free_list):
                if existing_size == size:
                    merged = True
                    break
            if not merged:
                self._free_list.append((size, array))

    def get_stats(self) -> Dict:
        used_bytes = sum(arr.nbytes for arr in self._allocated_blocks.values())
        free_bytes = sum(size for size, _ in self._free_list)
        return {
            'total': self.total_size,
            'used': used_bytes,
            'free': free_bytes,
            'num_allocated_blocks': len(self._allocated_blocks),
            'num_free_blocks': len(self._free_list)
        }

    def __del__(self):
        self._cupy_pool.free_all_blocks()


class TensorLifetime:
    def __init__(self, name: str, first_use: int, last_use: int, size: int = 0):
        self.name = name
        self.first_use = first_use
        self.last_use = last_use
        self.size = size


class LifetimeAnalyzer:
    def __init__(self):
        pass

    def analyze(self, nodes: List, edges: List) -> List[TensorLifetime]:
        tensor_uses: Dict[str, List[int]] = {}
        tensor_sizes: Dict[str, int] = {}
        
        for idx, node in enumerate(nodes):
            if isinstance(node, dict):
                inputs = node.get('inputs', [])
                outputs = node.get('outputs', [])
            else:
                inputs = getattr(node, 'inputs', [])
                outputs = getattr(node, 'outputs', [])
            
            for inp in inputs:
                if inp not in tensor_uses:
                    tensor_uses[inp] = []
                tensor_uses[inp].append(idx)
            
            for out in outputs:
                if out not in tensor_uses:
                    tensor_uses[out] = []
                tensor_uses[out].append(idx)
        
        lifetimes = []
        for tensor_name, uses in tensor_uses.items():
            if uses:
                lifetimes.append(TensorLifetime(
                    name=tensor_name,
                    first_use=min(uses),
                    last_use=max(uses),
                    size=tensor_sizes.get(tensor_name, 0)
                ))
        
        return sorted(lifetimes, key=lambda x: x.first_use)


class MemoryReuseScheduler:
    def __init__(self, memory_pool: DeviceMemoryPool):
        self.memory_pool = memory_pool
        self.tensor_slot_map: Dict[str, int] = {}
        self.slot_lifetimes: Dict[int, int] = {}

    def assign_slots(self, lifetimes: List[TensorLifetime]) -> Dict[str, int]:
        current_slots: List[Tuple[int, int]] = []
        
        for lifetime in lifetimes:
            current_slots = [(slot, end) for slot, end in current_slots if end > lifetime.first_use]
            
            if len(current_slots) < len(self.slot_lifetimes):
                for slot in self.slot_lifetimes:
                    if slot not in [s[0] for s in current_slots]:
                        assigned_slot = slot
                        break
            else:
                assigned_slot = len(self.slot_lifetimes)
            
            self.tensor_slot_map[lifetime.name] = assigned_slot
            self.slot_lifetimes[assigned_slot] = lifetime.last_use
            current_slots.append((assigned_slot, lifetime.last_use))
        
        return self.tensor_slot_map


class WeightPrefetcher:
    def __init__(self):
        self.host_weights: Dict[str, np.ndarray] = {}
        self.device_weights: Dict[str, cp.ndarray] = {}
        self.weight_usage: Dict[str, int] = {}

    def register_weight(self, name: str, data: np.ndarray):
        self.host_weights[name] = data
        self.weight_usage[name] = 0

    def prefetch(self, name: str):
        if name in self.device_weights:
            self.weight_usage[name] += 1
            return
        
        if name in self.host_weights:
            self.device_weights[name] = cp.array(self.host_weights[name])
            self.weight_usage[name] = 1

    def evict(self, name: str):
        if name in self.device_weights:
            del self.device_weights[name]

    def evict_unused(self, threshold: int = 0):
        to_evict = [name for name, usage in self.weight_usage.items() 
                    if usage <= threshold and name in self.device_weights]
        for name in to_evict:
            self.evict(name)

    def get_device_weight(self, name: str) -> Optional[cp.ndarray]:
        if name in self.device_weights:
            return self.device_weights[name]
        return None

    def clear(self):
        self.device_weights.clear()
        self.weight_usage.clear()


class DynamicMemoryManager:
    def __init__(self, max_gpu_memory: int = 16 * 1024 * 1024 * 1024):
        self.max_gpu_memory = max_gpu_memory
        self.prefetcher = WeightPrefetcher()
        self.memory_pool = DeviceMemoryPool(max_gpu_memory)
        self.current_usage = 0

    def register_weights(self, weights: Dict[str, np.ndarray]):
        for name, data in weights.items():
            self.prefetcher.register_weight(name, data)

    def get_weight(self, name: str) -> cp.ndarray:
        device_weight = self.prefetcher.get_device_weight(name)
        if device_weight is not None:
            self.prefetcher.weight_usage[name] += 1
            return device_weight
        
        if name in self.prefetcher.host_weights:
            required_bytes = self.prefetcher.host_weights[name].nbytes
            self._ensure_memory(required_bytes)
            self.prefetcher.prefetch(name)
            device_weight = self.prefetcher.get_device_weight(name)
            if device_weight is not None:
                self.prefetcher.weight_usage[name] += 1
        
        return device_weight

    def _ensure_memory(self, required_bytes: int):
        free_memory = self.max_gpu_memory - self._get_current_gpu_usage()
        
        max_retries = 20
        retries = 0
        
        while free_memory < required_bytes and retries < max_retries:
            self.prefetcher.evict_unused(threshold=0)
            cp.cuda.runtime.deviceSynchronize()
            new_free = self.max_gpu_memory - self._get_current_gpu_usage()
            
            if new_free == free_memory:
                retries += 1
            else:
                retries = 0
            
            free_memory = new_free
            if free_memory >= required_bytes:
                break

    def _get_current_gpu_usage(self) -> int:
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            return mem_info[1] - mem_info[0]
        except Exception:
            return self.current_usage

    def cleanup(self):
        self.prefetcher.clear()
        cp.cuda.runtime.deviceSynchronize()