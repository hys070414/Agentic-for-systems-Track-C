from typing import List, Dict, Tuple
from scheduler.core.graph import NodeInfo


class Stream:
    def __init__(self, stream_id: int, stream_type: str):
        self.stream_id = stream_id
        self.stream_type = stream_type


class WeightPrefetchScheduler:
    def __init__(self):
        pass

    def schedule_prefetch(self, nodes: List[NodeInfo], weights: Dict[str, str]) -> List[Dict]:
        schedule = []
        
        for i, node in enumerate(nodes):
            node_weights = [w for w in node.inputs if w in weights]
            
            if node_weights:
                for weight in node_weights:
                    schedule.append({
                        'type': 'prefetch',
                        'weight_name': weight,
                        'target_node': node.name,
                        'prefetch_before': max(0, i - 1)
                    })
            
            schedule.append({
                'type': 'compute',
                'node_name': node.name,
                'node_idx': i
            })
        
        return schedule


class StreamParallelScheduler:
    def __init__(self, num_streams: int = 2):
        self.num_streams = num_streams
        self.streams = [Stream(i, 'compute') for i in range(num_streams)]
        self.streams.append(Stream(num_streams, 'copy'))

    def assign_streams(self, nodes: List[NodeInfo], dependencies: List[Tuple[int, int]]) -> List[int]:
        stream_assignments = []
        node_deps: Dict[int, List[int]] = {i: [] for i in range(len(nodes))}
        
        for src, dst in dependencies:
            node_deps[dst].append(src)
        
        for i, node in enumerate(nodes):
            used_streams = set()
            for dep_idx in node_deps[i]:
                used_streams.add(stream_assignments[dep_idx])
            
            assigned_stream = 0
            while assigned_stream in used_streams and assigned_stream < self.num_streams:
                assigned_stream += 1
            
            stream_assignments.append(assigned_stream)
        
        return stream_assignments
