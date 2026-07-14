from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: List


@dataclass
class NodeInfo:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]


@dataclass
class EdgeInfo:
    src_node: str
    dst_node: str
    tensor: str


@dataclass
class GraphDAG:
    format_version: str = "1.0"
    graph_inputs: List[TensorInfo] = None
    graph_outputs: List[TensorInfo] = None
    nodes: List[NodeInfo] = None
    edges: List[EdgeInfo] = None

    def __post_init__(self):
        if self.graph_inputs is None:
            self.graph_inputs = []
        if self.graph_outputs is None:
            self.graph_outputs = []
        if self.nodes is None:
            self.nodes = []
        if self.edges is None:
            self.edges = []

    def to_dict(self):
        return {
            "format_version": self.format_version,
            "graph_inputs": [asdict(t) for t in self.graph_inputs],
            "graph_outputs": [asdict(t) for t in self.graph_outputs],
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    def validate(self):
        node_names = {n.name for n in self.nodes}
        output_tensors = set()
        for node in self.nodes:
            for out in node.outputs:
                output_tensors.add(out)

        for edge in self.edges:
            if edge.src_node not in node_names:
                raise ValueError(f"Edge src_node {edge.src_node} not in nodes")
            if edge.dst_node not in node_names:
                raise ValueError(f"Edge dst_node {edge.dst_node} not in nodes")

        return True
