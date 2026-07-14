from typing import List, Dict, Set, Tuple
from scheduler.core.graph import GraphDAG, NodeInfo, EdgeInfo


class FusionPattern:
    def __init__(self, name: str, pattern: List[str], fused_op: str):
        self.name = name
        self.pattern = pattern
        self.fused_op = fused_op

    def matches(self, nodes: List[NodeInfo]) -> bool:
        if len(nodes) != len(self.pattern):
            return False
        for i, node in enumerate(nodes):
            if node.op_type != self.pattern[i]:
                return False
        return True


class FusionPass:
    def __init__(self):
        self.patterns = [
            FusionPattern("FusedMatMulBias", ["MatMul", "Add"], "FusedMatMulBias"),
            FusionPattern("FusedMatMulBias", ["Gemm", "Add"], "FusedMatMulBias"),
            FusionPattern("FusedConv2dBatchNorm", ["Conv", "BatchNormalization"], "FusedConv2dBatchNorm"),
            FusionPattern("FusedSoftmaxDropout", ["Softmax", "Dropout"], "FusedSoftmaxDropout"),
            FusionPattern("FusedResidualNorm", ["Add", "LayerNormalization"], "FusedResidualNorm"),
            FusionPattern("FusedGemmRelu", ["Gemm", "Relu"], "FusedGemmRelu"),
            FusionPattern("FusedConvRelu", ["Conv", "Relu"], "FusedConvRelu"),
        ]
        self.fusion_log = []

    def run(self, dag: GraphDAG) -> GraphDAG:
        self.fusion_log = []
        new_nodes = []
        new_edges = []
        visited = set()
        
        i = 0
        while i < len(dag.nodes):
            matched = False
            
            for pattern in self.patterns:
                if i + len(pattern.pattern) > len(dag.nodes):
                    continue
                
                candidate_nodes = dag.nodes[i:i+len(pattern.pattern)]
                
                if not pattern.matches(candidate_nodes):
                    continue
                
                if not self._check_adjacent(candidate_nodes, dag):
                    continue
                
                fused_node = self._create_fused_node(pattern, candidate_nodes)
                new_nodes.append(fused_node)
                visited.update(range(i, i + len(pattern.pattern)))
                
                self.fusion_log.append({
                    "pattern": pattern.name,
                    "nodes": [n.name for n in candidate_nodes],
                    "fused_into": fused_node.name
                })
                
                i += len(pattern.pattern)
                matched = True
                break
            
            if not matched:
                if i not in visited:
                    new_nodes.append(dag.nodes[i])
                i += 1
        
        new_edges = self._rebuild_edges(new_nodes, dag.edges)
        
        return GraphDAG(
            format_version=dag.format_version,
            graph_inputs=dag.graph_inputs,
            graph_outputs=dag.graph_outputs,
            nodes=new_nodes,
            edges=new_edges
        )

    def _check_adjacent(self, nodes: List[NodeInfo], dag: GraphDAG) -> bool:
        for j in range(len(nodes) - 1):
            current_node = nodes[j]
            next_node = nodes[j+1]
            
            connected = False
            for edge in dag.edges:
                if edge.src_node == current_node.name and edge.dst_node == next_node.name:
                    connected = True
                    break
            
            if not connected:
                return False
        
        return True

    def _create_fused_node(self, pattern: FusionPattern, nodes: List[NodeInfo]) -> NodeInfo:
        inputs = []
        outputs = []
        
        for node in nodes:
            inputs.extend(node.inputs)
            outputs.extend(node.outputs)
        
        inputs = list(set(inputs))
        outputs = list(set(outputs))
        
        return NodeInfo(
            name=f"{pattern.name}_{nodes[0].name}",
            op_type=pattern.fused_op,
            inputs=inputs,
            outputs=outputs
        )

    def _rebuild_edges(self, nodes: List[NodeInfo], original_edges: List[EdgeInfo]) -> List[EdgeInfo]:
        node_names = {n.name for n in nodes}
        node_output_map = {}
        for node in nodes:
            for output in node.outputs:
                node_output_map[output] = node.name
        
        new_edges = []
        for edge in original_edges:
            if edge.src_node in node_names and edge.dst_node in node_names:
                if edge.src_node in node_output_map:
                    new_edges.append(edge)
        
        return new_edges


class EWChainFusionPass:
    def __init__(self):
        self.elementwise_ops = {"Add", "Mul", "Relu", "Div", "Sub"}
        self.fusion_log = []

    def run(self, dag: GraphDAG) -> GraphDAG:
        self.fusion_log = []
        new_nodes = []
        visited = set()
        
        i = 0
        while i < len(dag.nodes):
            if i in visited:
                i += 1
                continue
            
            chain = []
            j = i
            while j < len(dag.nodes):
                if dag.nodes[j].op_type in self.elementwise_ops:
                    chain.append(dag.nodes[j])
                    visited.add(j)
                    j += 1
                else:
                    break
            
            if 2 <= len(chain) <= 5:
                fused_node = self._create_fused_chain(chain)
                new_nodes.append(fused_node)
                self.fusion_log.append({
                    "pattern": "FusedEWChain",
                    "nodes": [n.name for n in chain],
                    "fused_into": fused_node.name
                })
            else:
                for node in chain:
                    new_nodes.append(node)
            
            i = j
        
        new_edges = self._rebuild_edges(new_nodes, dag.edges)
        
        return GraphDAG(
            format_version=dag.format_version,
            graph_inputs=dag.graph_inputs,
            graph_outputs=dag.graph_outputs,
            nodes=new_nodes,
            edges=new_edges
        )

    def _create_fused_chain(self, chain: List[NodeInfo]) -> NodeInfo:
        inputs = []
        outputs = []
        
        for node in chain:
            inputs.extend(node.inputs)
            outputs.extend(node.outputs)
        
        inputs = list(set(inputs))
        outputs = list(set(outputs))
        
        op_types = "_".join(n.op_type for n in chain)
        return NodeInfo(
            name=f"FusedEWChain_{chain[0].name}",
            op_type=f"FusedEWChain_{op_types}",
            inputs=inputs,
            outputs=outputs
        )

    def _rebuild_edges(self, nodes: List[NodeInfo], original_edges: List[EdgeInfo]) -> List[EdgeInfo]:
        node_names = {n.name for n in nodes}
        new_edges = []
        for edge in original_edges:
            if edge.src_node in node_names and edge.dst_node in node_names:
                new_edges.append(edge)
        return new_edges


class GraphPassPipeline:
    def __init__(self, enable_fusion: bool = True):
        self.enable_fusion = enable_fusion
        self.passes = []
        if enable_fusion:
            self.passes.append(FusionPass())
            self.passes.append(EWChainFusionPass())

    def run(self, dag: GraphDAG) -> Dict:
        pass_results = {
            'Fusion': {
                'stats': {
                    'fusion_log': []
                }
            }
        }
        
        current_dag = dag
        for pass_instance in self.passes:
            current_dag = pass_instance.run(current_dag)
            if hasattr(pass_instance, 'fusion_log'):
                pass_results['Fusion']['stats']['fusion_log'].extend(pass_instance.fusion_log)
        
        pass_results['optimized_graph'] = current_dag
        pass_results['Fusion']['stats']['original_nodes'] = len(dag.nodes)
        pass_results['Fusion']['stats']['optimized_nodes'] = len(current_dag.nodes)
        
        return pass_results
