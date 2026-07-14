import onnx
from onnx import helper, TensorProto
from typing import Dict, Set, List, Tuple
from .graph import GraphDAG, TensorInfo, NodeInfo, EdgeInfo


def dtype_to_str(dtype: int) -> str:
    dtype_map = {
        TensorProto.FLOAT: "FLOAT",
        TensorProto.FLOAT16: "FLOAT16",
        TensorProto.DOUBLE: "DOUBLE",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
        TensorProto.UINT8: "UINT8",
        TensorProto.INT8: "INT8",
        TensorProto.BOOL: "BOOL",
    }
    return dtype_map.get(dtype, f"UNKNOWN_{dtype}")


def shape_to_list(shape) -> List:
    result = []
    for dim in shape.dim:
        if dim.HasField("dim_param"):
            result.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(None)
    return result


def import_onnx_graph(model_path: str) -> GraphDAG:
    model = onnx.load(model_path)
    graph = model.graph

    initializer_names = {init.name for init in graph.initializer}
    constant_names = set()
    for node in graph.node:
        if node.op_type == "Constant":
            for output in node.output:
                constant_names.add(output)

    graph_inputs = []
    for input_tensor in graph.input:
        if input_tensor.name not in initializer_names:
            dtype = dtype_to_str(input_tensor.type.tensor_type.elem_type)
            shape = shape_to_list(input_tensor.type.tensor_type.shape)
            graph_inputs.append(TensorInfo(
                name=input_tensor.name,
                dtype=dtype,
                shape=shape
            ))

    graph_outputs = []
    for output_tensor in graph.output:
        dtype = dtype_to_str(output_tensor.type.tensor_type.elem_type)
        shape = shape_to_list(output_tensor.type.tensor_type.shape)
        graph_outputs.append(TensorInfo(
            name=output_tensor.name,
            dtype=dtype,
            shape=shape
        ))

    nodes = []
    node_output_map: Dict[str, str] = {}

    for node in graph.node:
        node_name = node.name if node.name else f"{node.op_type}_{len(nodes)}"
        inputs = list(node.input)
        outputs = list(node.output)

        nodes.append(NodeInfo(
            name=node_name,
            op_type=node.op_type,
            inputs=inputs,
            outputs=outputs
        ))

        for output in outputs:
            node_output_map[output] = node_name

    edges = []
    for node in nodes:
        for input_name in node.inputs:
            if input_name in node_output_map:
                src_node = node_output_map[input_name]
                dst_node = node.name
                edges.append(EdgeInfo(
                    src_node=src_node,
                    dst_node=dst_node,
                    tensor=input_name
                ))

    dag = GraphDAG(
        format_version="1.0",
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        nodes=nodes,
        edges=edges
    )

    return dag
