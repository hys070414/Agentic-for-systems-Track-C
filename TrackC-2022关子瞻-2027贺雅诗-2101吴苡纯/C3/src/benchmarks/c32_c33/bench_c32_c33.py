import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scheduler.core.onnx_parser import import_onnx_graph
from scheduler.strategies import strategy, precision_strategy, hardware
from scheduler.graph_passes.fusion import GraphPassPipeline


def benchmark_c32(dag):
    results = [] 
    
    for node in dag.nodes:
        precision = precision_strategy.select_precision(node, dag)
        
        kernels = strategy.decompose(node, dag, precision)
        
        tuning_params = []
        for kernel in kernels:
            problem_size = (1024, 1024)
            params = strategy.tune_kernel(kernel, precision, problem_size)
            tuning_params.append({
                'kernel_name': kernel.name,
                'block_x': params.block_x,
                'grid_x': params.grid_x,
                'smem_bytes': params.smem_bytes
            })
        
        intermediate_tensors = []
        for kernel in kernels:
            inter = set(kernel.outputs) - set(node.outputs)
            intermediate_tensors.extend(list(inter))
        
        results.append({
            'node_name': node.name,
            'op_type': node.op_type,
            'precision': precision.precision,
            'kernels': [k.name for k in kernels],
            'tuning_params': tuning_params,
            'intermediate_tensors': intermediate_tensors
        })
    
    return results


def benchmark_c33(dag):
    pipeline = GraphPassPipeline(enable_fusion=True)
    pass_results = pipeline.run(dag)
    
    optimized_dag = pass_results['optimized_graph']
    fusion_log = pass_results['Fusion']['stats']['fusion_log']
    
    result = {
        'original_nodes': len(dag.nodes),
        'optimized_nodes': len(optimized_dag.nodes),
        'fusion_log': fusion_log,
        'fusion_patterns': list(set(entry['pattern'] for entry in fusion_log))
    }
    
    return result, optimized_dag


def main():
    parser = argparse.ArgumentParser(description='Benchmark C3.2 and C3.3')
    parser.add_argument('--models', nargs='+', required=True, help='Model names to benchmark')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_scores = {}
    all_results = {}
    
    for model_name in args.models:
        model_path = f"testcases/release_to_competitors/models/{model_name}.onnx"
        
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}", file=sys.stderr)
            continue
        
        dag = import_onnx_graph(model_path)
        
        c32_results = benchmark_c32(dag)
        c33_results, optimized_dag = benchmark_c33(dag)
        
        model_result = {
            'c32': c32_results,
            'c33': c33_results
        }
        
        all_results[model_name] = model_result
        
        with open(os.path.join(args.output_dir, f"bench_{model_name}.json"), 'w') as f:
            json.dump(model_result, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'scores.json'), 'w') as f:
        json.dump(all_scores, f, indent=2)
    
    print(f"Benchmark results written to {args.output_dir}")


if __name__ == '__main__':
    main()
