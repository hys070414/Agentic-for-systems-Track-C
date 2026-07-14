import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.runtime.cupy_engine import CuPyInferenceEngine, load_input_from_dir, write_output_to_dir


def main():
    print("READY", flush=True)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            task = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error parsing task JSON: {e}", file=sys.stderr)
            print(json.dumps({"status": "error", "error": f"JSON parse error: {e}"}), flush=True)
            continue
        
        if 'cmd' in task and task['cmd'] == 'exit':
            sys.exit(0)
        
        if 'onnx' not in task or 'input' not in task or 'output' not in task:
            print("Error: missing required fields", file=sys.stderr)
            print(json.dumps({"status": "error", "error": "missing required fields"}), flush=True)
            continue
        
        try:
            batch_size = task.get('batch_size', 256)
            
            print(f"Loading model: {task['onnx']}", file=sys.stderr)
            engine = CuPyInferenceEngine(task['onnx'])
            
            total_weight_bytes = sum(
                engine.prefetcher.host_weights[name].nbytes 
                for name in engine.prefetcher.host_weights
            ) if hasattr(engine, 'prefetcher') else 0
            
            if total_weight_bytes > 10 * 1024 * 1024 * 1024:
                batch_size = min(batch_size, 8)
            elif total_weight_bytes > 2 * 1024 * 1024 * 1024:
                batch_size = min(batch_size, 64)
            
            print(f"Model weight size: {total_weight_bytes / (1024**3):.2f} GB, adjusted batch_size: {batch_size}", file=sys.stderr)
            
            print(f"Loading input from: {task['input']}", file=sys.stderr)
            inputs = load_input_from_dir(task['input'])
            
            print(f"Running inference with batch_size={batch_size}", file=sys.stderr)
            outputs = engine.run_batch(inputs, batch_size=batch_size)
            
            print(f"Writing output to: {task['output']}", file=sys.stderr)
            write_output_to_dir(task['output'], outputs)
            
            sample_count = 0
            for name, tensor in outputs.items():
                sample_count = max(sample_count, tensor.shape[0])
            
            print(json.dumps({"status": "ok", "samples": sample_count}), flush=True)
            print(f"Inference completed, samples: {sample_count}", file=sys.stderr)
            
        except Exception as e:
            print(f"Inference error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"status": "error", "error": str(e)}), flush=True)


if __name__ == '__main__':
    main()
