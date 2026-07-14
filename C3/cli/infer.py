import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.runtime.cupy_engine import CuPyInferenceEngine, load_input_from_dir, write_output_to_dir


def main():
    parser = argparse.ArgumentParser(description='Run CuPy-based GPU inference on ONNX model')
    parser.add_argument('--onnx', required=True, help='Input ONNX model file path')
    parser.add_argument('--input', required=True, help='Input directory containing manifest.json and .npy files')
    parser.add_argument('--output', required=True, help='Output directory to write results')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    
    args = parser.parse_args()

    try:
        engine = CuPyInferenceEngine(args.onnx)
        
        inputs = load_input_from_dir(args.input)
        
        outputs = engine.run_batch(inputs, batch_size=args.batch_size)
        
        write_output_to_dir(args.output, outputs)
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
