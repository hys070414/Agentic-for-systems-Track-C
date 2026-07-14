import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.core.onnx_parser import import_onnx_graph


def main():
    parser = argparse.ArgumentParser(description='Export ONNX model to DAG JSON')
    parser.add_argument('--onnx', required=True, help='Input ONNX model file path')
    parser.add_argument('--output', required=True, help='Output DAG JSON file path')
    
    args = parser.parse_args()

    try:
        dag = import_onnx_graph(args.onnx)
        dag.validate()
        
        with open(args.output, 'w') as f:
            json.dump(dag.to_dict(), f, indent=2)
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
