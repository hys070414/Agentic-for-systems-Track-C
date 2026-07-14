import subprocess
import sys
import os


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if result.returncode == 0:
        print("[OK] Command succeeded")
        if result.stdout:
            print("Output:")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return True
    else:
        print("[FAIL] Command failed")
        if result.stderr:
            print("Error:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False


def main():
    print("=" * 60)
    print("C3-Scheduler Full Test Suite (CuPy GPU Mode)")
    print("=" * 60)
    
    python = sys.executable
    
    all_passed = True
    
    print("\n" + "="*60)
    print("[TEST 1] C3.1 - ONNX Graph Parsing")
    print("="*60)
    
    mlp_result = run_command([
        python, 'cli/export_dag.py',
        '--onnx', 'testcases/release_to_competitors/models/mlp_v1.onnx',
        '--output', 'testcases/mlp_dag_cupy.json'
    ], "Export MLP DAG")
    
    resnet_result = run_command([
        python, 'cli/export_dag.py',
        '--onnx', 'testcases/release_to_competitors/models/resnet_v1.onnx',
        '--output', 'testcases/resnet_dag_cupy.json'
    ], "Export ResNet DAG")
    
    transformer_result = run_command([
        python, 'cli/export_dag.py',
        '--onnx', 'testcases/release_to_competitors/models/transformer_v1.onnx',
        '--output', 'testcases/transformer_dag_cupy.json'
    ], "Export Transformer DAG")
    
    all_passed &= (mlp_result and resnet_result and transformer_result)
    
    print("\n" + "="*60)
    print("[TEST 2] C3.5 - CuPy GPU Inference")
    print("="*60)
    
    mlp_infer = run_command([
        python, 'cli/infer.py',
        '--onnx', 'testcases/release_to_competitors/models/mlp_v1.onnx',
        '--input', 'testcases/release_to_competitors/testdata/c35/mlp_v1/input',
        '--output', 'testcases/output_cupy/mlp_v1',
        '--batch-size', '256'
    ], "MLP Inference")
    
    resnet_infer = run_command([
        python, 'cli/infer.py',
        '--onnx', 'testcases/release_to_competitors/models/resnet_v1.onnx',
        '--input', 'testcases/release_to_competitors/testdata/c35/resnet_v1/input',
        '--output', 'testcases/output_cupy/resnet_v1',
        '--batch-size', '256'
    ], "ResNet Inference")
    
    transformer_infer = run_command([
        python, 'cli/infer.py',
        '--onnx', 'testcases/release_to_competitors/models/transformer_v1.onnx',
        '--input', 'testcases/release_to_competitors/testdata/c35/transformer_v1/input',
        '--output', 'testcases/output_cupy/transformer_v1',
        '--batch-size', '256'
    ], "Transformer Inference")
    
    all_passed &= (mlp_infer and resnet_infer and transformer_infer)
    
    print("\n" + "="*60)
    print("[TEST 3] Precision and Accuracy Validation")
    print("="*60)
    
    validate_result = run_command([
        python, '-c', '''
import numpy as np

print("=== MLP Validation ===")
out = np.load("testcases/output_cupy/mlp_v1/logits.npy")
gold = np.load("testcases/release_to_competitors/testdata/c35/mlp_v1/golden/logits.npy")
lab = np.load("testcases/release_to_competitors/testdata/c35/mlp_v1/labels.npy")
print(f"Shape: {out.shape}")
precision_pass = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
accuracy = (out.argmax(1) == lab).mean()
print(f"Precision: {precision_pass}")
print(f"Accuracy: {accuracy*100:.2f}%")

print("\\n=== ResNet Validation ===")
out = np.load("testcases/output_cupy/resnet_v1/logits.npy")
gold = np.load("testcases/release_to_competitors/testdata/c35/resnet_v1/golden/logits.npy")
lab = np.load("testcases/release_to_competitors/testdata/c35/resnet_v1/labels.npy")
print(f"Shape: {out.shape}")
precision_pass = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
accuracy = (out.argmax(1) == lab).mean()
print(f"Precision: {precision_pass}")
print(f"Accuracy: {accuracy*100:.2f}%")

print("\\n=== Transformer Validation ===")
out = np.load("testcases/output_cupy/transformer_v1/logits.npy")
gold = np.load("testcases/release_to_competitors/testdata/c35/transformer_v1/golden/logits.npy")
print(f"Shape: {out.shape}")
precision_pass = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
print(f"Precision: {precision_pass}")
'''
    ], "Validate Results")
    
    all_passed &= validate_result
    
    print("\n" + "="*60)
    if all_passed:
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed!")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
