import subprocess
import os
import sys


def check_cuda_path():
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path and os.path.exists(cuda_path):
        print("[OK] CUDA_PATH:", cuda_path)
        nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
        if os.path.exists(nvcc_path):
            result = subprocess.run([nvcc_path, '--version'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print("[OK] CUDA Version:", line.strip())
                    break
        return True
    else:
        print("[ERROR] CUDA_PATH not found or invalid")
        return False


def check_cudnn():
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        cudnn_dll = os.path.join(cuda_path, 'bin', 'cudnn64_*.dll')
        import glob
        files = glob.glob(cudnn_dll)
        if files:
            print("[OK] cuDNN found:", files[0])
            return True
        else:
            print("[WARNING] cuDNN not found in CUDA bin directory")
    return False


def check_nvidia_smi():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[:5]
            for line in lines:
                print("[OK]", line.strip())
            return True
        else:
            print("[ERROR] nvidia-smi failed:", result.stderr)
            return False
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found")
        return False


def main():
    print("=" * 50)
    print("C3-Scheduler GPU Environment Check")
    print("=" * 50)
    
    print("\n[1] Checking NVIDIA GPU...")
    gpu_ok = check_nvidia_smi()
    
    print("\n[2] Checking CUDA installation...")
    cuda_ok = check_cuda_path()
    
    print("\n[3] Checking cuDNN...")
    check_cudnn()
    
    print("\n[4] Checking Python version...")
    print("[OK] Python:", sys.version)
    
    print("\n" + "=" * 50)
    if gpu_ok and cuda_ok:
        print("[PASS] GPU environment is ready!")
    else:
        print("[FAIL] GPU environment has issues. Please fix before proceeding.")


if __name__ == '__main__':
    main()
