import subprocess
import sys


def get_cuda_version():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if 'CUDA Version' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'Version':
                        return parts[i+1]
    except:
        pass
    return None


def install_requirements(cuda_version):
    print(f"Detected CUDA version: {cuda_version}")
    
    if cuda_version is None:
        print("[ERROR] Cannot detect CUDA version. Please install manually.")
        return False
    
    major = int(cuda_version.split('.')[0])
    minor = int(cuda_version.split('.')[1])
    
    cupy_package = "cupy-cuda12x>=13.0.0"
    
    if major == 12:
        if minor >= 8:
            cupy_package = "cupy-cuda13x>=13.0.0"
        elif minor >= 4:
            cupy_package = "cupy-cuda12x>=13.0.0"
        else:
            cupy_package = "cupy-cuda121>=13.0.0"
    elif major == 11:
        cupy_package = "cupy-cuda11x>=12.0.0"
    else:
        print("[ERROR] Unsupported CUDA version:", cuda_version)
        return False
    
    print(f"\n[1] Installing core dependencies with {cupy_package}...")
    cmd1 = [sys.executable, '-m', 'pip', 'install', 
            'numpy>=1.24.0', 'onnx>=1.15.0', 
            'scipy>=1.10.0', 'tabulate>=0.9.0',
            cupy_package, '-q']
    result = subprocess.run(cmd1)
    if result.returncode != 0:
        print("[ERROR] Failed to install core dependencies")
        return False
    
    return True


def verify_installation():
    print("\n[2] Verifying installation...")
    
    try:
        import cupy as cp
        print(f"[OK] CuPy: {cp.__version__}")
        
        if cp.cuda.is_available():
            print(f"[OK] CUDA available: {cp.cuda.runtime.getDeviceName(0)}")
            print(f"[OK] CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        else:
            print("[ERROR] CUDA not available in CuPy")
            return False
        
        print("\n[PASS] All dependencies installed and verified!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False


def main():
    print("=" * 60)
    print("C3-Scheduler CuPy GPU Environment Setup")
    print("=" * 60)
    
    cuda_version = get_cuda_version()
    
    if install_requirements(cuda_version):
        verify_installation()
    else:
        print("\n[FAIL] Installation failed.")


if __name__ == '__main__':
    main()
