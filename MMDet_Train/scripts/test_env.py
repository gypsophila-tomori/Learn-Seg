# 创建一个test_cuda.py文件
import torch
import subprocess
import os

print("=== CUDA Diagnostics ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

print("\n=== Environment ===")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

print("\n=== Library check ===")
libs = ['libcuda.so', 'libcudnn.so.8', 'libcudnn_cnn_infer.so.8']
for lib in libs:
    result = subprocess.run(['find', '/usr', '-name', lib, '-type', 'f'], 
                          capture_output=True, text=True)
    if result.stdout:
        print(f"{lib}: Found")
        for path in result.stdout.strip().split('\n'):
            print(f"  {path}")
    else:
        print(f"{lib}: Not found")