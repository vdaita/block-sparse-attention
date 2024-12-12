import torch
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
torch.manual_seed(42)

def get_file_with_settings(name, source_file, cpp_file="extension/bsa_cpp_def.cpp"):
    return load(
        name=name,
        sources=[
            source_file,
            cpp_file
        ],
        with_cuda=True,
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-O2', '-std=c++17'],
        extra_ldflags=['-lc10', '-ltorch', '-ltorch_cuda'],
        build_directory="./build",
        verbose=True,
    )

# Validate stage 1 decoding
B = 32
T = 32
D = 128

q = torch.randn((B, 1, D))
k = torch.randn((B, T, D))
v = torch.randn((B, T, D))

def baseline_calc(q, k, v):
    attention = torch.matmul(q, k.transpose(-2, -1))
    attention = torch.nn.functional.softmax(attention, dim=-1)
    output = torch.matmul(attention, v)
    return output

implementation = get_file_with_settings("flash_decoding_stage_1", "flash_decoding.cu", cpp_file="flash_decoding_def.cpp")

baseline_result = baseline_calc(q, k, v)
kernel_result = implementation.forward(q, k, v)

print(torch.allclose(kernel_result, baseline_result))

print(baseline_result)
print("===========")
print(kernel_result)