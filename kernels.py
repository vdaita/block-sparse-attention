import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

def get_file_with_settings(name, source_file):
    return load(
        name=name,
        sources=[
            source_file,
            "extension/bsa_cpp_def.cpp"
        ],
        with_cuda=True,
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-O2', '-std=c++17'],
        extra_ldflags=['-lc10', '-ltorch', '-ltorch_cuda'],
        build_directory="./extension/build",
        verbose=True,
    )

def get_v1():
    return get_file_with_settings("block_sparse_attention_v1", "extension/bsa_shared_memory.cu")

def get_v2():
    return get_file_with_settings("block_sparse_attention_v2", "extension/bsa_global_memory.cu")

def get_v3():
    return get_file_with_settings("block_sparse_attention_v3", "extension/bsa_tensor_core.cu")

def get_v4():
    return get_file_with_settings("block_sparse_attention_v4", "extension/bsa_transposed_mixed_memory.cu")

def get_v5():
    return get_file_with_settings("block_sparse_attention_v5", "extension/bsa_split_computation_atomic.cu")