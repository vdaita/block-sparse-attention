import os
import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from kernels import get_file_with_settings
from torch.profiler import profile, ProfilerActivity, schedule

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
torch.manual_seed(42)

implementation = get_file_with_settings("bsa_tensor_core", "extension/bsa_tensor_core.cu")

T = 1024
D = 64
B = 8
block_size = 32

# NOTE: need to make sure that configuration settings line up

num_query_blocks = (T + block_size - 1) // block_size
num_blocks_selected = 4  # Number of blocks selected per query block

q = torch.randn(B, T, D).cuda()
k = torch.randn(B, T, D).cuda()
v = torch.randn(B, T, D).cuda()
block_indices = torch.randint(0, num_query_blocks, (B, num_query_blocks, num_blocks_selected)).cuda().int()

with profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        try:
            minimal_result = implementation.forward(q, k, v, block_indices)
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        except RuntimeError as e:
            print(f"Kernel runtime error {e}")

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
