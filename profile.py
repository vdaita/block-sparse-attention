import os
import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torch
from kernels import get_v1, get_v2, get_v3, get_v4, get_v5

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
torch.manual_seed(42)

implementations = [get_v1(), get_v2(), get_v3(), get_v4(), get_v5()] 
implementation_names = ["shared memory", "global memory", "tensor core", "transposed mixed memory", "split computation"]  

T = 16384
D = 64
B = 32
block_size = 16

num_query_blocks = (T + block_size - 1) // block_size
num_blocks_selected = 4  # Number of blocks selected per query block

q = torch.randn(B, T, D).cuda()
k = torch.randn(B, T, D).cuda()
v = torch.randn(B, T, D).cuda()
block_indices = torch.randint(0, num_query_blocks, (B, num_query_blocks, num_blocks_selected)).cuda().int()

print("Block indices shape: ", block_indices.shape)

for implementation_name, implementation_idx, implementation in zip(implementation_names, range(len(implementations)), implementations):
    print(f"Profiling implementation {implementation_name}")
    
    # Warmup
    for _ in range(10):  # Perform several warmup iterations
        _ = implementation.forward(q, k, v, block_indices)
    torch.cuda.synchronize()

    # Profiling multiple runs
    repeated_runs = 50
    total_time = 0.0

    for _ in range(repeated_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()  # Start timing
        _ = implementation.forward(q, k, v, block_indices)
        end_event.record()  # End timing
        
        # Synchronize and measure elapsed time
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        total_time += elapsed_time

    # Calculate the average time
    avg_time = total_time / repeated_runs
    print(f"    Average CUDA time over {repeated_runs} runs: {avg_time:.3f} ms")