import os
import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torch
from kernels import get_v1, get_v2, get_v3, get_v4, get_v5, get_v7
from flash_attn import flash_attn_func

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
torch.manual_seed(42)

# implementations = [get_v1(), get_v2(), get_v3(), get_v4(), get_v5(), get_v7(), flash_attn_func] 
# implementation_names = ["shared memory", "global memory", "tensor core", "transposed mixed memory", "split computation with atomic operations", "coalesced memory accesses", "sdpa"]  

implementations = [get_v1(), flash_attn_func]
implementation_names = ["shared memory", "sdpa"]

T = 8192
D = 64
B = 32
block_size = 32

num_query_blocks = (T + block_size - 1) // block_size
num_blocks_selected = 32  # Number of blocks selected per query block

q = torch.randn(B, T, D).cuda()
k = torch.randn(B, T, D).cuda()
v = torch.randn(B, T, D).cuda()
block_indices = torch.randint(0, num_query_blocks, (B, num_query_blocks, num_blocks_selected)).cuda().int()

q_fa = q.unsqueeze(0).to(torch.float16)
k_fa = k.unsqueeze(0).to(torch.float16)
v_fa = v.unsqueeze(0).to(torch.float16)

print("Block indices shape: ", block_indices.shape)

print("Profiling FlashAttention for reference")

for implementation_name, implementation_idx, implementation in zip(implementation_names, range(len(implementations)), implementations):
    print(f"Profiling implementation {implementation_name}")
    
    # Warmup
    for _ in range(10):  # Perform several warmup iterations
        if implementation_name != "sdpa":
            _ = implementation.forward(q, k, v, block_indices)
        else:
            _ = implementation(q_fa, k_fa, v_fa)
    torch.cuda.synchronize()

    # Profiling multiple runs
    repeated_runs = 50
    total_time = 0.0

    for _ in range(repeated_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()  # Start timing
        if implementation_name != "sdpa":
            _ = implementation.forward(q, k, v, block_indices)
        else:
            _ = implementation(q_fa, k_fa, v_fa)
        end_event.record()  # End timing
        
        # Synchronize and measure elapsed time
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        total_time += elapsed_time

    # Calculate the average time
    avg_time = total_time / repeated_runs
    print(f"    Average CUDA time over {repeated_runs} runs: {avg_time:.3f} ms")