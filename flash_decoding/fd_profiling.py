import os
import torch
from torch.utils.cpp_extension import load
from flash_attn import flash_attn_func

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

def baseline(q, k, v):
    attention = torch.matmul(q, k.transpose(-2, -1))
    attention = torch.nn.functional.softmax(attention, dim=-1)
    output = torch.matmul(attention, v)

@torch.compile
def baseline_compiled(q, k, v):
    attention = torch.matmul(q, k.transpose(-2, -1))
    attention = torch.nn.functional.softmax(attention, dim=-1)
    output = torch.matmul(attention, v)

custom_flash_decoding = load(
    name="custom_flash_decoding",
    sources=[
        "flash_decoding_def.cpp",
        "flash_decoding_torch.cu",
    ],
    with_cuda=True,
    extra_cflags=['-std=c++17'],
    extra_cuda_cflags=['-O2', '-std=c++17', '-Xptxas=-maxrregcount=32'],
    extra_ldflags=['-lc10', '-ltorch', '-ltorch_cuda'],
    build_directory="./build",
    verbose=True,
)

B = 12
D = 128
T = 8192

q = torch.randn((B, 1, D)).cuda()
k = torch.randn((B, T, D)).cuda()
v = torch.randn((B, T, D)).cuda()

q_fa = q.unsqueeze(0).transpose(1, 2).to(torch.float16).contiguous()
k_fa = k.unsqueeze(0).transpose(1,2).to(torch.float16).contiguous()
v_fa = v.unsqueeze(0).transpose(1,2).to(torch.float16).contiguous()

implementation_names = ["sdpa", "baseline", "baseline_compiled", "custom"]
implementations = [flash_attn_func, baseline, baseline_compiled, custom_flash_decoding.forward]

print(f"Config: B={B} D={D} T={T}")
for implementation_name, implementation in zip(implementation_names, implementations):
    print(f"Profiling implementation {implementation_name}")
    # Warmup
    for _ in range(10):  # Perform several warmup iterations
        if implementation_name == "sdpa":
            _ = implementation(q_fa, k_fa, v_fa)
        else:
            _ = implementation(q, k, v)
    torch.cuda.synchronize()

    # Profiling multiple runs
    repeated_runs = 50
    total_time = 0.0

    for _ in range(repeated_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()  # Start timing
        if implementation_name == "sdpa":
            _ = implementation(q_fa, k_fa, v_fa)
        else:
            _ = implementation(q, k, v)
        end_event.record()  # End timing
        
        # Synchronize and measure elapsed time
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        total_time += elapsed_time

    # Calculate the average time
    avg_time = total_time / repeated_runs
    print(f"    Average CUDA time over {repeated_runs} runs: {avg_time:.3f} ms")