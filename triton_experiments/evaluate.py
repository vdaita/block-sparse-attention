from chunked_bsa import triton_block_sparse_attention_chunked
from baseline_minference import triton_block_sparse_attention
import torch
import math
import triton

# create a rudimentary test and check that the results are all the same
NUM_CHUNKS = 1

B = 4
H = 4
N = 16384
D = 64
num_query_blocks = (N + 64 - 1) // 64
num_blocks_selected = 8

q = torch.randn(B, H, N, D, dtype=torch.float32)
k = torch.randn(B, H, N, D, dtype=torch.float32)
v = torch.randn(B, H, N, D, dtype=torch.float32)
seqlens = torch.tensor([N] * B, dtype=torch.int32)
block_indices = torch.randint(0, num_query_blocks, (B, num_query_blocks, num_blocks_selected)).cuda().int()
sm_scale = math.sqrt(1 / D)

o_chunked = triton_block_sparse_attention_chunked(q, k, v, seqlens, block_indices, sm_scale, num_chunks=NUM_CHUNKS)
o_baseline = triton_block_sparse_attention(q, k, v, seqlens, block_indices, sm_scale)   

if torch.allclose(o_chunked, o_baseline):
    print("All outputs are the same")
else:
    print("Outputs are different")
    exit(1)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[(2 ** 10)*i for i in range(1, 2**7, 2**2)],
        line_arg="implementation",
        line_vals=["baseline", "1_chunk", "2_chunk", "4_chunk", "8_chunk"],
        line_names=["baseline", "1 chunk", "2 chunks", "4 chunks", "8 chunks"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-"), ("purple", "-")],
        ylabel="runtime (ms)",
        plot_name="block-sparse-attention",
        args={}
    )
)
def benchmark(N, implementation):
    B, H, D = 4, 4, 64
    num_query_blocks = (N + 64 - 1) // 64
    num_blocks_selected = 8

    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    block_indices = torch.randint(0, num_query_blocks, (B, num_query_blocks, num_blocks_selected)).cuda().int()
    sm_scale = math.sqrt(1 / D)

    quantiles = [0.5, 0.2, 0.8]
    if(implementation == "baseline"):
        ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_block_sparse_attention(q, k, v, seqlens, block_indices, sm_scale)
            )
    else:
        num_chunks = int(implementation.split("_")[0])
        ms, min_ms, max_ms = triton.testin.do_bench(
            triton_block_sparse_attention_chunked(q, k, v, seqlens, block_indices, sm_scale, num_chunks=num_chunks)
        )
    
    return ms, min_ms, max_ms

benchmark.run(save_path=".", print_data=True)