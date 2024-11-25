from torch.utils.cpp_extension import load
import torch

block_sparse_attention = load(name='block_sparse_attention', sources=['extension/block_sparse_attention.cpp', 'extension/block_sparse_attention_fwd.cu'])

def baseline_block_sparse_attention(q, k, v, block_indices, block_size):
    B, H, T, D = q.shape
    O = torch.zeros_like(v)

    for b in range(B):
        for h in range(H):
            bh_output = []
            for query_block_index in range((T + block_size - 1) // block_size):
                query_block = q[b, h, query_block_index * block_size : (query_block_index + 1) * block_size, ...]
                key_blocks = []
                value_blocks = []
                for block_indices in block_indices[b][h][query_block_index]:
                    key_block = k[b, h, block_indices * block_size : (block_indices + 1) * block_size, ...]
                    key_blocks.append(key_block)

                    value_block = v[b, h, block_indices * block_size : (block_indices + 1) * block_size, ...]
                    value_blocks.append(value_block)
                key_block = torch.cat(key_blocks, dim=0)
                value_block = torch.cat(value_blocks, dim=0)

                attention = torch.matmul(query_block, key_block.transpose(-2, -1))
                attention = attention / (D ** 0.5)
                attention = torch.nn.functional.softmax(attention, dim=-1)
                output = torch.matmul(attention, value_block)
                bh_output.append(output)
            bh_output = torch.cat(bh_output, dim=0)
            O[b, h, ...] = bh_output

    return O

def test_sample_matrices():
    B = 2
    H = 4
    T = 100
    D = 128
    block_size = 16

    q = torch.randn(B, H, T, D).cuda()
    k = torch.randn(B, H, T, D).cuda()
    v = torch.randn(B, H, T, D).cuda()
    block_indices = torch.randint(0, T // block_size, (B, H, T // block_size, 4)).cuda()

    O = block_sparse_attention.block_sparse_attention(q, k, v, block_indices, block_size)
    O_baseline = baseline_block_sparse_attention(q, k, v, block_indices, block_size)

    print(torch.allclose(O, O_baseline))