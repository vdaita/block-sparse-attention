import torch
import os
torch.manual_seed(42)

B = 1
D = 64
T = 128

q = torch.randn((B, 1, D))
k = torch.randn((B, T, D))
# v = torch.randn((B, T, D))
v = torch.cat(
    (torch.randn((B, T // 2, D)),
    torch.zeros((B, T // 2, D))),
    dim=1
)

print("Queries: ", q)
print("Keys: ", k)
print("Values: ", v)

with open("test.in", "w+") as f:
    for x in q.flatten():
        f.write(str(float(x)) + "\n")

    for x in k.flatten():
        f.write(str(float(x)) + "\n")
        
    for x in v.flatten():
        f.write(str(float(x)) + "\n")

    # Compute attention across the batch
    attention = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, seq_len, seq_len)

    # Flatten the attention matrix for each batch element
    flat_attention = attention.flatten(start_dim=1)  # Flatten per sequence for each batch
    print(flat_attention.shape)

    # Print attention weights per token for each batch (optional)
    for batch_idx in range(B):
        print(f"Batch {batch_idx}:")
        for i in range(T):
            print(f"  Token: {i}, weight: {flat_attention[batch_idx, i]}")

    # Apply softmax to the attention across the last dimension (sequence length)
    attention = torch.nn.functional.softmax(attention, dim=-1)  # (batch_size, seq_len, seq_len)

    # Compute the output by applying attention to the value vectors
    output = torch.matmul(attention, v)  # (batch_size, seq_len, dim)

    for x in output.flatten():
        f.write(str(float(x)) + "\n")