import torch
import torch.nn.functional as F

# Reshaped appearance: (bsz, length, num_heads, head_dim)
def get_top_blocks_regular(query_states, key_states, block_size):
    B, H, L, D = query_states.shape
    query_reshaped = query_states.reshape(-1, L, D)
    key_reshaped = key_states.reshape(-1, L, D)
    attention = torch.einsum('bld,bld->bll', query_reshaped, key_reshaped)
    attention = attention.softmax(dim=-1)
    attention = attention.reshape(B * H, L, L // block_size, block_size) # how much of the attention weight is in each of these blocks?
    attention = attention.sum(dim=-1)
    return attention

def compare_divergence(top_blocks_regular, top_blocks_generated, block_size):
    B, Tadj, D = top_blocks_generated.shape
    B, T, D = top_blocks_regular.shape
    g_reshaped = top_blocks_generated.unsqueeze(2).repeat(1, 1, block_size, 1).view(B, Tadj * block_size, D)
    # calculate the KL divergence between the two distributions
    kl_div = F.kl_div(F.log(top_blocks_regular + 1e-8), g_reshaped, reduction='mean')
    return kl_div.item()

def avg_softmax_pooling(query_states, key_states, block_size):
    B, H, L, D = query_states.shape
    query_reshaped = query_states.reshape(-1, L, D)
    key_reshaped = key_states.reshape(-1, L, D)
    query_reshaped = query_states.reshape(B * H, L // block_size, block_size, D)
    key_reshaped = key_states.reshape(B * H, L // block_size, block_size, D)
    query_reshaped = query_reshaped.mean(dim=-2)
    key_reshaped = key_reshaped.mean(dim=-2)
    attn_weights = torch.einsum("bld,bld->bll", query_reshaped, key_reshaped)
    # Create and add causal mask
    mask = torch.triu(torch.ones(L // block_size, L // block_size), diagonal=1).to(attn_weights.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    attn_weights = attn_weights.masked_fill(mask == 1, float('-inf'))
    attn_weights = attn_weights.softmax(dim=-1)
    return attn_weights

def softmax_avg_pooling(query_states, key_states, top_k, block_size):
    ...

def max_softmax_pooling(query_states, key_states, top_k, block_size):
    ...

def softmax_max_pooling(query_states, key_states, top_k, block_size):
    ...

def avg_pooling(query_states, key_states, top_k, block_size):
    ...

def max_pooling(query_states, key_states, top_k, block_size):
    ...

pooling_methods = {
    "avg_softmax_pooling": avg_softmax_pooling,
}