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

def compare(top_blocks_regular, top_blocks_generated, block_size):
    g_reshaped = top_blocks_generated...


def avg_softmax_pooling(query_states, key_states, block_size):
    B, H, L, D = query_states.shape
    query_reshaped = query_states.reshape(-1, L, D)
    key_reshaped = key_states.reshape(-1, L, D)
    query_reshaped = query_states.reshape(B * H, L // block_size, block_size, D)
    key_reshaped = key_states.reshape(B * H, L // block_size, block_size, D)
    query_reshaped = query_reshaped.mean(dim=-2)
    key_reshaped = key_reshaped.mean(dim=-2)
    attention = torch.einsum("bld,bld->bll", query_reshaped, key_reshaped)
    attention = attention.softmax(dim=-1)
    return attention

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