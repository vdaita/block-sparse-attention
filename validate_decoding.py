import torch

# Validate stage 1 decoding
B = 32
T = 32
D = 128

q = torch.randn((B, 1, D))