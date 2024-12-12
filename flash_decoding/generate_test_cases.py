import torch
import os
torch.manual_seed(42)

B = 1
D = 32
T = 64

q = torch.randn((B, 1, D))
k = torch.randn((B, T, D))
v_1 = torch.randn((B, T // 2, D))
v_2 = torch.zeros((B, T // 2, D))
v = torch.cat((v_1, v_2), dim=1)

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

    attention = torch.matmul(q, k.transpose(-2, -1))
    
    flat_attention = attention.flatten()
    for i in range(T):
        print("Token: ", i, " weight: ", str(float(flat_attention[i])))

    attention = torch.nn.functional.softmax(attention, dim=-1)
    output = torch.matmul(attention, v)

    for x in output.flatten():
        f.write(str(float(x)) + "\n")
