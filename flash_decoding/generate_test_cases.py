import torch
import os
torch.manual_seed(42)

B = 1
D = 64
T = 128


with open("test.in", "w+") as f:
    outputs = []
    for b in range(B):
        q = torch.randn((1, D))
        k = torch.randn((T, D))
        v = torch.randn((T, D))

        print("Queries: ", q.shape)
        print("Keys: ", k.shape)
        print("Values: ", v.shape)

        for x in q.flatten():
            f.write(str(float(x)) + "\n")

        for x in k.flatten():
            f.write(str(float(x)) + "\n")
            
        for x in v.flatten():
            f.write(str(float(x)) + "\n")

        attention = torch.matmul(q, k.transpose(-2, -1))
        
        flat_attention = attention.flatten()
        # for i in range(T):
        #     print("Token: ", i, " weight: ", str(float(flat_attention[i])))

        attention = torch.nn.functional.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)
        for x in output.flatten():
            outputs.append(str(float(x)))

    for output in outputs:
        print(output)
