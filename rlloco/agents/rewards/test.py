import torch

x = torch.rand(20).unsqueeze(1)
y = torch.rand(20, 3)
print(x.shape)

# dot product of x and y across dim 1
# xy = torch.sum(x * y, dim=1)
# print(xy.shape)
# print((x > y) + (x < y))
# print((x > y).shape)

print((x*y).shape)