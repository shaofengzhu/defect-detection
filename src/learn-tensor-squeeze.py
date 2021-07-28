import torch

# one dimensional array
a = torch.tensor([1,2,3])
print(a)
print(a.shape)

# convert to two dimensional array
b = torch.unsqueeze(a, dim=0)
print(b)
print(b.shape)

# convert to three dimensional array
c = torch.unsqueeze(b, dim=0)
print(c)
print(c.shape)

# convert back to two dimensional array
d = c.squeeze(dim = 0)
print(d)
print(d.shape)

# the squeeze could cause data loss
e = torch.tensor([[10,20,30],[40,50,60]])
f = c.squeeze(dim = 0)
print(f)
print(f.shape)
