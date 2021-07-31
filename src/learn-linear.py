import torch
import torch.nn as nn

l1 = nn.Linear(3, 2)

input = torch.tensor([[1.0,2.0,3.0], [1.0,2.0,3.0]])
output = l1(input)
print(output)

input = torch.tensor([1.0,2.0,3.0])
output = l1(input)
print(output)
