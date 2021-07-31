import torch
import torch.nn as nn

loss_fun = nn.CrossEntropyLoss()

a = torch.tensor([2.0,3.0,4.0])
b = torch.softmax(a, dim=0)

a = torch.tensor([[1.0, 4.0, 3.0], [4.0, 1.0, 3.0]])
b = torch.softmax(a, dim=1)
# the [1, 4, 3] are calcualted together for softmax
print(b)

# the [1,4] are calcuatled together for softmax
b = torch.softmax(a, dim=0)
print(b)


pred = torch.tensor([2.0, 3.0, 4.0])
good_result = loss_fun(pred, torch.tensor([2]))
print(f"good_result={good_result}")
bad_result = loss_fun(pred, torch.tensor([1]))
print(f"bad_result={bad_result}")