import torch

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y = x*x + 3 x
# the derivative should be (2 x + 3)
b = a * a + 3 * a
loss = b.sum()
loss.backward()

# now could print the value
print(a)
# and the derivative
print(a.grad)

a = torch.tensor([[[1.0, 2.0], [0.0, 4], [0.0, 3]]])
b = torch.tensor([[[0.0, 2.0], [0.0, 4], [2., 3]]])
intersection = a.logical_and(b)
print(intersection)
print(intersection.sum())
union = a.logical_or(b)
print(union)
print(union.sum())
accuracy = intersection.sum().item() / union.sum().item()
print(accuracy)