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


