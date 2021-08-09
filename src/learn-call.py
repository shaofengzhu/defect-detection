import numpy as np

class Foo:
    def __call__(self, x):
        return x * 2

f = Foo()
result = f(3)
print(result)
