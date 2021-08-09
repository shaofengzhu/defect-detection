def getResizeFunc(ratio):
    func = lambda x : ratio * x
    return func


f = getResizeFunc(2)
result = f(10)
print(result)
