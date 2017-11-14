import numpy as np

print np.random.multinomial(1000, [1. / 6] * 6, 1)

a = np.array([1, 2])
b = np.log(a) / 1.2
print b
print np.exp(b)

a = [1, 2]
b = [2, 3]
a.extend(b)
print a
print b[:-1]


