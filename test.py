import numpy as np

a = np.zeros([3, 4])

a = np.expand_dims(a, -1)

print a.shape
print np.divide([1, 2], 1.)