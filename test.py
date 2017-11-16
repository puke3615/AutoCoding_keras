import numpy as np
import hashlib


def generator(length=10, depth=10):
    md5 = hashlib.md5()
    while True:
        random_data = [str(c) for c in np.random.randint(0, depth, length).tolist()]
        x = ' '.join(random_data)
        md5.update(''.join(random_data))
        y = ' '.join(md5.hexdigest())
        yield x, y


g = generator(length=3)
for i, (x, y) in enumerate(g):
    print x, ' => ', y
    if i > 100:
        break
