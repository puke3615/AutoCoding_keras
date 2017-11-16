import numpy as np
import hashlib

def get_md5(content):
    md5 = hashlib.md5()
    content = content.encode('utf-8')
    md5.update(content)
    return md5.hexdigest()

def generator(length=10, depth=10):
    while True:
        random_data = [str(c) for c in np.random.randint(0, depth, length).tolist()]
        x = ' '.join(random_data)
        y = ' '.join(get_md5(''.join(random_data)))
        yield x, y



in_size = 1
g = generator(length=in_size, depth=in_size)
for i, (x, y) in enumerate(g):
    print(x, ' => ', y)
    if i > 1:
        break

print(get_md5('0'))
print(get_md5('0'))
print(get_md5('0'))
