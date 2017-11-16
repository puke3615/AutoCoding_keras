import numpy as np
import hashlib

def get_md5(content):
    md5 = hashlib.md5()
    content = content.encode('utf-8')
    md5.update(content)
    return md5.hexdigest()


def md5_generator(length=10, depth=10, batch_size=32):
    while True:
        xs, ys = [], []
        for _ in range(batch_size):
            random_data = [str(c) for c in np.random.randint(0, depth, length).tolist()]
            x = ''.join(random_data)
            y = ''.join(get_md5(''.join(random_data)))
            xs.append(x)
            ys.append(y)
        yield xs, ys


def vectorize_stories(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, vocab_size):
    x_set = []
    X = np.zeros((len(input_list), input_maxlen, vocab_size))
    Y = np.zeros((len(tar_list), tar_maxlen, vocab_size))
    for _sent in input_list:
        x = [word_idx[w] for w in _sent]
        x_set.append(x)
    for s_index, input_tmp in enumerate(input_list):
        for t_index, token in enumerate(input_tmp):
            X[s_index, t_index, word_idx[token]] = 1
    for s_index, tar_tmp in enumerate(tar_list):
        for t_index, token in enumerate(tar_tmp):
            Y[s_index, t_index, word_idx[token]] = 1

    return X, Y


def data_generator(in_steps=10, in_depth=10, batch_size=32):
    m_generator = md5_generator(length=in_steps, depth=in_depth, batch_size=batch_size)
    while True:
        X, Y = next(m_generator)
        size = len(X)
        xs = np.zeros([size, in_steps, in_depth])
        ys = np.zeros([size, out_size, out_depth])
        for i, (x, y) in enumerate(zip(X, Y)):
            try:
                for j, x_j in enumerate(x):
                    xs[i, j, char2id[x_j]] = 1
                for q, y_q in enumerate(y):
                    ys[i, q, char2id[y_q]] = 1
            except Exception as e:
                print(e)
        yield xs, ys


out_size = 32
out_depth = 16
chars = u'0123456789abcdef'
char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))

if __name__ == '__main__':
    in_steps = 1
    in_depth = in_steps
    batch_size = 1

    m_generator = md5_generator(length=in_steps, depth=in_depth, batch_size=batch_size)
    generator = data_generator(in_steps, in_depth, batch_size)
    for i, (x, y) in enumerate(m_generator):
        print(x, ' => ', y)
        if i > 1:
            break
    # for i, (x, y) in enumerate(generator):
    #     print(x, ' => ', y)
    #     if i > 1:
    #         break
