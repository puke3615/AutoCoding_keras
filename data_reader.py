import os
import time
import utils
import numpy as np


def choose(file):
    return file.endswith('.py')


def find_files(path, func_choose=None):
    if not os.path.exists(path):
        raise Exception('File "%s" not found.' % path)
    elif os.path.isfile(path):
        return [path] if func_choose is None or func_choose(path) else None
    targets = []
    for f in os.listdir(path):
        file = os.path.join(path, f)
        sub_targets = find_files(file, func_choose)
        if sub_targets:
            targets.extend(sub_targets)
    return targets


def generate_sequence(files, maxlen=10, step=3):
    chars = set()
    X, Y = [], []
    for file in files:
        with open(file) as f:
            text = f.read()
            try:
                text = unicode(text, encoding='utf-8')
            except Exception:
                print('Read file "%s" failure.' % file)
                continue
            len_text = len(text)
            if len_text - maxlen <= 0:
                continue
            chars = chars.union(text)
            for i, c in enumerate(range(0, len_text - maxlen, step)):
                X.append(text[i: i + maxlen])
                Y.append(text[i + maxlen])
    chars = list(sorted(chars))
    return X, Y, chars


def sequence2ids(X, Y, char2id):
    n_samples = sum([len(data_per_file) for data_per_file in X])
    n_steps = len(X[0])
    depth = len(char2id) + 1
    xs = np.zeros([n_samples, n_steps, depth])
    ys = np.zeros([n_samples, depth])
    for i, (x, y) in enumerate(zip(X, Y)):
        ys[i, char2id[y]] = 1
        for j, x_j in enumerate(x):
            xs[i, j, char2id[x_j]] = 1
    return xs, ys


# path_data = '/Users/zijiao/tensorflow'
PATH_DATA = '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy'
MAX_LEN = 100
STEP = 3


def parse_data(path=PATH_DATA, n_files=None, max_len=MAX_LEN, step=STEP, log=True):
    start = time.time()
    target_files = find_files(path, choose)
    if log:
        print '%d files found.' % len(target_files)

    if n_files:
        target_files = target_files[:n_files]
    X, Y, chars = generate_sequence(target_files, max_len, step)
    char2id = {c: i + 1 for i, c in enumerate(chars)}
    id2char = {v: k for k, v in char2id.items()}
    if log:
        print('%d chars found.' % len(chars))
        print(utils.char_str(chars))
        print('Preprocess sequence data...')
    xs, ys = sequence2ids(X, Y, char2id)
    if log:
        print('X.shape = (%d, %d, %d)' % xs.shape)
        print('Y.shape = (%d, %d)' % ys.shape)
        print('Take %.2f s with processing data.' % (time.time() - start))

    return xs, ys, char2id, id2char


if __name__ == '__main__':
    xs, ys, char2id, id2char = parse_data(n_files=10)
    # print(xs[0, :1, :])
