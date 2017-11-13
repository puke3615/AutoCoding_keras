import os


def choose(file):
    return file.endswith('.py')


def calcalute(path, func_choose=None):
    if not os.path.exists(path):
        raise Exception('File "%s" not found.' % path)
    elif os.path.isfile(path):
        return func_choose is None or func_choose(path)
    n_files = 0
    for f in os.listdir(path):
        file = os.path.join(path, f)
        n_files += calcalute(file, func_choose)
    return n_files


def generate_sequence(files, maxlen=10, step=3):
    chars = set()
    X, Y = [], []
    for file in files:
        with open(file) as f:
            text = f.read()
            len_text = len(text)
            if len_text - maxlen <= 0:
                continue
            # for i, c in enumerate(range(0, len_text - maxlen, step)):
            #     X.append(text[])



# path_data = '/Users/zijiao/tensorflow'
maxlen = 100
path_data = '/Users/zijiao/Documents/WorkSpace/PyCharm/AutoCoding_keras'
if __name__ == '__main__':
    n_files = calcalute(path_data, choose)
    print n_files
