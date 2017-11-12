# coding=utf-8
import os
import keras
import numpy as np
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.models import Sequential

PATH_DATA = 'data.txt'
PATH_PARAMS = 'params/weights.h5'
START, END = u'å', u'ß'
N_SEQUENCE = 5
STEP = 1
N_HIDDEN = 32
LEARNING_RATE = 1e-2
BATCH_SIZE = 10
EPOCH = 100
TRAIN = True


def predict(func_predict, inputs):
    prediction = func_predict(np.divide(inputs, float(len(words))))[0]
    prob = np.max(prediction)
    output = np.argmax(prediction)
    inputs_str = ''
    for i in inputs[0]:
        inputs_str += words[i[0]].encode('utf-8')
    print('%s --> %s  %f' % (inputs_str, words[output].encode('utf-8'), prob))
    if output != word2id[END]:
        inputs[0].__delitem__(0)
        inputs[0].append([output])
        predict(func_predict, inputs)


with open(PATH_DATA) as f:
    data = unicode(f.read(), 'utf-8')
data = START * N_SEQUENCE + data + END
words = list(set(data))
word2id = {w: i for i, w in enumerate(words)}

X, y = [], []
for i in range(0, len(data) - N_SEQUENCE, STEP):
    inputs = data[i: i + N_SEQUENCE]
    outputs = data[i + N_SEQUENCE]
    print('%s --> %s' % (inputs, outputs))
    X.append(list(map(word2id.get, inputs)))
    y.append(word2id[outputs])

print('Train data size is %d' % len(X))
X = np.array(X).astype(np.float32)
X = np.expand_dims(np.divide(X, len(words)), -1)
y = np.array(y).astype(np.float32)
y = keras.utils.to_categorical(y, len(words))

model = Sequential()
model.add(LSTM(N_HIDDEN, input_shape=(N_SEQUENCE, 1)))
model.add(Dense(len(words), activation='softmax'))

if os.path.isfile(PATH_PARAMS):
    model.load_weights(PATH_PARAMS)
    print('Load params successfully.')
dir_name = os.path.dirname(PATH_PARAMS)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

optimizer = Adam(LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

if TRAIN:
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2, callbacks=[ModelCheckpoint(PATH_PARAMS)])

inputs = START * N_SEQUENCE
inputs = [[[word2id[w]] for w in inputs]]
predict(lambda x: model.predict(x), inputs)
