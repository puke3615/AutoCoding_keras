import os
import numpy as np
import data_reader
from keras.engine import *
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *

PATH_PARAMS = 'params/client2.h5'
MAX_LEN = 50
STEP = 3
BATCH_SIZE = 512
LAYERS = 3
N_HIDDEN = 128
LR = 1e-2
EPOCH = 200

xs, ys, char2id, id2char = data_reader.parse_data(n_files=1000, max_len=MAX_LEN, step=STEP)
depth = len(id2char) + 1

model = Sequential()
model.add(LSTM(N_HIDDEN, input_shape=(MAX_LEN, depth), return_sequences=True))
for _ in range(LAYERS - 2):
    model.add(LSTM(N_HIDDEN, return_sequences=True))
if LAYERS > 2:
    model.add(LSTM(N_HIDDEN))
else:
    model.add(Flatten())
model.add(Dense(depth, activation='softmax'))

optimizer = Adam(LR)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
if os.path.exists(PATH_PARAMS):
    model.load_weights(PATH_PARAMS)
    print('Load params successfully.')

model.fit(xs, ys, BATCH_SIZE, EPOCH, callbacks=[ModelCheckpoint(PATH_PARAMS)], validation_split=0.2)
