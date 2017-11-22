# coding=utf-8
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import *
from keras.callbacks import *
from keras.metrics import *
from keras.layers import *
from callback import *
import numpy as np
import md5_data
import re
import os


def build_model(input_maxlen, in_depth, out_size, out_depth, hidden_size):
    model = Sequential()

    # Encoder(第一个 LSTM)
    # model.add(LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(LSTM(hidden_size, input_shape=(input_maxlen, in_depth)))
    model.add(BatchNormalization())

    model.add(Dense(hidden_size, activation="relu"))

    # 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
    model.add(RepeatVector(out_size))
    model.add(BatchNormalization())

    # Decoder(第二个 LSTM)
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(BatchNormalization())

    # TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
    model.add(TimeDistributed(Dense(out_depth, activation="softmax")))

    return model

def time_print(y_true, y_pred):
    return time.time()

if __name__ == '__main__':
    n_hidden = 256
    path_weights = './params/weights.h5'
    in_steps = 2
    in_depth = in_steps
    batch_size = 32
    lr = 1e-4
    epochs = 1000
    steps_per_epoch = 200

    generator = md5_data.data_generator(in_steps, in_depth, batch_size)

    model = build_model(in_steps, in_depth, md5_data.out_size, md5_data.out_depth, n_hidden)
    model.compile(Nadam(lr), 'categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(path_weights):
        model.load_weights(path_weights)
        print('Load weights successfully.')
    dir_weights = os.path.dirname(path_weights)
    if not os.path.exists(dir_weights):
        os.makedirs(dir_weights)
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=0,
                        callbacks=[
                            ModelCheckpoint(path_weights),
                            TensorBoard(),
                            ProgbarLogger(count_mode='steps')
                        ])
    model.save_weights(path_weights)
