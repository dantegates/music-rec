import itertools
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import config as cf


batch_size = 125
epochs = 25

files = [os.path.join(cf.TRAIN_DIR, f) for f in os.listdir(cf.TRAIN_DIR)]
train, test = train_test_split(files, test_size=0.2, random_state=0)
train_size = len(train)
test_size = len(test)
train_steps = int(train_size / batch_size)
val_steps = int(test_size / batch_size)


def batch_gen(filenames, n_features=cf.EXPECTED_SHAPE[0], batch_size=30):
    batch = np.zeros((batch_size, n_features))
    for i, f in enumerate(filenames, start=1):
        i = i % batch_size
        arr = np.load(f)
        batch[i-1, :] = arr
        if i % batch_size == 0:
            batch *= 2000
            yield batch, batch
            batch = np.zeros((batch_size, n_features))

def save_model(model):
    model.save(cf.MODEL_PATH)


train_gen = batch_gen(itertools.cycle(train), cf.EXPECTED_SHAPE[0], batch_size)
test_gen = batch_gen(itertools.cycle(test),  cf.EXPECTED_SHAPE[0], batch_size)


def get_model():
    model = Sequential([
        Dense(2**10, input_shape=cf.EXPECTED_SHAPE),
        Activation('relu'),
        Dense(2**9),
        Activation('relu'),
        Dense(2**8),
        Activation('relu'),
        Dense(2**7),
        Activation('relu'),
        Dense(2**6),
        Activation('relu'),
        Dense(2**5),
        Activation('sigmoid'),
        Dense(2**6),
        Activation('relu'),
        Dense(2**7),
        Activation('relu'),
        Dense(2**8),
        Activation('relu'),
        Dense(2**9),
        Activation('relu'),
        Dense(2**10),
        Activation('relu'),
        Dense(cf.EXPECTED_SHAPE[0]),
        Activation('relu')
    ])
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    return model

def main():
    save = False
    load_model = input('use loaded model? (y/n):') == 'y'
    if load_model:
        ae = keras.models.load_model(cf.MODEL_PATH)
        shutil.copy(cf.MODEL_PATH, cf.MODEL_PATH + '.bak')
    else:
        ae = get_model()
    try:
        ae.fit_generator(train_gen, train_steps, validation_data=test_gen,
                         validation_steps=val_steps, epochs=epochs)
    except KeyboardInterrupt:
        save = input('save model? (y/n):') == 'y'
    else:
        save = True
    if save:
        ae.save(cf.MODEL_PATH)


if __name__ == '__main__':
    main()
