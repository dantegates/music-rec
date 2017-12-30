import itertools
import os
import shutil

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import config as cf


batch_size = 125
epochs = 50

train_files = [os.path.join(cf.TRAIN_PROC_DIR, f) for f in os.listdir(cf.TRAIN_PROC_DIR)]
test_files = [os.path.join(cf.TEST_PROC_DIR, f) for f in os.listdir(cf.TEST_PROC_DIR)]
train_size = len(train_files)
test_size = len(test_files)
train_steps = int(train_size / batch_size)
val_steps = int(test_size / batch_size)
TIME_SAMPLES = 9
SIZE = cf.EXPECTED_SHAPE[0] * (TIME_SAMPLES + 1)


def batch_gen(filenames, n_features, batch_size=30):
    for i, f in enumerate(filenames, start=1):
        if i % batch_size == 1:
            batch = np.zeros((batch_size, n_features))
        arr = np.load(f)
        j = (i % batch_size) - 1
        batch[j, :] = arr
        if i % batch_size == 0:
             yield batch, batch


def save_model(model):
    model.save(cf.MODEL_PATH)


train_gen = batch_gen(itertools.cycle(train_files), SIZE, batch_size)
test_gen = batch_gen(itertools.cycle(test_files), SIZE, batch_size)


def get_model():
    model = Sequential([
        Dense(2**11, input_shape=(SIZE,)),
        Activation('relu'),
        Dense(2**10),
        Activation('relu'),
        Dense(2**9),
        Activation('relu'),
        Dense(2**8),
        Activation('relu'),
        Dense(2**7),
        Activation('relu'),
        Dense(2**6),
        Activation('sigmoid'),
        Dense(2**7),
        Activation('relu'),
        Dense(2**8),
        Activation('relu'),
        Dense(2**9),
        Activation('relu'),
        Dense(2**10),
        Activation('relu'),
        Dense(2**11),
        Activation('relu'),
        Dense(SIZE),
        Activation('relu')
    ])
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    return model

def main():
    save = False
    load_model = input('use loaded model? (y/n): ').lower() == 'y'
    if load_model:
        model = keras.models.load_model(cf.MODEL_PATH)
        shutil.copy(cf.MODEL_PATH, cf.MODEL_PATH + '.bak')
    else:
        model = get_model()
    print(model.summary())
    try:
        model.fit_generator(train_gen, train_steps,
                            validation_data=test_gen, validation_steps=val_steps,
                            epochs=epochs)
    except KeyboardInterrupt:
        save = input('save model? (y/n):' ).lower() == 'y'
    else:
        save = True
    if save:
        model.save(cf.MODEL_PATH)


if __name__ == '__main__':
    main()
