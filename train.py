from functools import partial
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import config as cf


batch_size = 125
epochs = 100

files = list(glob.glob('/home/dante_gates/repos/music-rec/data/train/*.npy'))
train, test = train_test_split(files, test_size=0.2)
train_size = len(train)
test_size = len(test)
train_steps = int(train_size / batch_size)
val_steps = int(test_size / batch_size)


def batch_gen(filenames, n_features=cf.EXPECTED_SHAPE[0], batch_size=30):
    n = 0
    batch = np.zeros((batch_size, n_features))
    for i, f in enumerate(filenames, start=1):
        i = i % 30
        arr = np.load(f)
        batch[i-1] = arr
        if i % batch_size == 0:
            yield batch, batch
            batch = np.zeros((batch_size, n_features))

def repeat_generator(g):
    while 1:
        for item in g():
            yield item


train_gen = repeat_generator(partial(batch_gen, train))
test_gen = repeat_generator(partial(batch_gen, test))

def save_model(model):
    model.save('model.h5')


ae = Sequential([
        Dense(2**10, input_shape=cf.EXPECTED_SHAPE),
        Activation('sigmoid'),
        Dense(2**9),
        Activation('sigmoid'),
        Dense(2**8),
        Activation('sigmoid'),
        Dense(2**7),
        Activation('sigmoid'),
        Dense(2**6),
        Activation('sigmoid'),
        Dense(2**7),
        Activation('sigmoid'),
        Dense(2**8),
        Activation('sigmoid'),
        Dense(2**9),
        Activation('sigmoid'),
        Dense(2**10),
        Activation('sigmoid'),
        Dense(cf.EXPECTED_SHAPE[0]),
        Activation('sigmoid')
    ])
ae.compile(optimizer='adadelta', loss='mean_squared_error')

def main():
    save = False
    try:
        ae.fit_generator(train_gen, train_steps, validation_data=test_gen,
                         validation_steps=val_steps, epochs=epochs)
    except KeyboardInterrupt:
        save = input('save model? (y/n):') == 'y'
    else:
        save = True
    if save:
        ae.save('model.h5')


if __name__ == '__main__':
    main()
