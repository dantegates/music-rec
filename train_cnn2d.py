from functools import partial
import os

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import Lambda, Conv2D, UpSampling2D, AveragePooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import maxabs_scale


def binarize(x):
    return x + K.stop_gradient(K.round(x) - x)
Binarize = partial(Lambda, binarize, output_shape=lambda x: x)

model = Sequential([
    Conv2D(400, (64, 64), padding='same', activation='relu', input_shape=(512,184, 1)),
    AveragePooling2D((2, 2), padding='same'),
    Conv2D(300, (64, 32), padding='same', activation='relu'),
    AveragePooling2D((2, 2), padding='same'),
    Conv2D(200, (64, 32), padding='same', activation='relu'),
    AveragePooling2D((2, 2), padding='same'),
    Conv2D(150, (32, 23), padding='same', activation='relu'),
    AveragePooling2D((2, 1), padding='same'),
    Conv2D(128, (32, 23), padding='valid', activation='sigmoid'),
    Binarize(name='encoder'),
    UpSampling2D((8, 23)),
    Conv2D(256, (4, 4), padding='same', activation='relu'),
    UpSampling2D((4, 2)),
    Conv2D(256, (32, 32), padding='same', activation='relu'),
    UpSampling2D((2, 2)),
    Conv2D(256, (64, 32), padding='same', activation='relu'),
    UpSampling2D((2, 2)),
    Conv2D(300, (64, 32), padding='same', activation='relu', input_shape=(256, 184)),
    UpSampling2D((2, 1)),
    Conv2D(400, (64, 64), padding='same', activation='relu'),
    UpSampling2D((2, 1)),
    Conv2D(1, (1, 1), activation='sigmoid'),
])


def load(f):
    X = np.load(f).items()[0][-1]
    return maxabs_scale(X).transpose().reshape((512, 184, 1))


def data_gen(files, batch_size, steps=1):
    if steps == -1:
        condition = lambda: True
    else:
        i = 0
        condition = lambda: i <= steps
    while condition():
        np.random.shuffle(files)
        X = np.zeros((batch_size, 512, 184, 1))
        for i, f in enumerate(files):
            j = i % batch_size
            X[j,:] = load(f)
            if j == batch_size - 1:
                yield X, X
                X = np.zeros((batch_size, 512, 184, 1))

class DisplayProgress(Callback):
    def __init__(self, X, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = X
        self.plots = {}
        
    def on_epoch_end(self, epoch, logs):
        preds = model.predict(self.X)
        f, ax = plt.subplots(self.X.shape[0], 2, figsize=(20,20))
        plt.suptitle('epoch: %d' % epoch, fontsize=24)
        for i, (x, p) in enumerate(zip(self.X, preds)):
            ax[i][0].imshow(x.transpose(),  cmap='hot', interpolation='nearest')
            ax[i][1].imshow(p.transpose(),  cmap='hot', interpolation='nearest')
        plt.savefig('logs/epoch-%d.png' % epoch)
        self.plots[epoch] = (f, ax)
        plt.close()

        
if __name__ == '__main__':
    spec_dir = 'data/spec'
    spectrograms = [os.path.join(spec_dir, f) for f in os.listdir(spec_dir)]

    # train configs
    epochs = 120
    batch_size = 30
    steps = len(spectrograms) // batch_size
    gen = data_gen(spectrograms, batch_size, steps=-1)

    # callbacks
    sample = next(data_gen(spectrograms, 5, steps=5))[0]
    display_progress = DisplayProgress(sample)
    early_stop = EarlyStopping(patience=10, monitor='loss', verbose=1)
    model_checkpoint = ModelCheckpoint('autoencoder2.h5', period=3)    

    # train
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs,
                        callbacks=[early_stop, model_checkpoint])
