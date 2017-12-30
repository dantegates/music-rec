import os

import numpy as np

from train import load, SIZE, TIME_SAMPLES
import config as cf

TRAIN_FILES = [os.path.join(cf.TRAIN_DIR, f) for f in os.listdir(cf.TRAIN_DIR)]
TEST_FILES = [os.path.join(cf.TEST_DIR, f) for f in os.listdir(cf.TEST_DIR)]
TRAIN_PROC_DIR = 'data/train_proc'
TEST_PROC_DIR = 'data/test_proc'


def min_max(X, upper_lim):
    mx, mn = X.max(), X.min()
    if mx == mn:
        return X
    return (X -  mn) / ((mx - mn) / upper_lim)


def load(f, time_samples):
    X = np.load(f)
    y = X.shape[1]
    x =  np.add.reduceat(X, list(range(0, y, y // time_samples)), axis=1)
    return min_max(x.flatten(), 2000)


def process(files, output_dir, expected_shape, skip):
    for i, f in enumerate(files):
        if i % 1000 == 0: print(i)
        basename = os.path.basename(f)
        if basename in skip:
            continue
        try:
            features = load(f, TIME_SAMPLES)
        except ValueError:
            continue
        else:
            if features.shape == expected_shape:
                dest = os.path.join(output_dir, basename)
                np.save(dest, features)


if __name__ == '__main__':
    skip = {f for f in os.listdir(TRAIN_PROC_DIR) + os.listdir(TEST_PROC_DIR)}
    print('train')
    process(TRAIN_FILES, TRAIN_PROC_DIR, (SIZE,), skip)
    print('test')
    process(TEST_FILES, TEST_PROC_DIR, (SIZE,), skip)
