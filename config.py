import glob
import os

import numpy as np
import scipy.signal

_dir = os.path.dirname(__file__)

# boilerplate
N_CLIPS = 20       # number of samples per song
WINDOW_LENGTH = 6  # length in seconds of sample
NFFT = 2**12       # determines frequency bins
DESIRED_SAMPLE_RATE = 44100  # in Hz
DOWNMIX = True
MAXSIZE = 40                 # in megabytes
MIN_HZ = 20
MAX_HZ = 2000
MUSIC_DIR = os.path.join(_dir, 'data/raw')
TRAIN_DIR = os.path.join(_dir, 'data/preprocessed')
MODEL_PATH = os.path.join(_dir, 'autoencoder.h5')
SUCCESSFILE = os.path.join(_dir, 'success.txt')
FAILFILE = os.path.join(_dir, 'failed.txt')
SKIPFILE = os.path.join(_dir, 'skip.txt')

# computed configs
def _find_start_stop(sr, min_hz, max_hz, nfft):
    f, *_ = scipy.signal.spectrogram(np.random.uniform(size=sr), sr, nfft=nfft)
    return abs(f - min_hz).argmin(), abs(f - max_hz).argmin()

MIN_FQ_BIN, MAX_FQ_BIN = _find_start_stop(DESIRED_SAMPLE_RATE, MIN_HZ, MAX_HZ, NFFT)
EXPECTED_SHAPE = (MAX_FQ_BIN - MIN_FQ_BIN,)
FILES = glob.glob('{}/**/.wav'.format(MUSIC_DIR), recursive=True) \
      + glob.glob('{}/**/*mp3'.format(MUSIC_DIR), recursive=True)
