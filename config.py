import glob
import numpy as np
import scipy.signal

# boilerplate
N_CLIPS = 20       # number of samples per song
WINDOW_LENGTH = 6  # length in seconds of sample
NFFT = 2**12       # determines frequency bins
DESIRED_SAMPLE_RATE = 44100  # in Hz
DOWNMIX = True
MAXSIZE = 40                 # in megabytes
MIN_HZ = 20
MAX_HZ = 2000
MUSIC_DIR = '/home/dante_gates/music/Music'
TRAIN_DIR = '/home/dante_gates/repos/music-rec/data/train2'
MODEL_PATH = 'model.h5'
SUCCESSFILE = 'success.txt'
FAILFILE = 'failed.txt'
SKIPFILE = 'skip.txt'

# computed configs
def _find_start_stop(sr, min_hz, max_hz, nfft):
    f, *_ = scipy.signal.spectrogram(np.random.uniform(size=sr), sr, nfft=nfft)
    return abs(f - min_hz).argmin(), abs(f - max_hz).argmin()

START, STOP = _find_start_stop(DESIRED_SAMPLE_RATE, MIN_HZ, MAX_HZ, NFFT)
EXPECTED_SHAPE = (STOP - START,)
FILES = glob.glob('{}/**/.wav'.format(MUSIC_DIR), recursive=True) \
      + glob.glob('{}/**/*mp3'.format(MUSIC_DIR), recursive=True)
