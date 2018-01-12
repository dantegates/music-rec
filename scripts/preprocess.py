import logging
import os
import random
import sys

import numpy as np

from musicrec.audio import read_audio, spectrogram
from musicrec import utils


N_SAMPLES = 30
SAMPLE_LENGTH = 8
DESIRED_SAMPLE_RATE = 44100
OUTPUT_DIR = 'data/spec'
LOG_FILE = 'preprocess.log'
NFFT = 2**12
MIN_HZ = 20
MAX_HZ = 2000


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=LOG_FILE, level='INFO', filemode='w',
    format='[%(asctime)s] %(msg)s')


def get_directory():
    try:
        d = sys.argv[1]
    except IndexError:
        d = input('raw directory:')
    return d


def use_track(sr, audio):
    if sr != DESIRED_SAMPLE_RATE or \
       (sr * SAMPLE_LENGTH) > audio.shape[0]:
        return False
    return True


def load_raw_features(f):
    sr, audio = read_audio(f, downmix=True)
    if use_track(sr, audio):
        window_length = sr * SAMPLE_LENGTH
        n_audio_signal_samples = audio.shape[0]
        feature_samples = random.sample(list(range(n_audio_signal_samples - window_length)), k=N_SAMPLES)
        for clip_begin in feature_samples:
            clip = audio[clip_begin:clip_begin+window_length]
            clip_begin_seconds = clip_begin // sr
            clip_id = '%s - %s sec' % (os.path.basename(f), clip_begin_seconds)
            yield clip_id, sr, clip


def make_feature(sr, audio, nfft=NFFT):
    f, t, Sxx = spectrogram(audio, sr, nfft)
    min_hz_bin, max_hz_bin = abs(f - MIN_HZ).argmin(), abs(f - MAX_HZ).argmin()
    return Sxx[min_hz_bin:max_hz_bin,:]


def save_feature(X, filename):
    np.save(filename, X)


def main(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    logger.info('processing started. %d files remain.' % len(files))    
    for i, f in enumerate(files):
        utils.refresh_print('%d files remain. processing %s'
                            % (len(files) - i, f))
        raw_features = load_raw_features(f)
        for feature_id, sr, audio in raw_features:
            feature = make_feature(sr, audio)
            f_out = os.path.join(OUTPUT_DIR, '%s.npy' % feature_id)
            save_feature(feature, f_out)
        logger.info('completed: %s' % f)
    logger.info('processing complete')


if __name__ == '__main__':
    raw_dir = get_directory()
    try:
        main(raw_dir)
    except:
        logger.exception('processing failed')
