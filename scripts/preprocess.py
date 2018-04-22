import logging
import os
import random
import sys

import numpy as np

from musicrec.audio import read_audio, spectrogram
from musicrec import utils


N_SAMPLES = 20
DESIRED_SAMPLE_RATE = 44100
OUTPUT_DIR = 'data/spec'
LOG_FILE = 'preprocess.log'
NFFT = 2**12
MIN_HZ = 20
MAX_HZ = 2000
N_TIME_BINS = 512
NPERSEG = 256
NOVERLAP = 256 / 8
N_AUDIO_SAMPLES = int((N_TIME_BINS * NPERSEG) - ((N_TIME_BINS-1) * NOVERLAP))


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
           N_AUDIO_SAMPLES > audio.shape[0]:
        return False
    return True


def load(f):
    sr, audio = read_audio(f, downmix=True)
    return sr, audio

    
def sample_from_audio(sr, audio, window_length, n_samples):
    window_len = window_length
    audio_len = audio.shape[0]
    indices = random.sample(list(range(audio_len - window_len)), k=n_samples)
    for i in indices:
        start, stop = i, i+window_len
        sample = audio[start:stop]
        sample_begin_seconds = i // sr
        yield sample_begin_seconds, sample


def process_audio(sr, audio, *, nfft, nperseg, noverlap):
    f, t, Sxx = spectrogram(audio, sr, nfft=nfft, nperseg=nperseg,
                            noverlap=noverlap)
    min_hz_bin, max_hz_bin = abs(f - MIN_HZ).argmin(), abs(f - MAX_HZ).argmin()
    return Sxx[min_hz_bin:max_hz_bin,:]


def save_feature(X, filename):
    np.savez_compressed(filename, X)


def _main(file):
    sr, audio = load(file)
    if use_track(sr, audio):
        logger.info('processing: %s' % file)
        samples = sample_from_audio(sr, audio, N_AUDIO_SAMPLES, N_SAMPLES)
        for sec, sample in samples:
            feature = process_audio(sr, sample, nfft=NFFT, nperseg=NPERSEG, noverlap=NOVERLAP)
            feature_id = '%s - %s' % (os.path.basename(file), sec)
            f_out = os.path.join(OUTPUT_DIR, '%s.npz' % feature_id)
            save_feature(feature, f_out)
        logger.info('completed: %s' % file)
    else:
        logger.info('skipped: %s' % file)

def main(directory):
    completed = {os.path.basename(L.replace('\n', '')) for L in open('completed.log')}
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if not os.path.basename(f) in completed]
    logger.info('processing started. %d files remain.' % len(files))    
    for file in files:
        try:
            _main(file)
        except Exception as err:
            logger.exception('could not process %s' % file)
    logger.info('processing complete')


if __name__ == '__main__':
    raw_dir = get_directory()
    try:
        main(raw_dir)
    except:
        logger.exception('processing failed')
