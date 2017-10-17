import glob
import io
import random
import os

import librosa
import numpy as np
import pydub
import scipy.io


WINDOW_LENGTH = 3  # in seconds
DESIRED_SAMPLE_RATE = 44100
HOP_LENGTH = 2048
N_CLIPS = 20
OUTPUT_DIR = '/home/dante_gates/repos/music-rec/data/train'
MAXSIZE = 40  # in megabytes
SUCCESSFILE = 'success.txt'
FAILFILE = 'failed.txt'
SKIPFILE = 'skip.txt'

FILES = glob.glob('/home/dante_gates/music/Music/**/.wav', recursive=True) \
      + glob.glob('/home/dante_gates/music/Music/**/*mp3', recursive=True)


class UnreadableMP3Error(ValueError): pass
class InvalidSampleRateError(ValueError): pass
class InvalidShapeError(ValueError): pass

def _mp3_hook(f):
    try:
        audio_segment = pydub.AudioSegment.from_mp3(f)
    except (OSError, pydub.exceptions.CouldntDecodeError) as err:
        raise UnreadableMP3Error('could not read mp3') from err
    byte_stream = io.BytesIO()
    audio_segment.export(byte_stream, 'wav')
    byte_stream.seek(0)
    return byte_stream

def _read_wav(f):
    """Read wav file and return the sample rate and audio."""
    sr, audio = scipy.io.wavfile.read(f)
    if len(audio.shape) == 2:
        audio = (audio.sum(axis=1) / 2).astype("int16")
    return sr, audio

# TODO: not doing anything with right channel now.
def _make_features(sr, audio):
    n_samples = audio.shape[0]
    window = sr * WINDOW_LENGTH
    samples = random.sample(list(range(n_samples - window)), k=N_CLIPS)
    melspecs = []
    for clip_begin in samples:
        clip = audio[clip_begin:clip_begin+window]
        melspec = librosa.feature.melspectrogram(clip, sr, hop_length=HOP_LENGTH)
        melspecs.append(np.reshape(melspec, -1))
    return melspecs

_megabyte = 2**20
def _valid_size(f):
    """Return size of `f` in megabytes."""
    size = os.path.getsize(f) / _megabyte
    return size <= MAXSIZE

_expected_shape = (8320,)  # for sanity check
def create_training_data(files):
    successes = []
    failures = []
    success = 0
    skipped = 0
    msg = ''
    for file in files:
        try:
            print('\r', ' ' * len(msg), end='', flush=True)
            msg = '\r%s files processed. %s skipped. Currently processing %s' \
                  % (success, skipped, os.path.basename(file))
            print(msg, end='', flush=True)
            _create_training_data(file)
        except (UnreadableMP3Error, InvalidSampleRateError) as err:
            skipped += 1
            failures.append((file, err))
            with open(SKIPFILE, 'a') as f:
                f.write('%s\n' % file)
        except ValueError as err:
            skipped += 1
            failures.append((file, err))
            with open(FAILFILE, 'a') as f:
                f.write('%s: %s\n' % (err, file))
        else:
            success += 1
            successes.append(file)
            with open(SUCCESSFILE, 'a') as f:
                f.write('%s\n' % file)


def _create_training_data(file):
    f = _mp3_hook(file) if file.endswith('.mp3') else file
    sr, audio = _read_wav(f)
    features = _make_features(sr, audio)
    basename = os.path.basename(file)
    for i, feature in enumerate(features, start=1):
        if feature.shape == _expected_shape:
            saveto = '%s - sample %s.npy' % (basename, i)
            saveto = os.path.join(OUTPUT_DIR, saveto)
            np.save(saveto, feature)
        else:
            raise InvalidShapeError('Unexpected feature shape, %s' % feature.shape)


def _filter_input_files(files):
    skip = set()
    with open(SUCCESSFILE) as f:
        skip |= {L for L in f.read().split('\n')}
    with open(SKIPFILE) as f:
        skip |= {L for L in f.read().split('\n')}
    files = (f for f in files if not f in skip)
    files = [f for f in files if _valid_size(f)]
    return files


if __name__ == '__main__':
    print('found %s files total' % len(FILES))
    files = _filter_input_files(FILES[3000:])
    print('processing %s files' % len(files))
    create_training_data(files)

