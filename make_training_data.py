import glob
import io
import random
import os

import librosa
import numpy as np
import pydub
import scipy.io

import config as cf


class UnreadableMP3Error(ValueError): pass
class InvalidSampleRateError(ValueError): pass
class InvalidShapeError(ValueError): pass
class InvalidPCMError(ValueError): pass


def _mp3_hook(f):
    try:
        audio_segment = pydub.AudioSegment.from_mp3(f)
    except (OSError, pydub.exceptions.CouldntDecodeError) as err:
        raise UnreadableMP3Error('could not read mp3') from err
    byte_stream = io.BytesIO()
    audio_segment.export(byte_stream, 'wav')
    byte_stream.seek(0)
    return byte_stream
    
def _down_mix(audio):
    audio = (audio.sum(axis=1) / 2).astype(audio.dtype)
    return audio

_pcm_map = {
    'int32': 2**31,
    'int16': 2**15
}
def _normalize_pcm(audio):
    pcm_scalar = _pcm_map.get(audio.dtype.name, None)
    if pcm_scalar is None:
        raise InvalidPCMError('cannot read pcm: %s' % audio.dtype)
    return audio / pcm_scalar

def _read_wav(f):
    """Read wav file and return the sample rate and audio."""
    sr, audio = scipy.io.wavfile.read(f)
    if not audio.dtype is np.float32:
        audio = _normalize_pcm(audio)
    if len(audio.shape) == 2:
        audio = _down_mix(audio)
    return sr, audio

def _make_features(sr, audio):
    window_length = sr * cf.WINDOW_LENGTH
    n_samples = audio.shape[0]
    samples = random.sample(list(range(n_samples - window_length)), k=cf.N_CLIPS)
    output = []
    for clip_begin in samples:
        clip = audio[clip_begin:clip_begin+window_length]
        *_, S = scipy.signal.spectrogram(clip, sr, nfft=cf.NFFT)
        mean = S.mean(axis=1)
        min_, max_ = mean.min(), mean.max()
        if min_ != max_:
            features = (mean - min_) / (max_ - min_)
        else:
            features = mean / max_
        output.append(features)
    return output

def _valid_size(f):
    """Return size of `f` in megabytes."""
    size = os.path.getsize(f) / cf.MEGABYTE
    return size <= cf.MAXSIZE

def create_training_data(files):
    successes = []
    failures = []
    success = 0
    skipped = 0
    msg = ''
    report = ''
    ttl = len(files)
    msg = '\r{remain} files remain. {complete} completed (success={s}, fail={f}). Currently processing {file}'
    for file in files:
        n_processed = success + skipped
        try:
            print('\r', ' ' * len(report), end='', flush=True)
            report = msg.format(
                remain=ttl - n_processed,
                complete=n_processed,
                s=success,
                f=skipped,
                file=os.path.basename(file))
            print(report, end='', flush=True)
            _create_training_data(file)
        except (UnreadableMP3Error, InvalidSampleRateError) as err:
            skipped += 1
            failures.append((file, err))
            with open(cf.SKIPFILE, 'a') as f:
                f.write('%s\n' % file)
            with open(cf.FAILFILE, 'a') as f:
                f.write('%s: %s\n' % (err, file))
        except ValueError as err:
            skipped += 1
            failures.append((file, err))
            with open(cf.FAILFILE, 'a') as f:
                f.write('%s: %s\n' % (err, file))
        else:
            success += 1
            successes.append(file)
            with open(cf.SUCCESSFILE, 'a') as f:
                f.write('%s\n' % file)

def _create_training_data(file):
    f = _mp3_hook(file) if file.endswith('.mp3') else file
    sr, audio = _read_wav(f)
    if sr != cf.DESIRED_SAMPLE_RATE:
        raise InvalidSampleRateError('invalid sample rate: %s' % sr)
    features = _make_features(sr, audio)
    basename = os.path.basename(file)
    for i, feature in enumerate(features, start=1):
        if feature.shape == cf.EXPECTED_SHAPE:
            saveto = '%s - sample %s.npy' % (basename, i)
            saveto = os.path.join(cf.OUTPUT_DIR, saveto)
            np.save(saveto, feature)
        else:
            raise InvalidShapeError('Unexpected feature shape: %s != %s'
                                    % (feature.shape, cf.EXPECTED_SHAPE))

def _filter_input_files(files):
    skip = set()
    with open(cf.SUCCESSFILE) as f:
        skip |= {L for L in f.read().split('\n')}
    with open(cf.SKIPFILE) as f:
        skip |= {L for L in f.read().split('\n')}
    files = (f for f in files if not f in skip)
    files = [f for f in files if _valid_size(f)]
    return files

def main():
    if not os.path.exists(cf.SUCCESSFILE):
        with open(cf.SUCCESSFILE, 'w'): pass
    if not os.path.exists(cf.SKIPFILE):
        with open(cf.SKIPFILE, 'w'): pass
    with open(cf.FAILFILE, 'w'): pass
    print('found %s files total' % len(cf.FILES))
    files = _filter_input_files(cf.FILES)
    print('processing %s files' % len(files))
    create_training_data(files)    


if __name__ == '__main__':
    main()
