import random
import os

import numpy as np
import scipy.io

import config as cf
import utils


class InvalidSampleRateError(ValueError): pass
class InvalidShapeError(ValueError): pass
class InvalidFeatureError(ValueError): pass
    

def make_feature(audio, sr, nfft):
    *_, Sxx = scipy.signal.spectrogram(audio, sr, nfft=cf.NFFT)
    X = Sxx.mean(axis=1)
    mn, mx = X.min(), X.max()
    # be careful to handle divide by zero errors here
    if mn != mx:
        X = (X - mn) / (mx - mn)
    elif mx != 0:
        X = X / mx
    else:
        X = X
    X = X[cf.START:cf.STOP]
    return X

def _validate_features(arr):
    if not (arr >= 0).all() and (arr <= 1).all():
        raise InvalidFeatureError('features not scaled to [0, 1]: %s' % arr)

def _make_features(sr, audio):
    window_length = sr * cf.WINDOW_LENGTH
    n_samples = audio.shape[0]
    samples = random.sample(list(range(n_samples - window_length)), k=cf.N_CLIPS)
    output = []
    for clip_begin in samples:
        clip = audio[clip_begin:clip_begin+window_length]
        feature = make_feature(clip, sr, cf.NFFT)
        _validate_features(feature)
        output.append(feature)
    return output

def create_training_data(files):
    successes = []
    failures = []
    success = skipped = 0
    msg = report = ''
    ttl = len(files)
    msg = '\r{remain} files remain. {complete} completed (success={s}, fail={f}). Currently processing {file}'
    for file in files:
        n_processed = success + skipped
        try:
            print('\r', ' ' * len(report), end='', flush=True)
            report = msg.format(
                remain=ttl - n_processed, complete=n_processed, s=success,
                f=skipped, file=os.path.basename(file))
            print(report, end='', flush=True)
            _create_training_data(file)
        except (utils.UnreadableMP3Error, InvalidSampleRateError, Exception) as err:
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
    sr, audio = utils.read_audio(file, downmix=cf.DOWNMIX)
    if sr != cf.DESIRED_SAMPLE_RATE:
        raise InvalidSampleRateError('invalid sample rate: %s' % sr)
    features = _make_features(sr, audio)
    basename = os.path.basename(file)
    for i, feature in enumerate(features, start=1):
        if feature.shape == cf.EXPECTED_SHAPE:
            saveto = '%s - sample %s.npy' % (basename, i)
            saveto = os.path.join(cf.TRAIN_DIR, saveto)
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
    files = [f for f in files if utils.getsize(f) <= cf.MAXSIZE]
    return files
 
def main():
    utils.init_fs()
    print('found %s files total' % len(cf.FILES))
    files = _filter_input_files(cf.FILES)
    print('processing %s files' % len(files))
    create_training_data(files)


if __name__ == '__main__':
    main()
