import random
import os

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

import config as cf
import utils
from musicrec.audio import spectrogram


class InvalidSampleRateError(ValueError): pass
class InvalidShapeError(ValueError): pass
class InvalidFeatureError(ValueError): pass
    

def make_feature(audio, sr, nfft, min_fq_bin=0, max_fq_bin=-1):
    Sxx = spectrogram(audio, sr, nfft)
    X = Sxx[min_fq_bin:max_fq_bin, :]
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
        feature = make_feature(clip, sr, cf.NFFT, cf.MIN_FQ_BIN, cf.MAX_FQ_BIN)
        _validate_features(feature)
        output.append((feature, clip_begin // sr))
    return output

def create_training_data(files, output_dir):
    successes = []
    failures = []
    success = skipped = 0
    ttl = len(files)
    msg = '{remain} files remain. {complete} completed (success={s}, fail={f}). Currently processing {file}'
    for file in files:
        n_processed = success + skipped
        try:
            utils.rprint(msg, remain=ttl - n_processed,
                         complete=n_processed, s=success,
                         f=skipped, file=os.path.basename(file))
            _create_training_data(file, output_dir)
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

def _create_training_data(file, output_dir):
    sr, audio = utils.read_audio(file, downmix=cf.DOWNMIX)
#    if sr != cf.DESIRED_SAMPLE_RATE:
#        raise InvalidSampleRateError('invalid sample rate: %s' % sr)
    features = _make_features(sr, audio)
    basename = os.path.basename(file)
    for feature, clip_begin in features:
#        if feature.shape == cf.EXPECTED_SHAPE:
        saveto = '%s - %s sec.npy' % (basename, clip_begin)
        saveto = os.path.join(output_dir, saveto)
        np.save(saveto, feature)
#        else:
#            raise InvalidShapeError('Unexpected feature shape: %s != %s'
#                                    % (feature.shape, cf.EXPECTED_SHAPE))

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
    train, test = train_test_split(cf.FILES, test_size=0.2, random_state=0)
    train = _filter_input_files(train)
    test = _filter_input_files(test)
    print('processing %s files' % (len(train) + len(test)))
    create_training_data(train, cf.TRAIN_DIR)
    create_training_data(test, cf.TEST_DIR)


if __name__ == '__main__':
    main()
