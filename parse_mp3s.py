import glob
import logging
import multiprocessing as mp
import os
import shutil
import sys

import pydub


_logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO', filename='conversion.log')
OUTPUT_DIR = '/home/dante_gates/repos/music-rec/data/raw'

def main(root):
#    pool = mp.Pool(processes=mp.cpu_count())
    mp3s, wavs = find_audio_files(root)
    for mp3 in mp3s:
        _logger.info('processing %s' % mp3)
        basename = os.path.basename(mp3).rpartition('.')[0]
        output_file = os.path.join(OUTPUT_DIR, basename + '.wav')
        convert_mp3(mp3, output_file)
    for wav in wav:
        _logger.info('processing %s' % mp3)
        copy_wav(wav)


def find_audio_files(root):
    search_mp3s = '{root}/**/*.mp3'.format(root=root)
    search_wavs = '{root}/**/*.wav'.format(root=root)
    mp3s = glob.glob(search_mp3s, recursive=True)
    wavs = glob.glob(search_wavs, recursive=True)
    return mp3s, wavs


def convert_mp3(mp3_file, output_file):
    try:
        audio_segment = pydub.AudioSegment.from_mp3(mp3_file)
        _logger.info('writing %s to %s' % (mp3_file, output_file))
        audio_segment.export(output_file, format='wav')
    except Exception as e:
        _logger.error('could not write %s: %s' % (mp3_file, e))


def copy_wav(wav_file, output_file):
    _logger.info('copying %s to %s' % (wav_file, output_file))
    shutil.copyfile(wav_file, output_file)


if __name__ == '__main__':
    root = sys.argv[1]
    main(root)
