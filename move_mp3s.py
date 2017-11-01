import collections
import glob
import logging
import readline
import os
import shutil

import pydub

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO', filename='move.log', filemode='a')


MetaData = collections.namedtuple('MetaData', 'artist,album,track')
def get_meta_data(f):
    try:
        info = pydub.utils.mediainfo(f)['TAG']
    except Exception as err:
        raise ValueError('could not read metadata: %s' % f) from err
    artist, album, track = info['artist'], info['album'], info['title']
    return MetaData(artist, album, track)


_path_format = '{artist}-{album}-{track}.mp3'
def rename(metadata):
    return '-'.join(metadata) + '.mp3'


def main(root_dir, output_dir):
    files = glob.glob('{}/**/*mp3'.format(root_dir), recursive=True)
    logger.info('found %s files' % len(files))
    for f in files:
        try:
            metadata = get_meta_data(f)
            pathname = rename(metadata)
            dst = os.path.join(output_dir, pathname)
            shutil.move(f, dst)
            logger.info('moved %s to %s' % (f, dst))
        except Exception as err:
            logger.error('could not move %s: %s' % (f, err))
        



if __name__ == '__main__':
    root_dir = input('root directory: ')
    output_dir = input('output directory: ')
    main(root_dir, output_dir)
