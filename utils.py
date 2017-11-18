import collections
import io
import os

import numpy as np
import pydub
import scipy.io.wavfile

import config as cf


class UnreadableMP3Error(ValueError): pass
class InvalidPCMError(ValueError): pass


_MEGABYTE = 2**20

def getsize(f):
    """Return size of `f` in megabytes."""
    size = os.path.getsize(f) / _MEGABYTE
    return size


def init_fs():
    if not os.path.exists(cf.SUCCESSFILE):
        with open(cf.SUCCESSFILE, 'w'):
            pass
    if not os.path.exists(cf.SKIPFILE):
        with open(cf.SKIPFILE, 'w'):
            pass
    with open(cf.FAILFILE, 'w'):
        pass
    if not os.path.exists(cf.TRAIN_DIR):
        os.makedirs(cf.TRAIN_DIR)


def _mp3_hook(f):
    try:
        audio_segment = pydub.AudioSegment.from_mp3(f)
    except (OSError, pydub.exceptions.CouldntDecodeError) as err:
        raise UnreadableMP3Error('could not read mp3: %s' % err)
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

def read_audio(f, downmix):
    """Read wav file and return the sample rate and audio."""
    if f.endswith('.mp3'):
        f = _mp3_hook(f)
    sr, audio = scipy.io.wavfile.read(f)
    if not audio.dtype is np.float32:
        audio = _normalize_pcm(audio)
    if downmix and len(audio.shape) == 2:
        audio = _down_mix(audio)
    return sr, audio

MetaData = collections.namedtuple('MetaData', 'artist,album,track')
def get_meta_data(f):
    try:
        info = pydub.utils.mediainfo(f)['TAG']
    except Exception:
        pass
    artist, album, track = info['artist'], info['album'], info['title']
    return MetaData(artist, album, track)


class _Printer:
    last = ''
    console_width = os.get_terminal_size().columns - 1
    @classmethod
    def rprint(cls, text, **kwargs):
        print('\r', ' ' * min(cls.console_width, len(cls.last)), end='', flush=True)
        text = '\r%s' % text.format(**kwargs)
        cls.last = text
        print(text[:cls.console_width], end='', flush=True)


rprint = _Printer.rprint
