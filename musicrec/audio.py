import collections
import io

import numpy as np
import pydub
import scipy
import scipy.signal
import scipy.io.wavfile


PCM_SCALARS = {
    'int32': 2**31,
    'int16': 2**15
}


class MP3Error(ValueError):
    pass


def down_mix(audio):
    audio = (audio.sum(axis=1) / 2).astype(audio.dtype)
    return audio


def _normalize_pcm(audio):
    pcm_scalar = PCM_SCALARS.get(audio.dtype.name, None)
    if pcm_scalar is None:
        raise MP3Error('Unkown PCM: %s' % audio.dtype)
    return audio / pcm_scalar


def _mp3_hook(f):
    try:
        audio_segment = pydub.AudioSegment.from_mp3(f)
    except (OSError, pydub.exceptions.CouldntDecodeError) as err:
        raise MP3Error('could not read mp3: %s' % err)
    byte_stream = io.BytesIO()
    audio_segment.export(byte_stream, 'wav')
    byte_stream.seek(0)
    return byte_stream


def read_audio(f, downmix):
    """Read wav file and return the sample rate and audio."""
    if f.endswith('.mp3'):
        f = _mp3_hook(f)
    sr, audio = scipy.io.wavfile.read(f)
    if not audio.dtype is np.float32:
        audio = _normalize_pcm(audio)
    if downmix and len(audio.shape) == 2:
        audio = down_mix(audio)
    return sr, audio


MetaData = collections.namedtuple('MetaData', 'artist,album,track')
def get_meta_data(f):
    try:
        info = pydub.utils.mediainfo(f)['TAG']
        artist, album, track = info['artist'], info['album'], info['title']
    except KeyError as err:
        raise MP3Error('aritst, album and/or title infor missing from metadata: %s'
                       % f) from err
    except Exception as err:
        raise MP3Error('could not read metadata: %s' % f) from err
    return MetaData(artist, album, track)


def spectrogram(audio, sr, nfft):
    f, t, Sxx = scipy.signal.spectrogram(audio, sr, nfft=nfft)
    return f, t, Sxx
