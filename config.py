import glob

# boilerplate
N_CLIPS = 20       # number of samples per song
WINDOW_LENGTH = 6  # length in seconds of sample
NFFT = 2**12       # determines frequency bins
DESIRED_SAMPLE_RATE = 44100  # in Hz
MAXSIZE = 40                 # in megabytes
MUSIC_DIR = '/home/dante_gates/music/Music'
OUTPUT_DIR = '/home/dante_gates/repos/music-rec/data/train'
SUCCESSFILE = 'success.txt'
FAILFILE = 'failed.txt'
SKIPFILE = 'skip.txt'

# computed configs
EXPECTED_SHAPE = ((NFFT // 2) + 1,)
FILES = glob.glob('{}/**/.wav'.format(MUSIC_DIR), recursive=True) \
      + glob.glob('{}/**/*mp3'.format(MUSIC_DIR), recursive=True)
MEGABYTE = 2**20
