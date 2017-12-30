import collections
import io
import os


MEGABYTE = 2**20

def getsize(f):
    """Return size of `f` in megabytes."""
    size = os.path.getsize(f) / MEGABYTE
    return size


class Printer:
    last = ''
    console_width = os.get_terminal_size().columns - 1

    @classmethod
    def refresh_print(cls, text, **kwargs):
        print('\r', ' ' * min(cls.console_width, len(cls.last)), end='', flush=True)
        text = '\r%s' % text.format(**kwargs)
        cls.last = text
        print(text[:cls.console_width], end='', flush=True)


refresh_print = Printer.refresh_print
