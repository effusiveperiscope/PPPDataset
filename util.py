import string
import re
import math
import os

def remove_punc(text):
    return re.sub(r'[^\w\s]+',' ',text)

def signif(x, digits=6):
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)

from pydub import AudioSegment
def check_file_dur_ms(path):
    audio = AudioSegment.from_file(path)
    audio_length_ms = len(audio)
    return audio_length_ms

def longpath(path):
    import platform
    path = os.path.abspath(path)
    if 'Windows' in platform.system() and not path.startswith('\\\\?\\'):
        path = u'\\\\?\\'+path.replace('/','\\')
        return path
    else:
        return path