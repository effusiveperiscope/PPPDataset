import string
import re
import math

def remove_punc(text):
    return re.sub(r'[^\w\s]+',' ',text)

def signif(x, digits=6):
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)
