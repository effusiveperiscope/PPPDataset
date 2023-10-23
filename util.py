import string
import re
def remove_punc(text):
    return re.sub(r'[^\w\s]+',' ',text)
