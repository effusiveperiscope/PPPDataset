import string
def remove_punc(text):
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)
