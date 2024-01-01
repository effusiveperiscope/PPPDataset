import gruut
from gruut.const import default_split_words
from pathlib import Path
import re
import eng_to_ipa as ipa
CMU_IPA_MAPPING = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "ʧ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "ʤ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ"
}
HORSEWORDS_DICTIONARY = 'horsewords.clean'

def load_dictionary(dict_path, remove_nums=True):
    arpadict = dict()
    with open(dict_path, "r") as f:
        for line in f.readlines():
            word = line.split("  ")
            assert len(word) == 2
            maps_to = word[1].strip()
            if remove_nums:
                maps_to = re.sub(r'\d+', '', maps_to)
            arpadict[word[0].strip().upper()] = maps_to
    return arpadict

exceptions_dictionary = load_dictionary(HORSEWORDS_DICTIONARY)

def arpabet_to_ipa(arpabet : str):
    return ' '.join(CMU_IPA_MAPPING[token] for token in arpabet.split(' '))

def _word_conv_to_ipa(word : str):
    return word

def conv_to_ipa(text : str, print_unhandled : bool = False,
    normalize : bool = False):
    output = []
    # 1. Check if all words are in CMU. If so, just use gruut
    phoneme_join = ' ' if normalize else ''
    if ipa.isin_cmu(text):
        sentence = gruut.sentences(text)
        for sent in sentence:
            for word in sent.words:
                if not word.is_spoken:
                    output.append(word.text)
                elif word.phonemes:
                    word_str = ''.join(word.phonemes)
                    if normalize:
                        word_str = phoneme_join.join(word_str)+' ▁'
                    output.append(word_str)
        return ' '.join(output)

    # 2. If not, handle each word individually
    for word in default_split_words(text):
        if word.upper() in exceptions_dictionary:
            ipa_sub = arpabet_to_ipa(exceptions_dictionary[word.upper()])
            if normalize:
                ipa_sub = phoneme_join.join(ipa_sub)
            output.append(ipa_sub)
        else:
            if print_unhandled and not ipa.isin_cmu(text):
                print("Found word not in dictionary "+word)
                print("Using gruut fallback")
            sentence = gruut.sentences(word)
            for sent in sentence:
                for word in sent.words:
                    if not word.is_spoken:
                        output.append(word.text)
                    elif word.phonemes:
                        word_str = ''.join(word.phonemes)
                        if normalize:
                            word_str = phoneme_join.join(word_str)
                        output.append(word_str)
    output = ' '.join(output)
    return output
