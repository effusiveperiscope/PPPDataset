LEWD_WORDS = {
    "cum" : 3, "cock" : 3, "pussy" : 3, "vagina" : 3, "marehood" : 3,
    "ass" : 2, "ponut" : 3, "lips" : 1, "moan" : 2, "moans" : 2,
    "clit" : 3, "clitoris" : 3, "tongue-fucking" : 3,
    "moaning" : 2, "womb" : 3, "orgasm" : 3, "came" : 1, "kiss" : 2,
    "marehood" : 2, "flank" : 2,
    "flankhole" : 3, "member" : 1, "pounding" : 1, "thighs" : 2,
    "groin" : 2, "pulse" : 1, "pulsing" : 1, "climax" : 1, "thrust" : 1,
    "thrusts" : 1, "shaft" : 1, "erection" : 1, "surge" : 1, "bury" : 1,
    "rear" : 1, "thrusting" : 1, "rump" : 2, "climaxing" : 3, "sweat" : 1,
    "panting" : 2, "squealed" : 1, "sucking" : 1, "suck" : 1, "hips" : 2,
    "dick" : 2, "loins" : 1, "rub" : 1, "pelvis" : 1}
def lewd_word_count(paragraph : str):
    if paragraph is None or len(paragraph) == 0:
        return 0
    words = paragraph.lower().split()
    return sum(LEWD_WORDS[word] for word in words if word in LEWD_WORDS)

def humanized_detector(paragraph : str): # because people don't tag their shit
    return (" her fingers" in paragraph) or (" her hands" in paragraph)

