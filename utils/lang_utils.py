import nltk
from functools import lru_cache
from itertools import product as iterprod

arpabet = nltk.corpus.cmudict.dict()

@lru_cache()
def lookup_phon(s):
    s = s.lower()
    try:
        return arpabet[s]
    except KeyError:
        middle = len(s)/2
        partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
        for i in partition:
            pre, suf = (s[:i], s[i:])
            if pre in arpabet and lookup_phon(suf) is not None:
                return [x+y for x,y in iterprod(arpabet[pre], lookup_phon(suf))]
        return None


def check_word2phon(word, phones):
    if word == 'cheque':
        return phones[:-2]
    else:
        return phones


def compute_phoneme_edit_distance(phonemes_a, phonemes_b):
    # gets phonemic levenshtein distance
    len_a = len(phonemes_a)
    len_b = len(phonemes_b)
    d = [[i] for i in range(1, len_a + 1)]
    d.insert(0, list(range(0, len_b + 1)))
    for j in range(1, len_b + 1):
        for i in range(1, len_a + 1):
            if phonemes_a[i - 1] == phonemes_b[j - 1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i].insert(j, min(d[i - 1][j] + 1,
                               d[i][j - 1] + 1,
                               d[i - 1][j - 1] + substitutionCost))
    return d[-1][-1]