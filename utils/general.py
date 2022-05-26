import os
from nltk.corpus import cmudict

# nltk.download('cmudict')
corpus_syl = cmudict.dict()

def listdir_nohidden(path):
    x = os.listdir(path)
    def gen_nohidden(x):
        for f in x:
            if not f.startswith('.'):
                yield f
    return list(gen_nohidden(x))

def get_nsyl(word):
    def syllables(word):
        # referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        return count
    try:
        # some words can be pronounced differently: this with the full syllable and this as 's' (like Dutch het and 't)
        # in the latter case there will be 0 syllables, that's why corpus_syl['this'] returns two elements
        # to simplify things, just take the first output
        x = corpus_syl[word.lower()][0]
        return len(list(y for y in x if y[-1].isdigit()))
    except KeyError:
        #if word not found in cmudict
        return syllables(word)


def get_longest_frame(dictionary):
    return max([i.shape[0] for i in dictionary.values()])



# def get_longest(dictionary):
#     biggest = 0
#     for word in dictionary:
#         length = len(dictionary[word].at[0, 'ULx'])
#         if length > biggest:
#             biggest = length
#
#     return biggest


# def frobenius_norm(matrix_a, matrix_b):
#     difference_matrix = np.subtract(matrix_a, matrix_b)
#     frob_norm = LA.norm(difference_matrix)
#     return frob_norm