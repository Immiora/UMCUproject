import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.general import get_longest_frame
from numpy import pad, mod, linalg, zeros

def separate_by_syl(frames, nsyl, normalize):
    sensors = ['ULx', 'ULy', 'LLx', 'LLy',
               'JWx', 'JWy', 'TDx', 'TDy',
               'TBx', 'TBy', 'TTx', 'TTy']

    scaler = StandardScaler()
    scaler.fit(pd.concat(frames)) # global normalization
    syl_frames = {}

    for count in tqdm.trange(len(frames)):
        if  frames[count].syl == nsyl:
            if normalize:
                # standardize the data to have a mean of 0 and approx. a SD of 1
                data = scaler.transform(frames[count])
                df = pd.DataFrame(data, columns=sensors)

                # set meta-data, at this point we only need the word and the sentence it came from
                df.word = frames[count].word
                df.sent = frames[count].sent

                syl_frames[count] = df
            else:
                syl_frames[count] = frames[count]
    return syl_frames


def pad_data(frame):
    # target length is the the word with the most samples in that syllable category
    target_length = get_longest_frame(frame)
    scaler = StandardScaler()
    scaler.fit(pd.concat(frame)) # global normalization
    global_mean = scaler.mean_
    padded = {}

    for t, word in zip(tqdm.trange(len(frame)), frame.keys()):
        current_length = frame[word].shape[0]
        pad_length1 = int((target_length - current_length) / 2)

        if mod((target_length - current_length),2) == 1:
            pad_length2 = pad_length1 + 1
        else:
            pad_length2 = pad_length1

        x = pad(frame[word].values, ((pad_length1,pad_length2), (0,0)),
                                   mode='constant',
                                    constant_values=((global_mean,global_mean), (0,0)))
        padded[word] = pd.DataFrame(x, columns=frame[word].columns).transpose()
        padded[word].word = frame[word].word
        padded[word].sent = frame[word].sent

    return padded


def compute_word_difference(data):
    difference_matrix = zeros((len(data), len(data)))

    for a, row_word in enumerate(data.values()):
        for b, column_word in enumerate(data.values()):
            diff_mat = row_word.subtract(column_word).to_numpy()
            frob_norm = linalg.norm(diff_mat)

            difference_matrix[a, b] = frob_norm
            difference_matrix[b, a] = frob_norm

    df = pd.DataFrame(difference_matrix)
    df.columns, df.index = [data[i].word for i in data.keys()], [data[i].word for i in data.keys()]
    correlations = df.corr()

    return correlations