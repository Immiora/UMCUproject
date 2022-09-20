'''
get missing data per subject,
create a common word list,
save individual timestamps for common words per subject
'''


from utils.read_data import get_sensor_data, get_pos_data
import pandas as pd
import argparse
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def remove_missing(subject, n_total, idx_missing):
    ts = pd.read_csv(os.path.join(subject + '_timestamps_mfa.txt'), sep=',', header=None)
    assert ts.shape[0] == n_total, 'Number of entries in timestamps does not match the length of frames'
    ts.drop(index=idx_missing, inplace=True)
    pd.DataFrame(ts).to_csv(os.path.join(subject + '_timestamps_mfa_no_missing.txt'),
                                    sep=',', header=False, index=False)
    return ts

def main(subjects):

    # make new timestamps per subject without missing sensor data
    ts_all = dict.fromkeys(subjects)
    for subject in subjects:
        print(f'Subject: {subject}')
        sensors = get_sensor_data(subject)
        frames = get_pos_data(subject, sensors, 'timestamps_mfa')

        missing = []
        for frame in frames:
            if frames[frame].empty or frames[frame].isnull().values.any(): #F5 2785 should be empty
                missing.append(frame)

        print(f'Number of all dataframes: {len(frames)}')
        print(f'Number of dataframes with missing data: {len(missing)}')

        ts_all[subject] = remove_missing(subject, len(frames), missing)

    # make common list of words across subjects
    # find intersection of word-rows (file, sentence, word and word-num-in-sent across all subjects
    # save the list of common words
    temp = pd.concat(ts_all.values(), axis=0, join='inner')
    temp = temp.drop(columns=[2, 3]).reset_index(drop=True)
    temp = temp[temp.duplicated(keep=False)]
    df1 = (temp.groupby(temp.columns.tolist()).apply(lambda x: tuple(x.index)).reset_index(name='idx'))
    n_subjs = df1['idx'].apply(len)
    indx = n_subjs.index[n_subjs == len(subjects)]
    common = df1.loc[indx].drop(columns='idx')

    pd.DataFrame(common).to_csv('common_words_mfa_no_missing.txt', sep=',', header=False, index=False)

    # remove most common words
    import nltk
    stopwords = nltk.corpus.stopwords.words('english')
    [stopwords.append(i) for i in ["they're", "it's", "we'll", "don't", "haven't", "i'd", "i'll"]]
    common_nostop = common[~common[1].isin(stopwords)]
    pd.DataFrame(common_nostop).to_csv('common_words_mfa_no_missing_no_stopwords.txt',
                                    sep=',', header=False, index=False)
    pd.DataFrame(common_nostop[1].sort_values()).to_csv('common_words_mfa_no_missing_no_stopwords_wordlist.txt',
                                    sep=',', header=False, index=False)

##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--subjects', '-s', type=str,  nargs="+",
                        choices=['F1', 'F5', 'M1', 'M3'],
                        default=['F1', 'F5', 'M1', 'M3'],
                        help='Subject to run')
    args = parser.parse_args()
    main(args.subjects)