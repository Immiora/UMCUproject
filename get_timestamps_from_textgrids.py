'''
get timestamps over words for each subject
using MFA forced alignment results

M3: need to rerun 086-090 (missing sentence in transcript), 201_205
(331_335 were also missing but added already)

F5: remove sorry from 001-005, remove extra sentence in 356-360

F1: 146-150 missing (extra sentence at the end), 176-180

python get_orig_timestamps_from_textgrids
python get_orig_timestamps_from_textgrids -s M3
python get_orig_timestamps_from_textgrids -s F1 F5
'''

import os
import textgrids
import argparse
import pandas as pd


def main(subjects):

    for subject in subjects:
        annot_directory = os.path.join('data/Data', subject, 'mfa_textgrids')
        sent_directory = os.path.join('data/Data', subject, 'text')
        timestamps = {'file': [], 'text': [], 'xmin': [], 'xmax': [], 'sentence': [], 'word_in_sentence':[]}
        mismatched = []
        sent_count = 0

        for ifile, filename in enumerate(sorted(os.listdir(annot_directory))):
            grid = textgrids.TextGrid(os.path.join(annot_directory, filename))
            with open(os.path.join(sent_directory, filename).replace('.TextGrid', '.txt'), 'r') as sent_file:
                txt = sent_file.read()
            sentences = txt.rstrip('\n').lower().split('. ')
            words = txt.rstrip('\n').replace('.', '').lower().split(' ')
            words = [x for x in words if x]
            num_words_sent = [len(i.split(' ')) for i in sentences]
            sent_id = [i for i in range(len(sentences)) for j in range(num_words_sent[i]) ]
            word_id = [j for i in range(len(sentences)) for j in range(num_words_sent[i]) ]
            sent_id = [i + sent_count for i in sent_id]

            if ifile == 6: # barb's
                ind1 = [i for i in range(len(grid['words'])) if grid['words'][i].text == 'barb'][0]
                ind2 = [i for i in range(len(grid['words'])) if grid['words'][i].text == "'s"][0]
                grid['words'][ind1].text = grid['words'][ind1].text + "'s"
                grid['words'][ind1].xmax = grid['words'][ind2].xmax
                grid['words'].pop(ind2)

            assert len([itext.text for itext in grid['words'] if itext.text != '']) == len(words),\
                        'word length in text annotation and textgrid does not match'

            iword = 0
            for index, itext in enumerate(grid['words']):
                print(itext.text)

                if itext.text == '':
                    continue
                else:
                    assert itext.text.replace("'", '') == words[iword].replace("'", ''), 'annot and textgrid words do not match'
                    timestamps['file'].append(ifile)
                    timestamps['text'].append(itext.text)
                    timestamps['xmin'].append(itext.xmin)
                    timestamps['xmax'].append(itext.xmax)
                    timestamps['sentence'].append(sent_id[iword] + 1)
                    timestamps['word_in_sentence'].append(word_id[iword])
                    iword += 1
            sent_count += len(sentences)
            print(sent_count)

        pd.DataFrame(timestamps).to_csv(os.path.join(subject + '_timestamps_mfa.txt'),
                                        sep=',', header=True, index=False)

##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--subjects', '-s', type=str,  nargs="+",
                        choices=['F1', 'F5', 'M1', 'M3'],
                        default=['F1', 'F5', 'M1', 'M3'],
                        help='Subject to run')
    args = parser.parse_args()
    main(args.subjects)