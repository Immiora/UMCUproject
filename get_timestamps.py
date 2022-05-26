'''
get timestamps over words for each subject

python get_timestamps
python get_timestamps -s F1
python get_timestamps -s F1 F5
'''

import os
import argparse

def main(subjects):

    for subject in subjects:
        directory = os.path.join('data/Data', subject, 'trans')
        word_timestamps = []
        index_number = 0

        for filename in sorted(os.listdir(directory)):
            f = os.path.join(directory, filename)

            with open(f, 'r') as tag_file:
                lines = tag_file.read().splitlines()
                counter = -1
                for x in range(len(lines) - 1):
                    # skips over the silences (n=?)
                    # and phonemes that are not linked to a word (n=126)
                    if (lines[x].split(',')[3] == ''):
                        continue

                    word = lines[x].split(',')[3].lower()
                    counter += 1
                    if lines[x + 1].split(',')[3] != word:
                        # counts back to the first phoneme of current word to set start time
                        start_time = lines[x - counter].split(',')[0]
                        counter = -1
                        result = [index_number, word, start_time, lines[x].split(',')[1], lines[x].split(',')[-1]]
                        word_timestamps.append(result)

            index_number += 1

        #
        all_words = []
        for x in range(len(word_timestamps)):
            word = word_timestamps[x][1]
            all_words.append(word)

        unique_words = set(all_words)

        #
        with open(os.path.join(subject + '_timestamps.txt'), 'w') as file:
            for word_plus_time in word_timestamps:
                counter = 1
                for elements in word_plus_time:
                    if counter % 5 == 0:
                        file.write(elements + '\n')
                        counter += 1
                        continue

                    file.write(str(elements) + ',')
                    counter += 1

##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--subjects', '-s', type=str,  nargs="+",
                        choices=['F1', 'F5', 'M1', 'M3'],
                        default=['F1', 'F5', 'M1', 'M3'],
                        help='Subject to run')
    args = parser.parse_args()
    main(args.subjects)