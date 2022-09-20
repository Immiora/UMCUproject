'''
For the forced aligner to work I need transcription files
Read trans files and resave unique sentences separated by full stops.
Adjust capitalization (lower)?
'''


import os
import argparse
import pandas as pd

def main(subjects):

    for subject in subjects:
        data_dir = os.path.join('data/Data', subject, 'trans')
        out_dir = data_dir.replace('trans', 'text')
        if not os.path.isdir(out_dir): os.makedirs(out_dir)

        for filename in sorted(os.listdir(data_dir)):
            f = os.path.join(data_dir, filename)
            df = pd.read_csv(f, header=None)
            text = '. '.join(df[4].dropna().unique()) + '.'

            with open(os.path.join(out_dir, filename.replace('.trans', '.txt')).replace('mri', 'ema'), 'w') as file: # for M3!
                file.write(text)
            print('Done')


##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--subjects', '-s', type=str,  nargs="+",
                        choices=['F1', 'F5', 'M1', 'M3'],
                        default=['F1', 'F5', 'M1', 'M3'],
                        help='Subject to run')
    args = parser.parse_args()
    main(args.subjects)