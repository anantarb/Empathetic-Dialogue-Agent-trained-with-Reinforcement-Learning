"""
For each of the preprocessed datasets (after process_data.py: '...model.csv'), sample 70% of the data as train,
fetch the remaining 30% of the data as test
and save them as '..._model_train.csv' and '..._model_test.csv'
"""

import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    seed = 7

    reactions_filepath = os.path.join(ROOT_DIR, 'dataset/emotional-reactions-reddit_model.csv')
    explorations_filepath = os.path.join(ROOT_DIR, 'dataset/explorations-reddit_model.csv')
    interpretations_filepath = os.path.join(ROOT_DIR, 'dataset/interpretations-reddit_model.csv')

    filepaths = [reactions_filepath, explorations_filepath, interpretations_filepath]

    for filepath in filepaths:
        # read csv, sample 70% of the data as train, the remaining 30% as test and save them as train/test
        df = pd.read_csv(filepath, delimiter=',', quotechar='"')
        train = df.sample(frac=0.7, random_state=seed)
        test = df.drop(train.index)
        train.to_csv("{0}_train{1}".format(*os.path.splitext(filepath)))
        test.to_csv("{0}_test{1}".format(*os.path.splitext(filepath)))
