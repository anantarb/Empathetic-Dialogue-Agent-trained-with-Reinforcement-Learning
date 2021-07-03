import json
import random
import re

import ftfy


def make_query(source_path, history_len=4):
    
    df = open(source_path).readlines()
    max_hist_len = history_len
    data = []
    history = []
    for i in range(1, len(df)):
        cparts = df[i - 1].strip().split(",")
        sparts = df[i].strip().split(",")
        if cparts[0] == sparts[0]:
            prevsent = cparts[5].replace("_comma_", ",").strip()
            history.append(prevsent)
            temp = {}
            context = " <SOC> ".join(history[-max_hist_len :])
            temp['context'] = context 
            data.append(temp)                
        else:
            history = []
    
    return data

def dialog_generator(mode, seed=0, shuffle=False, comm=None):

    if mode == 'test':
        datas = make_query('datasets/empatheticdialogues/test.csv', history_len=4)
    else:
        datas = make_query('datasets/empatheticdialogues/train.csv', history_len=4)
        if shuffle:
            random.seed(seed)
            random.shuffle(datas)

    print("Total Samples:", len(datas))
    for data in datas:
        text = data['context']
        yield text