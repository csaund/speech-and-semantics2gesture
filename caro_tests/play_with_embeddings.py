from bert_embedding import BertEmbedding
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import os
import pickle
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import random


def get_random_closest_gestures(df):
    i = random.randint(0, len(df))
    a = df.iloc[i]
    b = get_closest_gesture(df, i)
    print(' '.join(a.transcript))
    print(' '.join(b.transcript))
    return a, b


def get_nonzero_min_ind(l):
    m = max(l)
    ind = 0
    for i in range(len(l)):
        if l[i] < m and l[i] != 0:
            m = l[i]
            ind = i
    return m, ind


def get_closest_gesture(df, i):
    df['features'] = df.apply(lambda row: row['encoding'][1], axis=1)
    interest_vector = df.iloc[i].features
    distances = df.apply(lambda row: get_encoding_dist(interest_vector, row['features']), axis=1)
    val, i = get_nonzero_min_ind(distances)
    return df.iloc[i]


def get_encoding_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.linalg.norm(v1 - v2)
    return dist


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--pickle', '-orig', default="",
                                   help="Path where original motion files (in BVH format) are stored")

    params = parser.parse_args()

    df = pd.read_pickle(params.pickle)
