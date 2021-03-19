## takes as input train_with_text.csv and does various levels of semantic
# analysis on text for each gesture,
# and outputs the resulting csv into train_with_semantics.csv.
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm
import re
import seaborn as sns
from data.wp.category_assignment import get_categories

# todo explore difference sentence encoding model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
print("loading module: ", module_url)
vectorization_model = hub.load(module_url)
print("module %s loaded" % module_url)


def embed(sentence):
    vectorization = None
    try:
        vectorization = vectorization_model(sentence)
    except Exception as e:
        print('unable to vectorize ', sentence)
        print(e)
        return None
    return vectorization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_csv', '--train_csv', help='where the training csv lives')
    parser.add_argument('-base_path', '--base_path',
                        help='where is the base path for TRAINING data? is it Gestures/<speaker> ?')
    #parser.add_argument('-speaker', '--speaker',
    #                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
    args = parser.parse_args()

    BASE_PATH = args.base_path

    df = pd.read_csv(args.train_csv)

    tqdm.pandas()       # watch this bad boy

    transcripts = df['transcript'].fillna('')
    df = df.fillna('')
    embeddings = vectorization_model(transcripts)
    # so nervous about this could this be a bug????
    if 'sentence_vectorization' not in df.columns:
        print("getting sentence embeddings")
        embeddings = vectorization_model(transcripts)
        df['sentence_vectorization'] = list(embeddings)
    if 'semantic_categories' not in df.columns:
        print("getting semantic categories")
        df['semantic_categories'] = df.progress_apply(lambda row: get_categories(row['transcript']), axis=1)

    output_csv = 'training_with_semantics.csv'
    df.to_csv(os.path.join(BASE_PATH, output_csv))
