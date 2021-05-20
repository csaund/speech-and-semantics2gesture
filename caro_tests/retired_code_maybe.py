import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.distance import cdist, pdist
import random
from chord import Chord
from shutil import copyfile
from caro_tests.ontology_generator import CII
from tqdm import tqdm
import string
tqdm.pandas()
import spacy


SEMANTIC_CATEGORIES = {
    'FEELINGS': ['angry', 'sad', 'happy', 'love', 'passion', 'anxious', 'stress', 'worry', 'worried', 'anger.', 'anger'],
    'REFLEXIVE': [' me', ' my', 'I'],
    'SIZE': [' big', 'small', 'simple', 'little', 'short', 'long', ' all ', 'ultimate', 'important'],
    'DIRECTION': [' up', ' down', 'left', 'right', 'top', 'bottom', 'side', 'sideways', 'aside'],
    'TIME': ['start', 'finish', 'begin', ' end', 'forever', 'recent', 'years', 'always', ' now', \
            'behind', 'before', 'after', 'as soon as', 'grew up', 'last night', 'yesterday', 'then'],
    'SEPARATION': ['more than', 'separate from', ' other', 'over there', 'aside', 'different', 'another', \
                  'withdraw', 'between', 'outside', 'also'],
    'TOGETHER': ['together', 'bring in', 'incorporate', 'interact', 'association', 'associate'],
    'UNCERTAIN': ['maybe', 'kind of', 'sort of'],
    'DISMISSAL': ['stupid', 'whatever', 'weird', 'ridiculous'],
    'GOOD': ['best', 'good', 'most', 'tasty', 'fantastic', 'perfect'],
    'BAD': ['worst', ' bad ', ' evil', 'wicked', 'gross', 'torture', 'failure'],
    'RELATIONSHIPS': ['brother', 'friend', 'sister', 'mother', 'father', 'girlfriend', 'boyfriend'],
    'TPV': ['gave', 'care', 'interact', 'associate', 'association', ' caring'],
    'SEEING': ['watch', 'look'],
    'PERSONAL_HISTORY': ['remember', 'recount', ' did', ' was', ' were', 'grew up'],
    'LIST_OF': [' two', ' some ', ' some.'],
    'THEM': ['somebody', 'something', 'you guys'],
    'CATEGORICAL': ['whole', 'everything', 'everyone', ' all ', 'crammed', 'same place', 'stuff', 'all over'],
    'TIME_CYCLICAL': [' cycle', ' cycles', 'cyclical'],
    'NEGATION': [' not ', ' nothing', 'nobody', 'noone', 'no ', ' not.'],
    'QUESTIONS': ['where are', 'what are', 'what is', 'how are', 'how is', 'how do', 'where are', 'where is'],
    'PLACES': ['back at', 'back in', 'home', 'over there', 'over here', 'right here', 'get there', 'behind'],
    'ENUMERATION': ['one', 'two', 'three'],
    'PATHS': ['going', 'went', 'journey', 'becoming', 'go off', 'come back', 'goes'],
    'CAUSE': ['because', 'due to', 'owing to']
}


def get_set_categories(row):
    return set([k for k in list(row.keys()) if k in SEMANTIC_CATEGORIES.keys() and row[k] != '-'])



def get_closest_gesture_from_row_sets(row, df, n=10):
    """
    Given a df and a row within that df, calculates the closest
    gesture semantically given the semantic categories assigned to that given row.

    Does this by finding gestures in the DF with maximal intersection and
    minimal difference of the categories.
    """
    # get the categories of the row
    target_cats = set([k for k in list(row.keys()) if k in SEMANTIC_CATEGORIES.keys() and row[k] != '-'])

    # get the categories of every row in the df
    tdf = df.copy()
    tdf['category_set'] = tdf.apply(lambda r: \
            set([k for k in list(r.keys()) if k in SEMANTIC_CATEGORIES.keys() and r[k] != '-']),
            axis=1)

    # get intersection and prioritize those with NO non-existing other-categories
    tdf['intersection'] = tdf.apply(lambda r: set.intersection(target_cats, r['category_set']), axis=1)
    tdf['intersection_len'] = tdf.apply(lambda r: len(r['intersection']), axis=1)
    tdf['difference'] = tdf.apply(lambda r: r['category_set'] - target_cats - r['intersection'], axis=1)
    # make this difference negative to be a bit hacky around the sorting
    tdf['difference_len'] = tdf.apply(lambda r: len(r['difference']), axis=1)
    tdf['sort'] = tdf.apply(lambda r: r['intersection_len'] - r['difference_len'], axis=1)
    # sort by max intersection len and minimum difference overlap
    tdf = tdf.sort_values('sort', ascending=False)

    # take top n and choose randomly from it
    tdf = tdf[tdf['video_fn'] != row['video_fn']]   # remove our exact match
    ret = get_closest_timing_to_row(row, tdf[:10])  # send our top 10 candidates
    return ret, ret['sort']


def get_set_overlap(r1, r2):
    r1_cats = get_set_categories(r1)
    r2_cats = get_set_categories(r2)
    int = set.intersection(r1_cats, r2_cats)
    diff = r2_cats - r1_cats - int
    return int, diff


# from DF, builds adjacency matrix of co-occurrences of categories
def build_adjacency_matrix(df):
    names = list(SEMANTIC_CATEGORIES.keys())
    n_cats = len(names)
    M = [[0 for x in range(n_cats)] for y in range(n_cats)]

    details = [[0 for x in range(n_cats)] for y in range(n_cats)]

    for i in range(n_cats):
        cur_key = names[i]
        co_counts = []
        deets = []
        for j in range(n_cats):
            #print('j == %s' % j)
            if i == j:
                co_counts.append(0)
                deets.append([])
            else:
                j_key = names[j]
                mask = (df[cur_key] != '-') & (df[j_key] != '-')
                co_occurring = df[mask]
                deets.append(list(co_occurring['PHRASE'].values))
                co_counts.append(len(co_occurring))
        details[i] = deets
        M[i] = co_counts
    return M, details


# uses chord library, which you have to pay for????? fuck that.
def build_chord_diagram(df, show_details=False, output='category_chord.html'):
    M, details = build_adjacency_matrix(df)
    names = list(SEMANTIC_CATEGORIES.keys())
    if show_details:
        Chord(M, names,
            details=details,
            colors="d3.schemeSet1",
            opacity=0.8,
            padding=0.01,
            width=600,
            label_color="#454545",
            wrap_labels=False,
            margin=100,
            credit=False,
            font_size="12px",
            font_size_large="16px",
            allow_download=True).to_html(output)
    else:
        Chord(M, names,
            colors="d3.schemeSet1",
            opacity=0.8,
            padding=0.01,
            width=600,
            label_color="#454545",
            wrap_labels=False,
            margin=100,
            credit=False,
            font_size="12px",
            font_size_large="16px",
            allow_download=True).to_html(output)


# will create clusters in form of df
def create_category_clusters(df):
    clusters = {}
    for k in SEMANTIC_CATEGORIES.keys():
        clusters[k] = {'df': None, 'len': 0}
        clusters[k]['df'] = df[df[k] != '-']
        clusters[k]['len'] = len(clusters[k]['df'])
    return clusters



# gives two gestures that have no overlapping categories
def get_fully_non_overlapping_gestures(df, max_iter=1000):
    lim = 0
    r = df.iloc[random.randint(0, len(df)-1)]
    cats = set([k for k in list(SEMANTIC_CATEGORIES.keys()) if r[k] != '-'])     # this is technically a set
    j = df.iloc[random.randint(0, len(df)-1)]
    j_cats = set([k for k in list(SEMANTIC_CATEGORIES.keys()) if j[k] != '-'])
    while cats.intersection(j_cats) and lim < max_iter:
        j = df.iloc[random.randint(0, len(df) - 1)]
        j_cats = set([k for k in list(SEMANTIC_CATEGORIES.keys()) if j[k] != '-'])
        lim += 1
    if lim == max_iter:
        return get_fully_non_overlapping_gestures(df, max_iter)
    else:
        return r, j


def assign_categories(df):
    for k in list(SEMANTIC_CATEGORIES.keys()):
        df[k] = df.apply(lambda row: [s for s in SEMANTIC_CATEGORIES[k] if s in row['PHRASE']] \
                         if [s for s in SEMANTIC_CATEGORIES[k] if s in row['PHRASE']] \
                         else np.NaN, axis=1)
    df = df.dropna(subset=list(SEMANTIC_CATEGORIES.keys()), how='all')  # toss all gestures that have none of our semantics
    df = df.fillna('-')
    return df


def get_far_gesture_by_encoding(df, r=None, exclude=[], sample=10):
    v = r['encoding']
    if len(df) < 2:
        return df.iloc[0]
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda row: np.linalg.norm(row['encoding'][0] - v[0]), axis=1)
    tdf = tdf.sort_values(by='comp_dists', ascending=False)
    tdf = tdf[:sample]      # sample from top N furthest
    si = random.randint(0, sample-1)
    if si >= len(tdf):
        return tdf.iloc[0], tdf.iloc[0]['comp_dists']   # if we can't sample from furthest N, just return furthest.
    return tdf.iloc[si], tdf.iloc[si]['comp_dists'] # guaranteed to not be the same bc


# adds Cluster key of value k to df
def add_semantic_key(df, k):
    df['cluster'] = k
    return df


def get_shallow_ontology_gesture_match(df, row):
    """
    Based on row, looks for closest match based on shallow ontology in df.
    Gets top 10 matches and chooses the one that is the most similar in length.
    """
    transcript_ont = row['shallow_ont']
    tdf = df.copy()
    tdf['ont_overlap'] = tdf.apply(lambda r: len(transcript_ont.intersection(r['shallow_ont'])), axis=1)
    tdf = tdf.sort_values('ont_overlap', ascending=False)
    tdf = tdf[tdf['video_fn'] != row['video_fn']]       # remove original
    r = get_closest_timing_to_row(row, tdf[:10])
    return r


def get_deep_ontology_gesture_match(df, row):
    """
    Based on row, looks for closest match based on shallow ontology in df.
    Gets top 10 matches and chooses the one that is the most similar in length.
    """
    transcript_ont = row['deep_ont']
    if not row['deep_ont']:         ## empty set of deep ontology
        print('NO DEEP ONTOLOGY FOUND')
        return get_shallow_ontology_gesture_match(row, df)
    tdf = df.copy()
    tdf['ont_overlap'] = tdf.apply(lambda r: len(transcript_ont.intersection(r['deep_ont'])), axis=1)
    tdf = tdf.sort_values('ont_overlap', ascending=False)
    tdf = tdf[tdf['video_fn'] != row['video_fn']]       # remove original
    r = get_closest_timing_to_row(row, tdf[:10])
    return r

# save this shit!!!
# print them if you want
outname = params.cluster_output
dfs = [clusters[k]['df'] for k in clusters.keys()]
comb_df = pd.concat([add_semantic_key(clusters[k]['df'], k) for k in clusters.keys()])

comb_df.to_pickle(outname + 'clusters.pkl')
profile_df.to_pickle(outname + 'clusters_profile.pkl')