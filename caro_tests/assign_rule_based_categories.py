import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.distance import cdist, pdist
import random
from chord import Chord


SEMANTIC_CATEGORIES = {
    'FEELINGS': ['angry', 'sad', 'happy', 'love', 'passion', 'anxious', 'stress', 'worry', 'worried', 'anger.', 'anger'],
    'REFLEXIVE': [' me', ' my', 'I'],
    'SIZE': [' big', 'small', 'simple', 'little', 'short', 'long', ' all ', 'ultimate', 'important'],
    'DIRECTION': [' up', ' down', 'left', 'right', 'top', 'bottom', 'side', 'sideways', 'aside'],
    'TIME': ['start', 'finish', 'begin', ' end', 'forever', 'recent', 'years', 'always', ' now', \
            'behind', 'before', 'after', 'as soon as', 'grew up', 'last night', 'yesterday', 'then'],
    'SEPARATION': ['more than', 'separate from', 'other', 'over there', 'aside', 'different', 'another', \
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
            margin=70,
            credit=False,
            font_size="12px",
            font_size_large="16px").to_html(output)
    else:
        Chord(M, names,
            colors="d3.schemeSet1",
            opacity=0.8,
            padding=0.01,
            width=600,
            label_color="#454545",
            wrap_labels=False,
            margin=70,
            credit=False,
            font_size="12px",
            font_size_large="16px").to_html(output)


# uses vectors to build a difference matrix
def build_distance_matrix(df):
    dist_mat = {}
    for i in range(len(df)):
        dists = {}
        v1 = df.iloc[i]['vector']
        for j in range(len(df)):
            v2 = df.iloc[j]['vector']
            dists[df.iloc[j]['SPLIT']] = np.linalg.norm(v1 - v2)
        dist_mat[df.iloc[i]['SPLIT']] = dists
    return dist_mat


def dist_lambda(row, comp_row):
    sims = 0
    for k in row.keys():
        if row[k] == '-' and comp_row[k] == '-':
            sims += 1
        elif row[k] != '-' and comp_row[k] != '-':
            sims += 1
        elif row[k] == '-':
            sims -= 1
        elif comp_row[k] == '-':
            sims -= 1
    return sims


def get_value_vector(row):
    # ugh can't get the listcomp working
    vec = []
    for k in SEMANTIC_CATEGORIES.keys():
        vec.append(len(list(row[k])) if row[k] != '-' else -1)
    return np.array(vec)


def get_avg_dist_of_vectors(vecs):
    dists = []
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            dists.append(np.linalg.norm(vecs[i] - vecs[j]))
    return np.mean(dists)


# using the vectors and distance to get nearest thing
def get_nearest_row_vectors(df, row):
    v = row['vector']
    ndf = df.copy()
    ndf['comp_dists'] = ndf.apply(lambda r: np.linalg.norm(r['vector'] - v), axis=1)
    ndf = ndf.sort_values(by='comp_dists')
    if len(ndf) == 1:               # there's only one in this category
        return ndf.iloc[0], 0
    ret_row = ndf.iloc[1]                       # highest match is probably the second one
    if ret_row['PHRASE'] != row['PHRASE']:     # but if there's a perfect match, make sure to return a different one
        return ret_row, ret_row['comp_dists']
    else:
        return ndf.iloc[0], ndf.iloc[0]['comp_dists']        # if it's a perfect match!


def get_nearest_row(df, i):
    comp_row = df.iloc[i]
    ndf = df.copy()
    ndf['comp_dists'] = ndf.apply(lambda row: dist_lambda(row, comp_row), axis=1)
    ndf = ndf.sort_values(by='comp_dists', ascending=False)
    # need to return the highest match that isn't the same phrase
    ret_row = ndf.iloc[1]                       # highest match is probably the second one
    print(comp_row['PHRASE'], ret_row['PHRASE'])
    if ret_row['PHRASE'] != comp_row['PHRASE']:     # but if there's a perfect match, make sure to return a different one
        return ret_row, ret_row['comp_dists']
    else:
        return ndf.iloc[0], ndf.iloc[0]['comp_dists']        # if it's a perfect match!


# will create clusters in form of df
def create_category_clusters(df):
    clusters = {}
    for k in SEMANTIC_CATEGORIES.keys():
        clusters[k] = {'df': None, 'len': 0}
        clusters[k]['df'] = df[df[k] != '-']
        clusters[k]['len'] = len(clusters[k]['df'])
    return clusters


# given a df of gesture/semantic with vectors, finds the maximum similarity between different gestures
def get_max_similarity_score(df):
    m_dist = 1000
    for i in range(len(df)):
        row = df.iloc[i]
        _, d = get_nearest_row_vectors(df, row)
        if d < m_dist:
            m_dist = d
    return m_dist


def get_vector_distances(vecs):
    dists = []
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            dists.append(np.linalg.norm(vecs[i] - vecs[j]))
    return dists


def get_cluster_profiles(clusters, df):
    semantic_category = []
    lengths = []
    silhouette_scores = []
    custom_silhouette_scores = []
    max_sim_scores = []
    min_sim_scores = []
    mean_sim_scores = []
    median_sim_scores = []

    for k in clusters.keys():
        semantic_category.append(k)
        l = clusters[k]['len']
        lengths.append(l)
        custom_score = get_custom_cluster_silhouette_score(clusters, k, df)
        custom_silhouette_scores.append(custom_score)
        sc = get_cluster_silhouette_score(clusters, k, df)
        silhouette_scores.append(sc)
        dists = get_vector_distances(clusters[k]['df']['vector'].values)
        max_sim_scores.append(min(dists))
        min_sim_scores.append(max(dists))
        mean_sim_scores.append(np.mean(dists))
        median_sim_scores.append(np.median(dists))
        print(k, "(%s)" % l, sc, 'min/max: %s/%s' % (min(dists), max(dists)), 'median: %s' % np.median(dists), 'mean: %s' % np.mean(dists))
        print('%s custom score: %s' % (k, custom_score))
    ret_df = pd.DataFrame(list(zip(semantic_category, lengths, silhouette_scores, custom_silhouette_scores, \
                           max_sim_scores, min_sim_scores, mean_sim_scores, median_sim_scores)),
                          columns=['semantic_category', 'length', 'silhouette_score', 'custom_silhouette_score', \
                           'min_distance', 'max_distance', 'mean_distance', 'median_distance'])
    return ret_df


# this.... isn't quite right.
def get_cluster_silhouette_score(clusters, k, df):
    vecs = clusters[k]['df']['vector'].values
    avg_within = np.mean(get_vector_distances(vecs))
    avg_between = np.mean(get_vector_distances(df['vector'].values))

    # (b-a) / max(a,b)
    # a = average intra-cluster distance (avg dist between each point in a cluster)
    # b = average inter-cluster distance (avg dist between point and next-closest)
    sil = (avg_between - avg_within) / max([avg_between, avg_within])
    return sil


"""
silhouette score doesn't really apply for overlapping clusters. I would say the closest we could do is find
the mean intra-cluster distance and the mean distance between each point in the cluster and its closest point 
that it isn't a shared cluster of... 
"""
def get_custom_cluster_silhouette_score(clusters, k, df):
    cluster_df = clusters[k]['df']
    As = []
    Bs = []
    for i in range(len(cluster_df)):
        v = cluster_df.iloc[i]['vector']
        # get average distance between this vector and all other vectors in the cluster
        a = np.mean([np.linalg.norm(v - cluster_df.iloc[j]['vector']) for j in range(len(cluster_df)) if i != j])
        # get the distance between this vector and next closest non-overlapping gesture
        b, d = get_nearest_non_overlapping_gesture(df, cluster_df.iloc[i])
        As.append(a)
        Bs.append(d)

    a = np.mean(As)
    b = np.mean(Bs)
    print('%s WITHIN cluster avg: %s' % (k, a))
    print('%s BETWEEN cluster avg: %s' % (k, b))
    sil = (b - a) / max(a, b)
    return sil


def get_nearest_non_overlapping_gesture(df, row):
    cats = set([k for k in list(SEMANTIC_CATEGORIES.keys()) if row[k] != '-'])     # this is technically a set
    nolap_df = df
    for c in cats:
        nolap_df = nolap_df[nolap_df[c] == '-']
    nolap_df['comp_dists'] = nolap_df.apply(lambda r: np.linalg.norm(r['vector'] - row['vector']), axis=1)
    nolap_df = nolap_df.sort_values(by='comp_dists', ascending=True)
    return nolap_df.iloc[0], nolap_df.iloc[0]['comp_dists']


# from cluster k, get a random pair of closest gestures
def get_random_closest_gestures_within_cluster(clusters, k):
    df = clusters[k]['df']
    r = df.iloc[random.randint(0, len(df)-1)]
    nr, d = get_nearest_row_vectors(df, r)
    return r, nr, d


def get_random_gesture_within_cluster(clusters, k):
    df = clusters[k]['df']
    ind1 = random.randint(0, len(df)-1)
    ind2 = random.randint(0, len(df)-1)
    while ind1 == ind2:
        ind2 = random.randint(0, len(df) - 1)

    r1 = df.iloc[ind1]
    r2 = df.iloc[ind2]
    d = np.linalg.norm(r1['vector'] - r2['vector'])
    return r1, r2, d


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


# given a cluster, get a transcript from that cluster,
# a nearby gesture also in that cluster,
# and a far-away gesture, not in that cluster
def get_transcript_close_and_far_gesture(clusters, k, df):
    r1, r2, d = get_random_closest_gestures_within_cluster(clusters, k)
    r3, _ = get_max_different_gesture(df, r1)
    return r1, r2, d, r3


def get_transcript_random_and_far_gesture(clusters, k, df):
    r1, r2, d = get_random_gesture_within_cluster(clusters, k)
    r3, _ = get_max_different_gesture(df, r1)
    return r1, r2, d, r3


def get_max_different_gesture(df, row):
    v = row['vector']
    ndf = df.copy()
    ndf['comp_dists'] = ndf.apply(lambda r: np.linalg.norm(r['vector'] - v), axis=1)
    ndf = ndf.sort_values(by='comp_dists', ascending=False)
    if len(ndf) == 1:               # there's only one in this category
        return ndf.iloc[0], 0
    ret_row = ndf.iloc[1]                       # highest match is probably the second one
    if ret_row['PHRASE'] != row['PHRASE']:     # but if there's a perfect match, make sure to return a different one
        return ret_row, ret_row['comp_dists']
    else:
        return ndf.iloc[0], ndf.iloc[0]['comp_dists']        # if it's a perfect match!


def assign_categories(df):
    for k in list(SEMANTIC_CATEGORIES.keys()):
        df[k] = df.apply(lambda row: [s for s in SEMANTIC_CATEGORIES[k] if s in row['PHRASE']] \
                         if [s for s in SEMANTIC_CATEGORIES[k] if s in row['PHRASE']] \
                         else np.NaN, axis=1)
    df = df.dropna(subset=list(SEMANTIC_CATEGORIES.keys()), how='all')  # toss all gestures that have none of our semantics
    df = df.fillna('-')
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', '-f', default="",
                                   help="txt file of transcript with split lines")
    parser.add_argument('--embedding_file', '-ef', default="",
                                   help="embedding file of transcript and gestures")
    parser.add_argument('--semantic_output', '-o', default="category_output.csv",
                                   help="what to name the output csv")
    parser.add_argument('--cluster_output', '-co', default="cluster_output.pkl",
                                   help="what to name the cluster output")
    params = parser.parse_args()
    f = params.file
    embeddings = params.embedding_file
    df = pd.DataFrame(columns=['PHRASE'] + list(SEMANTIC_CATEGORIES.keys()))

    if embeddings:
        df = pd.read_pickle(embeddings)
        df['PHRASE'] = df.apply(lambda row: ' '.join(row['transcript']), axis=1)
        df['SPLIT'] = df.apply(lambda row: row['fn'].split('_')[3], axis=1)
        df = assign_categories(df)

    elif f:
        with open(f, 'r') as file:
            l = file.readline()
            while l:
                splitnum = int(l.split('\t')[0]) - 1
                l = l.replace('\t', ' ').replace('\n', '')
                categories = {'SPLIT': splitnum, 'PHRASE': l}
                for k in SEMANTIC_CATEGORIES.keys():
                    matches = [s for s in SEMANTIC_CATEGORIES[k] if s in l]     # see if any semcats are in
                    if matches:                             # if there is a match in a particular category
                        categories[k] = matches             # add it to the category and what triggered it
                # print(categories)
                df = df.append(categories, ignore_index=True)
                l = file.readline()

        # clean up the data
        df = df.dropna(subset=list(SEMANTIC_CATEGORIES.keys()), how='all')      # toss all gestures that have none of our semantics
        df = df.fillna('-')

    # get the feature vector
    df['vector'] = df.apply(get_value_vector, axis=1)

    # df.to_csv(params.output)

    # get the clusters
    clusters = create_category_clusters(df)

    # print them if you want
    with open(params.cluster_output, 'w') as out:
        json.dump(clusters, out)

    # view it
    build_chord_diagram(df, show_details=False, output='category_chord.html')

    # get some stats
    profile_df = get_cluster_profiles(clusters, df)

    # and get some examples
    k = 'CATEGORICAL'

    r, nr, d_closest = get_random_closest_gestures_within_cluster(clusters, k)
    print(r['PHRASE'], ' // ', nr['PHRASE'], '(%s)' % d_closest)
    print(r['fn'], ' // ', nr['fn'])

    r1, r2, d_close, r3 = get_transcript_close_and_far_gesture(clusters, k, df)
    print('Phrase to match: %s' % r1['PHRASE'])
    print('gesture options: ')
    print(r2['fn'])
    print(r3['fn'])
    print('(nearest gesture is %s away)' % d_close)

    r4, r5, d_random, r6 = get_transcript_random_and_far_gesture(clusters, k, df)
    print('Phrase to match: %s' % r1['PHRASE'])
    print('gesture options: ')
    print(r2['fn'])
    print(r3['fn'])
    print('(nearest gesture is %s away)' % d_random)

