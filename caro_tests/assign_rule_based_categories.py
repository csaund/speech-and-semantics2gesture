import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.distance import cdist, pdist
import random
from chord import Chord
from shutil import copyfile


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


def get_random_row(df):
    row = df.sample(1).iloc[0]
    return row


def get_transcript_embedding_v_random(clusters, k, df):
    row = get_random_row(clusters[k]['df'])
    embedding_gesture = get_closest_gesture_from_row_embeddings(row, df)
    samp = df.sample(100)
    samp = samp[samp['video_fn'] != row['video_fn']]
    random_gesture = get_closest_timing_to_row(row, samp)  # control for timing
    return row, embedding_gesture, random_gesture


def get_transcript_set_v_embedding(clusters, k, df):
    row = get_random_row(clusters[k]['df'])
    set_gesture, d = get_closest_gesture_from_row_sets(row, df)
    embedding_gesture = get_closest_gesture_from_row_embeddings(row, df)
    return row, set_gesture, embedding_gesture


def get_closest_gesture_from_row_embeddings(row, df):
    v = row['encoding']
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda r: np.linalg.norm(r['encoding'][0] - v[0]), axis=1)
    tdf = tdf.sort_values(by='comp_dists', ascending=True)
    tdf = tdf[tdf['video_fn'] != row['video_fn']]   # remove our exact match
    gest = get_closest_timing_to_row(row, tdf[:10])
    return gest


def get_transcript_close_random_gesture_sets(clusters, k, df):
    row = get_random_row(clusters[k]['df'])
    # get close gesture
    close_gesture, d = get_closest_gesture_from_row_sets(row, df)
    # get random gesture
    samp = df.sample(100)
    samp = samp[samp['video_fn'] != row['video_fn']]
    random_gesture = get_closest_timing_to_row(row, samp)  # control for timing
    return row, close_gesture, random_gesture


"""
Given an experimental transcript T, the viewer distinguishes between two videos which would better
match T. These videos are split into the following goups: 
- random gesture vs. random (R) -- expect 50/50
- close gesture based on transcript embedding (E) vs. random -- expect embedding preferred
- close gesture based on shallow ontology groups (SO)  vs. random -- expect shallow preferred
- close gesture based on deep ontology groups (DO) vs. random -- expect deep preferred
- E vs. DO -- H0 is embedding preferred
- E vs. SO -- H0 is embedding preferred
- SO vs. DO -- H0 is no difference btw groups
"""


def get_transcript_gesture_match(cluster_df, full_df, matching_fxn1, matching_fxn2):
    """
    Given a sub-df to get a random gesture from, gets a random gesture
    and two gestures from the full DF according to the matching functions passed.
    For a description of potential matches see the above comment.
    """
    row = get_random_row(cluster_df)
    t0 = matching_fxn1(row, full_df)
    t1 = matching_fxn2(full_df)
    return row, t0, t1


# todo implement this
def get_shallow_ontology_gesture_match(row, df):
    """
    Based on row, looks for closest match based on shallow ontology in df.
    Gets top 10 matches and chooses the one that is the most similar in length.
    """
    return row


# todo implement
def get_deep_ontology_gesture_match(row, df):
    """
    Based on row, looks for closest match based on deep ontology in df.
    Gets top 10 matches and chooses the one that is the most similar in length.
    """
    return row


def get_set_categories(row):
    return set([k for k in list(row.keys()) if k in SEMANTIC_CATEGORIES.keys() and row[k] != '-'])


def get_set_overlap(r1, r2):
    r1_cats = get_set_categories(r1)
    r2_cats = get_set_categories(r2)
    int = set.intersection(r1_cats, r2_cats)
    diff = r2_cats - r1_cats - int
    return int, diff


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


# given a df and a row, gets the row from df closest to length of original row
def get_closest_timing_to_row(row, df):
    match_time = row['time_length']
    tdf = df.copy()
    tdf['timing_diff'] = tdf.apply(lambda r: abs(r['time_length'] - match_time), axis=1)
    tdf = tdf.sort_values('timing_diff')
    return tdf.iloc[0]


# TODO fix the pipeline so this hacky garbage doesn't need to be here.
def get_video_fn_from_json_fn(jfn, vid_dir):
    vid_files = os.listdir(vid_dir)
    splits = jfn.split('_')
    # need to match the first 4 splits
    candidates = [v for v in vid_files if v.split('_')[:4] == splits[:4]]
    vid_fn = [f for f in candidates if f.endswith('.mp4')]
    if len(vid_fn) > 1:
        vid_fn = [f for f in vid_fn if 'sound' not in vid_fn]
    if len(vid_fn) < 1:
        print('couldnt get a video fn for ', jfn)
        print(vid_fn)
        return ''
    return vid_fn[0]


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
    if len(vecs) == 0 or len(vecs) == 1:
        return [0]
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
    if len(df) < 2:
        print("LEN DF: ", len(df))
        return df.iloc[0], df.iloc[0], 0
    r = df.iloc[random.randint(0, len(df)-1)]
    nr, d = get_nearest_row_vectors(df, r)
    return r, nr, d


def get_random_gesture_within_cluster(clusters, k):
    df = clusters[k]['df']
    if len(df) < 2:
        return df.iloc[0], df.iloc[0], 0
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


def get_nearest_gesture_by_encoding_dist(df, r):
    v = r['encoding']
    if len(df) < 2:
        print('WEE BABBY DF FOUND')
        return df.iloc[0], 0
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda row: np.linalg.norm(row['encoding'][0] - v[0]), axis=1)
    tdf = tdf.sort_values(by='comp_dists', ascending=True)
    if tdf.iloc[0]['PHRASE'] == r['PHRASE']:
        return tdf.iloc[1], tdf.iloc[1]['comp_dists']
    else:
        return tdf.iloc[0], tdf.iloc[0]['comp_dists']      # we got a perfect match, this would only happen if exact same transcript.


def get_far_gesture_by_encoding(df, r, sample=10):
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


def get_transcript_close_and_far_embedding(clusters, k, df):
    cdf = clusters[k]['df']
    ind1 = random.randint(0, len(cdf)-1)
    r1 = cdf.iloc[ind1]
    r2, d2 = get_nearest_gesture_by_encoding_dist(df, r1)   # cluster agnostic!!
    r3, _ = get_far_gesture_by_encoding(df, r1)            # cluster agnostic!!
    return r1, r2, d2, r3


def get_transcript_random_and_far_embedding(clusters, k, df):
    cdf = clusters[k]['df']
    ind1 = random.randint(0, len(cdf)-1)
    ind2 = random.randint(0, len(df)-1)
    r1 = cdf.iloc[ind1]
    r2 = df.iloc[ind2]
    # cluster agnostic!!
    d = np.linalg.norm(r1['encoding'][0] - r2['encoding'][0])
    r3, _ = get_far_gesture_by_encoding(df, r1)            # cluster agnostic!!
    return r1, r2, d, r3


# adds Cluster key of value k to df
def add_semantic_key(df, k):
    df['cluster'] = k
    return df


def get_time_lambda(row):
    sp = row['fn'].split('_')
    t0 = sp[5]
    t1 = sp[6].split('.json')[0]
    if t1 == '+':
        return 10            # todo lol this is just absolutely made up.
    else:
        return float(t1) - float(t0)


def get_embedding_distances(r1, r2):
    """
    given two rows, gets the distance between their vector embedding
    """
    v1 = r1['encoding']
    v2 = r2['encoding']
    return np.linalg.norm(v1[0] - v2[0])


def write_fns_in_df_to_folder(df, host_dirname="combo_fillers", dirname="stimuli_fns"):
    """
    For convenience when working with all these files to upload only relevant stimuli
    dirname is directory,
    df contains video fns of videos to be moved to dirname.
    """
    src_fns = [os.path.join('speech-and-semantics2gesture', 'Splits', host_dirname, f) for f in list(df['video_fn']) if f != '']
    dst_fns = [os.path.join('speech-and-semantics2gesture', dirname, f) for f in list(df['video_fn']) if f != '']
    assert(len(src_fns) == len(dst_fns))
    for src, dest in list(zip(src_fns, dst_fns)):
        copyfile(src, dest)


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
    parser.add_argument('--video_dir', '-cd', default="Splits/combo_fillers",
                                   help="file where all the video fns are")
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

    # TODO ah yes here is the hacky garbage.
    # TEMP for testing ONLY
    video_dir = os.path.join("speech-and-semantics2gesture", "Splits", "combo_fillers")
    df['video_fn'] = df.apply(lambda row: get_video_fn_from_json_fn(row['fn'], video_dir), axis=1)

    # TODO exclude gestures that are too short?
    df['time_length'] = df.apply(lambda row: get_time_lambda(row), axis=1)
    df = df[df['time_length'] >= 1.8]   # arbitrary...

    # get the clusters
    clusters = create_category_clusters(df)

    # view it
    build_chord_diagram(df, show_details=False, output='category_chord.html')

    # get some stats
    profile_df = get_cluster_profiles(clusters, df)

    COLS = ['randomise_trials', 'display', 'transcripts', 'video1_fn', 'video2_fn',
            'video_relation', 'category', 'correct_video',
            'video1_transcript', 'video2_transcript', 'vidA_overlap', 'vidB_overlap',
            'transcript_categories', 'transcript_length', 'vidA_length', 'vidB_length',
            'vidA_embedding_distance', 'vidB_embedding_distance',
            'vidA_set_diff', 'vidB_set_diff']

    exp_df = pd.DataFrame(columns=COLS)
    # build up a df of examples
    for k in SEMANTIC_CATEGORIES.keys():
        print("WORKING ON CATEGORY: ", k)
        # get some random closest gestures within a given cluster
        if len(clusters[k]['df']) < 2:
            print('NOT ENOUGH GESTURES FOR CLUSTER ', k)
            continue

        ## TODO: ensure no duplicate videos!!!!
        fxns = [
            get_closest_gesture_from_row_embeddings,
            get_shallow_ontology_gesture_match,
            get_deep_ontology_gesture_match
        ]
        """
        Given an experimental transcript T, the viewer distinguishes between two videos which would better
        match T. These videos are split into the following goups: 
        - random gesture vs. random (R) -- expect 50/50
        - close gesture based on transcript embedding (E) vs. random -- expect embedding preferred
        - close gesture based on shallow ontology groups (SO)  vs. random -- expect shallow preferred
        - close gesture based on deep ontology groups (DO) vs. random -- expect deep preferred
        - E vs. DO -- H0 is embedding preferred
        - E vs. SO -- H0 is embedding preferred
        - SO vs. DO -- H0 is no difference btw groups
        """
        predicted_gesture = []          # either 'A' or 'B'
        video_relation = []

        T_rows = []
        A_rows = []
        B_rows = []

        for f1 in fxns:
            # first the fxn vs. random
            r0, r1, r2 = get_transcript_gesture_match(clusters[k]['df'], df, f1, get_random_row)
            indexes = [0, 1]
            order = [0, 1]
            vids = [(r1, 0), (r2, 1)]     # we know r1 is the 'good' one
            random.shuffle(vids)
            predicted_i = 'B' if vids[0][1] else 'A'
            predicted_gesture.append(predicted_i)
            T_rows.append(r0)
            A_rows.append(vids[0][0])
            B_rows.append(vids[1][0])
            video_relation.append(str(f1.__name__ + '_random'))

            for f2 in fxns:             # then the fxn vs. the other fxns
                if f1.__name__ == f2.__name__:
                    continue
                r3, r4, r5 = get_transcript_gesture_match(clusters[k]['df'], df, f1, f2)
                indexes = [0, 1]
                order = [0, 1]
                vids = [(r4, 0), (r5, 1)]  # we know r4 is the 'good' one
                random.shuffle(vids)
                predicted_i = 'B' if vids[0][1] else 'A'
                predicted_gesture.append(predicted_i)
                T_rows.append(r3)
                A_rows.append(vids[0][0])
                B_rows.append(vids[1][0])
                video_relation.append(str(f1.__name__ + '_' + f2.__name__))

        # format it for the df
        videoA_fn = [r['video_fn'] for r in A_rows]
        videoB_fn = [r['video_fn'] for r in B_rows]
        transcripts = [r['PHRASE'] for r in T_rows]
        videoA_transcript = [r['PHRASE'] for r in A_rows]
        videoB_transcript = [r['PHRASE'] for r in B_rows]
        vidA_shallow_ontology = [r['shallow_ont'] for r in A_rows]
        vidB_shallow_ontology = [r['shallow_ont'] for r in B_rows]
        vidA_deep_ontology = [r['deep_ont'] for r in A_rows]
        vidB_deep_ontology = [r['deep_ont'] for r in B_rows]
        transcript_shallow = [r['shallow_ont'] for r in T_rows]
        transcript_deep = [r['deep_ont'] for r in T_rows]
        transcript_length = [r['time_length'] for r in T_rows]
        vidA_length = [r['time_length'] for r in A_rows]
        vidB_length = [r['time_length'] for r in B_rows]
        vidA_embedding_distances = [get_embedding_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
        vidB_embedding_distances = [get_embedding_distances(T_rows[i], B_rows[i]) for i in range(len(B_rows))]

        randomise_trials = [random.randint(1, 9)]
        display = ['video_participant_view'] * 9
        category = [k] * 9

        """
            COLS = ['randomise_trials', 'display', 'transcripts', 'video1_fn', 'video2_fn',
                    'video_relation', 'category', 'correct_video',
                    'video1_transcript', 'video2_transcript', 'vidA_overlap', 'vidB_overlap',
                    'transcript_categories', 'transcript_length', 'vidA_length', 'vidB_length',
                    'vidA_embedding_distance', 'vidB_embedding_distance']
        """
        ndf = pd.DataFrame(list(zip(randomise_trials, display, transcripts, video1_fn, video2_fn,
                                    video_relation, category, correct_gesture,
                                    video1_transcript, video2_transcript, vidA_overlap, vidB_overlap,
                                    transcript_categories, transcript_length, vidA_length, vidB_length,
                                    vidA_embedding_distances, vidB_embedding_distances,
                                    vidA_set_diff, vidB_set_diff)),
                           columns=COLS)
        exp_df = exp_df.append(ndf)

    # create an experimental block?
    exp_df.sample(25)


    # save this shit!!!
    # print them if you want
    outname = params.cluster_output
    dfs = [clusters[k]['df'] for k in clusters.keys()]
    comb_df = pd.concat([add_semantic_key(clusters[k]['df'], k) for k in clusters.keys()])

    comb_df.to_pickle(outname + 'clusters.pkl')
    profile_df.to_pickle(outname + 'clusters_profile.pkl')
