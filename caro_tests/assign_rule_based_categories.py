import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.distance import cdist, pdist
import random
from chord import Chord
from shutil import copyfile
# from caro_tests.ontology_generator import CII
from tqdm import tqdm
import string
tqdm.pandas()
import spacy

import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
nlp = spacy.load("en_core_web_lg")


def compare_all_sentence_functions(df, row=None, fxns=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        print("Phrase to match: ")
        print(row['PHRASE'])
        print("=======================")
    if not fxns:
        fxns = [
            get_closest_gesture_from_row_embeddings,
            get_most_similar_sentence_USE,
            get_ontology_pos_match,
            get_ontology_set_match,
            get_ontology_sequence_match,
            get_random_row,
            get_least_similar_sentence_USE,
            get_farthest_match_embedding
        ]
    for f in fxns:
        name = f.__name__
        tr = f(df, row)
        print(f'{name}: \n {tr["PHRASE"]}')
        print("------------------------")


def get_random_row(df, row=None, exclude=[]):
    if row is not None:
        samp = df.sample(25)
        samp = samp[samp['video_fn'] != row['video_fn']]
        if exclude:
            exclude_fns = [r['video_fn'] for r in exclude]
            samp = samp[~samp['video_fn'].isin(exclude_fns)]
        r = get_closest_timing_to_row(row, samp)
        return r
    return df.sample(1).iloc[0]


def get_closest_gesture_from_row_embeddings(df, row=None, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        # print(row['PHRASE'])
    v = row['encoding']
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda r: np.linalg.norm(r['encoding'][0] - v[0]), axis=1)
    gest = sort_exclude_timing(tdf, row, by='comp_dists', ascending=True, exclude=exclude, n=8)
    return gest


# takes [set(), set()] and returns superset
# ex [{A, B}, {C}, {D}]
# returns {A, B, C, D}
def combine_list_of_sets(S):
    super = set()
    for s in S:
        super.update(s)
    return super


def sort_exclude_timing(df, row, by, ascending=False, exclude=[], n=10):
    tdf = df.sort_values(by=by, ascending=ascending)
    tdf = tdf[tdf['video_fn'] != row['video_fn']]   # remove our exact match
    if exclude:
        exclude_fns = [r['video_fn'] for r in exclude]
        tdf = tdf[~tdf['video_fn'].isin(exclude_fns)]
    gest = get_closest_timing_to_row(row, tdf[:n])
    return gest


def get_ontology_pos_overlaps(r1, r2, ont_level='ont_sequence'):
    # for the different pos in gestures,
    # get the % of overlaps within those pos.
    r1_pos = [r['pos'] for r in r1['parse']]
    r2_pos = [r['pos'] for r in r2['parse']]
    overlaps = []
    for pos in r1_pos:
        all_r1 = [r['word'] for r in r1['parse'] if r['pos'] == pos]
        all_r2 = [r['word'] for r in r2['parse'] if r['pos'] == pos]
        r1_pos_onts = combine_list_of_sets([r[0] for r in r1[ont_level] if r[1] in all_r1])
        r2_pos_onts = combine_list_of_sets([r[0] for r in r2[ont_level] if r[1] in all_r2])
        # TODO do intersection here, and maybe difference as well?
        intersect = len(r1_pos_onts.intersection(r2_pos_onts))
        o1_diff = len(r1_pos_onts - r2_pos_onts)
        o2_diff = len(r2_pos_onts - r1_pos_onts)
        overlaps.append(intersect - o1_diff - o2_diff)
    return sum(overlaps)


def get_ontology_pos_match(df, row=None, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        print(row['PHRASE'])
    tdf = df.copy()
    tdf['set_overlaps'] = tdf.apply(lambda r: get_ontology_pos_overlaps(row, r), axis=1)
    gest = sort_exclude_timing(tdf, row, by='set_overlaps', ascending=False, exclude=exclude, n=5)
    return gest


def get_extont_pos_match(df, row=None, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        print(row['PHRASE'])
    tdf = df.copy()
    tdf['set_overlaps'] = tdf.apply(lambda r: get_ontology_pos_overlaps(row, r, ont_level='extont_sequence'), axis=1)
    gest = sort_exclude_timing(tdf, row, by='set_overlaps', ascending=False, exclude=exclude, n=5)
    return gest


def get_num_ontology_overlaps(r1, r2):
    ont1 = set()
    ont2 = set()
    for o in r1['ont_sequence']:
        ont1 = ont1.union(o[0])         # todo separate this bc has word?
    for o in r2['ont_sequence']:        # todo want to make this just operate on sets?
        ont2 = ont2.union(o[0])

    intersect = len(ont1.intersection(ont2))
    o1_diff = len(ont1 - ont2)
    o2_diff = len(ont2 - ont1)
    return intersect - o1_diff - o2_diff


# don't bother with sequence matching, just match up the categories that
# appear in each one
def get_ontology_set_match(df, row, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
    tdf = df.copy()
    tdf['set_overlaps'] = tdf.apply(lambda r: get_num_ontology_overlaps(row, r), axis=1)
    gest = sort_exclude_timing(tdf, row, by='set_overlaps', ascending=False, exclude=exclude, n=8)
    return gest


def get_farthest_match_embedding(df, row, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        # print(row['PHRASE'])
    v = row['encoding']
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda r: np.linalg.norm(r['encoding'][0] - v[0]), axis=1)
    gest = sort_exclude_timing(tdf, row, by='comp_dists', ascending=False, exclude=exclude, n=8)
    return gest


def get_least_similar_sentence_USE(df, row, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        # print(row['PHRASE'])
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda r: np.linalg.norm(r['use_embedding'] - row['use_embedding']), axis=1)
    gest = sort_exclude_timing(tdf, row, by='comp_dists', ascending=False, exclude=exclude, n=8)
    return gest


def check_use_vs_bert(df):
    row = df.sample(1).iloc[0]
    print("PHRASE TO MATCH: \n", row['PHRASE'])
    r1 = get_closest_gesture_from_row_embeddings(df, row)
    r2 = get_most_similar_sentence_USE(df, row)
    print("==============================")
    print('embedding match: \n', r1['PHRASE'])
    print('use match: \n', r2['PHRASE'])


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
def get_overlap_sim_starting_at_ij(S1, S2, i, j):
    tot = 0
    while i != len(S1) and j != len(S2):
        tot += len(S1[i].intersection(S2[j]))
        i += 1
        j += 1
    return tot


def calculate_set_sequence_similarity(S1, S2):
    m = 0
    for i in range(len(S1)):
        for j in range(len(S2)):
            overlap_pos = get_overlap_sim_starting_at_ij(S1, S2, i, j)
            m = max(m, overlap_pos)
    return m


def get_total_ontologies(sequence):
    tot = 0
    if not sequence:
        return 0
    for s in sequence:
        tot += len(s)
    return tot


def get_ontology_distances(r1, r2, ont_level='ont_sequence'):
    r1_ont_only = [s[0] for s in r1[ont_level]]
    r2_ont_only = [s[0] for s in r2[ont_level]]
    sim = calculate_set_sequence_similarity(r1_ont_only, r2_ont_only)
    tot = min(get_total_ontologies(r1_ont_only), get_total_ontologies(r2_ont_only))
    if not tot:
        print('0 ont sequence for either ', r1['video_fn'], ' or ', r2['video_fn'])
        return 0
    return float(sim / tot)


def get_transcript_gesture_match(df, matching_fxn1, matching_fxn2):
    """
    Given a sub-df to get a random gesture from, gets a random gesture
    and two gestures from the full DF according to the matching functions passed.
    For a description of potential matches see the above comment.
    """
    row = get_random_row(df)
    t0 = matching_fxn1(df, row)
    t1 = matching_fxn2(df, row, exclude=[row, t0])
    return row, t0, t1


def get_ontology_sequence_match(df, row, exclude=[]):
    transcript_ont = row['ont_sequence']
    tdf = df.copy()
    tdf['sequence_val'] = tdf.apply(lambda r: get_ontology_distances(row, r), axis=1)
    tdf = tdf.sort_values('sequence_val', ascending=False)
    tdf = tdf[tdf['video_fn'] != row['video_fn']]       # remove original
    if exclude:
        exclude_fns = [r['video_fn'] for r in exclude]
        tdf = tdf[~tdf['video_fn'].isin(exclude_fns)]
    r = get_closest_timing_to_row(row, tdf[:10])
    return r


def get_extont_sequence_match(df, row, exclude=[]):
    transcript_ont = row['extont_sequence']
    tdf = df.copy()
    tdf['sequence_val'] = tdf.apply(lambda r: get_ontology_distances(row, r, ont_level='extont_sequence'), axis=1)
    tdf = tdf.sort_values('sequence_val', ascending=False)
    tdf = tdf[tdf['video_fn'] != row['video_fn']]       # remove original
    if exclude:
        exclude_fns = [r['video_fn'] for r in exclude]
        tdf = tdf[~tdf['video_fn'].isin(exclude_fns)]
    r = get_closest_timing_to_row(row, tdf[:10])
    return r


def get_ontology_sequence(row, cere, feat_set=None):
    p = row['PHRASE']
    if not feat_set:
        feat_set = cere.generate(p)
    words = p.rstrip().split(' ')
    words = [s.translate(str.maketrans('', '', string.punctuation)) for s in words]
    ont_sequence = []
    ont_words = []
    for w in words:
        if w in feat_set.keys():
            if 'Ont' in feat_set[w].keys():
                ont_sequence.append(feat_set[w]['Ont'][1])
                ont_words.append(w)
            elif 'ExtOnt' in feat_set[w].keys():        # if there's no ontology, use the extont.
                ont_sequence.append(feat_set[w]['ExtOnt'][1])
                ont_words.append(w)
    return list(zip(ont_sequence, ont_words))


# extra ont
def get_extont_sequence(row, cere, feat_set=None):
    p = row['PHRASE']
    if not feat_set:
        feat_set = cere.generate(p)
    words = p.rstrip().split(' ')
    words = [s.translate(str.maketrans('', '', string.punctuation)) for s in words]
    ont_sequence = []
    ont_words = []
    for w in words:
        if w in feat_set.keys():
            ontologies = feat_set[w]
            if 'ExtOnt' in ontologies.keys():
                original_ext_ont = ontologies['ExtOnt'][1]
                ext_ont = original_ext_ont
                # if items in extont appear in ontology, dont add them!
                extont_cats = [k.split(':')[0] for k in list(original_ext_ont)]
                if 'Ont' in ontologies.keys():
                    original_ont = ontologies['Ont'][1]
                    ont_cats = [k.split(':')[0] for k in list(original_ont)]
                    ext_ont = original_ont
                    extra_keys = [k for k in extont_cats if k not in ont_cats]
                    extra_keys += 'type'
                    for ek in extra_keys:
                        original_keys = list(original_ext_ont)
                        updates = [o for o in original_keys if ek in o]
                        ext_ont.update(updates)
                ont_sequence.append(ext_ont)
                ont_words.append(w)
            elif 'Ont' in feat_set[w].keys():
                original_ont = ontologies['Ont'][1]
                ont_sequence.append(original_ont)
                ont_words.append(w)
    return list(zip(ont_sequence, ont_words))


def get_hypernyms(row, cere, feat_set=None):
    p = row['PHRASE']
    if not feat_set:
        feat_set = cere.generate(p)
    words = p.rstrip().split(' ')
    words = [s.translate(str.maketrans('', '', string.punctuation)) for s in words]
    hypernym_sequence = []
    hypernym_words = []
    for w in words:
        if w in feat_set.keys():
            if 'Hyper_Synonyms' in feat_set[w].keys():
                hypernym_sequence.append(feat_set[w]['Hyper_Synonyms'][1])
                hypernym_words.append(w)
    return list(zip(hypernym_sequence, hypernym_words))


def get_parsed_sentence(row, cere, feat_set=None):
    p = row['PHRASE']
    if not feat_set:
        feat_set = cere.generate(p)
    doc = nlp(p)
    word_seq = []
    for token in doc:
        if token.text in feat_set.keys():
            word_seq.append({
                'word': token.text,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_
            })
    return word_seq


def get_wn_attributes_for_df(row, cere):
    p = row['PHRASE']
    feat_set = cere.generate(p)
    ont_sequence = get_ontology_sequence(row, cere, feat_set=feat_set)
    extont_sequence = get_extont_sequence(row, cere, feat_set=feat_set)
    hypernyms = get_hypernyms(row, cere, feat_set=feat_set)
    parsed_sentence = get_parsed_sentence(row, cere, feat_set=feat_set)
    return ont_sequence, extont_sequence, hypernyms, parsed_sentence


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


def get_time_lambda(row):
    sp = row['fn'].split('_')
    t0 = sp[5]
    t1 = sp[6].split('.json')[0]
    if t1 == '+':
        return 10            # todo lol this is just absolutely made up.
    else:
        return float(t1) - float(t0)


def write_fns_in_df_to_folder(df, host_dirname="combo_fillers", dirname="stimuli_fns"):
    """
    For convenience when working with all these files to upload only relevant stimuli
    dirname is directory,
    df contains video fns of videos to be moved to dirname.
    """
    hostdir = os.path.join('speech-and-semantics2gesture', 'Splits', host_dirname)
    src_fns = [os.path.join(hostdir, f) for f in os.listdir(hostdir) if (f.endswith('mp4') and 'sound' not in f)]
    dst_fns = [os.path.join(hostdir, dirname, f) for f in os.listdir(hostdir) if (f.endswith('mp4') and 'sound' not in f)]
    assert(len(src_fns) == len(dst_fns))
    for src, dest in list(zip(src_fns, dst_fns)):
        copyfile(src, dest)


def insert_same_video(ind, df):
    r = df.iloc[ind]
    d = r.to_dict()
    d['videoA_fn'] = d['videoB_fn']
    d['video_relation'] = 'CONTROL_SAME_VIDEO'
    line = pd.DataFrame(d, index=[ind+1])
    df2 = pd.concat([df.iloc[:ind], line, df.iloc[ind:]]).reset_index(drop=True)
    return df2


def insert_missing_video(ind, df):
    r = df.iloc[ind]
    d = r.to_dict()
    d['videoA_fn'] = 'broken_link.mp4'
    d['video_relation'] = 'CONTROL_BROKEN_VIDEO'
    line = pd.DataFrame(d, index=[ind+1])
    df2 = pd.concat([df.iloc[:ind], line, df.iloc[ind:]]).reset_index(drop=True)
    return df2


# add n rows to beginning for intro
def buffer_intro_rows(df, n=1, view_name='Intro'):
    fn1 = 'NaturalTalking_015_split_219_frame_27995_28139.mp4'
    fn2 = 'NaturalTalking_015_split_64_frame_7409_7553.mp4'
    t = 'that is the last thing that I wanted'
    t2 = 'and there was an older gentleman who said'
    data = {
        'videoA_fn': [fn1] * n,
        'videoB_fn': [fn2] * n,
        'videoA_transcript': [t2] * n,
        'transcripts': [t] * n,
        'predicted_video': ['broken_link.mp4'] * n,
        'display': [view_name] * n
    }
    ndf = pd.DataFrame(data)
    return pd.concat([ndf, df]).reset_index(drop=False)


def buffer_debrief_rows(df, n=1):
    data = {
        'display': ['debrief'] * n
    }
    ndf = pd.DataFrame(data)
    return pd.concat([df, ndf])


def get_embedding_distances(r1, r2):
    """
    given two rows, gets the distance between their vector embedding
    """
    v1 = r1['encoding']
    v2 = r2['encoding']
    return np.linalg.norm(v1[0] - v2[0])


def add_use_embeddings(df):
    print('generating embeddings for all phrases (this can take awhile).')
    df['use_embedding'] = embed(list(df['PHRASE'].values))
    # now turn it into a vector instead of a tuble of tensors?
    df['use_embedding'] = df.apply(lambda row: np.array([t.numpy() for t in row['use_embedding']]), axis=1)
    return df


def get_most_similar_sentence_USE(df, row=None, exclude=[]):
    if row is None:
        row = df.sample(1).iloc[0]
        # print(row['PHRASE'])
    tdf = df.copy()
    tdf['comp_dists'] = tdf.apply(lambda r: np.linalg.norm(r['use_embedding'] - row['use_embedding']), axis=1)
    gest = sort_exclude_timing(tdf, row, by='comp_dists', ascending=True, exclude=exclude, n=8)
    return gest


def get_USE_distances(r1, r2):
    return np.linalg.norm(r1['use_embedding'] - r2['use_embedding'])


# does the list of functions ONLY against actual and random gestures.
def get_actual_random_experimental_df(data_df, view_name='video_participant_view'):
    COLS = ['randomise_trials', 'display', 'transcripts', 'videoA_fn', 'videoB_fn',
            'video_relation', 'category', 'predicted_video',
            'videoA_transcript', 'videoB_transcript',
            # 'vidA_shallow_ont', 'vidB_shallow_ont', 'vidA_deep_ont', 'vidB_deep_ont',
            # 'transcript_shallow', 'transcript_deep',
            'transcript_length',
            'vidA_length', 'vidB_length',
            'vidA_embedding_distance', 'vidB_embedding_distance',
            'ShowProgressBar', 'A_function', 'B_function',
            'A_ontology_match', 'B_ontology_match']
    # build up a df of examples

    fxns = [
        get_closest_gesture_from_row_embeddings,
        get_most_similar_sentence_USE,
        get_ontology_pos_match,
        get_ontology_set_match,
        get_extont_pos_match,
        get_ontology_sequence_match,
        get_extont_sequence_match,
        get_least_similar_sentence_USE,
        get_farthest_match_embedding
    ]


    predicted_video = []  # either 'A' or 'B'
    video_relation = []

    T_rows = []
    A_rows = []
    B_rows = []
    a_fxns = []
    b_fxns = []

    # add an actual v random
    r0, r1, r2 = get_transcript_gesture_match(data_df, get_random_row, get_random_row)
    vids = [(r0, 0), (r2, 1)]  # we predict r0 in this case
    random.shuffle(vids)
    predicted_i = 'A' if vids[0][1] == 0 else 'B'
    predicted_video.append(predicted_i)
    a_fxns.append(['original_gesture' if predicted_i == 'A' else 'random'])
    b_fxns.append(['original_gesture' if predicted_i == 'B' else 'random'])
    T_rows.append(r0)
    A_rows.append(vids[0][0])
    B_rows.append(vids[1][0])
    video_relation.append('original_gesture_v_random')

    for i in tqdm(range(len(fxns))):
        f1 = fxns[i]
        # first the fxn vs. random
        r0, r1, r2 = get_transcript_gesture_match(data_df, f1, get_random_row)

        vids = [(r1, 0), (r2, 1)]  # we know r1 is the 'good' one
        random.shuffle(vids)
        predicted_i = 'A' if vids[0][1] == 0 else 'B'
        predicted_video.append(predicted_i)
        a_fxns.append([f1.__name__ if predicted_i == 'A' else 'get_random_row'])
        b_fxns.append([f1.__name__ if predicted_i == 'B' else 'get_random_row'])
        T_rows.append(r0)
        A_rows.append(vids[0][0])
        B_rows.append(vids[1][0])
        video_relation.append(str(f1.__name__ + '_v_random'))

        # then the fxn vs. actual
        r0, r1, r2 = get_transcript_gesture_match(data_df, f1, get_random_row)
        vids = [(r0, 0), (r1, 1)]  # we predict r0 in this case
        random.shuffle(vids)
        predicted_i = 'A' if vids[0][1] == 0 else 'B'
        predicted_video.append(predicted_i)
        a_fxns.append(['original_gesture' if predicted_i == 'A' else f1.__name__])
        b_fxns.append(['original_gesture' if predicted_i == 'B' else f1.__name__])
        T_rows.append(r0)
        A_rows.append(vids[0][0])
        B_rows.append(vids[1][0])
        video_relation.append(str(f1.__name__ + '_v_original_gesture'))

        # format it for the df
    videoA_fn = [r['video_fn'] for r in A_rows]
    videoB_fn = [r['video_fn'] for r in B_rows]
    transcripts = [r['PHRASE'] for r in T_rows]
    videoA_transcript = [r['PHRASE'] for r in A_rows]
    videoB_transcript = [r['PHRASE'] for r in B_rows]
    transcript_length = [r['time_length'] for r in T_rows]
    vidA_length = [r['time_length'] for r in A_rows]
    vidB_length = [r['time_length'] for r in B_rows]
    vidA_embedding_distances = [get_embedding_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidB_embedding_distances = [get_embedding_distances(T_rows[i], B_rows[i]) for i in range(len(B_rows))]
    vidA_ontology_match = [get_ontology_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidB_ontology_match = [get_ontology_distances(T_rows[i], B_rows[i]) for i in range(len(B_rows))]

    randomise_trials = [random.randint(1, len(A_rows))] * len(A_rows)
    display = [view_name] * len(A_rows)
    show_progress = [1] * len(A_rows)
    category = [None] * len(A_rows)

    exp_df = pd.DataFrame(list(zip(randomise_trials, display, transcripts, videoA_fn, videoB_fn,
                                   video_relation, category, predicted_video,
                                   videoA_transcript, videoB_transcript,
                                   # vidA_shallow_ontology, vidB_shallow_ontology, vidA_deep_ontology, vidB_deep_ontology,
                                   # transcript_shallow, transcript_deep,
                                   transcript_length,
                                   vidA_length, vidB_length,
                                   vidA_embedding_distances, vidB_embedding_distances,
                                   show_progress, a_fxns, b_fxns,
                                   vidA_ontology_match, vidB_ontology_match)),
                          columns=COLS)
    return exp_df



def get_likert_df(data_df, view_name='video_participant_view'):
    COLS = ['randomise_trials', 'display', 'transcripts', 'videoA_fn',
            'video_relation', 'category',
            'videoA_transcript',
            # 'vidA_shallow_ont', 'vidB_shallow_ont', 'vidA_deep_ont', 'vidB_deep_ont',
            # 'transcript_shallow', 'transcript_deep',
            'transcript_length',
            'vidA_length',
            'vidA_embedding_distance', 'vidA_USE_distances',
            'ShowProgressBar', 'A_function',
            'A_ontology_match', 'A_extontology_match',
            'A_ontpos_match', 'A_extontpos_match']
    # build up a df of examples

    fxns = [
        get_closest_gesture_from_row_embeddings,
        get_most_similar_sentence_USE,
        get_ontology_pos_match,
        get_ontology_set_match,
        get_extont_pos_match,
        get_ontology_sequence_match,
        get_extont_sequence_match
    ]

    predicted_video = []  # either 'A' or 'B'
    video_relation = []

    T_rows = []
    A_rows = []
    a_fxns = []

    # add an actual
    r0, r1, r2 = get_transcript_gesture_match(data_df, get_random_row, get_random_row)
    a_fxns.append(['original_gesture'])
    T_rows.append(r0)
    A_rows.append(r0)
    video_relation.append('original_gesture')

    # add a random
    a_fxns.append(['random'])
    T_rows.append(r1)
    A_rows.append(r2)
    video_relation.append('random')

    # add one per fxn
    for f in fxns:
        # first the fxn vs. random
        r0, r1, r2 = get_transcript_gesture_match(data_df, f, get_random_row)
        a_fxns.append([f.__name__])
        T_rows.append(r0)
        A_rows.append(r1)
        video_relation.append(f.__name__)

        # format it for the df
    videoA_fn = [r['video_fn'] for r in A_rows]
    transcripts = [r['PHRASE'] for r in T_rows]
    videoA_transcript = [r['PHRASE'] for r in A_rows]
    transcript_length = [r['time_length'] for r in T_rows]
    vidA_length = [r['time_length'] for r in A_rows]
    vidA_embedding_distances = [get_embedding_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_USE_distances = [get_USE_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_ontology_match = [get_ontology_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_extontology_match = [get_ontology_distances(T_rows[i], A_rows[i], ont_level='extont_sequence') for i in range(len(A_rows))]
    vidA_ontpos_match = [get_ontology_pos_overlaps(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_extontpos_match = [get_ontology_pos_overlaps(T_rows[i], A_rows[i], ont_level='extont_sequence') for i in range(len(A_rows))]

    randomise_trials = [random.randint(1, len(A_rows))] * len(A_rows)
    display = [view_name] * len(A_rows)
    show_progress = [1] * len(A_rows)
    category = [None] * len(A_rows)

    exp_df = pd.DataFrame(list(zip(randomise_trials, display, transcripts, videoA_fn,
                                   video_relation, category,
                                   videoA_transcript,
                                   # vidA_shallow_ontology, vidB_shallow_ontology, vidA_deep_ontology, vidB_deep_ontology,
                                   # transcript_shallow, transcript_deep,
                                   transcript_length,
                                   vidA_length,
                                   vidA_embedding_distances, vidA_USE_distances,
                                   show_progress, a_fxns,
                                   vidA_ontology_match, vidA_extontology_match,
                                   vidA_ontpos_match, vidA_extontpos_match)),
                          columns=COLS)
    return exp_df


def get_likert_df_between_subj(data_df, view_name='video_participant_view', num_groups=2):
    COLS = ['randomise_trials', 'display', 'transcripts', 'videoA_fn',
            'video_relation', 'category',
            'videoA_transcript',
            # 'vidA_shallow_ont', 'vidB_shallow_ont', 'vidA_deep_ont', 'vidB_deep_ont',
            # 'transcript_shallow', 'transcript_deep',
            'transcript_length',
            'vidA_length',
            'vidA_embedding_distance', 'vidA_USE_distances',
            'ShowProgressBar', 'A_function',
            'A_ontology_match', 'A_extontology_match',
            'A_ontpos_match', 'A_extontpos_match']
    # build up a df of examples

    fxns = [
        get_closest_gesture_from_row_embeddings,
        get_most_similar_sentence_USE,
        get_ontology_pos_match,
        get_ontology_set_match,
        get_extont_pos_match,
        get_ontology_sequence_match,
        get_extont_sequence_match
    ]

    predicted_video = []  # either 'A' or 'B'
    video_relation = []

    T_rows = []
    A_rows = []
    a_fxns = []

    # add an actual
    r0, r1, r2 = get_transcript_gesture_match(data_df, get_random_row, get_random_row)
    a_fxns.append(['original_gesture'])
    T_rows.append(r0)
    A_rows.append(r0)
    video_relation.append('original_gesture')

    # add a random
    a_fxns.append(['random'])
    T_rows.append(r1)
    A_rows.append(r2)
    video_relation.append('random')

    # add one per fxn
    for f in fxns:
        # first the fxn vs. random
        r0, r1, r2 = get_transcript_gesture_match(data_df, f, get_random_row)
        a_fxns.append([f.__name__])
        T_rows.append(r0)
        A_rows.append(r1)
        video_relation.append(f.__name__)

        # format it for the df
    videoA_fn = [r['video_fn'] for r in A_rows]
    transcripts = [r['PHRASE'] for r in T_rows]
    videoA_transcript = [r['PHRASE'] for r in A_rows]
    transcript_length = [r['time_length'] for r in T_rows]
    vidA_length = [r['time_length'] for r in A_rows]
    vidA_embedding_distances = [get_embedding_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_USE_distances = [get_USE_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_ontology_match = [get_ontology_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_extontology_match = [get_ontology_distances(T_rows[i], A_rows[i], ont_level='extont_sequence') for i in range(len(A_rows))]
    vidA_ontpos_match = [get_ontology_pos_overlaps(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidA_extontpos_match = [get_ontology_pos_overlaps(T_rows[i], A_rows[i], ont_level='extont_sequence') for i in range(len(A_rows))]

    randomise_trials = [random.randint(1, len(A_rows))] * len(A_rows)
    display = [view_name] * len(A_rows)
    show_progress = [1] * len(A_rows)
    category = [None] * len(A_rows)

    exp_df = pd.DataFrame(list(zip(randomise_trials, display, transcripts, videoA_fn,
                                   video_relation, category,
                                   videoA_transcript,
                                   # vidA_shallow_ontology, vidB_shallow_ontology, vidA_deep_ontology, vidB_deep_ontology,
                                   # transcript_shallow, transcript_deep,
                                   transcript_length,
                                   vidA_length,
                                   vidA_embedding_distances, vidA_USE_distances,
                                   show_progress, a_fxns,
                                   vidA_ontology_match, vidA_extontology_match,
                                   vidA_ontpos_match, vidA_extontpos_match)),
                          columns=COLS)
    return exp_df


# gets a df of 10 that has one from each category
def get_representative_df(data_df, view_name='video_participant_view'):
    COLS = ['randomise_trials', 'display', 'transcripts', 'videoA_fn', 'videoB_fn',
            'video_relation', 'category', 'predicted_video',
            'videoA_transcript', 'videoB_transcript',
            # 'vidA_shallow_ont', 'vidB_shallow_ont', 'vidA_deep_ont', 'vidB_deep_ont',
            # 'transcript_shallow', 'transcript_deep',
            'transcript_length',
            'vidA_length', 'vidB_length',
            'vidA_embedding_distance', 'vidB_embedding_distance',
            'ShowProgressBar', 'A_function', 'B_function',
            'A_ontology_match', 'B_ontology_match']
    # build up a df of examples

    """
    Given an experimental transcript T, the viewer distinguishes between two videos which would better
    match T. These videos are split into the following goups: 
    - random gesture vs. random (R) -- expect 50/50
    - close gesture based on transcript embedding (E) vs. random -- expect embedding preferred
    - close gesture based on shallow ontology groups (SO)  vs. random -- expect shallow preferred
    - close gesture based on deep ontology groups (DO) vs. random -- expect deep preferred
    - E vs. DO -- H0 is no difference btw groups
    - E vs. SO -- H0 is no difference btw groups
    - SO vs. DO -- H0 is no difference btw groups
    """
    fxns = [
        get_closest_gesture_from_row_embeddings,
        get_most_similar_sentence_USE,
        get_ontology_pos_match,
        get_ontology_set_match,
        get_ontology_sequence_match,
        get_extont_sequence_match,
        get_random_row,
        get_least_similar_sentence_USE,
        get_farthest_match_embedding
    ]
    """
            fxns = [
            get_closest_gesture_from_row_embeddings,
            get_most_similar_sentence_USE,
            get_ontology_pos_match,
            get_ontology_set_match,
            get_ontology_sequence_match,
            get_random_row,
            get_least_similar_sentence_USE,
            get_farthest_match_embedding
        ]
    """


    predicted_video = []  # either 'A' or 'B'
    video_relation = []

    T_rows = []
    A_rows = []
    B_rows = []
    a_fxns = []
    b_fxns = []

    # add actual v random once per round
    r0, r1, r2 = get_transcript_gesture_match(data_df, get_random_row, get_random_row)
    vids = [(r0, 0), (r2, 1)]  # we predict r0 in this case
    random.shuffle(vids)
    predicted_i = 'A' if vids[0][1] == 0 else 'B'
    predicted_video.append(predicted_i)
    a_fxns.append(['original_gesture' if predicted_i == 'A' else 'random'])
    b_fxns.append(['original_gesture' if predicted_i == 'B' else 'random'])
    T_rows.append(r0)
    A_rows.append(vids[0][0])
    B_rows.append(vids[1][0])
    video_relation.append('original_gesture_v_random')

    for i in tqdm(range(len(fxns))):
        f1 = fxns[i]
        # first the fxn vs. random
        r0, r1, r2 = get_transcript_gesture_match(data_df, f1, get_random_row)

        vids = [(r1, 0), (r2, 1)]  # we know r1 is the 'good' one
        random.shuffle(vids)
        predicted_i = 'A' if vids[0][1] == 0 else 'B'
        predicted_video.append(predicted_i)
        a_fxns.append([f1.__name__ if predicted_i == 'A' else 'get_random_row'])
        b_fxns.append([f1.__name__ if predicted_i == 'B' else 'get_random_row'])
        T_rows.append(r0)
        A_rows.append(vids[0][0])
        B_rows.append(vids[1][0])
        video_relation.append(str(f1.__name__ + '_v_random'))

        # first the fxn vs. FAR
        r0, r1, r2 = get_transcript_gesture_match(data_df, f1, get_random_row)

        vids = [(r1, 0), (r2, 1)]  # we know r1 is the 'good' one
        random.shuffle(vids)
        predicted_i = 'A' if vids[0][1] == 0 else 'B'
        predicted_video.append(predicted_i)
        a_fxns.append([f1.__name__ if predicted_i == 'A' else 'get_random_row'])
        b_fxns.append([f1.__name__ if predicted_i == 'B' else 'get_random_row'])
        T_rows.append(r0)
        A_rows.append(vids[0][0])
        B_rows.append(vids[1][0])
        video_relation.append(str(f1.__name__ + '_v_random'))

        # and a few with the actual gesture there
        r0, r1, r2 = get_transcript_gesture_match(data_df, f1, get_random_row)
        vids = [(r0, 0), (r1, 1)]  # we predict r0 in this case
        random.shuffle(vids)
        predicted_i = 'A' if vids[0][1] == 0 else 'B'
        predicted_video.append(predicted_i)
        a_fxns.append(['original_gesture' if predicted_i == 'A' else f1.__name__])
        b_fxns.append(['original_gesture' if predicted_i == 'B' else f1.__name__])
        T_rows.append(r0)
        A_rows.append(vids[0][0])
        B_rows.append(vids[1][0])
        video_relation.append(str(f1.__name__ + '_v_original_gesture'))

        for j in range(i + 1, len(fxns)):
            f2 = fxns[j]  # then the fxn vs. the other fxns
            if f1.__name__ == f2.__name__:
                continue
            r3, r4, r5 = get_transcript_gesture_match(data_df, f1, f2)
            vids = [(r4, 0), (r5, 1)]  # r4 is our 'predicted' one by default
            random.shuffle(vids)
            predicted_i = 'A' if vids[0][1] == 0 else 'B'
            a_fxns.append([f1.__name__ if predicted_i == 'A' else f2.__name__])
            b_fxns.append([f1.__name__ if predicted_i == 'B' else f2.__name__])
            predicted_video.append(predicted_i)
            T_rows.append(r3)
            A_rows.append(vids[0][0])
            B_rows.append(vids[1][0])
            video_relation.append(str(f1.__name__ + '_v_' + f2.__name__))

        # format it for the df
    videoA_fn = [r['video_fn'] for r in A_rows]
    videoB_fn = [r['video_fn'] for r in B_rows]
    transcripts = [r['PHRASE'] for r in T_rows]
    videoA_transcript = [r['PHRASE'] for r in A_rows]
    videoB_transcript = [r['PHRASE'] for r in B_rows]
    transcript_length = [r['time_length'] for r in T_rows]
    vidA_length = [r['time_length'] for r in A_rows]
    vidB_length = [r['time_length'] for r in B_rows]
    vidA_embedding_distances = [get_embedding_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidB_embedding_distances = [get_embedding_distances(T_rows[i], B_rows[i]) for i in range(len(B_rows))]
    vidA_ontology_match = [get_ontology_distances(T_rows[i], A_rows[i]) for i in range(len(A_rows))]
    vidB_ontology_match = [get_ontology_distances(T_rows[i], B_rows[i]) for i in range(len(B_rows))]

    randomise_trials = [random.randint(1, len(A_rows))] * len(A_rows)
    display = [view_name] * len(A_rows)
    show_progress = [1] * len(A_rows)
    category = [None] * len(A_rows)

    exp_df = pd.DataFrame(list(zip(randomise_trials, display, transcripts, videoA_fn, videoB_fn,
                                   video_relation, category, predicted_video,
                                   videoA_transcript, videoB_transcript,
                                   # vidA_shallow_ontology, vidB_shallow_ontology, vidA_deep_ontology, vidB_deep_ontology,
                                   # transcript_shallow, transcript_deep,
                                   transcript_length,
                                   vidA_length, vidB_length,
                                   vidA_embedding_distances, vidB_embedding_distances,
                                   show_progress, a_fxns, b_fxns,
                                   vidA_ontology_match, vidB_ontology_match)),
                          columns=COLS)
    return exp_df


def get_experimental_df(data_df, view_name=None, simplified=False, n=3):
    exp_f = get_representative_df
    if simplified:
        exp_f = get_actual_random_experimental_df
    dfs = []
    for i in range(n):
        dfs.append(exp_f(data_df, view_name=view_name))

    exp_df = pd.concat(dfs)
    #shuffle
    experimental_df = exp_df.sample(frac=1).reset_index()
    # insert missing video link at 13
    experimental_df = insert_missing_video(13, experimental_df)
    # insert same video at 3
    experimental_df = insert_same_video(3, experimental_df)
    # insert same video at 22
    experimental_df = insert_same_video(23, experimental_df)

    experimental_df = buffer_intro_rows(experimental_df, n=1)
    experimental_df = buffer_debrief_rows(experimental_df, n=1)
    return experimental_df


def get_vid_pairs_per_group(df, group):
    df = df[df['analysis_group'] == group]
    pairs = set(zip(list(df['videoA_fn']), list(df['videoB_fn'])))
    return pairs


def check_spreadsheet_video_pairs(df=None):
    if df is None:
        # load all the dfs from output
        df = load_data()
    df = add_analysis_category(df)
    for ag in df.analysis_group.unique():
        pairs = get_vid_pairs_per_group(df, ag)
        print(f'unique {ag} pairs:', len(set(pairs)))


def check_full_data():
    fns =  ['participant_likert0.csv',
            'participant_likert1.csv',
            'participant_likert2.csv',
            'participant_likert3.csv',
            'participant_likert4.csv',
            'participant_likert5.csv',
            'participant_likert6.csv',
            'participant_likert7.csv',
            'participant_likert8.csv',
            'participant_likert9.csv',
            'participant_likert10.csv',
            'participant_likert11.csv',
            'participant_likert12.csv',
            'participant_likert13.csv',
            'participant_likert14.csv',
            'participant_likert15.csv',
            'participant_likert16.csv',
            'participant_likert17.csv',
            'participant_likert18.csv',
            'participant_likert19.csv',
            'participant_likert20.csv',
            'participant_likert21.csv',
            'participant_likert22.csv',
            'participant_likert23.csv',
            'participant_likert24.csv']
    dfs = []
    for f in fns:
        dfs.append(pd.read_csv(f))
    full_exp_df = pd.concat(dfs)

    for k in full_exp_df['video_relation'].unique():
        cat_df = full_exp_df[full_exp_df.video_relation == k]
        if len(cat_df):
            print(k, len(cat_df), (len(cat_df.videoA_fn.unique()) / len(cat_df)) * 100, '% unique')
    return full_exp_df


def generate_likert_experimental_df(data_df, view_name='participant_likert_view', n=1, num_trials=5):
    exp_f = get_likert_df
    dfs = []
    for i in range(n):
        dfs.append(exp_f(data_df, view_name=view_name))

    exp_df = pd.concat(dfs)
    #shuffle
    experimental_df = exp_df.sample(num_trials)
    experimental_df = buffer_intro_rows(experimental_df, n=1, view_name='likert_intro')
    experimental_df = buffer_debrief_rows(experimental_df, n=1)
    return experimental_df


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
    pf = params.embedding_file
    # df = pd.DataFrame(columns=['PHRASE'] + list(SEMANTIC_CATEGORIES.keys()))

    if pf:
        # pf = os.path.join('speech-and-semantics2gesture', 'Splits', 'combo_fillers', '_transcript_encodings.pkl')
        df = pd.read_pickle(pf)
        df['PHRASE'] = df.apply(lambda row: ' '.join(row['transcript']), axis=1)
        df['SPLIT'] = df.apply(lambda row: row['fn'].split('_')[3], axis=1)
        # df = assign_categories(df)
    """
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
        """

    cere = CII()
    # df['shallow_ont'] = df.progress_apply(lambda row: get_shallow_ontology(row, cere), axis=1)
    # df['deep_ont'] = df.apply(lambda row: get_deep_ontology(row, cere), axis=1)
    # df['hypernyms'] = df.apply(lambda row: get_hypernyms(row, cere), axis=1)

    # TODO exclude gestures that are too short?
    df['time_length'] = df.apply(lambda row: get_time_lambda(row), axis=1)
    df = df[df['time_length'] >= 1.8]   # arbitrary...

    # TODO ah yes here is the hacky garbage.
    print("adding video fns")
    video_dir = os.path.join("speech-and-semantics2gesture", "Splits", "combo_fillers")
    df['video_fn'] = df.progress_apply(lambda row: get_video_fn_from_json_fn(row['fn'], video_dir), axis=1)

    print("getting wordnet properties")
    wn_properties = df.progress_apply(lambda row: get_wn_attributes_for_df(row, cere), axis=1)
    df['ont_sequence'] = [t[0] for t in wn_properties]
    df['extont_sequence'] = [t[1] for t in wn_properties]
    df['hypernyms'] = [t[2] for t in wn_properties]
    df['parse'] = [t[3] for t in wn_properties]

    # ditch the ones without ont_sequences because send it
    df['ontlen'] = df.apply(lambda row: len(row['ont_sequence']), axis=1)
    df = df[df['ontlen'] != 0]

    # df['ont_sequence'] = df.progress_apply(lambda row: get_ontology_sequence(row, cere), axis=1)
    # df['extont_sequence'] = df.progress_apply(lambda row: get_extont_sequence(row, cere), axis=1)

    # get the feature vector

    # add use!!
    df = add_use_embeddings(df)

    # get the clusters
    # clusters = create_category_clusters(df)

    # view it
    # build_chord_diagram(df, show_details=False, output='category_chord.html')

    # get some stats
    # profile_df = get_cluster_profiles(clusters, df)

    # build up a df of examples
    x = 0
    num_samples = 15
    for i in tqdm(range(x, num_samples)):
        ex_df = get_experimental_df(df, view_name='video_participant_view', simplified=True, n=2)
        # ex_df = generate_likert_experimental_df(df, view_name='participant_likert_view', n=1)
        ex_df.to_csv(f'participant_simple{i}.csv')

