import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import scipy


def get_correct(row):
    if row['Response'] == 'Video A' and row['predicted_video'] == 'A':
        return True
    elif row['Response'] == 'Video B' and row['predicted_video'] == 'B':
        return True
    if row['Response'] == 'Video A' and row['predicted_video'] == 'B':
        return False
    elif row['Response'] == 'Video B' and row['predicted_video'] == 'A':
        return False
    return np.nan


def get_correct_embedding_distance(row):
    return row['vidA_embedding_distance'] if row['predicted_video'] == 'A' else row['vidB_embedding_distance']

def get_incorrect_embedding_distance(row):
    return row['vidA_embedding_distance'] if row['predicted_video'] == 'B' else row['vidB_embedding_distance']


def plot_embedded_vs_semantic_distances(exp_df):
    embedding_dists = list(exp_df['vidA_embedding_distance']) + list(exp_df['vidB_embedding_distance'])
    ontology_dists = np.array(list(exp_df['A_ontology_match']) + list(exp_df['B_ontology_match'])) * 10
    fig, ax = plt.subplots(1, 1)
    ax.scatter(embedding_dists, ontology_dists)
    ax.set_title('Embedding vs. Ontology distances')
    ax.set_xlabel('Embedding Distance (G and transcript)')
    ax.set_ylabel('Ontology Distance (G and transcript)')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(embedding_dists, ontology_dists)
    print(r_value ** 2)
    print(p_value)
    fig.show()


if __name__ == "__main__":
    data_df = pd.read_csv('trial_2_data.csv')
    answer_df = data_df.copy()
    answer_df['correct'] = data_df.apply(get_correct, axis=1)
    answer_df = answer_df.dropna(subset=['correct'])
    answer_df['correct_embedding_dist'] = answer_df.apply(get_correct_embedding_distance, axis=1)
    answer_df['incorrect_embedding_dist'] = answer_df.apply(get_incorrect_embedding_distance, axis=1)

    # plot x dist vs. y dist where x is correct answer
    # green is predicted answers, red is deviant answers
    fig, ax = plt.subplots(1, 1)
    xs = answer_df['correct_embedding_dist'].values
    ys = answer_df['incorrect_embedding_dist'].values
    color = answer_df.apply(lambda row: "green" if row['correct'] else "red", axis=1)

    ax.scatter(xs, ys, color=color)
    ax.set_title('Correct vs. incorrect embedding distances')
    ax.set_xlabel('Correct video embedding distance to original transcript')
    ax.set_ylabel('Incorrect video embedding distance to original transcript')
    ax.legend(['Correct responses', 'incorrect responses'])
    # fig.show()

    # plot % overlapping categories with color
    set_df = answer_df[['transcript_categories', 'vidA_overlap', 'vidB_overlap', 'correct_video']]

    row = set_df.iloc[3]

    s = row['transcript_categories']
    l = list(set(re.split("[" + "\\".join(d) + "]", s)))
    l.remove('')
    set(l)

    """
    Analyses: 
    - how often did they choose our predicted video in:
        - E  vs. R
        - SO vs. R
        - DO vs. R
        ** compared to R vs. R baseline
    - In our experimental conditions (EvSO, EvDO, SOvDO) which video was preferred most often? 
    - Overall, how much CLOSER was the chosen video's semantic EMBEDDING to the transcript than the not chosen gesture?
    - Overall, how many MORE set overlaps were in the chosen video's SEMANTIC SET to the transcript than not chosen?
        - for SO
        - for DO
    - Semantic category breakdown:
        - if we isolate certain semantics, can we make it better? 
        - e.g. analyses without container, without dynamic, etc. 
    - How detrimental are semantics that are NOT present in transcript that ARE present in gesture?
        - remove cases in which predicted video contains categories not present in transcript
        - look at average INTERSECTION of chosen video vs not chosen (SO, DO)
        - look at average DIFFERENCE of chosen video vs. not chosen (SO, DO)
    """
    video_relations = answer_df['video_relation'].unique()
    """
    array(['get_deep_ontology_gesture_match_v_get_shallow_ontology_gesture_match',
       'get_closest_gesture_from_row_embeddings_v_get_shallow_ontology_gesture_match',
       'get_shallow_ontology_gesture_match_v_random',
       'get_closest_gesture_from_row_embeddings_v_random',
       'get_closest_gesture_from_row_embeddings_v_get_deep_ontology_gesture_match',
       'get_shallow_ontology_gesture_match_v_get_closest_gesture_from_row_embeddings',
       'get_shallow_ontology_gesture_match_v_get_deep_ontology_gesture_match',
       'get_deep_ontology_gesture_match_v_random',
       'get_deep_ontology_gesture_match_v_get_closest_gesture_from_row_embeddings']
    """
    vs_embedding_df = answer_df[answer_df.video_relation.str.contains('get_closest_gesture_from_row_embeddings')]
    vs_random_df = answer_df[answer_df.video_relation.str.contains('random')]

