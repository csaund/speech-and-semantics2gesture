import pandas as pd
import os
import matplotlib.pyplot as plt
from pylab import savefig
import re
import numpy as np
import scipy
import scipy.stats
plt.switch_backend('agg')
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

## TODO FIRST THING:
## TODO DOUBLE CHECK THAT THE RESULTS ARE FROM
## TODO DIFFERENT ANALYSIS PAIRS!!!
## todo aka make sure it's not the same videos being compared within a comparison group

"""
Stacy notes 
each ont element, there is a class and an instance
-- printing ontology for a sentence, see categories like 'agency', and some identifyer
"if CONTAINER- tag, should ignore all other container tags in extont"
"Agency (active, inactive, partial, etc), ignore all AGENCY tags in extont"
        -- only add anything from extont is when CATEGORY is not mentioned in ontology
        -- categories are identified as CATEGORY:value

So maybe should rank by sets, or like only select 
"""



"""
map these names to our actual experimental groups:
actual_random
actual_embedded
actual_shallow
actual_deep
random_embedded
random_shallow
random_deep
embedded_shallow
embedded_deep
shallow_deep
"""
category_mapper = {
    'get_extont_sequence_match_v_original_gesture': 'actual_deep',
    'get_ontology_sequence_match_v_get_closest_gesture_from_row_embeddings': 'embedded_shallow',
    'CONTROL_SAME_VIDEO': 'control',
    'get_closest_gesture_from_row_embeddings_v_original_gesture': 'actual_embedded',
    'get_extont_sequence_match_v_get_ontology_sequence_match': 'shallow_deep',
    'get_extont_sequence_match_v_get_closest_gesture_from_row_embeddings': 'embedded_deep',
    'get_ontology_sequence_match_v_original_gesture': 'actual_shallow',
    'get_closest_gesture_from_row_embeddings_v_get_extont_sequence_match': 'embedded_deep',
    'get_closest_gesture_from_row_embeddings_v_get_ontology_sequence_match': 'embedded_shallow',
    'CONTROL_BROKEN_VIDEO': 'control',
    'get_extont_sequence_match_v_random': 'random_deep',
    'get_ontology_sequence_match_v_get_extont_sequence_match': 'shallow_deep',
    'get_closest_gesture_from_row_embeddings_v_random': 'random_embedded',
    'get_ontology_sequence_match_v_random': 'random_shallow',
    'original_gesture_v_random': 'actual_random',
    'get_least_similar_sentence_USE_v_get_farthest_match_embedding': 'bad_bad',
    'get_farthest_match_embedding_v_random': 'bad_bad',
    'get_closest_gesture_from_row_embeddings_v_get_least_similar_sentence_USE': 'embedded_bad'
}

def cat_function(k):
    n = ''
    if 'original_gesture' in k:
        n += 'actual_'
    if 'random' in k:
        n += 'random_'
    if 'get_closest_gesture_from_row_embeddings' in k:
        n += 'embeddedsim_'
    if 'most_similar_sentence_USE' in k:
        n += 'usesim_'
    if 'ontology_sequence_match' in k:
        n += 'ontseq'
    if 'extont_sequence_match' in k:
        n += 'extontseq_'
    if 'ontology_set_match' in k:
        n += 'ontset_'
    if 'get_farthest_match_embedding' in k:
        n += 'embeddedfar_'
    if 'ontology_pos_match' in k:
        n += 'ontpos_'
    if 'get_extont_pos_match' in k:
        n += 'extontpos_'
    if 'least_similar_sentence_USE' in k:
        n += 'usefar_'
    return n[:-1]


category_to_group_mapper = {
    'actual': ['actual_random', 'actual_embedded', 'actual_shallow', 'actual_deep'],
    'random': ['actual_random', 'random_embedded', 'random_shallow', 'random_deep'],
    'embedded': ['actual_embedded', 'random_embedded', 'embedded_shallow', 'embedded_deep'],
    'shallow': ['actual_shallow', 'random_shallow', 'embedded_shallow', 'shallow_deep'],
    'deep': ['actual_deep', 'random_deep', 'embedded_deep', 'shallow_deep']
}


def print_analysis_categories(df):
    tdf = df.copy()
    tdf = tdf.dropna(subset=['video_relation'])
    tdf['analysis_group'] = tdf.apply(lambda row: cat_function(row['video_relation']) if row['video_relation'] != np.nan else None, axis=1)
    ##
    for ag in tdf['analysis_group'].unique():
        gdf = len(tdf[tdf['analysis_group'] == ag])
        print(f'{ag}:', gdf)


def add_analysis_category(df):
    tdf = df.copy()
    tdf = tdf.dropna(subset=['video_relation'])
    tdf['analysis_group'] = tdf.apply(lambda row: cat_function(row['video_relation']) if row['video_relation'] != np.nan else None, axis=1)
    return tdf


def get_analysis_category_df(cat_name, df):
    tdf = df.copy()
    tdf = tdf.dropna(subset=['video_relation'])
    tdf['analysis_group'] = tdf.apply(lambda row: category_mapper[row['video_relation']] if row['video_relation'] != np.nan else None, axis=1)
    ##
    ret_df = tdf[tdf['analysis_group'] == cat_name]
    return ret_df


def plot_analysis_rando(cat_name, cat1, cat2, df):
    rel_df = get_analysis_category_df(cat_name, df)
    rel_df['chose_random'] = rel_df.apply(lambda row: True if ('random' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('random' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    rel_df['chose_actual'] =  rel_df.apply(lambda row: True if ('original_gesture' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('original_gesture' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    random_count = len(rel_df[rel_df['chose_random']])
    actual_count = len(rel_df[rel_df['chose_actual']])
    tot = float(random_count) + float(actual_count)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    methods = ['random', 'actual']
    perc = [float(random_count)/tot, float(actual_count)/tot]
    ax.bar(methods, perc)
    plt.show()


# https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# cat_name == one of the analysis categories:
"""
map these names to our actual experimental groups:
actual_random
actual_embedded
actual_shallow
actual_deep
random_embedded
random_shallow
random_deep
embedded_shallow
embedded_deep
shallow_deep
"""
# cat1 and cat2 are one of
# ['get_extont_sequence_match']",
#        "['get_closest_gesture_from_row_embeddings']",
#        "['original_gesture']", "['get_ontology_sequence_match']",
#        "['get_random_row']", "['random']

def plot_analysis_category(cat_name, cat1, cat2, df):
    rel_df = get_analysis_category_df(cat_name, df)
    rel_df['chose_cat1'] = rel_df.apply(lambda row: True if (cat1 in row['A_function'] and row['Response'] == 'Video A') or
                                                              (cat1 in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    rel_df['chose_cat2'] =  rel_df.apply(lambda row: True if (cat2 in row['A_function'] and row['Response'] == 'Video A') or
                                                              (cat2 in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    cat1_count = len(rel_df[rel_df['chose_cat1']])
    cat2_count = len(rel_df[rel_df['chose_cat2']])
    tot = float(cat1_count) + float(cat2_count)
    print('Chose %s: %s' % (cat1, cat1_count))
    print('Chose %s: %s' % (cat2, cat2_count))

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    methods = [cat1, cat2]
    perc = [float(cat1_count)/tot, float(cat2_count)/tot]
    ax.bar(methods, perc)
    plt.show()

# ['get_extont_sequence_match']",
#        "['get_closest_gesture_from_row_embeddings']",
#        "['original_gesture']", "['get_ontology_sequence_match']",
#        "['get_random_row']", "['random']



"""
plot_comparisons(sem_df, 'actual')
plot_comparisons(sem_df, 'random')
plot_comparisons(sem_df, 'embedded')
plot_comparisons(sem_df, 'shallow')
plot_comparions(sem_df, 'deep')
plot_comparions(sem_df, 'all')
I want it to be able to take the analysis group and plot all the relevant items 
for that analysis group. so for 'actual' I want it to plot 
actual_random
actual_embedded
actual_shallow
actual_deep

def plot_comparisons(df, analysis_groups):
    function_names = get_function_from_analysis_group(analysis_groups)
    sub_df = 
    rel_df = df[df['analysis_group'].isin(category_to_group_mapper(analysis_groups))]
    rel_df['chose_actual'] = rel_df.apply(lambda row: True if ('original_gesture' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('original_gesture' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
"""


def get_group_stats(group_name, choice1, choice2, df):
    temp_df = df[df['analysis_group'] == group_name]
    choice1_count = float(len(temp_df[temp_df[choice1]]))
    choice2_count = float(len(temp_df[temp_df[choice2]]))
    total = choice1_count + choice2_count
    if total == 0:
        print('ERROR: no trials in analysis group ', group_name, 'found.')
        return [0, 0]
    choice1_perc = choice1_count / total
    choice2_perc = choice2_count / total
    print(f'{choice1}: {choice1_count / total}, {choice2}: {choice2_count / total}')
    return [choice1_perc, choice2_perc]


def get_data_all_comparison(df):
    rel_df = df
    rel_df['chose_actual'] = rel_df.apply(lambda row: True if ('original_gesture' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('original_gesture' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    rel_df['chose_random'] = rel_df.apply(lambda row: True if ('random' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('random' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    rel_df['chose_embedded'] = rel_df.apply(lambda row: True if ('get_closest_gesture_from_row_embeddings' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('get_closest_gesture_from_row_embeddings' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    rel_df['chose_shallow'] = rel_df.apply(lambda row: True if ('get_ontology_sequence_match' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('get_ontology_sequence_match' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    rel_df['chose_deep'] = rel_df.apply(lambda row: True if ('get_extont_sequence_match' in row['A_function'] and row['Response'] == 'Video A') or
                                                              ('get_extont_sequence_match' in row['B_function'] and row['Response'] == 'Video B')
                                                            else False, axis=1)
    print(len(rel_df))
    actual_count = float(len(rel_df[rel_df['chose_actual']]))
    random_count = float(len(rel_df[rel_df['chose_random']]))
    embedded_count = float(len(rel_df[rel_df['chose_embedded']]))
    shallow_count = float(len(rel_df[rel_df['chose_shallow']]))
    deep_count = float(len(rel_df[rel_df['chose_deep']]))
    tot = actual_count + random_count + embedded_count + shallow_count + deep_count
    other_count = len(rel_df) - tot

    # 10 x 2 matrix
    data = [
        get_group_stats('actual_random', 'chose_actual', 'chose_random', rel_df),
        get_group_stats('actual_embedded', 'chose_actual', 'chose_embedded', rel_df),
        get_group_stats('actual_shallow', 'chose_actual', 'chose_shallow', rel_df),
        get_group_stats('actual_deep', 'chose_actual', 'chose_deep', rel_df),
        get_group_stats('random_embedded', 'chose_random', 'chose_embedded', rel_df),
        get_group_stats('random_shallow', 'chose_random', 'chose_shallow', rel_df),
        get_group_stats('random_deep', 'chose_random', 'chose_deep', rel_df),
        get_group_stats('embedded_shallow', 'chose_embedded', 'chose_shallow', rel_df),
        get_group_stats('embedded_deep', 'chose_embedded', 'chose_deep', rel_df),
        get_group_stats('shallow_deep', 'chose_shallow', 'chose_deep', rel_df)
    ]

    return data


def plot_all_bars(df):
    data = get_data_all_comparison(df)
    # Numbers of pairs of bars you want
    N = 10

    # Data on X-axis

    # Specify the values of blue bars (height)
    blue_bar = [d[0] for d in data]
    # Specify the values of orange bars (height)
    orange_bar = [d[1] for d in data]

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.3

    # Plotting
    plt.bar(ind, blue_bar, width, label='first category')
    plt.bar(ind + width, orange_bar, width, label='second category')

    plt.xlabel('Category group')
    plt.ylabel("% preference")
    plt.title('percentage preferred in head-to-head categories')

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width / 2, ('actual_random', 'actual_embedded', 'actual_shallow', 'actual_deep',
                                 'random_embedded', 'random_shallow', 'random_deep',
                                 'embedded_shallow', 'embedded_deep',
                                 'shallow_deep'), rotation=20, fontsize='small')

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.show()


def plot_chosen_vs_not_chosen_scores_embedded(df):
    df = strip_equal_evals(df)
    chosen_embedding_dists = df.apply(lambda row: row['vidA_embedding_distance'] if row['Response'] == 'Video A'
                                             else row['vidB_embedding_distance'], axis=1)
    not_chosen_embedding_dists = df.apply(lambda row: row['vidA_embedding_distance'] if row['Response'] == 'Video B'
                                             else row['vidB_embedding_distance'], axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(chosen_embedding_dists, not_chosen_embedding_dists)
    ax.set_title('chosen vs not chosen embedding distance')
    ax.set_xlabel('chosen distance')
    ax.set_ylabel('not chosen distance')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(chosen_embedding_dists, not_chosen_embedding_dists)
    print(r_value ** 2)
    print(p_value)
    fig.show()


def strip_equal_evals(df):
    df = df[df['Response'].isin(['Video A', 'Video B'])]
    return df


def plot_chosen_vs_not_chosen_scores_ontology(df):
    df = strip_equal_evals(df)
    chosen_embedding_dists = df.apply(lambda row: row['A_ontology_match'] if row['Response'] == 'Video A'
                                             else row['B_ontology_match'], axis=1)
    not_chosen_embedding_dists = df.apply(lambda row: row['A_ontology_match'] if row['Response'] == 'Video B'
                                             else row['B_ontology_match'], axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(chosen_embedding_dists, not_chosen_embedding_dists)
    ax.set_title('chosen vs not chosen embedding distance')
    ax.set_xlabel('chosen distance')
    ax.set_ylabel('not chosen distance')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(chosen_embedding_dists, not_chosen_embedding_dists)
    print(r_value ** 2)
    print(p_value)
    fig.show()


def plot_random_embedded_vs_selected_embedded(df):
    tdf = df[df['analysis_group'].isin(['random_embedded', 'random_shallow', 'random_deep'])]
    random_embedded_values = tdf.apply(lambda row:
                                             row['vidA_embedding_distance'] if any(['random' in r for r in row['A_function']])
                                                                           else row['vidB_embedding_distance'],
                                      axis=1)
    our_selection_embedded_values = tdf.apply(lambda row:
                                             row['vidA_embedding_distance'] if not any(['random' in r for r in row['A_function']])
                                                                           else row['vidB_embedding_distance'],
                                      axis=1)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(our_selection_embedded_values, random_embedded_values)
    ax.set_title('random vs. selected embedding distances')
    ax.set_xlabel('selection distance')
    ax.set_ylabel('random distance')
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(chosen_embedding_dists, not_chosen_embedding_dists)
    # print(r_value ** 2)
    # print(p_value)
    fig.show()


def plot_random_vs_selected_ontology_distances(df):
    tdf = df[df['analysis_group'].isin(['random_shallow', 'random_embedded', 'random_shallow', 'random_deep'])]
    random_embedded_values = tdf.apply(lambda row:
                                             row['A_ontology_match'] if any(['random' in r for r in row['A_function']])
                                                                           else row['B_ontology_match'],
                                      axis=1)
    our_selection_embedded_values = tdf.apply(lambda row:
                                             row['A_ontology_match'] if not any(['random' in r for r in row['A_function']])
                                                                           else row['B_ontology_match'],
                                      axis=1)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(our_selection_embedded_values, random_embedded_values)
    ax.set_title('random vs. selected ontological distances')
    ax.set_xlabel('selection distance')
    ax.set_ylabel('random distance')
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(chosen_embedding_dists, not_chosen_embedding_dists)
    # print(r_value ** 2)
    # print(p_value)
    fig.show()


## TODO: try taking out when random and selected are close together and see
# if we can exaggerate the effects


def get_num_each_category(dfs=None, n=10):
    if not dfs:
        fns = [f'full_test{i}.csv' for i in range(n)]
        dfs = [pd.read_csv(fn) for fn in fns]
    df = pd.concat(dfs)
    print_analysis_categories(df)
    return df


def get_num_repeats(df):
    tdf = df.copy()
    tdf = tdf[tdf['videoA_fn'] == tdf['videoB_fn']]
    tdf = tdf[~tdf['video_relation'].isin(['CONTROL_SAME_VIDEO'])]
    return len(tdf)


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


# TODO what the fuck am I doing here, it should be chosen vs. not chosen, duh.
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


def get_answer_df(df):
    mask = df.apply(lambda row: True if (row['Zone Name'] == 'VideoA') or
                                        (row['Zone Name'] == 'Zone6')
                                else False, axis=1)
    df = df[mask]
    return df


def get_sem_and_energy_dfs(df):
    sem_df = df[df['Zone Name'] == 'VideoA']
    energy_df = df[df['Zone Name'] == 'Zone6']
    return sem_df, energy_df


def load_data_II(dirname='experiment_data'):
    fns = [f for f in os.listdir(dirname) if 'task' in f]
    dfs = []
    # f = 'data_exp_51799-v6_task-djbw.csv'
    for f in fns:
        print(f)
        df = pd.read_csv(os.path.join(dirname, f))
        if len(df) > 50 and passed_attention_checks(df):
            print('passed')
            dfs.append(df)
    df = pd.concat(dfs)
    return df


def passed_attention_checks(df):
    tdf = get_trial_df(df)
    passed_same_vid = tdf.apply(lambda row:
                            False if (row['videoA_fn'] == row['videoB_fn']) and
                                     (row['semantic_chosen'] != 'Match Equally' or
                                      row['energy_chosen'] != 'Equally Energetic')
                            else True,
                            axis=1)
    passed_same_vid = not any(~passed_same_vid)
    passed_broken = 'Zone8' in df[df['video_relation'] == 'CONTROL_BROKEN_VIDEO']['Zone Name'].unique()
    print(f'passed same: {passed_same_vid} / passed broken: {passed_broken}')
    return passed_same_vid and passed_broken


def load_data():
    fns =  ['data_exp_51799-v5_questionnaire-alfr.csv',
            'data_exp_51799-v5_questionnaire-anxz.csv',
            'data_exp_51799-v5_task-1qts.csv',
            'data_exp_51799-v5_task-9bgm.csv',
            'data_exp_51799-v5_task-djbw.csv',
            'data_exp_51799-v5_task-e4ub.csv',
            'data_exp_51799-v5_task-gg8f.csv',
            'data_exp_51799-v5_task-nxrv.csv',
            'data_exp_51799-v5_task-p5ld.csv',
            'data_exp_51799-v5_task-qrqe.csv',
            'data_exp_51799-v5_task-sujr.csv',
            'data_exp_51799-v5_task-zekk.csv']
    dfs = []
    for f in fns:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    return df


def get_main_dfs():
    # once
    # os.chdir('speech-and-semantics2gesture')
    raw_df = load_data()
    all_answers_df = get_answer_df(raw_df)
    sem_df, energy_df = get_sem_and_energy_dfs(all_answers_df)
    return add_analysis_category(sem_df), add_analysis_category(energy_df)


# format of all the df columns +
# 'semantic_response' (video A or video B)
# 'energetic_response' (video A or video B)
def get_trial_df(df=None):
    if df is None:
        df = load_data_II()
    # get only the full experiment
    df = df[df['Screen Name'] == 'Screen 1']
    df = df[df['display'] == 'video_participant_view']
    # df = df[~df['video_relation'].isin(['CONTROL_BROKEN_VIDEO', 'CONTROL_SAME_VIDEO'])]
    # get only the responses to the questions
    df['Response Type'] = df.apply(lambda row: 'semantic' if row['Zone Name'] == 'VideoA' \
                                                          else (
                                                          'energy' if row['Zone Name'] == 'Zone6'
                                                                   else np.nan), axis=1)
    df = df.dropna(subset=['Response Type'])
    # use Trial Number to
    df['trial_num'] = flatten([[n, n] for n in range(1, int(len(df)/2)+1)])
    # fuck it just create a new dataframe

    ndf = pd.DataFrame()
    cur_trial = 0
    nrow = None
    should_append = True
    for index, row in df.iterrows():
        if row['trial_num'] == cur_trial:
            # keep constructing the new row, aka add the energy answer
            # assert(row['Zone Name'] == 'Zone6')
            if row['Zone Name'] == 'VideoA':
                nrow['semantic_chosen'] = row['Response']
            elif row['Zone Name'] == 'Zone6':
                nrow['energy_chosen'] = row['Response']
            else:
                print('GOT UNEXPECTED ZONE: ', row['Zone Name'])

            if not row['videoA_fn'] == nrow['videoA_fn']:
                print('BAD TRIAL DATA', cur_trial, row['trial_num'])
            if not row['videoB_fn'] == nrow['videoB_fn']:
                print('BAD TRIAL DATA', cur_trial, row['trial_num'])
            # assert(row['videoA_fn'] == nrow['videoA_fn'])
            # assert(row['videoB_fn'] == nrow['videoB_fn'])
        else:
            if cur_trial != 0 and should_append:
                ndf = ndf.append(nrow)
            elif not should_append:
                print('AVOIDED appending trial number', cur_trial)
            cur_trial = row['trial_num']
            should_append = True
            nrow = row
            if row['Zone Name'] == 'VideoA':
                nrow['semantic_chosen'] = row['Response']
            elif row['Zone Name'] == 'Zone6':
                nrow['energy_chosen'] = row['Response']
            else:
                print('GOT UNEXPECTED ZONE: ', row['Zone Name'])

            if isNaN(row['videoA_fn']) or isNaN(row['videoB_fn']):
                print('FOUND NAN VIDEO FN: ', cur_trial)
                should_append = False

        # also remove duplicate videos
        mask = ndf.apply(lambda row: row['videoA_fn'] != row['videoB_fn'], axis=1)
        ndf = ndf[mask]

    # drop those who failed the attention check
    ndf = add_analysis_category(ndf)
    return ndf


def isNaN(num):
    return num != num


flatten = lambda t: [item for sublist in t for item in sublist]

actual_cats = ['actual_random', 'actual_embeddedsim', 'actual_usesim',
               'actual_ontse', 'actual_ontpos', 'actual_ontset',
               'actual_extontseq', 'actual_extontpos']

random_cats = ['actual_random', 'random_embeddedsim', 'random_usesim',
               'random_ontse', 'random_ontpos', 'random_ontset',
               'random_extontseq', 'random_extontpos']


def get_percentage_choices_random(df):
    random_perc = []
    other_perc = []
    equal_perc = []
    for cat in random_cats:
        ndf = df[df['analysis_group'] == cat]
        chose_random_mask = ndf.apply(lambda row:
                                      True if ('random' in row['A_function'] and row.semantic_chosen == 'Video A')
                                      or ('random' in row['B_function'] and row.semantic_chosen == 'Video B')
                                      else False, axis=1)
        num_chose_random = len(ndf[chose_random_mask]) / len(ndf)
        chose_other_mask = ndf.apply(lambda row:
                                      True if ('random' not in row['A_function'] and row.semantic_chosen == 'Video A')
                                      or ('random' not in row['B_function'] and row.semantic_chosen == 'Video B')
                                      else False, axis=1)
        num_chose_other = len(ndf[chose_other_mask]) / len(ndf)
        chose_equal_mask = ndf.apply(lambda row:
                                      True if (row.semantic_chosen == 'Match Equally')
                                      else False, axis=1)
        num_chose_equal = len(ndf[chose_equal_mask]) / len(ndf)
        random_perc.append(num_chose_random)
        other_perc.append(num_chose_other)
        equal_perc.append(num_chose_equal)

    return random_perc, other_perc, equal_perc


def get_percentage_choices(df):
    actual_perc = []
    other_perc = []
    equal_perc = []
    for cat in actual_cats:
        ndf = df[df['analysis_group'] == cat]
        chose_actual_mask = ndf.apply(lambda row:
                                      True if ('original_gesture' in row['A_function'] and row.semantic_chosen == 'Video A')
                                      or ('original_gesture' in row['B_function'] and row.semantic_chosen == 'Video B')
                                      else False, axis=1)
        num_chose_actual = len(ndf[chose_actual_mask]) / len(ndf)
        chose_other_mask = ndf.apply(lambda row:
                                      True if ('original_gesture' not in row['A_function'] and row.semantic_chosen == 'Video A')
                                      or ('original_gesture' not in row['B_function'] and row.semantic_chosen == 'Video B')
                                      else False, axis=1)
        num_chose_other = len(ndf[chose_other_mask]) / len(ndf)
        chose_equal_mask = ndf.apply(lambda row:
                                      True if (row.semantic_chosen == 'Match Equally')
                                      else False, axis=1)
        num_chose_equal = len(ndf[chose_equal_mask]) / len(ndf)
        actual_perc.append(num_chose_actual)
        other_perc.append(num_chose_other)
        equal_perc.append(num_chose_equal)

    return actual_perc, other_perc, equal_perc


def fake_stuff_quickly(comp_data, other_data, equal_data, comp_name='actual', fig_name='test'):
    N = len(comp_data)
    # Data on X-axis
    # Specify the values of blue bars (height)
    blue_bar = comp_data
    # Specify the values of orange bars (height)
    orange_bar = other_data
    equal_bar = equal_data

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.2

    # Plotting
    plt.bar(ind, blue_bar, width, label=f'preferred {comp_name}')
    plt.bar(ind + width, equal_bar, width, label='preferred equal')
    plt.bar(ind + width + width, orange_bar, width, label='preferred other')

    plt.xlabel('Category group')
    plt.ylabel("% preference")
    plt.title(f'percentage preferred in vs. {comp_name} across categories')

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width / 3, ('actual_random', f'{comp_name}_embeddedsim', f'{comp_name}_usesim',
               f'{comp_name}_ontseq', f'{comp_name}_ontpos', f'{comp_name}_ontset',
               f'{comp_name}_extontseq', f'{comp_name}_extontpos'),
               rotation=20, fontsize='small')

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.savefig(fig_name)


def get_energy_stats(df):
    chose_same_energy_mask = df.apply(lambda row:
                                 row.semantic_chosen == row.energy_chosen or
                                 (row.semantic_chosen == 'Match Equally' and
                                  row.energy_chosen == 'Equally Energetic'),
                                  axis=1)
    cse = len(df[chose_same_energy_mask]) / len(df)
    return cse


def print_choices(df):
    print('SEMANTIC')
    print('Chose A: ', len(df[df.semantic_chosen == 'Video A']) / len(df))
    print('Chose B: ', len(df[df.semantic_chosen == 'Video B']) / len(df))
    print('Chose Same: ', len(df[df.semantic_chosen == 'Match Equally']) / len(df))
    print('ENERGY')
    print('Chose A: ', len(df[df.energy_chosen == 'Video A']) / len(df))
    print('Chose B: ', len(df[df.energy_chosen == 'Video B']) / len(df))
    print('Chose Same: ', len(df[df.energy_chosen == 'Equally Energetic']) / len(df))

    lens = [len(df[df.semantic_chosen == 'Video A']),
            len(df[df.semantic_chosen == 'Video B'])]

    print('p =', scipy.stats.chisquare(lens).pvalue)
    print("insig. p means results were equally distributed btw A/B/Same")
    print('======================================')
    chose_same_energy_mask = df.apply(lambda row:
                                 row.semantic_chosen == row.energy_chosen or
                                 (row.semantic_chosen == 'Match Equally' and
                                  row.energy_chosen == 'Equally Energetic'),
                                  axis=1)
    cse_raw = len(df[chose_same_energy_mask])
    cse = cse_raw / len(df)
    oth = len(df) - cse_raw
    print('Chose same semantic/energy', cse, f'({cse_raw}, {oth})')
    sames = [cse_raw, oth]
    print('p =', scipy.stats.chisquare(sames).pvalue)
    print('significant p means people more likely to choose BOTH semantic and more energetic')


def last(l):
    t = list(l)
    t = [i for i in t if i is not None]
    return float(t[-1]) - 50


def analyse_likert_data(dirname='experiment_likert_data'):
    # dirname = 'experiment_likert_data'
    # dirname = 'experiment_likert_circular'
    keep_cols = ['Trial Number',
                 'video_relation',
                 'videoA_fn',
                 'vidA_length',
                 'transcripts',
                 'videoA_transcript',
                 'transcript_length',
                 'vidA_USE_distances',
                 'vidA_embedding_distance',
                 'A_ontology_match',
                 'A_extontology_match',
                 'A_ontpos_match',
                 'A_extontpos_match']
    likert_files = os.listdir(dirname)
    dfs = []
    for f in likert_files:
        if 'task' in f:
            try:
                ldf = pd.read_csv(os.path.join(dirname, f))
            except pd.errors.EmptyDataError as e:
                print('empty datafile:', f)
                continue
            print('processing:', f)
            if len(ldf) < 25:
                continue
            ldf = ldf[ldf['Screen Name'] == 'Screen 1']
            ldf = ldf[ldf['display'] == 'participant_likert_view']
            ldf['sem_res'] = ldf.apply(lambda row: row['Response'] if row['Zone Name'] == 'VideoA' else None, axis=1)
            ldf['eng_res'] = ldf.apply(lambda row: row['Response'] if row['Zone Name'] == 'Zone6' else None, axis=1)
            ldf = ldf.groupby(keep_cols, as_index=False).agg(
                semantic_response=pd.NamedAgg(column='sem_res', aggfunc=last),
                energy_response=pd.NamedAgg(column='eng_res', aggfunc=last)
            )
            ldf['num_words_transcript'] = ldf.apply(lambda row: len(row['transcripts'].split(' ')), axis=1)
            ldf['num_words_video'] = ldf.apply(lambda row: len(row['videoA_transcript'].split(' ')), axis=1)
            dfs.append(ldf)
    df = pd.concat(dfs)
    return df

# TODO this is good for qualitative
def get_qualitative_df(dirname='experiment_likert_qualitative_small'):
    keep_cols = ['Trial Number',
                 'video_relation',
                 'videoA_fn',
                 'vidA_length',
                 'transcripts',
                 'videoA_transcript',
                 'transcript_length',
                 'vidA_USE_distances',
                 'vidA_embedding_distance',
                 'A_ontology_match',
                 'A_extontology_match',
                 'A_ontpos_match',
                 'A_extontpos_match']
    likert_files = os.listdir(dirname)
    dfs = []
    for f in likert_files:
        if 'task' in f:
            try:
                ldf = pd.read_csv(os.path.join(dirname, f))
            except pd.errors.EmptyDataError as e:
                print('empty datafile:', f)
                continue
            print('processing:', f)
            if len(ldf) < 25:
                continue
            ldf = ldf[ldf['Screen Name'] == 'Screen 1']
            ldf = ldf[ldf['display'] == 'qualitative_view_set']
            ldf['separation'] = ldf.apply(lambda row: float(row['Response']) if row['Zone Name'] == 'separationslider' else None, axis=1)
            ldf['certainty'] = ldf.apply(lambda row: float(row['Response']) if row['Zone Name'] == 'certainslider' else None, axis=1)
            ldf['process'] = ldf.apply(lambda row: float(row['Response']) if row['Zone Name'] == 'processslider' else None, axis=1)
            ldf['positive'] = ldf.apply(lambda row: float(row['Response']) if row['Zone Name'] == 'personalslider' else None, axis=1)
            ldf = ldf.groupby(keep_cols, as_index=False).agg(
                separation=pd.NamedAgg(column='separation', aggfunc='last'),
                certainty=pd.NamedAgg(column='certainty', aggfunc='last'),
                process=pd.NamedAgg(column='process', aggfunc='last'),
                positive=pd.NamedAgg(column='positive', aggfunc='last')
            )
            ldf['num_words_transcript'] = ldf.apply(lambda row: len(row['transcripts'].split(' ')), axis=1)
            ldf['num_words_video'] = ldf.apply(lambda row: len(row['videoA_transcript'].split(' ')), axis=1)
            dfs.append(ldf)
    df = pd.concat(dfs)
    return df

## now pass that df here for overall by process...
def get_qualitative_stats(ldf):
    for t in ldf.transcripts.unique():
        print("====================================")
        print("Separation // Certainty // Positive // Process")
        print("Transcript: ", t)
        tdf = ldf[ldf.transcripts == t]
        # now we have all the different versions.
        dat = []
        relations = ['original_gesture',
                     'random',
                     'get_closest_gesture_from_row_embeddings',
                     # 'get_most_similar_sentence_USE',
                     'get_ontology_pos_match',
                     'get_extont_pos_match']
                     #'get_ontology_sequence_match',
                     #'get_extont_sequence_match']
        for r in relations:
            vdf = tdf[tdf.video_relation == r]
            if len(vdf) == 0:
                continue
            sep = np.array(vdf.separation.values)
            sep_m = np.round(sep.mean(), 3)
            sep_std = np.round(sep.std(), 3)
            cer = np.array(vdf.certainty.values)
            cer_m = np.round(cer.mean(), 3)
            cer_std = np.round(cer.std(), 3)
            pos = np.array(vdf.positive.values)
            pos_m = np.round(pos.mean(), 3)
            pos_std = np.round(pos.std(), 3)
            proc = np.array(vdf.process.values)
            proc_m = np.round(proc.mean(), 3)
            proc_std = np.round(proc.std(), 3)
            print(f'({len(vdf)}) v: {r}, {sep_m}/{sep_std}, {cer_m}/{cer_std}, \ '
                  f'{pos_m}/{pos_std}, {proc_m}/{proc_std},')


def get_qualitative_scores(df):
    sep = np.array(df.separation.values)
    sep_m = np.round(sep.mean(), 3)
    sep_std = np.round(sep.std(), 3)
    cer = np.array(df.certainty.values)
    cer_m = np.round(cer.mean(), 3)
    cer_std = np.round(cer.std(), 3)
    pos = np.array(df.positive.values)
    pos_m = np.round(pos.mean(), 3)
    pos_std = np.round(pos.std(), 3)
    proc = np.array(df.process.values)
    proc_m = np.round(proc.mean(), 3)
    proc_std = np.round(proc.std(), 3)
    return sep_m, sep_std, cer_m, cer_std, pos_m, pos_std, proc_m, proc_std


def get_qualitative_stats_summary(ldf, filter_by_concept=None):
    # on average, how much does each condition differ from original
    # mean and sd
    # collect original interpretations
    transcripts = ldf.transcripts.unique()
    relations = ['original_gesture',
                 'random',
                 'get_closest_gesture_from_row_embeddings',
                 'get_most_similar_sentence_USE',
                 'get_ontology_pos_match',
                 'get_extont_pos_match',
                 'get_ontology_sequence_match',
                 'get_extont_sequence_match']
    cols = ['video_relation', 'sep_m', 'sep_std',
            'cer_m', 'cer_std', 'pos_m', 'pos_std',
            'proc_m', 'proc_std', 'transcript', 'n']

    dfs = []
    ts = []
    vid_relations = []
    sep_ms = []
    sep_stds = []
    cer_ms = []
    cer_stds = []
    pos_ms = []
    pos_stds = []
    proc_ms = []
    proc_stds = []
    n_judgements = []
    for t in transcripts:
        tdf = ldf[ldf.transcripts == t]
        odf = tdf[tdf.video_relation == 'original_gesture']
        if len(odf) < 1:
            print('ERROR: NO ORIGINAL JUDGEMENTS FOUND FOR TRANSCRIPT: ', t)
            continue

        sep_m, sep_std, cer_m, cer_std, pos_m, pos_std, proc_m, proc_std = get_qualitative_scores(odf)

        for r in relations:
            cdf = tdf[tdf.video_relation == r]
            if filter_by_concept:
                cdf = cdf[cdf.have_concept == filter_by_concept]
            if len(cdf) < 1:
                continue
            vid_relations.append(r)
            sep_m_comp, sep_std_comp, cer_m_comp, cer_std_comp, pos_m_comp, pos_std_comp, proc_m_comp, proc_std_comp = get_qualitative_scores(cdf)
            sep_ms.append(abs(sep_m - sep_m_comp))
            sep_stds.append(abs(sep_std - sep_std_comp))
            cer_ms.append(abs(cer_m - cer_m_comp))
            cer_stds.append(abs(cer_std - cer_std_comp))
            pos_ms.append(abs(pos_m - pos_m_comp))
            pos_stds.append(abs(pos_std - pos_std_comp))
            proc_ms.append(abs(proc_m - proc_m_comp))
            proc_stds.append(abs(proc_std - proc_std_comp))
            ts.append(t)
            n_judgements.append(len(cdf))

        df = pd.DataFrame(list(zip(vid_relations,
                                    sep_ms,
                                    sep_stds,
                                    cer_ms,
                                    cer_stds,
                                    pos_ms,
                                    pos_stds,
                                    proc_ms,
                                    proc_stds,
                                    ts,
                                    n_judgements)),
                          columns=cols)
        dfs.append(df)
    if len(dfs) < 2:
        return None
    stats_df = pd.concat(dfs)
    return stats_df


def avg(l):
    return np.array(l).mean()


def aggregate_stats_summary(stats_df):
    agg_df = stats_df.groupby(['video_relation'], as_index=False).agg(
        sep_mean_diff=pd.NamedAgg(column='sep_m', aggfunc=avg),
        sep_std_diff=pd.NamedAgg(column='sep_std', aggfunc=avg),
        cer_m_diff=pd.NamedAgg(column='cer_m', aggfunc=avg),
        cer_std_diff=pd.NamedAgg(column='cer_std', aggfunc=avg),
        pos_m_diff=pd.NamedAgg(column='pos_m', aggfunc=avg),
        pos_std_diff=pd.NamedAgg(column='pos_std', aggfunc=avg),
        proc_m_diff=pd.NamedAgg(column='proc_m', aggfunc=avg),
        proc_std_diff=pd.NamedAgg(column='proc_std', aggfunc=avg),
        n=pd.NamedAgg(column='n', aggfunc=sum)
    )
    return agg_df


# Want to do this but only for samples which contain certain elements.
def do_stats(ldf=None):
    if ldf is None:
        ldf = get_qualitative_df()
    stats_df = get_qualitative_stats_summary(ldf)
    agg_df = aggregate_stats_summary(stats_df)
    return agg_df


def test_ont_features(tdf):
    feats = ['container', 'tangible', 'static', 'dynamic', 'intentional', 'agentic', 'trajectory', 'human']
    for f in feats:
        print("========================")
        print(f)
        adf = do_stats(tdf[tdf[f]])
        adf.to_csv(f'agg_data_{f}.csv')
        # show_density(tdf, plotname=f'density_responses{f}')


def get_ont_features_for_phrase(p):
    ont_feats = cere.generate(p)
    all_feats = set()
    for k in ont_feats.keys():
        if 'Ont' in ont_feats[k].keys():
            all_feats.update(ont_feats[k]['Ont'][1])
    return all_feats


def save_point():
    df = get_qualitative_df()
    # look at the qualitative stats
    get_qualitative_stats(df)
    # aggregate them all
    stats_df = get_qualitative_stats_summary(df)
    agg_df = aggregate_stats_summary(stats_df)

    # so the trick might be to look at this agg_df but only for subsets of trials in which
    # both the original (presented) transcript and the tested gesture transcript contain
    # particular semantic concepts.
    # THEN filter out ones where the 'random' selection is too close??
    # really just need to make 20 sets of 4 violin plots -- 1 for each transcript.
    make_violins_per_question(df)


def test_between_semantic_cats(df):
    cats_to_test = get_top_semantic_categories(df)
    all_cats_df = pd.DataFrame()
    for c in tqdm(cats_to_test):
        tdf = df.copy()
        tdf['have_concept'] = tdf.apply(lambda row: both_have_concept(row, c), axis=1)
        print("Qualitative scores for semantic concept: ", c)
        # now have to actually go through and filter

        print("FIRST HAS THE CONCEPT: ")
        # original has concept, not test
        stats1 = get_qualitative_stats_summary(tdf, filter_by_concept=1)
        agg_df1 = aggregate_stats_summary(stats1)
        agg_df1['concept_in'] = 'original_only'
        print(agg_df1)
        print("SECOND HAS THE CONCEPT: ")
        # test has concept, not original
        stats2 = get_qualitative_stats_summary(tdf, filter_by_concept=2)
        agg_df2 = aggregate_stats_summary(stats2)
        agg_df2['concept_in'] = 'test_only'
        print(agg_df2)
        print("BOTH HAVE THE CONCEPT: ")
        # both have concept
        stats3 = get_qualitative_stats_summary(tdf, filter_by_concept=3)
        agg_df3 = aggregate_stats_summary(stats3)
        agg_df3['concept_in'] = 'both'
        print(agg_df3)
        print("NEITHER HAS THE CONCEPT: ")
        # neither has concept
        stats4 = get_qualitative_stats_summary(tdf, filter_by_concept=4)
        agg_df4 = aggregate_stats_summary(stats4)
        agg_df4['concept_in'] = 'neither'
        print(agg_df4)

        full_df = pd.concat([agg_df1, agg_df2, agg_df3, agg_df4], ignore_index=True)
        full_df['concept'] = str(c)
        c = "".join(ch for ch in c if ch not in ('!', '.', ':', '-', '?'))

        all_cats_df = all_cats_df.append(full_df)

    all_cats_df.to_excel(f'filter_concepts_ALL_II.xlsx', engine='xlsxwriter')


def comp_test_gesture_across_originals(df):
    """
    for all *test* gestures, finds instances in which test gesture t is compared to an
    original gesture and calculates difference across semantic measurements. Aggregates and
    returns stats for all test gestures

    This shows the effect of a specific gesture on interpretation, across all transcripts
    (i.e. we would like to see consistency -- a particular gesture pushes interpretation in a similar way
    regardless of the transcript it is tested against)
    """
    lens = []
    tdfs = []
    for t in df.videoA_transcript.unique():
        tdf = df[df.videoA_transcript == t]
        lens.append(len(tdf))
        tdf = tdf[tdf.video_relation != 'original_gesture']
        if len(tdf) < 5:        # only do it for more than 5 judgements, preliminarily.
            continue
        tdfs.append(tdf)
        if len(tdf.transcripts.unique()) > 1:
            print('EXTRA TRANSCRIPT FOR ', t)
        else:
            continue

        orig_transcripts = list(tdf.transcripts.unique())  # get the transcript that was compared in this round
        print("DOING NEW MATCH FOR ", t)
        sep_diffs = []
        cer_diffs = []
        proc_diffs = []
        pos_diffs = []
        for o in orig_transcripts:
            comp_df = df[df.transcripts == o]
            comp_df = comp_df[comp_df.videoA_transcript != t]
            print('comparing to ', o, f'(n={len(comp_df)})')

            # find stats for all of these
            print(f"separation test: {tdf.separation.mean()} ----- {tdf.separation.std()}")
            print(f"separation orig: {comp_df.separation.mean()} ----- {comp_df.separation.std()}")
            sep_diffs.append(comp_df.separation.mean() - tdf.separation.mean())

            print(f"certainty test: {tdf.certainty.mean()} ----- {tdf.certainty.std()}")
            print(f"certainty orig: {comp_df.certainty.mean()} ----- {comp_df.certainty.std()}")
            cer_diffs.append(comp_df.certainty.mean() - tdf.certainty.mean())

            print(f"process test: {tdf.process.mean()} ----- {tdf.process.std()}")
            print(f"process orig: {comp_df.process.mean()} ----- {comp_df.process.std()}")
            proc_diffs.append(comp_df.process.mean() - tdf.process.mean())

            print(f"positive test: {tdf.positive.mean()} ----- {tdf.positive.std()}")
            print(f"positive orig: {comp_df.positive.mean()} ----- {comp_df.positive.std()}")
            pos_diffs.append(comp_df.positive.mean() - tdf.positive.mean())

        print("sep diffs: ", sep_diffs)
        print("cer diffs: ", cer_diffs)
        print("proc diffs: ", proc_diffs)
        print("pos diffs: ", pos_diffs)



# TODO: first: look at all the possible concepts we have in our original transcripts in df, and how often they occur
# TODO second: identify which ones are likely to overlap with gesture
# TODO need to test effect of SPECIFIC gesture as well
def get_top_semantic_categories(df):
    all_concepts = get_ont_features_in_df(df)
    dict(sorted(all_concepts.items(), reverse=True, key=lambda item: item[1]))
    # find we get
    # tangible:+, inentional:-,container:-,trajectory:-,information:-,container:+,time-span:EXTENDED,  etc...
    keepers = [k for k in all_concepts.keys() if all_concepts[k] >= 5]
    return keepers
    # just test the top 40, ish


# add columns to df:
# concepts_in_original
# concepts_in_test
def get_ont_features_in_df(df):
    all_concepts = {}
    for p in list(df.transcripts.unique()):
        shallow_concepts = get_ontology_sequence(p, cere)
        for s in shallow_concepts:
            for k in s[0]:
                if k in all_concepts.keys():
                    all_concepts[k] = all_concepts[k] + 1
                else:
                    all_concepts[k] = 1
    return all_concepts


# key: 1 = original has concept, 2 = test has concept, 3 = both, 4 = neither
def both_have_concept(row, concept):
    first_concepts = get_flat_ont_sequence(row['transcripts'])
    second_concepts = get_flat_ont_sequence(row['videoA_transcript'])
    if concept in first_concepts and concept not in second_concepts:
        return 1
    elif concept in second_concepts and concept not in first_concepts:
        return 2
    elif concept in second_concepts and concept in first_concepts:
        return 3
    else:
        return 4


def get_flat_ont_sequence(p):
    concepts = set()
    shallow_concepts = get_ontology_sequence(p, cere)
    for s in shallow_concepts:
        concepts = concepts.union(s[0])
    return concepts


# questions
# when there is a certain feature in the original transcript, if it's also in the guessed transcript,
# are those scores closer than when that feature isn't present in the guessed transcript?
# to test:
# for a given transcript, gather all the ont features
# for all alternative transcripts presented, gather all ont features
# compare similarity across the qualitative measures for MATCHING and NON-MATCHING transcripts for each feature

# relatedly,
# do the presence of certain keys predict higher values?
# for example, does the presence of the 'movable' ontological feature predict higher ratings in any qualitative domain?



def compare_matching_nonmatching_ont_features(tdf):
    closer_count = []
    in_relations = []
    out_relations = []
    maj_in_keys = []
    maj_out_keys = []
    equal_keys = []
    for t in tdf.transcripts.unique():
        print("=======================")
        print('transcript: ', t)
        cdf = tdf[tdf.transcripts == t]
        sep_m0, sep_std0, cer_m0, cer_std0, pos_m0, pos_std0, proc_m0, proc_std0 = get_qualitative_scores(cdf[cdf.video_relation == 'original_gesture'])

        t_feats = get_ont_features_for_phrase(t)
        cdf['comparison_feats'] = cdf.apply(lambda row: get_ont_features_for_phrase(row['videoA_transcript']), axis=1)
        ndf = cdf[cdf.video_relation != 'original_gesture']
        for f in t_feats:
            if 'type' in f:
                continue
            print(f'{f}')
            mask = ndf.apply(lambda row: f in row.comparison_feats, axis=1)
            in_df = ndf[mask]
            out_df = ndf[~mask]
            print('in_df: ', len(in_df), 'out_df: ', len(out_df))
            if len(in_df) < 1 or len(out_df) < 1:
                print('-------------------')
                continue
            sep_m1, sep_std1, cer_m1, cer_std1, pos_m1, pos_std1, proc_m1, proc_std1 = get_qualitative_scores(in_df)
            sep_m2, sep_std2, cer_m2, cer_std2, pos_m2, pos_std2, proc_m2, proc_std2 = get_qualitative_scores(out_df)
            sep_close = A_B_closer(sep_m0, sep_m1, sep_m2)
            cer_close = A_B_closer(cer_m0, cer_m1, cer_m2)
            pos_close = A_B_closer(pos_m0, pos_m1, pos_m2)
            proc_close = A_B_closer(proc_m0, proc_m1, proc_m2)
            closer_count += [sep_close, cer_close, pos_close, proc_close]
            print('sep: ', sep_m0, sep_m1, sep_m2, f'({A_B_closer(sep_m0, sep_m1, sep_m2)})')
            print('cer: ', cer_m0, cer_m1, cer_m2, f'({A_B_closer(cer_m0, cer_m1, cer_m2)})')
            print('pos: ', pos_m0, pos_m1, pos_m2, f'({A_B_closer(pos_m0, pos_m1, pos_m2)})')
            print('proc: ', proc_m0, proc_m1, proc_m2, f'({A_B_closer(proc_m0, proc_m1, proc_m2)})')
            print('in:', list(in_df.video_relation.values))
            print('out', list(out_df.video_relation.values))
            in_relations += list(in_df.video_relation.values)
            out_relations += list(out_df.video_relation.values)
            if len([l for l in [sep_close, cer_close, pos_close, proc_close] if l == 'A']) > 2:
                maj_in_keys.append(f)
            elif len([l for l in [sep_close, cer_close, pos_close, proc_close] if l == 'A']) < 2:
                maj_out_keys.append(f)
            else:
                equal_keys.append(f)
            print('--------------------')
    print('Matching gestures closer: ', len([l for l in closer_count if l == 'A']))
    print('Non-Matching gestures closer: ', len([l for l in closer_count if l == 'B']))
    print('Matching gestures by mechanism: ')
    print('random', len([l for l in in_relations if l == 'random']))
    print('original_gesture', len([l for l in in_relations if l == 'original_gesture']))
    print('get_closest_gesture_from_row_embeddings', len([l for l in in_relations if l == 'get_closest_gesture_from_row_embeddings']))
    print('get_extont_pos_match', len([l for l in in_relations if l == 'get_extont_pos_match']))
    print('get_ontology_pos_match', len([l for l in in_relations if l == 'get_ontology_pos_match']))
    print('Non-matching gestures by mechanism: ')
    print('random', len([l for l in out_relations if l == 'random']))
    print('original_gesture', len([l for l in out_relations if l == 'original_gesture']))
    print('get_closest_gesture_from_row_embeddings', len([l for l in out_relations if l == 'get_closest_gesture_from_row_embeddings']))
    print('get_extont_pos_match', len([l for l in out_relations if l == 'get_extont_pos_match']))
    print('get_ontology_pos_match', len([l for l in out_relations if l == 'get_ontology_pos_match']))
    print('.........')
    print('majority in keys: ', maj_in_keys)
    print('majority out keys: ', maj_out_keys)
    print('split keys: ', equal_keys)
    maj_in = set(maj_in_keys)
    maj_out = set(maj_out_keys)
    eq = set(equal_keys)
    print('intersection btw in/out: ', len(set.intersection(maj_in, maj_out)))


def A_B_closer(o, a, b):
    return 'A' if abs(o-a) < abs(o-b) else 'B'


def make_violins_per_question(df):
    trans = df.transcripts.unique()
    for t in trans:
        tdf = df[df.transcripts == t]
        fig = go.Figure
        measures = ['separation', 'certainty', 'process', 'positive']
        for m in measures:
            fig.add_trace(px.violin(tdf, y=m, color="video_relation",
                    violinmode='overlay', # draw violins on top of each other
                    # default violinmode is 'group' as in example above
                    hover_data=tdf.columns))
        fig.savefig('testquestions.html')


def show_density(df):
    # print out a plot for each transcript
    for t in df.transcripts.unique():
        tdf = df[df.transcripts == t]
        stripped = re.sub(r'[^\w\s]','', t)
        plot_title = "_".join(stripped.split(" ")[:4])

        tdf['video_relation'] = tdf.apply(lambda row:
                'original gesture' if row['video_relation'] == 'original_gesture'
           else ('BERT embedding' if row['video_relation'] == 'get_closest_gesture_from_row_embeddings'
           else ('ontology POS match' if row['video_relation'] == 'get_ontology_pos_match'
           else ('extended ontology POS match' if row['video_relation'] == 'get_extont_pos_match' else 'random'))),
                                       axis=1)
        color_discrete_map = {
            'original gesture': '#1f77b4',
            'BERT embedding': '#ff7f0e',
            'ontology POS match': '#2ca02c',
            'extended ontology POS match': '#9467bd',
            'random': '#d62728'
        }

        fig_sep = px.violin(tdf, y="separation", color="video_relation",
                            color_discrete_map=color_discrete_map,
                        violinmode='overlay', # draw violins on top of each other
                        # default violinmode is 'group' as in example above
                        hover_data=tdf.columns,
                            title="Separation")
        fig_sep_traces = []
        for trace in range(len(fig_sep["data"])):
            fig_sep_traces.append(fig_sep["data"][trace])

        fig_cer = px.violin(tdf, y="certainty", color="video_relation",
                            color_discrete_map=color_discrete_map,
                            violinmode='overlay', # draw violins on top of each other
                        # default violinmode is 'group' as in example above
                        hover_data=tdf.columns,
                            title="Certainty")
        fig_cer_traces = []
        for trace in range(len(fig_cer["data"])):
            fig_cer_traces.append(fig_cer["data"][trace])

        fig_proc = px.violin(tdf, y="process", color="video_relation",
                             color_discrete_map=color_discrete_map,
                             violinmode='overlay', # draw violins on top of each other
                        # default violinmode is 'group' as in example above
                        hover_data=tdf.columns,
                             title="Process")
        fig_proc_traces = []
        for trace in range(len(fig_proc["data"])):
            fig_proc_traces.append(fig_proc["data"][trace])

        fig_pos = px.violin(tdf, y="positive", color="video_relation",
                            color_discrete_map=color_discrete_map,
                            violinmode='overlay', # draw violins on top of each other
                        # default violinmode is 'group' as in example above
                        hover_data=tdf.columns,
                            title="Positive")
        fig_pos_traces = []
        for trace in range(len(fig_pos["data"])):
            fig_pos_traces.append(fig_pos["data"][trace])

        tf = sp.make_subplots(rows=2, cols=2,
                              subplot_titles=("Separation", "Certainty", "Process", "Positive"))
        for traces in fig_sep_traces:
            tf.append_trace(traces, row=1, col=1)
        for traces in fig_cer_traces:
            tf.append_trace(traces, row=1, col=2)
        for traces in fig_proc_traces:
            tf.append_trace(traces, row=2, col=1)
        for traces in fig_pos_traces:
            tf.append_trace(traces, row=2, col=2)

        tf.update_layout(title_text=f"{stripped}; n={len(tdf)}")
        tf.write_html(f'semantic_densities_{plot_title}.html')



def try_likert_filtering(ldf, energy_lim=50, energy_min=25):
    print('HIGH ENERGY')
    tdf = ldf[ldf['energy_response'] > energy_lim]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'energy.high_energy (p={p})')

    print('LOW ENERGY')
    tdf = ldf[ldf['energy_response'] < energy_min]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'energy.low_energy (p={p})')

    print('LONG')
    tdf = ldf[ldf['vidA_length'] >= 2.3]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'length over 2s (p={p})')

    print('SHORT')
    tdf = ldf[ldf['vidA_length'] <= 2.3]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'length under 2s (p={p})')

    print('LONG WORDS')
    tdf = ldf[ldf['num_words_transcript'] >= 9]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'wc over 8 words (p={p})')

    print('SHORT WORDS')
    tdf = ldf[ldf['num_words_transcript'] < 9]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'wc under 8 words (p={p})')

    print('HIGH SEMANTIC')
    tdf = ldf[ldf['semantic_response'] >= 70]
    p = np.round(print_likert_stats(tdf), 3)
    likert_violin_plots(tdf, plotname=f'wc under 8 words (p={p})')


"""
['get_most_similar_sentence_USE', 'original_gesture',
       'get_closest_gesture_from_row_embeddings', 'get_extont_pos_match',
       'get_extont_sequence_match', 'random', 'get_ontology_pos_match',
       'get_ontology_set_match', 'get_ontology_sequence_match']
"""
def print_likert_stats(ldf):
    all_sem = np.array([])
    all_eng = np.array([])
    for k in ldf.video_relation.unique():
        tdf = ldf[ldf.video_relation == k]
        sem_scores = np.array(tdf['semantic_response'])
        eng_scores = np.array(tdf['energy_response'])
        all_sem = np.append(all_sem, sem_scores)
        all_eng = np.append(all_eng, eng_scores)
        print('k:', k)
        print('avg/sd sem: ', sem_scores.mean(), sem_scores.std())
        print('avg/sd eng: ', eng_scores.mean(), eng_scores.std())
        print('--------------------------')

    print('Score correlations: ')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, all_eng)
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)

    model = ols('semantic_response ~ C(video_relation)', data=ldf).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print("====================================")
    print("ANOVA summary:")
    print(aov_table)
    return aov_table.values[0][3]


def print_likert_stats_details(ldf):
    all_sem = np.array(ldf.semantic_response.values)
    print('Score correlations: semantic vs. embedding dist')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, np.array(ldf.vidA_embedding_distance.values))
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)

    print('Score correlations: semantic vs. use dist')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, np.array(ldf.vidA_USE_distances.values))
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)

    print('Score correlations: semantic vs. ont match')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, np.array(ldf.A_ontology_match.values))
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)

    print('Score correlations: semantic vs. extont match')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, np.array(ldf.A_extontology_match.values))
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)

    print('Score correlations: semantic vs. ont pos match')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, np.array(ldf.A_ontpos_match.values))
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)

    print('Score correlations: semantic vs. extont pos match')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_sem, np.array(ldf.A_extontpos_match.values))
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('r2_value: ', r_value ** 2)
    print('p_value: ', p_value)


def transcript_has_key(row, target_key=""):
    ont_feats = cere.generate(row['transcripts'])
    for k in ont_feats.keys():
        if 'Ont' in ont_feats[k].keys():
            if target_key in ont_feats[k]['Ont'][1]:
                return True
    return False


def get_container_gestures(ldf):
    tdf = ldf.copy()
    tdf['container'] = tdf.apply(lambda row: transcript_has_key(row, target_key='container:+'), axis=1)
    tdf['tangible'] = tdf.apply(lambda row: transcript_has_key(row, target_key='tangible:+'), axis=1)
    tdf['static'] = tdf.apply(lambda row: transcript_has_key(row, target_key='aspect:STATIC'), axis=1)
    tdf['dynamic'] = tdf.apply(lambda row: transcript_has_key(row, target_key='aspect:DYNAMIC'), axis=1)
    tdf['intentional'] = tdf.apply(lambda row: transcript_has_key(row, target_key='intentional:+'), axis=1)
    tdf['agentic'] = tdf.apply(lambda row: transcript_has_key(row, target_key='cause:AGENTIVE'), axis=1)
    tdf['trajectory'] = tdf.apply(lambda row: transcript_has_key(row, target_key='trajectory:+'), axis=1)
    tdf['human'] = tdf.apply(lambda row: transcript_has_key(row, target_key='origin:HUMAN'), axis=1)
    return tdf


def likert_violin_plots(ldf, plotname='violin'):
    tdf = pd.melt(ldf, id_vars=['Trial Number', 'video_relation'], value_vars=['semantic_response', 'energy_response'],
                  var_name='rating_axis', value_name='user_value')

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10))
    plt.xlabel('Group', fontsize=18)
    plt.title(f'Semantic appropriateness {plotname}_split', fontsize=22)
    ax = sns.violinplot(x='video_relation', y="user_value",
                        hue='rating_axis',
                        scale='width',
                        split=True,
                        data=tdf,
                        cut=0,
                        palette='Set2',
                        order=["original_gesture", "random",
                               "get_closest_gesture_from_row_embeddings", "get_most_similar_sentence_USE",
                               "get_ontology_set_match",
                               "get_ontology_sequence_match", "get_extont_sequence_match",
                               "get_ontology_pos_match", "get_extont_pos_match"])
    fig = ax.get_figure()
    plt.xticks(rotation=20, fontsize='xx-small')

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    fig.savefig(f'{plotname}_split.png')

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10))
    plt.xlabel('Group', fontsize=18)
    plt.title(f'Semantic appropriateness {plotname}', fontsize=22)
    ax = sns.violinplot(x='video_relation', y="semantic_response",
                        scale='width',
                        data=ldf,
                        cut=0,
                        order=["original_gesture", "random",
                               "get_closest_gesture_from_row_embeddings", "get_most_similar_sentence_USE",
                               "get_ontology_set_match",
                               "get_ontology_sequence_match", "get_extont_sequence_match",
                               "get_ontology_pos_match", "get_extont_pos_match"])
    fig = ax.get_figure()
    plt.xticks(rotation=20, fontsize='xx-small')

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    fig.savefig(f'{plotname}.png')

color_map = {
    'original_gesture': '#7fc97f',
    'random': '#beaed4',
    'get_most_similar_sentence_USE': '#fdc086',
    'get_closest_gesture_from_row_embeddings': '#ffff99',
    'get_ontology_pos_match': '#386cb0',
    'get_extont_pos_match': '#f0027f',
    'get_ontology_sequence_match': '#bf5b17',
    'get_extont_sequence_match': '#666666',
     'get_ontology_set_match': 'black'
}

def plot_likert_sem_vs_energy(ldf, plotname='semantic vs. energy'):
    tdf = ldf.copy()
    tdf['color'] = tdf.apply(lambda row: color_map[row['video_relation']], axis=1)
    ax = ldf.plot.scatter(x='semantic_response',
                          y='energy_response',
                          c=tdf.color.values)
    fig = ax.get_figure()
    # plt.xticks(rotation=20, fontsize='xx-small')

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    fig.savefig(f'{plotname}.png')


## TODO CAROLYN HERE IS THE MAIN STUFF
raw_df = load_data_II()
answer_df = get_answer_df(raw_df)
trial_df = get_trial_df(answer_df)


#######################################
# COPY wn parsing stuff
def prune_ontology(ont_set):
    n_set = set()
    n_keys = []
    for el in ont_set:
        key = el.split(':')[0]
        if key not in n_keys:
            n_set.add(str(el))
            n_keys.append(key)
    return n_set


def get_ontology_sequence(p, cere, feat_set=None):
    if not feat_set:
        feat_set = cere.generate(p)
    words = p.rstrip().split(' ')
    words = [s.translate(str.maketrans('', '', string.punctuation)) for s in words]
    ont_sequence = []
    ont_words = []
    for w in words:
        if w in feat_set.keys():
            if 'Ont' in feat_set[w].keys():
                pruned = prune_ontology(feat_set[w]['Ont'][1])
                ont_sequence.append(pruned)
                ont_words.append(w)
            elif 'ExtOnt' in feat_set[w].keys():        # if there's no ontology, use the extont.
                pruned = prune_ontology(feat_set[w]['ExtOnt'][1])
                ont_sequence.append(pruned)
                ont_words.append(w)
    return list(zip(ont_sequence, ont_words))


# extra ont
def get_extont_sequence(p, cere, feat_set=None):
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


def get_hypernyms(p, cere, feat_set=None):
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


def get_parsed_sentence(p, cere, feat_set=None):
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

