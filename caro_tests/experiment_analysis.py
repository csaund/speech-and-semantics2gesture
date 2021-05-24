import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import scipy
import scipy.stats
plt.switch_backend('agg')

# TODO FIRST THING:
# TODO DOUBLE CHECK THAT THE RESULTS ARE FROM
# TODO DIFFERENT ANALYSIS PAIRS!!!
# todo aka make sure it's not the same videos being compared within a comparison group

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
    return float(t[-1])


def analyse_likert_data():
    likert_files = os.listdir('experiment_likert_data')
    dfs = []
    for f in likert_files:
        print(f)
        if 'task' in f:
            ldf = pd.read_csv(os.path.join('experiment_likert_data', f))
            if len(ldf) < 25:
                continue
            ldf = ldf[ldf['Screen Name'] == 'Screen 1']
            ldf = ldf[ldf['display'] == 'participant_likert_view']
            ldf['sem_res'] = ldf.apply(lambda row: row['Response'] if row['Zone Name'] == 'VideoA' else None, axis=1)
            ldf['eng_res'] = ldf.apply(lambda row: row['Response'] if row['Zone Name'] == 'Zone6' else None, axis=1)
            ldf = ldf.groupby(['Trial Number', 'video_relation'], as_index=False).agg(
                semantic_response=pd.NamedAgg(column='sem_res', aggfunc=last),
                energy_response=pd.NamedAgg(column='eng_res', aggfunc=last)
            )
            dfs.append(ldf)
    df = pd.concat(dfs)
    return df

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


## TODO CAROLYN HERE IS THE MAIN STUFF
raw_df = load_data_II()
answer_df = get_answer_df(raw_df)
trial_df = get_trial_df(answer_df)