import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import scipy

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
    tdf['analysis_group'] = tdf.apply(lambda row: category_mapper[row['video_relation']] if row['video_relation'] != np.nan else None, axis=1)
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
    if not df:
        df = load_data()
    # get only the full experiment
    df = df[df['Screen Name'] == 'Screen 1']
    df = df[df['display'] == 'video_participant_view']
    df = df[~df['video_relation'].isin(['CONTROL_BROKEN_VIDEO', 'CONTROL_SAME_VIDEO'])]
    # get only the responses to the questions
    df['Response Type'] = df.apply(lambda row: 'semantic' if row['Zone Name'] == 'VideoA' \
                                                          else (
                                                          'energy' if row['Zone Name'] == 'Zone6'
                                                                   else np.nan), axis=1)
    df = df.dropna(subset=['Response Type'])
    # use Trial Number to
    df['trial_num'] = flatten([[n, n] for n in range(1, int(len(df)/2)+1)])
    # fuck it just create a new dataframe
    # todo this is hacky as shit.
    # todo get rid of broken links / attention checks
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

    ndf = add_analysis_category(ndf)
    return ndf


def isNaN(num):
    return num != num


def plot_use_dist_vs_ont(df):



flatten = lambda t: [item for sublist in t for item in sublist]

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

