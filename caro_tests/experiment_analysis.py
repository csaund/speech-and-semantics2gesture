import pandas as pd
import os
import matplotlib.pyplot as plt
import re


if __name__ == "__main__":
    data_df = pd.read_csv(os.path.join('speech-and-semantics2gesture', 'trial_data.csv'))
    A_responses = data_df[data_df['Response'] == 'Video A']
    B_responses = data_df[data_df['Response'] == 'Video B']
    correctA = A_responses[A_responses['correct_video'] == 0]
    correctB = B_responses[B_responses['correct_video'] == 1]
    correct = correctA.append(correctB)
    incorrectA = A_responses[A_responses['correct_video'] == 1]
    incorrectB = B_responses[B_responses['correct_video'] == 0]
    incorrect = incorrectA.append(incorrectB)
    answer_df = correct.append(incorrect)

    # plot x dist vs. y dist where x is correct answer
    # green is predicted answers, red is deviant answers
    fig, ax = plt.subplots(1, 1)
    xs = answer_df.apply(lambda row: row['vidA_embedding_distance'] if row['correct_video'] == 0 else row['vidB_embedding_distance'], axis=1)
    ys = answer_df.apply(lambda row: row['vidA_embedding_distance'] if row['correct_video'] == 1 else row['vidB_embedding_distance'], axis=1)
    color = answer_df.apply(lambda row: "green" if row['correct_video'] == 0 and row['Response'] == 'Video A' else "red", axis=1)

    ax.scatter(xs, ys, color=color)
    ax.set_title('Correct vs. incorrect embedding distances')
    ax.set_xlabel('Correct video embedding distance to original transcript')
    ax.set_ylabel('Incorrect video embedding distance to original transcript')
    ax.legend(['Correct responses', 'incorrect responses'])
    fig.show()

    # plot % overlapping categories with color
    set_df = answer_df[['transcript_categories', 'vidA_overlap', 'vidB_overlap', 'correct_video']]
    d = "{''}"
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
