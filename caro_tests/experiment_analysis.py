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

    exit(0)