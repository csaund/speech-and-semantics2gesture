import json
import pandas as pd
import os
from argparse import ArgumentParser

SEMANTIC_CATEGORIES = {
    'FEELINGS': ['angry', 'sad', 'happy', 'love', 'passion', 'anxious', 'stress', 'worry', 'worried'],
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
    'LIST_OF': [' two', 'some'],
    'THEM': ['somebody', 'something', 'you guys'],
    'CATEGORICAL': ['whole', 'everything', 'everyone', ' all ', 'crammed', 'same place', 'stuff', 'all over'],
    'TIME_CYCLICAL': [' cycle', ' cycles', 'cyclical'],
    'NEGATION': [' not', ' nothing', 'nobody', 'noone', 'no '],
    'QUESTIONS': ['where are', 'what are', 'what is', 'how are', 'how is', 'how do', 'where are', 'where is'],
    'PLACES': ['back at', 'back in', 'home', 'over there', 'over here', 'right here', 'get there', 'behind'],
    'ENUMERATION': ['one', 'two', 'three'],
    'PATHS': ['going', 'went', 'journey', 'becoming', 'go off', 'come back', 'goes'],
    'CAUSE': ['because', 'due to', 'owing to']
}


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


def get_nearest_row(df, i):
    comp_row = df.iloc[i]
    ndf = df.copy()
    ndf['comp_dists'] = ndf.apply(lambda row: dist_lambda(row, comp_row), axis=1)
    ndf = ndf.sort_values(by='comp_dists', ascending=False)
    # need to return the highest match that isn't the same phrase
    ret_row = ndf.iloc[1]                       # highest match is probably the second one
    if ret_row['PHRASE'] != comp_row['PHRASE']:     # but if there's a perfect match, make sure to return a different one
        return ret_row
    else:
        return ndf.iloc[0]        # if it's a perfect match!


# will create clusters in form of df
def create_category_clusters(df):
    clusters = {}
    for k in SEMANTIC_CATEGORIES.keys():
        clusters[k] = {'df': None, 'len': 0}
        clusters[k]['df'] = df[df[k] != '-']
        clusters[k]['len'] = len(clusters[k]['df'])
    return clusters


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', '-f', default="",
                                   help="txt file of transcript with split lines")
    parser.add_argument('--output', '-o', default="category_output.csv",
                                   help="what to name the output csv")
    params = parser.parse_args()
    f = params.file
    df = pd.DataFrame(columns=['PHRASE'] + list(SEMANTIC_CATEGORIES.keys()))

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

    cols = list(df.columns)
    cols.remove('PHRASE')
    df = df.dropna(subset=cols, how='all')      # toss all gestures that have none of our semantics
    df = df.fillna('-')
    df.to_csv(params.output)




