from bert_embedding import BertEmbedding
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import os
import pickle
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt


def plot_distance_matrix(dif_mat):
    n = int(np.sqrt(R.size))
    C = R.reshape((n, n))

    # Plot the matrix
    plt.matshow(C, cmap="Reds")

    ax = plt.gca()

    # Set the plot labels
    xlabels = ["B%d" % i for i in xrange(n + 1)]
    ylabels = ["A%d" % i for i in xrange(n + 1)]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Add text to the plot showing the values at that point
    for i in xrange(n):
        for j in xrange(n):
            plt.text(j, i, C[i, j], horizontalalignment='center', verticalalignment='center')

    plt.show()


def get_distance_matrix(df):
    dist_mat = (df['encoding'].to_numpy(), df['encoding'].to_numpy())
    return dist_mat[1]


def get_encoding_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.linalg.norm(v1 - v2)
    return dist


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--pickle', '-orig', default="",
                                   help="Path where original motion files (in BVH format) are stored")

    params = parser.parse_args()

    df = pd.read_pickle(params.pickle)
