from argparse import ArgumentParser
import pandas as pd
import numpy as np
import random
import math
import os
import sys
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def print_transcripts_by_cluster_id(clusters, id):
    for t in clusters[id]:
        print(t)


# lol disgusting
def get_transcript_by_feature_vector(df, vec):
    el = df[df['features'].isin([vec])]
    if len(el) > 0:
        return el.iloc[0].transcript


def get_gesture_clusters(df):
    X, transcripts = df_to_kmeans_data(df)
    kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4,
                    random_state=0).fit(X)
    z_labs = zip(kmeans.labels_, transcripts)

    clusters = {}
    for l, t in z_labs:
        if l in clusters.keys():
            clusters[l].append(' '.join(t))
        else:
            clusters[l] = [' '.join(t)]
    return clusters


# get it in format for k means
# todo do this by df indeces
def df_to_kmeans_data(df):
    df['features'] = df.apply(lambda row: row['encoding'][1], axis=1)
    M = []
    trans = []
    for i in range(len(df)):
        M.append(df.iloc[i].features)
        trans.append(df.iloc[i].transcript)
    M = np.array(M)
    return M, trans



def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))


def test_kmeans(dat):
    kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4,
                    random_state=0)
    bench_k_means(kmeans=kmeans, name="k-means++", data=dat, labels=labels)


def get_random_closest_gestures(df):
    i = random.randint(0, len(df))
    a = df.iloc[i]
    b = get_closest_gesture(df, i)
    print(' '.join(a.transcript))
    print(' '.join(b.transcript))
    return a, b


def get_nonzero_min_ind(l):
    m = max(l)
    ind = 0
    for i in range(len(l)):
        if l[i] < m and l[i] != 0:
            m = l[i]
            ind = i
    return m, ind


def get_closest_gesture(df, i):
    df['features'] = df.apply(lambda row: row['encoding'][1], axis=1)
    interest_vector = df.iloc[i].features
    distances = df.apply(lambda row: get_encoding_dist(interest_vector, row['features']), axis=1)
    val, i = get_nonzero_min_ind(distances)
    return df.iloc[i]


def get_encoding_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.linalg.norm(v1 - v2)
    return dist


if __name__ == "__main))":

    parser = ArgumentParser()
    parser.add_argument('--dir', '-orig', default="",
                        help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--encoding_path', '-e', default="",
                        help="Path where original motion files (in BVH format) are stored")

    params = parser.parse_args()

    df = pd.read_pickle(params.encoding_path)
