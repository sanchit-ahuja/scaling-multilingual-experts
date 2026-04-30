import os

from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import random
import numpy as np
import pickle

import lang2vec.lang2vec as l2v
from iso639 import Lang  # this is the iso639-lang package
import math
import time

CLUSTER_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/clustering/"

SORTED_LANGS = [
    "en",
    "ru",
    "zh",
    "de",
    "es",
    "fr",
    "ja",
    "it",
    "pt",
    "el",
    "ko",
    "fi",
    "id",
    "tr",
    "ar",
    "vi",
    "th",
    "bg",
    "ca",
    "hi",
    "et",
    "bn",
    "ta",
    "ur",
    "sw",
    "te",
    "eu",
    "my",
    "ht",
    "pl",
    "nl",
    "cs",
    "fa",
    "uk",
    "ro",
    "sv",
    "hu",
    "da",
    "no",
    "mr",
    "sk",
    "he",
    "ms",
    "lt",
    "sl",
    "lv",
    "sr",
    "cy",
    "az",
    "kk",
    "sq",
    "ne",
    "mt",
    "mn",
    "ka",
    "hy",
    "ml",
    "be",
    "gl",
    "is",
    "mk",
    "fil",
    "tg",
    "af",
    "kn",
    "ky",
    "sd",
    "la",
    "so",
    "si",
    "km",
    "uz",
    "lb",
    "gu",
    "eo",
    "pa",
    "ps",
    "ga",
    "fy",
    "ku",
    "gd",
    "am",
    "yi",
    "jv",
    "ha",
    "zu",
    "hmn",
    "co",
    "mg",
    "ceb",
    "sn",
    "lo",
    "su",
    "ny",
    "mi",
    "ig",
    "sm",
    "st",
    "haw",
    "xh",
    "yo",
]


def _distance_to_centroid(X, centroid):
    mask = ~np.isnan(X)
    yy = np.nansum((centroid * mask) ** 2, axis=1)
    xx = np.nansum(X**2, axis=1)
    xy = np.nansum(X * centroid, axis=1)
    return 1 - (xy / (np.sqrt(xx) * np.sqrt(yy) + 1e-8))


def _initialize(X, num_clusters):
    data_size = X.shape[0]
    max_size = math.ceil(data_size / num_clusters)
    mod = data_size % num_clusters

    # k means initalization over the data
    def _kmeans_plusplus(X, num_clusters, data_size):
        # pick first random centroid
        c_id = np.random.choice(data_size, size=1, replace=False)
        centroid_ids = [c_id]
        centroids = [X[c_id]]

        # select rest of centroids
        while len(centroid_ids) < num_clusters:
            # remaining candidate data pts
            data_pts = [x for x in list(range(data_size)) if x not in centroid_ids]
            X_data = X[np.array(data_pts)]
            # calculate distance to chosen centroids
            dists = [_distance_to_centroid(X_data, c) for c in centroids]
            # min dist to chosen centroids for each data pt
            dists = np.stack(dists, axis=1).min(axis=1)
            # normalize into probabilities
            dists = dists**2
            sum_dists = np.sum(dists)
            dists = dists / sum_dists
            # choice new centroid
            c_id = np.random.choice(data_pts, size=1, replace=False, p=dists)
            centroid_ids.append(c_id)
            centroids.append(X[c_id])

        return centroids

    centroids = _kmeans_plusplus(X, num_clusters, data_size)

    # get balanced initalization based on kmeans++ centroids
    def _balanced_assignments(X, centroids, data_size):
        assigned_ids = []
        cluster_ids = ["--" for i in range(data_size)]
        full_clusters = []

        # assign each data pt to a cluster
        while len(assigned_ids) < data_size:
            # (0a) get unassigned data pts
            data_pts = [x for x in list(range(data_size)) if x not in assigned_ids]
            data_map = {i: x for i, x in enumerate(data_pts)}
            X_data = X[np.array(data_pts)]
            # (0b) get this cycle's available centroids and make mapping
            cluster_map = {}
            cycle_centroids = []
            for i, c in enumerate(centroids):
                if i not in full_clusters:
                    cycle_centroids.append(c)
                    cluster_map[len(cycle_centroids) - 1] = i
            # (1) calculate distances to centroids for all data
            dists = [_distance_to_centroid(X_data, c) for c in cycle_centroids]
            dists = np.stack(dists, axis=1)
            # (2) get utility metric = (min dist to centroid) - (max dist to a centroid)
            min_dists = dists.min(axis=1)
            max_dists = dists.max(axis=1)
            utility = min_dists - max_dists
            # (3) assign data point to cluster based on utility metric
            min_ids = dists.argmin(axis=1)
            d_id = utility.argmin()
            c_id = min_ids[utility.argmin()]
            d_id = data_map[d_id]
            c_id = cluster_map[c_id]
            assigned_ids.append(d_id)
            cluster_ids[d_id] = c_id
            # (4) if cluster is full, remove from centroid options for rest of assignments
            c_count = len([x for x in cluster_ids if x == c_id])
            if len(full_clusters) < mod:
                if c_count == max_size:
                    full_clusters.append(c_id)
                # handle clusters filled to max_size - 1 after meeting mod allotment
                if len(full_clusters) == mod:
                    for j in range(0, len(centroids)):
                        if (j not in full_clusters) and (
                            len([x for x in cluster_ids if x == j]) == (max_size - 1)
                        ):
                            full_clusters.append(j)
            elif c_count == (max_size - 1):
                full_clusters.append(c_id)
        return cluster_ids

    cluster_ids = _balanced_assignments(X, centroids, data_size)

    return cluster_ids


def _load_features(langs, feature_type):
    input_path = f"{CLUSTER_PREFIX}/{feature_type}_feats.pkl"
    with open(input_path, "rb") as f:
        v_dict = pickle.load(f)
    filter_dict = {}
    for lang in langs:
        filter_dict[lang] = v_dict[lang]
    return filter_dict


def _lang_features(langs, sim_metric):
    if sim_metric == "syntax":
        v_dict = l2v.get_features(langs, "syntax_average", minimal=True)
    if sim_metric == "geo":
        v_dict = l2v.get_features(langs, "geo", minimal=True)
    if sim_metric == "lexical":
        v_dict = _load_features(langs, "lexical")
    print("loaded features")
    return v_dict


def _within_cluster_cosine_distance(X, cluster_idxs, cluster_idx):
    # Choose only the rows belonging to the specified cluster
    X_idx = X[cluster_idxs == cluster_idx]

    # The centroid is the average of these columns

    centroid = np.nanmean(X_idx, axis=0, keepdims=True)

    # The centroid will only be nan when all rows are nan
    mask = ~np.isnan(X_idx)

    yy = np.nansum((centroid * mask) ** 2, axis=1)
    xx = np.nansum(X_idx**2, axis=1)
    xy = np.nansum(X_idx * centroid, axis=1)
    return np.sum(1 - (xy / (np.sqrt(xx) * np.sqrt(yy) + 1e-8)))


def cluster_once(X, num_clusters, initialization):
    if initialization == "kmeans++":
        # use balanced kmeans++ initialization
        cluster_idxs = _initialize(X, num_clusters)
        cluster_idxs = np.array(cluster_idxs)
    else:
        # initialize random clusters
        cluster_idxs = np.arange(len(X)) % num_clusters
        np.random.shuffle(cluster_idxs)

    # Calculate the set of initial distances
    distances = [
        _within_cluster_cosine_distance(X, cluster_idxs, idx)
        for idx in range(num_clusters)
    ]

    # sort to improve
    NUM_FLIPS = 50_000  # 100_000
    realized_flips = 0

    for itr in range(0, NUM_FLIPS):
        # Choose what pair of cluster labels indexes to flip
        a_idx, b_idx = np.random.choice(
            len(cluster_idxs), size=2, replace=False
        )  # The indexes in the array
        cluster_a, cluster_b = (
            cluster_idxs[a_idx],
            cluster_idxs[b_idx],
        )  # The actual cluster IDs
        cluster_idxs[a_idx], cluster_idxs[b_idx] = (
            cluster_b,
            cluster_a,
        )  # Perform the flip

        # Calculate the centroid for each cluster and cosine similarity within it
        da = _within_cluster_cosine_distance(X, cluster_idxs, cluster_a)
        db = _within_cluster_cosine_distance(X, cluster_idxs, cluster_b)

        # If the sum of distances is better now than before, make the flip
        if (da + db) < (distances[cluster_a] + distances[cluster_b]):
            realized_flips += 1
            distances[cluster_a] = da
            distances[cluster_b] = db

        # Otherwise reset the pair of labels
        else:
            cluster_idxs[a_idx], cluster_idxs[b_idx] = cluster_a, cluster_b

    return distances, cluster_idxs


def random_clusters(
    lang_codes, num_clusters, sim_metric, num_shuffles=50, initialization="random"
):
    print(num_clusters, sim_metric, num_shuffles)
    # skip learning clusters if every lang is a seperate cluster
    # (skips expensive process of calculating lang features)
    if num_clusters == len(lang_codes):
        cluster_map = {lang: i for i, lang in enumerate(lang_codes)}
        return cluster_map

    # get vector representations for each lang
    random.shuffle(lang_codes)
    if sim_metric in ["lexical"]:
        lang3_codes = lang_codes
    else:
        lang3_codes = [
            Lang(lang).pt3 if lang != "iw" else Lang("he").pt3 for lang in lang_codes
        ]
    v_dict = _lang_features(lang3_codes, sim_metric)
    X = np.array(
        [[v if v != "--" else np.nan for v in v_dict[lang]] for lang in lang3_codes]
    )

    f = delayed(cluster_once)
    results = Parallel(n_jobs=-1)(
        f(X, num_clusters, initialization) for _ in tqdm(range(num_shuffles))
    )

    best_cluster_idxs = None
    best_distances = [np.inf for i in range(num_clusters)]

    for distances, idxs in results:
        if sum(distances) < sum(best_distances):
            best_distances = distances
            best_cluster_idxs = idxs

    lang_codes = np.array(lang_codes)
    clusters = []
    for i in range(num_clusters):
        clusters.append(list(lang_codes[best_cluster_idxs == i]))

    for d, c in zip(best_distances, clusters):
        print(c, d)
    print(sum(best_distances))


"""
UPDATED FUNCTIONS TO USE REGRESSION MODEL 
AS DISTANCE METRIC FOR CLUSTERING
"""


def _regression_to_centroid(X, centroid, model):
    # calculate elementwise distance to centroid
    centroid_arr = [centroid] * X.shape[0]
    centroid = np.concatenate(centroid_arr, axis=0)
    X_dist = X - centroid

    # predict performance on cluster centroid
    y = model.predict(X_dist)
    # y[y>0] = 0
    return y


def _within_cluster_regression_distance(X, cluster_idxs, cluster_idx, model):
    # Choose only the rows belonging to the specified cluster
    X_idx = X[cluster_idxs == cluster_idx]

    # The centroid is the average of these columns
    centroid = np.mean(X_idx, axis=0, keepdims=True)
    centroid_arr = [centroid] * X_idx.shape[0]
    centroid = np.concatenate(centroid_arr, axis=0)

    X_dist = X_idx - centroid

    # predict performance on cluster centroid
    y = model.predict(X_dist)
    # y[y>0] = 0
    return np.average(y)


# perfrom kmeans++ initalization using regression model as distance
def _initialize_regression(X, num_clusters, model):
    data_size = X.shape[0]
    max_size = math.ceil(data_size / num_clusters)
    mod = data_size % num_clusters

    # k means initalization over the data
    def _kmeans_plusplus(X, num_clusters, data_size, model):
        # pick first random centroid
        c_id = np.random.choice(data_size, size=1, replace=False)
        centroid_ids = [c_id]
        centroids = [X[c_id]]

        # select rest of centroids
        while len(centroid_ids) < num_clusters:
            # remaining candidate data pts
            data_pts = [x for x in list(range(data_size)) if x not in centroid_ids]
            X_data = X[np.array(data_pts)]
            # calculate (regression) distance to chosen centroids
            dists = [_regression_to_centroid(X_data, c, model) for c in centroids]
            # bound at 0 for prob calcuations
            for d in dists:
                d[d > 0] = 0
            # min dist to chosen centroids for each data pt
            dists = np.stack(dists, axis=1).min(axis=1)
            # normalize into probabilities
            dists = dists**2
            sum_dists = np.sum(dists)
            dists = dists / sum_dists
            # choice new centroid
            c_id = np.random.choice(data_pts, size=1, replace=False, p=dists)
            centroid_ids.append(c_id)
            centroids.append(X[c_id])

        return centroids

    centroids = _kmeans_plusplus(X, num_clusters, data_size, model)

    # get balanced initalization based on kmeans++ centroids
    def _balanced_assignments(X, centroids, data_size, model):
        assigned_ids = []
        cluster_ids = ["--" for i in range(data_size)]
        full_clusters = []

        # assign each data pt to a cluster
        while len(assigned_ids) < data_size:
            # (0a) get unassigned data pts
            data_pts = [x for x in list(range(data_size)) if x not in assigned_ids]
            data_map = {i: x for i, x in enumerate(data_pts)}
            X_data = X[np.array(data_pts)]
            # (0b) get this cycle's available centroids and make mapping
            cluster_map = {}
            cycle_centroids = []
            for i, c in enumerate(centroids):
                if i not in full_clusters:
                    cycle_centroids.append(c)
                    cluster_map[len(cycle_centroids) - 1] = i
            # (1) calculate distances to centroids for all data
            dists = [_regression_to_centroid(X_data, c, model) for c in cycle_centroids]
            dists = np.stack(dists, axis=1)
            # (2) get utility metric = (min dist to centroid) - (max dist to a centroid)
            min_dists = dists.min(axis=1)
            max_dists = dists.max(axis=1)
            utility = min_dists - max_dists
            # (3) assign data point to cluster based on utility metric
            min_ids = dists.argmin(axis=1)
            d_id = utility.argmin()
            c_id = min_ids[utility.argmin()]
            d_id = data_map[d_id]
            c_id = cluster_map[c_id]
            assigned_ids.append(d_id)
            cluster_ids[d_id] = c_id
            # (4) if cluster is full, remove from centroid options for rest of assignments
            c_count = len([x for x in cluster_ids if x == c_id])
            if len(full_clusters) < mod:
                if c_count == max_size:
                    full_clusters.append(c_id)
                # handle clusters filled to max_size - 1 after meeting mod allotment
                if len(full_clusters) == mod:
                    for j in range(0, len(centroids)):
                        if (j not in full_clusters) and (
                            len([x for x in cluster_ids if x == j]) == (max_size - 1)
                        ):
                            full_clusters.append(j)
            elif c_count == (max_size - 1):
                full_clusters.append(c_id)
        return cluster_ids

    cluster_ids = _balanced_assignments(X, centroids, data_size, model)

    return cluster_ids


# TODO update this to use regression model as distance metric
# TODO finish implementing from after kmeans++ once tested
def cluster_once_regression(X, num_clusters, initialization, model):
    if initialization == "kmeans++":
        # use balanced kmeans++ initialization
        cluster_idxs = _initialize_regression(X, num_clusters, model)
        cluster_idxs = np.array(cluster_idxs)
    else:
        # initialize random clusters
        cluster_idxs = np.arange(len(X)) % num_clusters
        np.random.shuffle(cluster_idxs)

    # Calculate the set of initial distances
    # TODO implement this to use regression model as distance metric
    distances = [
        _within_cluster_regression_distance(X, cluster_idxs, idx, model)
        for idx in range(num_clusters)
    ]

    # sort to improve
    NUM_FLIPS = 100_000  # 50_000
    realized_flips = 0

    for itr in range(0, NUM_FLIPS):
        # Choose what pair of cluster labels indexes to flip
        a_idx, b_idx = np.random.choice(
            len(cluster_idxs), size=2, replace=False
        )  # The indexes in the array
        cluster_a, cluster_b = (
            cluster_idxs[a_idx],
            cluster_idxs[b_idx],
        )  # The actual cluster IDs
        cluster_idxs[a_idx], cluster_idxs[b_idx] = (
            cluster_b,
            cluster_a,
        )  # Perform the flip

        # Calculate the centroid for each cluster and cosine similarity within it
        # TODO implement this to use regression model as distance metric (reuse above function)
        da = _within_cluster_regression_distance(X, cluster_idxs, cluster_a, model)
        db = _within_cluster_regression_distance(X, cluster_idxs, cluster_b, model)

        # If the sum of distances is better now than before, make the flip
        if (da + db) < (distances[cluster_a] + distances[cluster_b]):
            realized_flips += 1
            distances[cluster_a] = da
            distances[cluster_b] = db

        # Otherwise reset the pair of labels
        else:
            cluster_idxs[a_idx], cluster_idxs[b_idx] = cluster_a, cluster_b

    return distances, cluster_idxs


def regression_clusters(
    lang_codes, num_clusters, sim_metric, num_shuffles=250, initialization="random"
):
    print(len(lang_codes), num_clusters, sim_metric, num_shuffles)
    # skip learning clusters if every lang is a seperate cluster
    # (skips expensive process of calculating lang features)
    if num_clusters == len(lang_codes):
        cluster_map = {lang: i for i, lang in enumerate(lang_codes)}
        return cluster_map

    # replace NaNs with mean value for that feature
    # TODO consider more complex imputation
    def _replace_nan(X):
        for i in range(0, X.shape[1]):
            X[np.isnan(X[:, i]), i] = np.nanmean(X[:, i])
        return X

    # get vector representations for each lang
    random.shuffle(lang_codes)
    # filepath = f"{CLUSTER_PREFIX}/regression_feats.pkl"
    v_dict = _load_features(lang_codes, "regression")
    v_keys = list(v_dict.keys())

    X = np.array(
        [[v if v != "--" else np.nan for v in v_dict[lang]] for lang in lang_codes]
    )
    X = _replace_nan(X)
    # load regression model
    input_path = f"{CLUSTER_PREFIX}/regression_model.pkl"
    with open(input_path, "rb") as f:
        model = pickle.load(f)

    f = delayed(cluster_once_regression)
    results = Parallel(n_jobs=-1)(
        f(X, num_clusters, initialization, model) for _ in tqdm(range(num_shuffles))
    )

    best_cluster_idxs = None
    best_distances = [0 for i in range(num_clusters)]

    for distances, idxs in results:
        if sum(distances) < sum(best_distances):
            best_distances = distances
            best_cluster_idxs = idxs

    lang_codes = np.array(lang_codes)
    clusters = []
    for i in range(num_clusters):
        clusters.append(list(lang_codes[best_cluster_idxs == i]))

    for d, c in zip(best_distances, clusters):
        print(c, d)
    print(sum(best_distances))
    print("")

    # use regression model to predict perfromance improvements for every language
    # construct centroid for every cluster id
    centroids_dict = {
        i: np.mean(X[best_cluster_idxs == i], axis=0, keepdims=True)
        for i in range(0, num_clusters)
    }

    # create centroid arr mapped to every data pt
    centroid_arr = [centroids_dict[i] for i in best_cluster_idxs]
    centroid = np.concatenate(centroid_arr, axis=0)

    # calculate distance
    X_dist = X - centroid

    # score with regression model, report for each lang
    print("Predicted BPC improvements per lang")
    y = model.predict(X_dist)
    lang_preds = {l: y_l for l, y_l in zip(lang_codes, y)}
    lo = " ".join([f"{lang_preds[l]:.5f}" for l in SORTED_LANGS])
    print(lo)


def main(args):
    LANG_OPTIONS = [
        "en",
        "ru",
        "es",
        "de",
        "fr",
        "ja",
        "it",
        "pt",
        "pl",
        "vi",
        "tr",
        "nl",
        "id",
        "ar",
        "cs",
        "fa",
        "uk",
        "ro",
        "el",
        "zh",
        "sv",
        "hu",
        "da",
        "hi",
        "fi",
        "bg",
        "no",
        "ko",
        "mr",
        "sk",
        "he",
        "th",
        "ms",
        "ca",
        "lt",
        "sl",
        "bn",
        "et",
        "lv",
        "sr",
        "cy",
        "az",
        "ta",
        "kk",
        "sq",
        "ne",
        "mt",
        "mn",
        "ka",
        "hy",
        "ur",
        "ml",
        "be",
        "gl",
        "is",
        "mk",
        "fil",
        "tg",
        "af",
        "te",
        "kn",
        "eu",
        "ky",
        "my",
        "sd",
        "la",
        "so",
        "si",
        "sw",
        "km",
        "uz",
        "lb",
        "gu",
        "eo",
        "pa",
        "ps",
        "ga",
        "fy",
        "ku",
        "gd",
        "am",
        "yi",
        "jv",
        "ha",
        "zu",
        "hmn",
        "co",
        "mg",
        "ceb",
        "ht",
        "sn",
        "lo",
        "su",
        "ny",
        "mi",
        "ig",
        "sm",
        "st",
        "haw",
        "xh",
        "yo",
    ]
    if args.metric == "regression":
        cluster_map = regression_clusters(
            LANG_OPTIONS, args.num_clusters, args.metric, initialization=args.init
        )
    else:
        cluster_map = random_clusters(
            LANG_OPTIONS, args.num_clusters, args.metric, initialization=args.init
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--num_clusters", type=int, choices=[8, 16], required=True)
    parser.add_argument(
        "--metric",
        type=str,
        choices=["geo", "syntax", "lexical", "regression"],
        required=True,
    )
    parser.add_argument(
        "--init", type=str, choices=["random", "kmeans++"], required=True
    )
    parser.add_argument("--rand_seed", default=42, type=int)

    args = parser.parse_args()

    # set random seeds
    os.environ["PYTHONHASHSEED"] = str(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)

    main(args)
