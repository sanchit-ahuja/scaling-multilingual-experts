import os
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np

import lang2vec.lang2vec as l2v
from iso639 import Lang  # this is the iso639-lang package
import math
from sklearn.cluster import KMeans

CLUSTER_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/clustering/"


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
    print(data_size, num_clusters, max_size, mod)

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

    # TODO print langauges by these clustr assignments to ensure balanced, than implement rest of algorithm
    print(cluster_ids)
    quit()
    return cluster_ids


# https://elki-project.github.io/tutorial/same-size_k_means
def balanced_kmeans(X, num_clusters, num_iter):
    # initalize clusters with kmeans++ and balanced greedy assignments
    cluster_ids = _initialize(X, num_clusters)

    # TODO: the rest of the dumb logic
    i = 0
    while i < num_iter:
        i += 1
        dists = []
        # For each cluster...
        for cluster_idx in range(0, num_clusters):
            # Choose only the rows belonging to the specified cluster
            X_idx = X[cluster_idxs == cluster_idx]
            # Calculate that cluster's centroid
            centroid = np.nanmean(X_idx, axis=0, keepdims=True)
            # Calculate distance from cluster centroid to all data points in X
            d = _distance_to_centroid(X, centroid)
            dists.append(d)
        dists = np.stack(dists, axis=1)  # data pts x cluster ids

    return -1


def kmeans_clusterer(lang_codes, num_clusters, sim_metric):
    print(num_clusters, sim_metric)
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

    return balanced_kmeans(X, num_clusters, num_iter=10)


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

# options: 'geo', 'syntax', 'lexical'
cluster_map = kmeans_clusterer(LANG_OPTIONS, 16, "syntax")
print(cluster_map)
# EOF
