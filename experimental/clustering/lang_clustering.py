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
    # if sim_metric == 'fam':
    #    v_dict = l2v.get_features(langs, "fam", minimal=True)
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


def cluster_once(X, num_clusters):
    # initialize random clusters
    cluster_idxs = np.arange(len(X)) % num_clusters
    np.random.shuffle(cluster_idxs)

    # Calculate the set of initial distances
    distances = [
        _within_cluster_cosine_distance(X, cluster_idxs, idx)
        for idx in range(num_clusters)
    ]

    # sort to improve
    NUM_FLIPS = 25_000  # 250_000
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
    lang_codes, num_clusters, sim_metric, num_shuffles=100
):  # 500): #agglut=False):
    print(num_clusters, sim_metric, num_shuffles)
    # skip learning clusters if every lang is a seperate cluster
    # (skips expensive process of calculating lang features)
    if num_clusters == len(lang_codes):
        cluster_map = {lang: i for i, lang in enumerate(lang_codes)}
        return cluster_map

    # get vector representations for each lang
    # random.shuffle(lang_codes)
    # saved state from 1M flips
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
    results = Parallel(n_jobs=-1)(f(X, num_clusters) for _ in tqdm(range(num_shuffles)))

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
    return -1


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

cluster_map = random_clusters(LANG_OPTIONS, 16, "geo")
