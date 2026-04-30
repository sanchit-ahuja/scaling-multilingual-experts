import argparse
import random
import math
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import r_regression, SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from sklearn.utils import shuffle

os.environ["HF_HOME"] = "/srv/home/users/blevinst24cs/.cache/huggingface/"

from transformers import XGLMTokenizerFast
import lang2vec.lang2vec as l2v
from iso639 import Lang  # this is the iso639-lang package
import unicodedataplus as ud
from tqdm import tqdm

CLUSTER_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/clustering/"

# cluster assignments (k=8)
CLUSTERS_K8 = {
    "syntax": [
        ["sq", "mk", "pl", "ca", "be", "el", "pt", "es", "it", "bg", "fr", "ro", "sl"],
        ["ig", "fil", "zu", "ny", "xh", "sr", "sw", "st", "sn", "sd", "sk", "co"],
        ["so", "pa", "ur", "ka", "bn", "hi", "si", "eu", "zh", "gu", "kk", "ko", "mr"],
        ["ru", "af", "no", "uk", "da", "sv", "en", "lb", "fy", "is", "nl", "de", "yi"],
        ["ms", "vi", "yo", "lo", "id", "jv", "su", "ha", "hmn", "ht", "km", "th"],
        ["ku", "fa", "eo", "hy", "hu", "et", "lt", "lv", "fi", "tg", "ps", "cs"],
        [
            "he",
            "gd",
            "la",
            "ceb",
            "cy",
            "mt",
            "ga",
            "mg",
            "sm",
            "gl",
            "haw",
            "ar",
            "mi",
        ],
        ["my", "tr", "ta", "az", "am", "uz", "kn", "mn", "ne", "ky", "ml", "te", "ja"],
    ],
    "geo": [
        ["sq", "mk", "hu", "ca", "la", "sr", "mt", "it", "lb", "sk", "co", "sl", "de"],
        ["fa", "pa", "uz", "ur", "mn", "ne", "hi", "ky", "gu", "kk", "tg", "sd", "ps"],
        ["he", "ru", "ku", "tr", "az", "hy", "ka", "el", "uk", "bg", "ar", "ro"],
        ["ms", "fil", "ceb", "id", "jv", "ja", "su", "zh", "sm", "ko", "haw", "mi"],
        ["pl", "be", "no", "et", "lt", "lv", "da", "fi", "sv", "fy", "nl", "cs"],
        ["my", "vi", "ta", "kn", "lo", "bn", "si", "ml", "te", "hmn", "km", "mr", "th"],
        ["so", "yo", "ig", "af", "am", "zu", "ny", "xh", "sw", "ha", "mg", "st", "sn"],
        ["gd", "eo", "cy", "eu", "pt", "es", "ga", "en", "gl", "ht", "fr", "is", "yi"],
    ],
    "lexical": [
        ["tr", "pl", "hu", "la", "et", "eu", "lt", "lv", "fi", "sk", "co", "sl", "cs"],
        ["he", "ru", "mk", "hy", "be", "ka", "mn", "sr", "ky", "uk", "kk", "tg", "bg"],
        ["my", "fa", "am", "ur", "lo", "bn", "ja", "zh", "ps", "km", "ar", "th", "yi"],
        ["eo", "af", "ca", "pt", "es", "ga", "en", "gl", "ht", "fr", "nl", "ro"],
        ["vi", "az", "fil", "zu", "uz", "ceb", "cy", "el", "jv", "su", "mg", "hmn"],
        ["ms", "so", "gd", "no", "id", "mt", "da", "sv", "lb", "fy", "sd", "is", "de"],
        ["sq", "yo", "ig", "ny", "xh", "it", "sw", "ha", "sm", "st", "sn", "haw", "mi"],
        ["ku", "ta", "pa", "kn", "ne", "hi", "si", "ml", "te", "gu", "ko", "mr"],
    ],
    "random": [
        ["sr", "vi", "bg", "he", "mi", "ga", "ha", "gd", "km", "mn", "de", "la", "jv"],
        ["hu", "uz", "fy", "tr", "es", "ar", "fr", "eu", "hy", "yo", "ru", "ne", "zh"],
        ["hi", "lt", "xh", "mg", "sk", "st", "sq", "da", "et", "en", "fa", "gu", "pa"],
        ["ky", "lv", "si", "kn", "mk", "ig", "sl", "kk", "so", "ms", "mt", "nl", "ht"],
        [
            "ku",
            "zu",
            "uk",
            "haw",
            "ta",
            "pl",
            "af",
            "sv",
            "sn",
            "ca",
            "mr",
            "no",
            "ceb",
        ],
        ["el", "bn", "yi", "pt", "fil", "lb", "az", "ps", "ko", "lo", "id", "sw"],
        ["am", "co", "it", "gl", "cs", "ny", "eo", "ka", "my", "fi", "cy", "te"],
        ["sm", "ja", "th", "ur", "is", "su", "ro", "hmn", "sd", "be", "tg", "ml"],
    ],
}
CLUSTER_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/clustering"

"""
CLUSTER OVERLAP ANALYSIS
"""


def _overlap(c1, c2):
    overlap = []
    for a in c1:
        row_overlap = []
        for b in c2:
            x = len(set(a).intersection(set(b)))
            row_overlap.append(x)
        overlap.append(row_overlap)
    return overlap


def cluster_overlap(cluster_arr):
    keys = list(CLUSTERS_K8.keys())
    for i in range(0, len(keys)):
        for j in range(i + 1, len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            overlap_matrix = _overlap(CLUSTERS_K8[k1], CLUSTERS_K8[k2])
            # report average set overlap
            flat_overlap = []
            max_rows = []
            for x in overlap_matrix:
                max_rows.append(max(x))
                for elm in x:
                    flat_overlap.append(elm)

            avg_overlap = sum(max_rows) / len(max_rows)
            max_overlap = max(flat_overlap)
            min_overlap = min(flat_overlap)
            print(
                "{} - {} ".format(k1, k2),
                avg_overlap,
                "[{}, {}]".format(min_overlap, max_overlap),
            )

            # visualize matrix (https://seaborn.pydata.org/generated/seaborn.heatmap.html)
            sns.heatmap(
                overlap_matrix,
                vmin=0,
                vmax=8,
                cmap=sns.color_palette("flare", as_cmap=True),
            )
            plt.ylabel(k1)
            plt.xlabel(k2)  # double check this
            plt.savefig("heatmap_{}_{}.png".format(k1, k2))
            plt.clf()
            # print(k1, k2, overlap_matrix[3][7], overlap_matrix[7][3])


"""
BPC DATA LOADING AND PREPROCESSING FOR REGRESSION
"""


def load_bpc_data(fp):
    with open(fp, "r") as dest_f:
        data_iter = csv.reader(dest_f, delimiter=",")
        data = [data for data in data_iter]
    # for d in data:
    # 	print(d)
    # 	input("....")
    langs = data[1][6:]
    langs = [l.strip().lower() for l in langs]
    scores = {
        "base": data[2][6:],
        "dense-ft": data[4][6:],
        "dense-fz": data[5][6:],
        "syntax-ft": data[8][6:],
        "syntax-fz": data[9][6:],
        "geo-ft": data[10][6:],
        "geo-fz": data[11][6:],
        "lexical-ft": data[12][6:],
        "lexical-fz": data[13][6:],
        "random-ft": data[14][6:],
        "random-fz": data[15][6:],
    }
    for k in scores.keys():
        scores[k] = [float(s) for s in scores[k]]
    return langs, scores


def _gen_lex_codes(tokenizer_name="facebook/xglm-564M", vocab_size=-1):
    # load tokenizer for model
    tokenizer = XGLMTokenizerFast.from_pretrained(tokenizer_name)
    # get surface form for every token
    codes = tokenizer.convert_ids_to_tokens([i for i in range(0, len(tokenizer))])
    # assert vocab size matches
    assert len(codes) == vocab_size
    return codes


def _consolidate_lex_feats_script(feats, codes, tokenizer_name="facebook/xglm-564M"):
    tokenizer = XGLMTokenizerFast.from_pretrained(tokenizer_name)
    special_tokens = tokenizer.all_special_tokens
    print(feats.shape)

    # map each feat code into a unicode script (or special tokens)
    mapping = []
    for c in codes:
        if c in special_tokens:
            mapping.append("Special**")
        else:
            # strip out leading space if present
            if len(c) > 1:
                c = c.replace("▁", "")
            # take most common script in token
            script_arr = [ud.script(x) for x in c]
            maj_script = max(set(script_arr), key=script_arr.count)
            mapping.append(maj_script)
    scripts = list(set(mapping))
    script_feats = []
    for s in scripts:
        s_map = [i for i, m in enumerate(mapping) if m == s]
        X_s = feats[:, s_map]
        # consolidate features along feat dim (axis=1) + add to script_feats
        X_s = np.sum(X_s, axis=1)
        script_feats.append(X_s)
    script_feats = np.stack(script_feats, axis=1)
    return script_feats, scripts


def _consolidate_lex_feats_pca(feats, codes, n_components=1):
    # lex_codes = codes
    # run pca on features
    pca = PCA(n_components=n_components, svd_solver="arpack", random_state=42)
    feats = pca.fit_transform(feats)
    codes = [f"lex_component_{i}" for i in range(0, n_components)]
    return feats, codes

    # component analysis
    # m = pca.components_
    # for i in range(0, n_components):
    # 	c_i = codes[i]
    # 	v_i = m[i]
    # 	top_feat_ids = np.argsort(v_i)[:10]
    # 	top_feats = [lex_codes[j] for j in top_feat_ids]
    # 	top_weights = [v_i[j] for j in top_feat_ids]
    # 	print(f"{c_i} "+" ".join([f"{a},{b:.3f}" for a, b in zip(top_feats, top_weights)]))


def _lang_features(langs, feat_type="syntax", consolidate_lex=""):
    print(feat_type)
    lang3_codes = [Lang(lang).pt3 if lang != "iw" else Lang("he").pt3 for lang in langs]
    if feat_type == "syntax":
        v_dict = l2v.get_features(
            lang3_codes, "syntax_average", minimal=True, header=True
        )
        feats = np.array(
            [[v if v != "--" else np.nan for v in v_dict[lang]] for lang in lang3_codes]
        )
        codes = v_dict["CODE"]
        # filter out features with only nans
        nan_cols = np.argwhere(np.isnan(feats).all(axis=1)).flatten()
        feats = np.delete(feats, nan_cols, axis=-1)
        codes = [c for i, c in enumerate(codes) if i not in nan_cols]
    elif feat_type == "geo":
        v_dict = l2v.get_features(lang3_codes, "geo", minimal=True, header=True)
        feats = np.array(
            [[v if v != "--" else np.nan for v in v_dict[lang]] for lang in lang3_codes]
        )
        codes = v_dict["CODE"]
        # filter out features with only nans
        # note: not actually used for this feature
        # nan_cols = np.argwhere(np.isnan(feats).all(axis=1)).flatten()
        # feats = np.delete(feats, nan_cols, axis=-1)
        # codes = [c for i,c in enumerate(codes) if i not in nan_cols]
    elif feat_type == "lexical":
        input_path = f"{CLUSTER_PREFIX}/{feat_type}_feats.pkl"
        with open(input_path, "rb") as f:
            v_dict = pickle.load(f)
        feats = np.array([v_dict[lang] for lang in langs])
        codes = _gen_lex_codes(vocab_size=feats.shape[-1])
        # remove dimensions where everything is 0
        zero_cols = np.argwhere((feats == 0).all(axis=0)).flatten()
        feats = np.delete(feats, zero_cols, axis=-1)
        codes = [c for i, c in enumerate(codes) if i not in zero_cols]
        # if this flag is given, consolidate lexical features into unicode block groupings
        if consolidate_lex == "script":
            feats, codes = _consolidate_lex_feats_script(feats, codes)
        elif "pca" in consolidate_lex:
            n = int(consolidate_lex.split("_")[1])
            feats, codes = _consolidate_lex_feats_pca(feats, codes, n_components=n)
    else:
        print("not implemented yet! ln 224")
        quit()
    return feats, codes


def _cluster_id_map(langs, clusters):
    maps = []
    for c in clusters:
        m = np.array([1 if l in c else 0 for l in langs])
        maps.append(m)
    return maps


def _coarse_distance(X, centroids):
    dists = []
    for i in range(X.shape[0]):
        x = X[i]
        y = centroids[i]
        mask = ~np.isnan(x)
        yy = np.nansum((x * mask) ** 2)
        xx = np.nansum(x**2)
        xy = np.nansum(x * y)
        d = 1 - (xy / (np.sqrt(xx) * np.sqrt(yy) + 1e-8))
        dists.append(d)
    dists = np.array(dists)
    return dists


# features is a settings: feat_type: (feature, language) matrix of feat scores
def calculate_feats(langs, settings, coarse=False, consolidate_lex=""):
    feats = {}
    feat_codes = {}
    # get each lang's features
    for feat_type in ["syntax", "geo", "lexical"]:
        f, codes = _lang_features(
            langs, feat_type=feat_type, consolidate_lex=consolidate_lex
        )
        feats[feat_type] = f
        feat_codes[feat_type] = codes

    # get centroids for each setting and calc element-wise distance in setting for ALL features
    feat_dists = {}
    # for each experimental setting
    for setting in settings:
        setting_dists = {}
        cluster_ids = _cluster_id_map(langs, CLUSTERS_K8[setting])
        # for each set of features...
        for feat_type in feats.keys():
            # feature vector and placeholder arr for distances
            X = feats[feat_type]
            centroids = [np.nan] * len(langs)

            # for each cluster of languages...
            for c_ids in cluster_ids:
                # cluster centroids
                X_cluster = X[c_ids == 1]
                c = np.nanmean(X_cluster, axis=0, keepdims=True)
                centroids = [
                    c if v == 1 else centroids[idx] for idx, v in enumerate(c_ids)
                ]
            centroids = np.array(centroids).squeeze()
            # distance based on if we want cosine dist or elementwise differences
            if coarse:
                dists = _coarse_distance(X, centroids)
            # else: dists = np.absolute(X-centroids)
            else:
                dists = X - centroids

            # save distances for each feature type
            setting_dists[feat_type] = dists

        # consolidate to a single feature set
        if coarse:
            all_feats = []
            all_codes = []
            for feat_type in feats.keys():
                all_feats.append(setting_dists[feat_type])
                all_codes.append(feat_type)
            all_feats = np.stack(all_feats, axis=1)
            feat_dists[setting] = {"coarse": all_feats}
            feat_codes["coarse"] = all_codes

        # save features across types for each setting
        else:
            feat_dists[setting] = setting_dists

    return feat_dists, feat_codes, feats


def normalize_scores(scores, keep_rows=[], normalize_on="base"):
    norm_row = scores[normalize_on]
    norm_scores = {}
    for row in keep_rows:
        row_scores = scores[row]
        # z = [(s-n)/n for s, n in zip(row_scores, norm_row)]
        # normalizing BPC so that improvements (relative decreasing in BPC) are postive values
        z = [(s - n) / n for s, n in zip(row_scores, norm_row)]
        norm_scores[row] = z
    return norm_scores


"""
REGRESSION(S)
"""


# replace NaNs with mean value for that feature
# TODO consider more complex imputation
def _replace_nan(X):
    for i in range(0, X.shape[1]):
        X[np.isnan(X[:, i]), i] = np.nanmean(X[:, i])
    return X


def _join_features(feats, codes, settings):
    joint_feats = {}
    joint_codes = []

    for i, s in enumerate(settings):
        s_feats = feats[s]
        j_feats = []
        feat_types = sorted(list(s_feats.keys()))
        for f in feat_types:
            if i == 0:
                joint_codes.extend([f"{f}:{c}" for c in codes[f]])
            j_feats.append(s_feats[f])
        joint_feats[s] = {"all": np.concatenate(j_feats, axis=1)}
    return joint_feats, joint_codes


def _plot_regression(X, Y, coef, bias, feat):
    plt.scatter(X, Y, color="#d63362", alpha=0.8)
    plt.plot(
        [np.min(X), np.max(X)],
        [coef * (np.min(X)) + bias, coef * (np.max(X)) + bias],
        color="gray",
    )
    plt.title(label=f"{coef:.3f} x + {bias:.3f} = y", fontdict={"fontsize": 12})
    plt.savefig(f"{feat}_reg.png", dpi=500)
    plt.clf()
    quit()  # DEBUGGING
    return


def regression(feats, scores, codes, joint=True, feat_type="all", corrcoef=False):
    settings = list(scores.keys())
    # make feature, score sets
    X = []
    Y = []
    if feat_type == "all":
        feats, codes = _join_features(feats, codes, settings)
    else:
        codes = codes[feat_type]
    for s in settings:
        setting_feats = feats[s]
        X.extend(setting_feats[feat_type])
        Y.extend(scores[s])

    X = np.array(X)
    Y = np.array(Y)

    if corrcoef:
        corr_matrix = np.ma.corrcoef(np.ma.masked_invalid(X.T))
    else:
        corr_matrix = None

    if joint:
        X = _replace_nan(X)
        model = LinearRegression()
        model.fit(X, Y)
        feat_coefs = model.coef_
        scores = [model.score(X, Y)] * len(feat_coefs)
        biases = [model.intercept_] * len(feat_coefs)
    else:
        # iterate over features
        feat_coefs = []
        scores = []
        biases = []
        for f_idx in range(0, X.shape[1]):
            X_col = X[:, f_idx]
            # remove nan values and corresponding label
            nan_mask = np.isfinite(X_col)
            X_col = X_col[nan_mask]
            Y_col = Y[nan_mask]

            # calculate regression, save coefs
            model = LinearRegression()
            model.fit(X_col.reshape(-1, 1), Y_col)
            feat_coefs.append(model.coef_.item())
            biases.append(model.intercept_)
            s = model.score(X_col.reshape(-1, 1), Y_col)
            scores.append(s)

            # plot any specififc features for viz
            viz_feats = ["Braille"]
            for v in viz_feats:
                if codes[f_idx] == v:
                    _plot_regression(
                        X_col, Y_col, model.coef_.item(), model.intercept_.item(), v
                    )

    return feat_coefs, biases, scores, corr_matrix, codes


def stepwise_regression(feats, scores, codes, n_feats, direction):
    # flatten to a single set of features
    settings = list(scores.keys())
    feats, codes = _join_features(feats, codes, settings)
    X = []
    Y = []
    for s in settings:
        setting_feats = feats[s]
        X.extend(setting_feats["all"])
        Y.extend(scores[s])
    X = np.array(X)
    Y = np.array(Y)
    X = _replace_nan(X)

    # run feature selection
    model = LinearRegression()
    selector = SequentialFeatureSelector(
        model, n_features_to_select=n_feats, direction=direction, cv=5, n_jobs=-1
    )
    X = selector.fit_transform(X, Y)
    codes = selector.get_feature_names_out(codes)
    scorer = LinearRegression()
    scorer.fit(X, Y)
    z = [c for c in codes if "syntax:" in c][0]
    z_idx = list(codes).index(z)

    # report selected features and regression score
    scorer = LinearRegression()
    scorer.fit(X, Y)
    score = scorer.score(X, Y)

    # get cross validation score on selected features
    model = LinearRegression()
    cv = cross_validate(model, X, Y, cv=5, n_jobs=-1)

    return score, codes, scorer, cv["test_score"]


def pca_selection(feats, scores, codes, n_components):
    # flatten to a single set of features
    settings = list(scores.keys())
    feats, codes = _join_features(feats, codes, settings)
    X = []
    Y = []
    for s in settings:
        setting_feats = feats[s]
        X.extend(setting_feats["all"])
        Y.extend(scores[s])
    X = np.array(X)
    Y = np.array(Y)
    X = _replace_nan(X)

    # run PCA on features
    pca = PCA(n_components=n_components, svd_solver="arpack", random_state=42)
    X = pca.fit_transform(X)

    # report regression score for PCA features
    scorer = LinearRegression()
    scorer.fit(X, Y)
    score = scorer.score(X, Y)

    # get cross validation score on selected features
    model = LinearRegression()
    cv = cross_validate(model, X, Y, cv=5, n_jobs=-1)

    return score, cv["test_score"]


"""
MAIN SCRIPT
"""

# calculate set overlaps, silhouette scores
# cluster_overlap(CLUSTERS_K8)

# import bpc data + clean spreadsheet
langs, scores = load_bpc_data("xelm-1GiB-bpc.csv")
settings = list(CLUSTERS_K8.keys())

# normalize bpc on baseline (base)
# (note: not normalizing on dense, but... could add in analysis with that)
ft_scores = normalize_scores(
    scores,
    keep_rows=["syntax-ft", "geo-ft", "lexical-ft", "random-ft"],
    normalize_on="base",
)
fz_scores = normalize_scores(
    scores,
    keep_rows=["syntax-fz", "geo-fz", "lexical-fz", "random-fz"],
    normalize_on="base",
)

# feature preprocessing (lang, setting, feature distance vector)
# feats, codes, lang_feats = calculate_feats(langs, settings)
feats, codes, lang_feats = calculate_feats(langs, settings, consolidate_lex="script")
# feats, codes, lang_feats = calculate_feats(langs, settings, consolidate_lex="pca_100")

# run coarse-grained single, joint regression (syn dist, geo dist, lex dist)
"""
feats, codes = calculate_feats(langs, settings, coarse=True)
ft_scores = {k.split('-')[0]:ft_scores[k] for k in ft_scores.keys()}
s1 = regression(feats, ft_scores, codes, feat_type='coarse')
s2 = regression(feats, ft_scores, codes, joint=False, feat_type='coarse')
data = zip(codes['coarse'], s2, s1)
data = sorted(data, key=lambda x: x[1], reverse=True)


for x, y, z in data:
	print(f"{x} {y:.5f} {z:.5f}".format(x,y,z))
"""

# run SYNTAX single feature regression, joint group level
"""
ft_scores = {k.split('-')[0]:ft_scores[k] for k in ft_scores.keys()}
c1, b1, s1, corr_syntax, syntax_codes = regression(feats, ft_scores, codes, feat_type='syntax')
c2, b2, s2, _, _ = regression(feats, ft_scores, codes, joint=False, feat_type='syntax')
data = zip(syntax_codes, s2, c2, c1, b2)
data = sorted(data, key=lambda x: x[1], reverse=True)
syntax_codes = [x for x, _, _, _, _ in data] #re-sorted
"""

# print("joint score = {}".format(s1[0]))
# print("joint bias = {}".format(b1[0]))
# for x, y, z, a, b in data
# 	print(f"{x} {y:.5f} {z:.5f} {b:.5f} {a:.5f}".format(x,y,z,b, a))


# run GEO single feature regression, joint group level
"""
ft_scores = {k.split('-')[0]:ft_scores[k] for k in ft_scores.keys()}
c1, b1, s1, corr_geo, geo_codes = regression(feats, ft_scores, codes, feat_type='geo')
c2, b2, s2, _, _ = regression(feats, ft_scores, codes, joint=False, feat_type='geo')
data = zip(geo_codes, s2, c2, c1, b2)
data = sorted(data, key=lambda x: x[1], reverse=True)
geo_codes = [x for x, _, _, _, _ in data] #re-sorted
"""

# print("joint score = {}".format(s1[0]))
# print("joint bias = {}".format(b1[0]))
# for x, y, z, a, b in data:
# 	print(f"{x} {y:.5f} {z:.5f} {b:.5f} {a:.5f}".format(x,y,z,b, a))


# run LEXICAL single feature regression -- see which individual features have an effect?

ft_scores = {k.split("-")[0]: ft_scores[k] for k in ft_scores.keys()}
c1, b1, s1, _, lex_codes = regression(
    feats, ft_scores, codes, feat_type="lexical", corrcoef=False
)
c2, b2, s2, _, _ = regression(
    feats, ft_scores, codes, joint=False, feat_type="lexical", corrcoef=False
)
data = zip(codes["lexical"], s2, c2, c1, b2)
data = sorted(data, key=lambda x: x[1], reverse=True)
lex_codes = [x for x, _, _, _, _ in data]  # re-sorted


# print("joint score = {}".format(s1[0]))
# print("joint bias = {}".format(b1[0]))
# for x, y, z, a, b in data:
# 	print(f"{x} {y:.5f} {z:.5f} {b:.5f} {a:.5f}".format(x,y,z,b, a))


"""
#run full joint regressions
ft_scores = {k.split('-')[0]:ft_scores[k] for k in ft_scores.keys()}
c1, s1, corr_all, all_codes = regression(feats, ft_scores, codes, feat_type='all', corrcoef=True)
print("joint score = {}".format(s1[0]))

data = {k:v for k, v in zip(all_codes, c1)}
for k in syntax_codes:
	v = data[f"syntax:{k}"]
	print(f"syntax:{k} {v:.5f}")
print(" ")

for k in geo_codes:
	v = data[f"geo:{k}"]
	print(f"geo:{k} {v:.5f}")
print(" ")

for k in lex_codes:
	v = data[f"lexical:{k}"]
	k=k.replace("{", "{{")
	print(f"lexical:{k} {v:.5f}")
"""
"""
#correlation matrix (corrcof) for all features across sets 
#corr_all = _syntax, _geo, _lex_pca_100
colors = sns.color_palette("rocket_r", as_cmap=True)
corr_abs = np.absolute(corr_all)
sns.heatmap(corr_abs, vmin=0, vmax=1, cmap=colors, cbar=False, xticklabels=False, yticklabels=False, square=True)
plt.savefig("feat_corr.png", dpi=500)

geo_ids = [i for i, x in enumerate(all_codes) if "geo:" in x]
lex_ids = [i for i, x in enumerate(all_codes) if "lexical:" in x]
syn_ids = [i for i, x in enumerate(all_codes) if "syntax:" in x]

corr_subset = corr_abs[syn_ids]
corr_subset = corr_subset[:, syn_ids]
print(corr_subset.shape)
sns.heatmap(corr_subset, vmin=0, vmax=1, cmap=colors, cbar=False, xticklabels=False, yticklabels=False, square=True)
plt.savefig("feat_subset.png", dpi=500)

data = []
for i in syn_ids:
	for j in lex_ids:
		f_i = all_codes[i]
		f_j = all_codes[j]
		c_ij = float(corr_all[i, j])
		if math.isnan(c_ij): c_ij = 0
		data.append((f_i, f_j, c_ij))

data = sorted(data, key=lambda x: abs(x[-1]), reverse=True)
for x, y, z in data[:25]:
	print(f"{x} {y} {z:.5f}".format(x,y,z))
"""


# construct as v_dict over langs with feats in order expected by regression model
def _construct_output_feats(filtered_codes, codes, lang_feats, langs):
    print(filtered_codes)
    X_arr = []
    for fc in filtered_codes:
        fc1, fc2 = fc.split(":")
        feat_id = codes[fc1].index(fc2)
        feat_vec = lang_feats[fc1][:, feat_id]
        X_arr.append(feat_vec)
    X = np.stack(X_arr, axis=1)

    v_dict = {}
    for i, l in enumerate(langs):
        v_dict[l] = X[i]
    return v_dict


# https://en.wikipedia.org/wiki/Stepwise_regression
# stepwise (forward) regression feature selection (w/ lex PCA preprocessing)
ft_scores = {k.split("-")[0]: ft_scores[k] for k in ft_scores.keys()}
# DEBUGGING
for n_feats in [100]:  # tqdm([10, 25, 50, 100, 150, 200, 250]):
    s, c, m, cv = stepwise_regression(
        feats, ft_scores, codes, n_feats, direction="forward"
    )
    cv = [f"{x:.5f}" for x in cv]
    cv = "[" + ",".join(cv) + "]"
    c = list(c)
    print(f"{n_feats} {s} {cv} {c}")
    sys.stdout.flush()
    # reconstruct feature set
    X_dict = _construct_output_feats(c, codes, lang_feats, langs)
    with open(f"{CLUSTER_PREFIX}/regression_feats.pkl", "wb") as f:
        pickle.dump(X_dict, f)
    with open(f"{CLUSTER_PREFIX}/regression_model.pkl", "wb") as f:
        pickle.dump(m, f)
quit()


# vs. PCA (over all)
feats, codes, lang_feats = calculate_feats(langs, settings, consolidate_lex="")
ft_scores = {k.split("-")[0]: ft_scores[k] for k in ft_scores.keys()}
pca_scores = []
for n_feats in tqdm([10, 25, 50, 100]):
    s, cv = pca_selection(feats, ft_scores, codes, n_feats)
    # print(s, cv)
    pca_scores.append(s)
    cv = [f"{x:.5f}" for x in cv]
    cv = "[" + ",".join(cv) + "]"
    print(f"{n_feats} {s:.5f} {cv}")
    sys.stdout.flush()


# pca_scores.append('-100')
# for a, b, c, d, in zip([10, 25, 50, 100, 250], fs_scores, pca_scores, fs_feats):
# 	d_str = ",".join(d)
# 	print(f"{a} {b:.5f} {c:.5f} {d_str}")


# EOF
