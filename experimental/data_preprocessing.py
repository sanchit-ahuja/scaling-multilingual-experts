import argparse
import torch
from torch.nn import functional as F
import random
import math
import numpy as np
import os
from tqdm import tqdm

# os.environ['HF_HOME'] = '/data/blevinst24dm/.cache/huggingface/'
os.environ["HF_HOME"] = "/srv/home/users/blevinst24cs/.cache/huggingface/"

from datasets import load_dataset, load_from_disk, Dataset

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
    "iw",
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
LANG_OPTIONS += ["he"]


# Code to download, subsample, and format data from mC4 for X-ELM experiments
def main(args):

    # iterate over langauges
    for lang in args.languages:
        if lang == "he":
            lang = "iw"

        # download data subset
        train_ds = load_dataset("allenai/c4", lang, streaming=True)["train"]
        train_size = 0
        train_docs = []
        for example in tqdm(train_ds):
            example_size = len(example["text"].encode())
            train_size += example_size
            train_docs.append(example)
            if train_size > args.train_threshold:
                break

        print(lang, "train", train_size, len(train_docs))
        train_ds = Dataset.from_list(train_docs)
        train_ds.save_to_disk(
            f"{args.data_prefix}/train/{lang}", num_shards=args.num_shards
        )

        # repeat process for validation data
        valid_ds = load_dataset("allenai/c4", lang, streaming=True)["validation"]
        valid_size = 0
        valid_docs = []
        for example in tqdm(valid_ds):
            example_size = len(example["text"].encode())
            valid_size += example_size
            valid_docs.append(example)
            if valid_size > args.valid_threshold:
                break

        print(lang, "valid", valid_size, len(valid_docs))
        valid_ds = Dataset.from_list(valid_docs)
        valid_ds.save_to_disk(f"{args.data_prefix}/valid/{lang}", num_shards=1)

        # Verifying loading from file
        train_ds = load_from_disk(f"{args.data_prefix}/train/{lang}")
        valid_ds = load_from_disk(f"{args.data_prefix}/valid/{lang}")
        print(train_ds)
        print(valid_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="*", required=True, choices=LANG_OPTIONS)
    parser.add_argument(
        "--train_threshold", type=int, default=1000000000  # 1000000000 = 1 GiB
    )
    parser.add_argument(
        "--valid_threshold", type=int, default=5000000  # 5000000 = 0.05 GiB, 5 MB
    )
    parser.add_argument(
        "--num_shards", type=int, default=20  # 20 w/ 1 GiB = 50 MB per shard
    )
    parser.add_argument("--data_prefix", type=str, default="")
    # parser.add_argument(
    #    "--flag", type=str, default=""
    # )

    args = parser.parse_args()

    print(args)
    main(args)

# 1 : mk mi mg lv lt lo lb la ky ku ko kn km kk ka jv ja it is ig id hy hu ht hmn hi he haw ha gu gl gd ga fy fr fil fi fa eu et es eo en
# 2 : ml mn mr ms mt my ne nl no ny pa pl ps pt ro ru sd si sk sl sm sn so sq sr st su sv sw ta te tg th tr uk ur uz vi xh yi yo zh zu
# rerun: he haw ha gu gl gd ga fy fr fil fi fa // eu et , es eo // en

# potential train thresholds = 1GB text would give roughtly 81% the same amount of data
# TODO - budget of how long to run with those data sizes (1Gb per lang)

"""
CALCULATING DATA SIZES (old)

train_d = load_dataset("allenai/c4", lang, streaming=True)["train"]
train_size = 0
train_chars = 0
train_docs = 0
for example in tqdm(train_d):
	example_size = len(example["text"].encode())
	train_size += example_size
	train_chars += len(example["text"])
	train_docs += 1
print(lang, "train", train_size, train_chars, train_docs)
with open('./mc4_sizes_{}.txt'.format(args.flag), 'a') as f:
	f.write("{} {} {} {} {}\n".format(lang, "train", train_size, train_chars, train_docs))

train_d = load_dataset("allenai/c4", lang, streaming=True)["validation"]
valid_size = 0
valid_chars = 0
valid_docs = 0
for example in train_d:
	example_size = len(example["text"].encode())
	valid_size += example_size
	valid_chars += len(example["text"])
	valid_docs += 1
print(lang, "valid", valid_size, valid_chars, valid_docs)
with open('./mc4_sizes_{}.txt'.format(args.flag), 'a') as f:
	f.write("{} {} {} {} {}\n".format(lang, "valid", valid_size, valid_chars, valid_docs))
"""
