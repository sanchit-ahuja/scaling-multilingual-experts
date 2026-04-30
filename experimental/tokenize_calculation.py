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

from datasets import load_from_disk
from transformers import XGLMTokenizerFast

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


# Code to download, subsample, and format data from mC4 for X-ELM experiments
def main(args):
    old_tokenizer = XGLMTokenizerFast.from_pretrained("facebook/xglm-564M")
    new_tokenizer = XGLMTokenizerFast.from_pretrained(args.tokenizer_path)

    # iterate over langauges
    old_token_counts = []
    new_token_counts = []
    char_arr = []
    byte_arr = []
    for lang in args.languages:

        # Load dataset
        valid_ds = load_from_disk(f"{args.data_prefix}/valid/{lang}")
        old_count = 0
        new_count = 0
        char_count = 0
        byte_count = 0
        # Tokenize every example and get token count
        for example in tqdm(valid_ds):
            text = example["text"]
            t = old_tokenizer(text, add_special_tokens=False)
            old_count += len(t["input_ids"])
            t = new_tokenizer(text, add_special_tokens=False)
            new_count += len(t["input_ids"])
            c = len(text)
            char_count += c
            b = len(text.encode())
            byte_count += b

        print(lang, "valid", old_count, new_count, char_count, byte_count)
        old_token_counts.append(old_count)
        new_token_counts.append(new_count)
        char_arr.append(char_count)
        byte_arr.append(byte_count)

    print(LANG_OPTIONS)
    print(old_token_counts)
    print(new_token_counts)
    print(char_arr)
    print(byte_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--languages", nargs="*", required=True, choices=LANG_OPTIONS)

    args = parser.parse_args()

    print(args)
    main(args)
