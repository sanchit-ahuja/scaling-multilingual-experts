# from pathlib import Path
import pickle
from tqdm import tqdm
import os

os.environ["HF_HOME"] = "/srv/home/users/blevinst24cs/.cache/huggingface/"

from transformers import XGLMTokenizerFast
from datasets import load_from_disk

# readme https://huggingface.co/datasets/allenai/c4/blob/mC4_3.1.0/README.md
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
DATA_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/data/mc4_1GB"
CLUSTER_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/clustering"


def main():
    tokenizer = XGLMTokenizerFast.from_pretrained("facebook/xglm-564M")
    feature_dict = {}
    # for each language...
    for lang in LANG_OPTIONS:
        # TODO load 1GiB dataset for language
        data_lang = lang if lang != "iw" else "he"
        dataset = load_from_disk(f"{DATA_PREFIX}/train/{lang}")
        token_counts = [0] * len(tokenizer)

        # TODO get counts for tokens over vocab,
        for example in tqdm(dataset):
            text = example["text"]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for t_id in token_ids:
                token_counts[t_id] += 1
        total_tokens = sum(token_counts)
        norm_counts = [x / total_tokens for x in token_counts]
        print(lang, total_tokens)
        feature_dict[lang] = norm_counts

    # write out feature dict for clustering
    output_path = f"{CLUSTER_PREFIX}/lexical_feats.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(feature_dict, f)
    return


if __name__ == "__main__":
    main()
