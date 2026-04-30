import argparse
import torch
from torch.nn import functional as F
import random
import math
import numpy as np
import os
from tqdm import tqdm

os.environ["HF_HOME"] = "/srv/home/users/blevinst24cs/.cache/huggingface/"

from transformers import (
    XGLMTokenizerFast,
    XGLMForCausalLM,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk

MODEL_OPTIONS = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-7.5B"]
# same tokenizer for models trained on 30 and on 134 languages? Looks like it...
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


# data processing from this guide: https://huggingface.co/docs/transformers/en/tasks/language_modeling
def _process_dataset(data, tokenizer, length):
    # extract text and tokenize
    def _tokenize(examples):
        return tokenizer([x for x in examples["text"]])

    data = data.map(
        _tokenize, batched=True, num_proc=4, remove_columns=data.column_names
    )

    # group texts
    def _group(examples, block_size=2048):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    data = data.map(_group, batched=True, num_proc=4)
    # return processed dataset
    return data


def _count_chars(data, tokenizer):
    count_arr = []
    for example in data:
        example_str = tokenizer.decode(example["labels"])
        count_arr.append(len(example_str))
    return sum(count_arr)


def load_eval_data(langs, data_prefix, tokenizer):
    eval_data = []
    for lang in langs:
        d = load_from_disk(f"{data_prefix}/valid/{lang}")
        # implement data packing myself, use DataCollator for Language Modeling
        d = _process_dataset(d, tokenizer, length=2048)
        # calculate num chars
        chars = _count_chars(d, tokenizer)
        eval_data.append((lang, d, chars))
    return eval_data


# ppl calculation from https://huggingface.co/docs/transformers/en/perplexity
def main(args):
    # set up model for training
    if args.tokenizer_dir:
        tokenizer = XGLMTokenizerFast.from_pretrained(
            f"{args.tokenizer_dir}/tokenizer.pt"
        )
    else:
        tokenizer = XGLMTokenizerFast.from_pretrained(args.model_name)
    if args.initalization_dir:
        model = XGLMForCausalLM.from_pretrained(args.initalization_dir)
    else:
        model = XGLMForCausalLM.from_pretrained(args.model_name)
    model = model.to("cuda:0").eval()

    # load and process datasets
    eval_data = load_eval_data(args.languages, args.data_prefix, tokenizer)

    # get data collator
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    lang_arr = []
    ppl_arr = []
    bpc_arr = []
    for lang, data, num_chars in eval_data:
        # construct DataLoader for language
        data = torch.utils.data.DataLoader(data, batch_size=1, collate_fn=data_collator)

        nll = 0.0
        num_tokens = 0
        for example in tqdm(data):
            # move example to GPU
            example["input_ids"] = example["input_ids"].to("cuda:0")
            example["attention_mask"] = example["attention_mask"].to("cuda:0")
            example["labels"] = example["labels"].to("cuda:0")
            # run example through model
            with torch.no_grad():
                score = model(
                    example["input_ids"],
                    example["attention_mask"],
                    labels=example["labels"],
                )["loss"].item()

            # count scored tokens for ppl
            valid_token_count = (example["labels"] != -100).sum().item()
            bsz = example["labels"].size(0)
            valid_token_count = valid_token_count - bsz

            # accum nll
            num_tokens += valid_token_count
            nll += score * valid_token_count

        ppl = math.exp(nll / num_tokens)
        bpc = nll / num_chars
        print(lang, ppl, bpc)
        lang_arr.append(lang)
        ppl_arr.append(ppl)
        bpc_arr.append(bpc)

    # this only works if langs are given in order of spreadsheet...
    # python 3.x formatting will auto round the floats
    print(" ".join(lang_arr))
    ppl_arr = [f"{ppl:.3f}" for ppl in ppl_arr]
    print(" ".join(ppl_arr))
    bpc_arr = [f"{bpc:.5f}" for bpc in bpc_arr]
    print(" ".join(bpc_arr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--initalization_dir", type=str, help="Path to starting model checkpoint"
    )
    parser.add_argument(
        "--model_name", default="facebook/xglm-564M", choices=MODEL_OPTIONS
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        help="Path to new tokenizer if vocabulary is reinitalized",
    )
    parser.add_argument("--rand_seed", default=42, type=int)
    parser.add_argument("--valid_bsz", type=int, default=4)
    parser.add_argument("--languages", nargs="*", choices=LANG_OPTIONS, required=True)
    # /srv/home/users/blevinst24cs/x-elm-v2/data/mc4_1GB
    parser.add_argument("--data_prefix", type=str, default="", required=True)

    args = parser.parse_args()

    # set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ["PYTHONHASHSEED"] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(args)

    main(args)
