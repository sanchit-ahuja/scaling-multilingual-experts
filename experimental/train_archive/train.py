import argparse
import torch
from torch.nn import functional as F
import random
import math
import numpy as np
import os
from tqdm import tqdm

# os.environ['HF_HOME'] = '/data/user/.cache/huggingface/'
os.environ["HF_HOME"] = "/home/user/.cache/huggingface/"

from transformers import XGLMTokenizerFast, XGLMForCausalLM
from datasets import load_from_disk, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from trl.trainer import ConstantLengthDataset

# rom evaluate import load as load_metric

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


# Freezes embedding layer of model (and LM head, since they are tied)
# this assues the XGLM model architecture, and may break with other HF models
def _freeze_embedding_layer(model):
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.model.embed_tokens.parameters():
        assert param.requires_grad == False
    for param in model.lm_head.parameters():
        assert param.requires_grad == False
    return model


# Unfreezes embedding layer of model (and LM head, since they are tied)
# We need to do this when reinitalizing the embeddings to ensure they train
def _unfreeze_embedding_layer(model):
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = True
    for param in model.model.embed_tokens.parameters():
        assert param.requires_grad == True
    for param in model.lm_head.parameters():
        assert param.requires_grad == True
    return model


# Freeze model layers
def _freeze_transformer(model):
    for param in model.model.parameters():
        param.requires_grad = False
    return model


# Unfreeze model layers
def _unfreeze_transformer(model):
    for param in model.model.parameters():
        param.requires_grad = True
    return model


# Reinitalizes tbe embeddings (and LM head) of the model by loading
# the new embedding layer from file (created with vocab_initalization.py)
# also loads the appropriate tokenizer for this embedding layer
def _reinit_embedding_layer(model, reinit_path, model_name):
    # load appropriate tokenizer
    tokenizer = XGLMTokenizerFast.from_pretrained(f"{reinit_path}/tokenizer.pt")
    pad_idx = tokenizer.encode("<pad>", add_special_tokens=False)[0]

    # load new embeddings
    model_name = model_name.replace("/", ".")
    new_embed = torch.load(f"{reinit_path}/{model_name}_embed.pt")
    new_embed = torch.nn.Embedding.from_pretrained(new_embed, padding_idx=pad_idx)

    # set the model's new embeddings, then tie weights to output layer
    model.set_input_embeddings(new_embed)
    model.tie_weights()
    model = _unfreeze_embedding_layer(model)
    # we are keeping vocab size fixed
    # model.config.vocab_size = new_vocab_size

    return model, tokenizer


def set_training_args(args):
    training_args = SFTConfig(
        output_dir=args.serialization_dir,
        overwrite_output_dir=True,
        do_train=True,
        max_seq_length=2048,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.valid_bsz,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        prediction_loss_only=True,
        max_steps=args.max_steps,
        warmup_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=args.rand_seed,
        fp16=True,
        fp16_full_eval=True,  # half precision eval = faster but worse
        dataloader_num_workers=2,
        disable_tqdm=False,
        greater_is_better=False,
        dataset_text_field="text",
        packing=True,
        torch_compile=True,  # TODO hopefully works?
        label_names=None,
        # remove_unused_columns=False
    )
    return training_args


def load_train_data(langs, data_prefix, rand_seed, tokenizer):
    train_data = []
    valid_data = []
    valid_langs = []

    for lang in langs:
        t = load_from_disk(f"{data_prefix}/train/{lang}")
        t = t.remove_columns("timestamp")
        train_data.append(t)
        v = load_from_disk(f"{data_prefix}/valid/{lang}")
        v = v.remove_columns("timestamp")
        valid_data.append(v)

    # trl.trainer.ConstantLengthDataset
    train_ds = (
        concatenate_datasets(train_data).shuffle(seed=rand_seed).flatten_indices()
    )
    valid_ds = (
        concatenate_datasets(valid_data).shuffle(seed=rand_seed).flatten_indices()
    )
    train_ds = ConstantLengthDataset(
        tokenizer, train_ds, "text", infinite=True, seq_length=2048
    )
    valid_ds = ConstantLengthDataset(
        tokenizer, valid_ds, "text", infinite=False, seq_length=2048
    )
    return train_ds, valid_ds


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
    print(model)

    if len(args.reinit_embed) > 0:
        model, tokenizer = _reinit_embedding_layer(
            model, args.reinit_embed, args.model_name
        )
    if args.freeze_embed:
        model = _freeze_embedding_layer(model)
    elif args.freeze_model:
        model = _freeze_transformer(model)
        model = _unfreeze_embedding_layer(model)

    # set up data for this training run
    # masking is NOT handled across samples (in diff languages) by ConstantLengthDataset
    train, valid = load_train_data(
        args.languages, args.data_prefix, args.rand_seed, tokenizer
    )

    # training set up
    training_args = set_training_args(args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        # compute_metrics=compute_metrics
        # compute metrics not implemented because of course not
    )

    # train model
    trainer.train()

    # perplexity sanity check v.2
    predictions = trainer.predict(valid)
    print(predictions)
    x = predictions[2][
        "test_loss"
    ]  # torch.nn.cross_entropy uses natural log, so (https://discuss.pytorch.org/t/why-does-torch-nn-functional-cross-entropy-use-the-natural-logarithm-instead-of-base-2/173753)
    print(math.exp(x))  # we use natural log as base here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--serialization_dir",
        default="./model_best",
        type=str,
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--initalization_dir", type=str, help="Path to starting model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        help="Path to new tokenizer if vocabulary is reinitalized",
    )
    parser.add_argument(
        "--model_name", default="facebook/xglm-564M", choices=MODEL_OPTIONS
    )
    parser.add_argument("--rand_seed", default=42, type=int)

    # Experimental Settings
    parser.add_argument("--freeze_embed", action="store_true")
    parser.add_argument("--freeze_model", action="store_true")
    parser.add_argument("--reinit_embed", type=str, default="")

    # training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--train_bsz", type=int, default=4)
    parser.add_argument("--valid_bsz", type=int, default=4)

    # TODO add clustering, data specific parameters
    parser.add_argument("--languages", nargs="*", choices=LANG_OPTIONS, required=True)
    # /home/user/project/data/mc4_1GB
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

# EOF
