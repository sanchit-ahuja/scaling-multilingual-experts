import copy
import torch
import random
from tqdm import tqdm
import os
from collections import defaultdict

os.environ["HF_HOME"] = "/srv/home/users/blevinst24cs/.cache/huggingface/"

from transformers import XGLMTokenizerFast, XGLMForCausalLM
from datasets import load_from_disk, concatenate_datasets
import sentencepiece as spm

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
TMP_DATA_PATH = "/srv/home/users/blevinst24cs/x-elm-v2/data/tmp.txt"
SPM_PATH = "/srv/home/users/blevinst24cs/x-elm-v2/models/reinit_vocab/spm/mc4_1GiB"
REINIT_PREFIX = "/srv/home/users/blevinst24cs/x-elm-v2/models/reinit_vocab"
MODEL_OPTIONS = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-7.5B"]
UNICODE_PATH = "/srv/home/users/blevinst24cs/x-elm-v2/models/unicode.txt"
_xglm_special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def hex2dec(hex_str):
    """Convert Unicode hexadecimal string to base-10 int."""
    return int(hex_str, 16)


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def get_ord2script(scriptfile):
    """Return dictionary (key: Unicode decimal, val: script of corresponding
    character according to Unicode documentation)"""
    with open(scriptfile, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    ord2script = dict()
    for line in lines:
        if line[0] != "#":
            items = line.split()
            if len(items) > 0:
                script = items[2]
                encoding = items[0]
                if ".." in encoding:
                    start_stop = encoding.split("..")
                    start = hex2dec(start_stop[0])
                    stop = hex2dec(start_stop[1])
                    for dec_encoding in range(start, stop + 1):
                        ord2script[dec_encoding] = script
                else:
                    dec_encoding = hex2dec(encoding)
                    ord2script[dec_encoding] = script

    return ord2script


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def top_script(token, ord2script):
    """Return most-used script within token (str), using ord2script (dict)
    to retrieve the script of each char in token."""
    script_counts = defaultdict(lambda: 0)

    for character in token:
        try:
            script = ord2script[ord(character)]
        except KeyError:
            script = "UNK"
        script_counts[script] += 1

    return max(script_counts, key=lambda x: script_counts[x])


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def get_script_to_ids(
    vocab: dict[str, int], ord_to_script: dict[int, str], word_position: bool
) -> dict[str, list[int]]:
    whitespace = "▁"
    # get script for each token in XLM-R's vocab
    script_to_ids = defaultdict(list)
    for token, index in vocab.items():
        if token in _xglm_special_tokens:
            script_to_ids["xlmr_special"].append(index)
        # leave out the preceding whitespace when identifying token script
        token_text = token[1:] if token[0] == whitespace and len(token) > 1 else token
        # identify top script for the token based on characters and Unicode mapping
        script = top_script(token_text, ord_to_script)
        if word_position == True:
            if token[0] == whitespace:
                script += "_initial"
            else:
                script += "_medial"
        script_to_ids[script].append(index)
    return script_to_ids


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def initialize_by_category_means(
    categories: list[str],
    means: torch.Tensor,
    stdevs: torch.Tensor,
    category_to_indices: dict[str, int],
    matrix: torch.Tensor,
    categories_to_omit: list[str] = ["xlmr_special"],
) -> torch.Tensor:
    # only initialize the categories that are in both the old and new data
    category_intersection = set(categories).intersection(
        set(category_to_indices.keys())
    )
    for category in category_intersection:
        if category in categories_to_omit:
            continue
        category_index = categories.index(category)
        # make sure standard devs > 0
        stdevs[category_index] = torch.where(
            stdevs[category_index] == 0.0,
            torch.full(stdevs[category_index].shape, 1e-8),
            stdevs[category_index],
        )
        category_distribution = torch.distributions.Normal(
            means[category_index], stdevs[category_index]
        )
        for index in category_to_indices[category]:
            matrix[index] = category_distribution.sample()

    return matrix


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def reinitialize_by_script(
    old_vocab,
    old_embeddings,
    new_vocab,
    new_embeddings,
    unicode_table_path,
    word_position=False,
):
    # get dictionary to map Unicode decimal to script
    ord_to_script = get_ord2script(unicode_table_path)

    old_script_to_ids = get_script_to_ids(old_vocab, ord_to_script, word_position)
    new_script_to_ids = get_script_to_ids(new_vocab, ord_to_script, word_position)

    all_old_scripts = list(old_script_to_ids.keys())

    # get mean and standard deviation of embeddings for each script
    old_script_stdevs = []
    old_script_means = []
    for script in all_old_scripts:
        script_embed_list = [old_embeddings[x] for x in old_script_to_ids[script]]
        script_embeddings = torch.stack(script_embed_list, dim=0)
        std_and_mean = torch.std_mean(script_embeddings, dim=0)
        old_script_stdevs.append(std_and_mean[0])
        old_script_means.append(std_and_mean[1])
    old_script_stdevs = torch.stack(old_script_stdevs, dim=0)
    old_script_means = torch.stack(old_script_means, dim=0)

    new_embeddings = initialize_by_category_means(
        all_old_scripts,
        old_script_means,
        old_script_stdevs,
        new_script_to_ids,
        new_embeddings,
    )
    return new_embeddings


# from https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
def reinitialize_by_identity(
    old_vocab, old_embeddings, new_vocab, new_embeddings, tokens_to_ignore
):
    identical_tokens = []
    for token, new_index in new_vocab.items():
        if token in old_vocab and token not in tokens_to_ignore:
            new_embeddings[new_index] = old_embeddings[old_vocab[token]]
            identical_tokens.append(token)

    return new_embeddings, identical_tokens


# def batch_iterator(dataset, batch_size=1028):
#    for batch in dataset.iter(batch_size=batch_size):
#        yield batch["text"]


def main():
    # load tokenizer, dataset
    old_tokenizer = XGLMTokenizerFast.from_pretrained("facebook/xglm-564M")
    vocab_size = old_tokenizer.vocab_size
    # dataset = []
    # for lang in LANG_OPTIONS:
    # 	d = load_from_disk(f"{DATA_PREFIX}/train/{lang}")
    # 	d = d.remove_columns("timestamp")
    # 	dataset.append(d)
    # dataset = concatenate_datasets(dataset).shuffle(seed=42).flatten_indices()
    # dataset_size = len(dataset)
    # sample_cutoff = int(len(dataset)*0.2)

    """
	# this doesn't work because HF doesn't support training tokenizers on "large" corpora 
	# https://github.com/huggingface/tokenizers/issues/821
	#train new tokenizer on mC4 1GiB data
	new_tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(dataset), vocab_size)
	print(new_tokenizer.vocab_size, vocab_size)
	assert new_tokenizer.vocab_size == vocab_size
	"""

    # write dataset out to text file to compensate for HF
    # (note that this is obvs take up a ton of disk space...)
    # with open(TMP_DATA_PATH, 'w') as f:
    # 	for idx, example in tqdm(list(enumerate(dataset))):
    # 		if idx > sample_cutoff: break #save out subset of data
    # 		f.write(example["text"]+"\n")

    """
	#use sentencepiece for get new vocab 
	spm.SentencePieceTrainer.Train(
	    input=TMP_DATA_PATH,
	    model_prefix=SPM_PATH,
	    vocab_size=256000,
	    bos_id=0,
	    pad_id=1,
	    eos_id=2,
	    unk_id=3,
	    bos_piece='<s>',
	    pad_piece='<pad>',
	    eos_piece='</s>',
	    unk_piece='<unk>',
	    model_type='unigram',
	    character_coverage=0.999,
	    train_extremely_large_corpus=True
	)
	os.remove(TMP_DATA_PATH)
	"""

    # convert spm to huggingface tokenizer
    new_tokenizer = XGLMTokenizerFast(vocab_file=f"{SPM_PATH}.model")
    print(vocab_size, new_tokenizer.vocab_size)
    assert vocab_size == new_tokenizer.vocab_size
    new_tokenizer.save_pretrained(f"{REINIT_PREFIX}/tokenizer.pt")

    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    new_vocab_size = new_tokenizer.vocab_size

    # create new XGLM embeddings with new vocabulary
    # code based on https://github.com/cmdowney88/EmbeddingStructure/blob/main/src/reinitialize_embeddings.py
    for model_name in MODEL_OPTIONS:
        # load xglm model (for old embeddings)
        model = XGLMForCausalLM.from_pretrained(model_name)
        embedding_size = model.config.hidden_size
        old_embeddings = copy.deepcopy(model.get_input_embeddings().weight).detach()
        print(model_name, vocab_size, new_vocab_size, embedding_size)

        # create new embeddings table
        new_pad_idx = new_tokenizer.encode("<pad>", add_special_tokens=False)[0]
        new_embeddings = torch.nn.Embedding(
            new_vocab_size, embedding_size, padding_idx=new_pad_idx
        ).weight.detach()

        # set the embeddings for special tokens to be identical to XLM-R
        for special_token in _xglm_special_tokens:
            old_token_index = old_vocab[special_token]
            new_token_index = new_vocab[special_token]
            new_embeddings[new_token_index] = old_embeddings[old_token_index]

        # Run reinitalization heuristics (by script and then by id)
        new_embeddings = reinitialize_by_script(
            old_vocab,
            old_embeddings,
            new_vocab,
            new_embeddings,
            UNICODE_PATH,
            word_position=False,
        )
        new_embeddings, identical_tokens = reinitialize_by_identity(
            old_vocab,
            old_embeddings,
            new_vocab,
            new_embeddings,
            tokens_to_ignore=_xglm_special_tokens,
        )

        # save embedding table
        m = model_name.replace("/", ".")
        output_path = f"{REINIT_PREFIX}/{m}_embed.pt"
        torch.save(new_embeddings, output_path)


if __name__ == "__main__":
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)
    torch.manual_seed(42)
    main()
