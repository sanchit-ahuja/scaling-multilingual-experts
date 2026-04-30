# Create dataset bucket script for different language families.

## Instructions to follow. Feel free to ask follow-up questions if you have any doubts.
1. I have data stored as `DatasetDict` for these language families: ```LANGS = {
    "Romance": ["es", "pt", "ro", "ga", "fr", "it"],
    "Slavic": ["mk", "hr", "ru", "sk", "sr", "uk"],
    "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Germanic": ["af", "fy", "lb", "da", "nl", "de", "en"],
}```. The path for the dataset for each language family is `/projects/lilac_lab/madlad-dataset/<language-family>`. The TOTAL upper limit budget for our dataset is 100 Billion tokens. This turns out to be 25B tokens per family upper limit.
2. The tokenizer that we are using for this exercise is Gemma 3 tokenizer. Path: `google/gemma-3-1b-pt`. We want to ensure that _each_ language gets proper representation. To ensure that, our objective is to have at least minimum tokens for a given language. For example, if ml in Indic only has 400M tokens and this is the minimum number of tokens across all languages for the given Indic family, then _all_ the languages would only have 400M tokens. Also, while sampling we stick to uniform sampling to reach to an upper limit of 25B tokens per language family. Do ask clarification heree if something is not clear!
3. Ensure that this run in parallel and runs fast. Ensure that we can save intermediate results when a given family is processed or done. Save the final processed madlad-dataset at `/scratch/ahuja.sanc/processed-madlad-dataset` in the same format as it was loaded i.e. HF format dataset.