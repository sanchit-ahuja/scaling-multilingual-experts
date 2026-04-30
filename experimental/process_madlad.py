import os
from datasets import load_dataset, DatasetDict

# Ensure token is available
if not (os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")):
    print("Warning: No Hugging Face token found in environment variables")



LANGS = {
    "Slavic": ["bg", "cs", "pl", "sl", "lt", "lv"],
    "Germanic": ["de", "sv", "no", "is"],
    "Indic": ["as", "gu", "or", "pa", "si", "ur", "sd"],
    "Austronesian": ["ilo", "war", "mg", "mi", "su"],
    "Romance": ["ca"],
}


def load_language_dataset(lang_group, lang):
    """Load a single language dataset"""
    print(f"Processing {lang} in {lang_group}...")
    try:
        dataset = load_dataset(
            "allenai/madlad-400",
            languages=[lang],
            split="clean[:10%]",  # Download only 10% of the data
            trust_remote_code=True,
            num_proc=10,  # Reduced per language to avoid resource conflicts
        )
        print(f"Completed {lang} in {lang_group} - loaded {len(dataset)} samples (10%)")
        return lang_group, lang, dataset
    except Exception as e:
        print(f"Error processing {lang} in {lang_group}: {e}")
        return lang_group, lang, None


def create_local_dataset() -> DatasetDict:
    final_dataset = DatasetDict()

    # Initialize nested structure
    for lang_group in LANGS.keys():
        final_dataset[lang_group] = DatasetDict()

    # Load datasets sequentially
    for lang_group, langs in LANGS.items():
        for lang in langs:
            lang_group_result, lang_result, dataset = load_language_dataset(lang_group, lang)
            if dataset is not None:
                final_dataset[lang_group_result][lang_result] = dataset
            else:
                print(f"Failed to load {lang} in {lang_group}")

    return final_dataset


if __name__ == "__main__":
    final_dataset = create_local_dataset()
    final_dataset.save_to_disk("/scratch/ahuja.sanc/madlad-dataset-held-out-langs")
