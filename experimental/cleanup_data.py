import numpy as np
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import psutil

# Configuration
INPUT_BASE_PATH = "/projects/lilac_lab/sanchit/processed-madlad-dataset"
OUTPUT_BASE_PATH = "/scratch/ahuja.sanc/cleanup-madlad-dataset"
CLEANUP_CHECKPOINT_PATH = "/scratch/ahuja.sanc/cleanup_checkpoints"
# Language families
LANG_FAMILIES = {
    # "Austronesian": ["sm", "jv", "ceb", "fil", "id", "ms"],
    # "Indic": ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te"],
    "Indic": ["hi", "kn", "ml", "mr", "ne", "ta", "te"],
    # "Germanic": ["af", "da", "fy", "lb"],
    # "Romance": ["ga", "ro"],
    # "Austronesian": ["ceb", "fil", "jv"],
}

PERCENTAGE_SAMPLES_TO_KEEP = {
    # "Slavic": [0.75, 0.85, 0.95, 0.8, 0.75, 0.85]
    "Indic": [0.98, 0.75, 0.80, 0.92, 0.81, 0.95, 0.76]
    # "Germanic": [0.89, 0.89, 0.79, 0.82],
    # "Romance": [0.73, 0.88],
    # "Austronesian": [0.90, 0.91, 0.89],
}


def main():
    for family, percentages in PERCENTAGE_SAMPLES_TO_KEEP.items():
        langs = LANG_FAMILIES[family]
        family_datasets = {}

        for i, percentage in enumerate(percentages):
            print(f"Processing {family}/{langs[i]} with {percentage}% of data")
            dataset = load_from_disk(f"{INPUT_BASE_PATH}/{family}/{langs[i]}")
            # do random sampling
            dataset = dataset.select(
                np.random.choice(
                    len(dataset), int(len(dataset) * percentage), replace=False
                )
            )
            # dataset = dataset.select(range(int(len(dataset) * percentage)))
            family_datasets[langs[i]] = dataset

        # Create DatasetDict for the entire family
        family_dataset_dict = DatasetDict(family_datasets)
        print(f"Saving {family} DatasetDict with {len(langs)} languages")
        family_dataset_dict.save_to_disk(f"{OUTPUT_BASE_PATH}/{family}", num_proc=24)


if __name__ == "__main__":
    main()
