"""Upload x-elm-v2 checkpoints to HuggingFace Hub.

Uploads 16 trained checkpoints (1 dense + 5 experts + 5 freeze + 5 layer-reg).
Reverted variants and the expert soup are not uploaded; each README points at
the reproducer script.

Usage:
    python upload_to_hf.py --namespace sanchit-ahuja --private --dry-run
    python upload_to_hf.py --namespace sanchit-ahuja --private --only expert --only-family Slavic
    python upload_to_hf.py --namespace sanchit-ahuja --private --skip-existing
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


CHECKPOINT_BASE = os.environ.get("CHECKPOINTS_ROOT", "./checkpoints")
GITHUB_URL = "https://github.com/sanchit-ahuja/x-elm-v2"
BASE_MODEL = "google/gemma-3-4b-pt"

# Family -> full language-code list (training + held-out; mirrors the commented
# FULL map in configs/constants.py). Kept here because constants.py is in flux.
FAMILY_LANGS = {
    "slavic":       ["mk", "hr", "ru", "sk", "sr", "uk", "bg", "cs", "pl", "sl", "lt", "lv"],
    "germanic":     ["af", "fy", "lb", "da", "nl", "de", "en", "sv", "no", "is"],
    "indic":        ["bn", "hi", "kn", "ml", "mr", "ne", "ta", "te", "as", "gu", "or", "pa", "si", "ur", "sd"],
    "austronesian": ["sm", "jv", "ceb", "fil", "id", "ms", "ilo", "war", "mg", "mi", "su"],
    "romance":      ["es", "pt", "fr", "gl", "it", "ro", "ca"],
}


@dataclass
class Entry:
    local_path: str
    repo_name: str
    group: str  # dense | expert | freeze | layer-reg
    family: str | None  # None for dense


MANIFEST: list[Entry] = [
    # Dense
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_dense_25b/final",
          "xelm-gemma-4b-dense", "dense", None),

    # Experts (Indic uses checkpoint-7000; that's what was evaluated in the paper)
    Entry(f"{CHECKPOINT_BASE}/slavic_gemma_4b_expert/final",
          "xelm-gemma-4b-slavic-expert",       "expert", "slavic"),
    Entry(f"{CHECKPOINT_BASE}/germanic_gemma_4b_expert/final",
          "xelm-gemma-4b-germanic-expert",     "expert", "germanic"),
    Entry(f"{CHECKPOINT_BASE}/Indic_gemma_4b_expert/checkpoint-7000",
          "xelm-gemma-4b-indic-expert",        "expert", "indic"),
    Entry(f"{CHECKPOINT_BASE}/austronesian_gemma_4b_expert/final",
          "xelm-gemma-4b-austronesian-expert", "expert", "austronesian"),
    Entry(f"{CHECKPOINT_BASE}/romance_gemma_4b_expert/final",
          "xelm-gemma-4b-romance-expert",      "expert", "romance"),

    # Freeze
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_slavic_freeze/final",
          "xelm-gemma-4b-slavic-freeze",       "freeze", "slavic"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_germanic_freeze/final",
          "xelm-gemma-4b-germanic-freeze",     "freeze", "germanic"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_indic_freeze/final",
          "xelm-gemma-4b-indic-freeze",        "freeze", "indic"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_austronesian_freeze/final",
          "xelm-gemma-4b-austronesian-freeze", "freeze", "austronesian"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_romance_freeze/final",
          "xelm-gemma-4b-romance-freeze",      "freeze", "romance"),

    # Layer-reg (layer_range_l2sp)
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_slavic_layer_reg/final",
          "xelm-gemma-4b-slavic-layer-reg",       "layer-reg", "slavic"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_germanic_layer_reg/final",
          "xelm-gemma-4b-germanic-layer-reg",     "layer-reg", "germanic"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_indic_layer_reg/final",
          "xelm-gemma-4b-indic-layer-reg",        "layer-reg", "indic"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_austronesian_layer_reg/final",
          "xelm-gemma-4b-austronesian-layer-reg", "layer-reg", "austronesian"),
    Entry(f"{CHECKPOINT_BASE}/gemma_4b_romance_layer_reg/final",
          "xelm-gemma-4b-romance-layer-reg",      "layer-reg", "romance"),
]


IGNORE_PATTERNS = [
    "runs/**",
    "checkpoint-*/**",
    "reverted*/**",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "*.log",
    "*.bin.index.tmp",
]


GROUP_BLURB = {
    "dense": (
        "Dense continual pre-training (CPT) of Gemma-3-4B on a concatenated "
        "25B-token mixture across Slavic, Germanic, Indic, Austronesian, and "
        "Romance language families. This is the no-regularization baseline."
    ),
    "expert": (
        "Single-family expert: CPT of Gemma-3-4B on one language family only. "
        "Used as a building block for model soup and for measuring per-family "
        "specialization."
    ),
    "freeze": (
        "Layer-freezing strategy: middle transformer layers are frozen at the "
        "base Gemma-3-4B weights; only the first and last layers are updated "
        "during CPT. Mitigates catastrophic forgetting of general capabilities."
    ),
    "layer-reg": (
        "Layer-range L2-SP regularization: middle layers receive a larger L2 "
        "penalty against the base Gemma-3-4B weights than the first/last "
        "layers. Soft equivalent of layer freezing."
    ),
}


GROUP_RECIPE_YAML = {
    "dense":     "configs/yaml/train_gemma_dense.yaml",
    "expert":    "configs/yaml/train_gemma_single_expert.yaml",
    "freeze":    "configs/yaml/train_gemma_freeze.yaml",
    "layer-reg": "configs/yaml/train_gemma_layer_range.yaml",
}


def _format_recipe(group: str) -> str:
    rel = GROUP_RECIPE_YAML[group]
    github_url = f"{GITHUB_URL}/blob/main/{rel}"
    return (
        f"The exact training recipe lives in [`{rel}`]({github_url}) in the "
        "code repo. The resolved config used for this specific run is also "
        "included in this model repo as `training_config.yaml` — load it with "
        "pyrallis to reproduce the run bit-for-bit:\n\n"
        "```bash\n"
        f"python train.py --config_path {rel}\n"
        "```\n"
    )


def _reproducer_block(entry: Entry, namespace: str) -> str:
    repo_id = f"{namespace}/{entry.repo_name}"
    if entry.group == "expert":
        return (
            "## Reproducing the reverted variant\n\n"
            "The *expert-reverted* variant restores middle-layer weights to the "
            "base Gemma-3-4B while keeping the trained first/last layers. "
            "It is not uploaded to the Hub; regenerate it with:\n\n"
            "```bash\n"
            "python train.py --config_path configs/yaml/revert_gemma_checkpoint.yaml \\\n"
            f"    --revert.checkpoint_path $(huggingface-cli download {repo_id}) \\\n"
            "    --revert.revert_output_path ./reverted\n"
            "```\n\n"
            "## Reproducing the expert soup\n\n"
            "See the `xelm-gemma-4b-dense` repo README for the full soup recipe.\n"
        )
    if entry.group == "dense":
        return (
            "## Reproducing the dense-reverted variant\n\n"
            "```bash\n"
            "python train.py --config_path configs/yaml/revert_gemma_checkpoint.yaml \\\n"
            f"    --revert.checkpoint_path $(huggingface-cli download {repo_id}) \\\n"
            "    --revert.revert_output_path ./reverted\n"
            "```\n\n"
            "## Reproducing the expert soup (uniform average of 5 experts)\n\n"
            "```bash\n"
            f"python model_soup.py \\\n"
            f"    --experts {namespace}/xelm-gemma-4b-slavic-expert \\\n"
            f"              {namespace}/xelm-gemma-4b-germanic-expert \\\n"
            f"              {namespace}/xelm-gemma-4b-indic-expert \\\n"
            f"              {namespace}/xelm-gemma-4b-austronesian-expert \\\n"
            f"              {namespace}/xelm-gemma-4b-romance-expert \\\n"
            f"    --output_dir ./xelm-gemma-4b-expert-soup \\\n"
            f"    --alpha 1.0\n"
            "```\n"
        )
    return ""


def _build_readme(entry: Entry, namespace: str) -> str:
    langs = FAMILY_LANGS.get(entry.family, []) if entry.family else sorted(
        {l for ls in FAMILY_LANGS.values() for l in ls}
    )
    tags = ["continual-pretraining", "multilingual", "x-elm", "gemma-3", entry.group]
    if entry.family:
        tags.append(entry.family)

    front_matter = ["---"]
    front_matter.append("license: gemma")
    front_matter.append(f"base_model: {BASE_MODEL}")
    if langs:
        front_matter.append("language:")
        for lc in langs:
            front_matter.append(f"  - {lc}")
    front_matter.append("tags:")
    for t in tags:
        front_matter.append(f"  - {t}")
    front_matter.append("library_name: transformers")
    front_matter.append("pipeline_tag: text-generation")
    front_matter.append("---")

    title = entry.repo_name
    family_line = (
        f"- **Language family**: {entry.family.capitalize()}\n"
        if entry.family else "- **Language families**: all 5 (Slavic, Germanic, Indic, Austronesian, Romance)\n"
    )

    recipe_md = _format_recipe(entry.group)
    reproducer_md = _reproducer_block(entry, namespace)

    body = f"""
# {title}

{GROUP_BLURB[entry.group]}

- **Base model**: [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
- **Strategy**: `{entry.group}`
{family_line}- **Code**: [{GITHUB_URL}]({GITHUB_URL})

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{namespace}/{entry.repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{namespace}/{entry.repo_name}")
```

## Training recipe

{recipe_md}
{reproducer_md}
## Citation

To be added upon release of the accompanying paper (COLM 2026).
"""
    return "\n".join(front_matter) + body


def _write_readme(entry: Entry, namespace: str, dry_run: bool) -> None:
    readme_path = Path(entry.local_path) / "README.md"
    content = _build_readme(entry, namespace)
    if dry_run:
        print(f"\n--- README preview for {entry.repo_name} ({readme_path}) ---")
        print(content[:800] + ("…" if len(content) > 800 else ""))
        return
    readme_path.write_text(content)


COLLECTIONS = {
    "expert":    ("x-elm Experts — Gemma-3-4B",
                  "Per-family expert checkpoints from continual pre-training of Gemma-3-4B."),
    "freeze":    ("x-elm Freeze — Gemma-3-4B",
                  "Layer-freezing checkpoints: middle layers kept at base, first/last trained."),
    "layer-reg": ("x-elm L2-Reg — Gemma-3-4B",
                  "Layer-range L2-SP regularization checkpoints."),
}


def _ensure_collection(api: HfApi, namespace: str, group: str, entries: list[Entry], dry_run: bool):
    title, description = COLLECTIONS[group]
    if dry_run:
        print(f"\n[dry-run] Collection '{title}' with {len(entries)} items:")
        for e in entries:
            print(f"  - {namespace}/{e.repo_name}")
        return
    slug = None
    # Look up existing collection by title on the user's profile.
    try:
        for coll in api.list_collections(owner=namespace):
            if coll.title == title:
                slug = coll.slug
                break
    except Exception as e:
        print(f"  [warn] list_collections failed: {e}")
    if slug is None:
        coll = api.create_collection(title=title, description=description, namespace=namespace, private=True)
        slug = coll.slug
        print(f"  created collection: {slug}")
    else:
        print(f"  reusing collection: {slug}")
    for e in entries:
        repo_id = f"{namespace}/{e.repo_name}"
        try:
            api.add_collection_item(collection_slug=slug, item_id=repo_id, item_type="model")
            print(f"    + {repo_id}")
        except HfHubHTTPError as err:
            if "already" in str(err).lower():
                print(f"    = {repo_id} (already in collection)")
            else:
                print(f"    [warn] failed to add {repo_id}: {err}")


def _filter(manifest: list[Entry], only: str | None, only_family: str | None) -> list[Entry]:
    out = manifest
    if only:
        out = [e for e in out if e.group == only]
    if only_family:
        fam = only_family.lower()
        out = [e for e in out if (e.family or "").lower() == fam]
    return out


def _should_skip(api: HfApi, repo_id: str, commit_msg: str) -> bool:
    try:
        refs = api.list_repo_refs(repo_id)
        branches = getattr(refs, "branches", []) or []
        for b in branches:
            if b.name == "main":
                # Compare the latest commit message on main.
                commits = api.list_repo_commits(repo_id)
                if commits and commits[0].title.strip() == commit_msg.strip():
                    return True
                return False
    except HfHubHTTPError:
        return False
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", required=True, help="HF user or org name")
    p.add_argument("--private", action="store_true", default=True,
                   help="Create repos as private (default: True)")
    p.add_argument("--public", dest="private", action="store_false",
                   help="Create repos as public instead")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", choices=["dense", "expert", "freeze", "layer-reg"])
    p.add_argument("--only-family", help="Filter by family (e.g. slavic)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip repos whose latest main commit message matches this run")
    p.add_argument("--skip-repo", action="append", default=[],
                   help="Skip a specific repo by name (repeatable). "
                        "Matches against either the short repo name or the full repo_id.")
    p.add_argument("--no-collections", action="store_true",
                   help="Don't create/update HF collections")
    args = p.parse_args()

    entries = _filter(MANIFEST, args.only, args.only_family)
    skip_set = {s.split("/")[-1] for s in args.skip_repo}
    if skip_set:
        entries = [e for e in entries if e.repo_name not in skip_set]
    if not entries:
        print("No entries matched filters.")
        return

    print(f"Target namespace: {args.namespace}")
    print(f"Private:          {args.private}")
    print(f"Dry-run:          {args.dry_run}")
    print(f"Entries:          {len(entries)}/{len(MANIFEST)}")
    print()

    api = HfApi()

    for entry in entries:
        repo_id = f"{args.namespace}/{entry.repo_name}"
        commit_msg = f"Upload {entry.repo_name}"
        print(f"[{entry.group}] {repo_id}")
        print(f"  source: {entry.local_path}")

        if not os.path.isdir(entry.local_path):
            print(f"  [skip] MISSING local path")
            continue

        if args.dry_run:
            print(f"  [dry-run] would create_repo(private={args.private}) + upload_folder")
            _write_readme(entry, args.namespace, dry_run=True)
            continue

        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=args.private, exist_ok=True)

        if args.skip_existing and _should_skip(api, repo_id, commit_msg):
            print(f"  [skip] latest commit already '{commit_msg}'")
            continue

        _write_readme(entry, args.namespace, dry_run=False)
        api.upload_folder(
            folder_path=entry.local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
            ignore_patterns=IGNORE_PATTERNS,
        )
        print(f"  uploaded")

    if not args.no_collections:
        print("\n=== Collections ===")
        # Collections use the full MANIFEST (respecting --only / --only-family
        # but ignoring --skip-repo and --skip-existing) so that repos uploaded
        # in a previous run are still added to the collection.
        coll_entries = _filter(MANIFEST, args.only, args.only_family)
        for group in ("expert", "freeze", "layer-reg"):
            group_entries = [e for e in coll_entries if e.group == group]
            if group_entries:
                _ensure_collection(api, args.namespace, group, group_entries, args.dry_run)


if __name__ == "__main__":
    main()
