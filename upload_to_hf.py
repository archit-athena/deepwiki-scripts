#!/usr/bin/env python3
"""
Upload token-aware dataset to Hugging Face Hub
"""

import json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import create_repo, HfApi


def load_jsonl(file_path: Path):
    """Load JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def upload_to_huggingface(dataset_path: Path, repo_name: str = "archit11/hyperswitch-token-aware-cpt"):
    """Upload dataset to Hugging Face"""
    print(f"\n📤 Uploading to Hugging Face: {repo_name}")

    # Load JSONL
    print("   Loading dataset...")
    samples = load_jsonl(dataset_path)
    print(f"   Loaded {len(samples)} samples")

    # Create HF dataset
    print("   Creating HF dataset...")
    dataset = Dataset.from_list(samples)

    # Print dataset info
    print(f"\n   Dataset info:")
    print(f"     Samples: {len(samples)}")
    print(f"     Features: {list(dataset.features.keys())}")

    # Calculate stats
    token_counts = [s['metadata']['token_count'] for s in samples]
    print(f"\n   Token statistics:")
    print(f"     Min: {min(token_counts):,} tokens")
    print(f"     Max: {max(token_counts):,} tokens")
    print(f"     Mean: {sum(token_counts)//len(token_counts):,} tokens")
    print(f"     Total: {sum(token_counts):,} tokens")

    # Create repo (if doesn't exist)
    print(f"\n   Creating/accessing repo: {repo_name}")
    try:
        create_repo(repo_name, repo_type="dataset", exist_ok=True, private=False)
        print("   ✓ Repo ready")
    except Exception as e:
        print(f"   Note: {e}")

    # Push to hub
    print("\n   Pushing to hub (this may take a while)...")
    dataset.push_to_hub(repo_name, private=False)

    print(f"\n✅ Successfully uploaded to https://huggingface.co/datasets/{repo_name}")

    # Create README content
    readme = f"""# Hyperswitch Token-Aware CPT Dataset

This dataset contains **{len(samples):,} samples** of Rust code from the [Hyperswitch](https://github.com/juspay/hyperswitch) payment router project, optimized for Continued Pre-Training (CPT) with the **Kwaipilot/KAT-Dev** tokenizer.

## Dataset Statistics

- **Total Samples**: {len(samples):,}
- **Total Tokens**: {sum(token_counts):,}
- **Mean Tokens per Sample**: {sum(token_counts)//len(token_counts):,}
- **Token Range**: {min(token_counts):,} - {max(token_counts):,}

### Token Distribution

- **< 4k tokens**: 38.1% of samples
- **4k-10k tokens**: 52.0% of samples
- **10k+ tokens**: 9.9% of samples

### Granularity Types

- **file**: 721 samples (single large files)
- **module**: 180 samples (multiple files from same module)
- **combined_files**: 160 samples (small files combined by crate)
- **crate**: 15 samples (entire small crates)

## Top Crates

1. **router** - 371 samples
2. **hyperswitch_connectors** - 336 samples
3. **analytics** - 54 samples
4. **diesel_models** - 39 samples
5. **api_models** - 28 samples

## Sample Structure

Each sample contains:
- `id`: Unique identifier
- `type`: Sample type (always "clm" for causal language modeling)
- `granularity`: Level of code organization (file/module/combined_files/crate)
- `content`: Full code with path metadata in format:
  ```
  <path>
  Repository: hyperswitch
  Crate: [crate_name]
  File: [file_path]
  Tokens: [token_count]
  </path>

  <file>
  [actual code content]
  </file>
  ```
- `metadata`: Contains crate, file info, and token count

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("archit11/hyperswitch-token-aware-cpt")

# Access samples
sample = dataset['train'][0]
print(f"Tokens: {{sample['metadata']['token_count']:,}}")
print(f"Crate: {{sample['metadata']['crate']}}")
print(f"Granularity: {{sample['granularity']}}")

# Filter by token count
medium_samples = dataset['train'].filter(
    lambda x: 4000 <= x['metadata']['token_count'] < 10000
)

# Filter by crate
router_samples = dataset['train'].filter(
    lambda x: x['metadata']['crate'] == 'router'
)
```

## Training Recommendations

- **Context Length**: 16k tokens (max sample is 15,609 tokens)
- **Tokenizer**: Kwaipilot/KAT-Dev
- **Suggested Batch Size**: 1-2 samples per batch (due to large context)
- **Format**: Samples are pre-formatted with `<path>` and `<file>` tags

## Source

- **Repository**: https://github.com/juspay/hyperswitch
- **Language**: Rust
- **License**: Apache 2.0

## Generation Method

Samples were generated using token-aware strategies:
1. **Large files** (2k-16k tokens) included as-is
2. **Small files** combined within same crate until reaching 2k+ tokens
3. **Module clusters** grouped by directory structure
4. **Complete crates** for small crates that fit within context

All token counts measured using the Kwaipilot/KAT-Dev tokenizer.
"""

    # Upload README
    api = HfApi()
    print("\n   Uploading README...")
    try:
        with open("/tmp/README.md", "w") as f:
            f.write(readme)
        api.upload_file(
            path_or_fileobj="/tmp/README.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
        )
        print("   ✓ README uploaded")
    except Exception as e:
        print(f"   Note: {e}")


def main():
    dataset_path = Path('/Users/architsinghai/code/repo_cpt_dataset_clean/token_aware_dataset.jsonl')

    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return

    print("=" * 80)
    print("UPLOAD TO HUGGING FACE")
    print("=" * 80)

    upload_to_huggingface(
        dataset_path=dataset_path,
        repo_name="archit11/hyperswitch-token-aware-cpt-fixed"
    )


if __name__ == '__main__':
    main()
