#!/usr/bin/env python3
"""
Push dataset to Hugging Face Hub
"""

from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

def push_to_hub(dataset_dir: Path, repo_id: str):
    """Push dataset to Hugging Face Hub."""

    print(f"Loading dataset from {dataset_dir}...")

    # Load from parquet file
    parquet_file = dataset_dir / "dataset.parquet"

    if not parquet_file.exists():
        print(f"Error: {parquet_file} not found!")
        return

    # Load dataset
    dataset = load_dataset("parquet", data_files=str(parquet_file))

    print(f"\nDataset loaded: {dataset}")
    print(f"Number of examples: {len(dataset['train'])}")

    # Push to hub
    print(f"\nPushing to {repo_id}...")
    dataset.push_to_hub(repo_id, private=False)

    print(f"\n✅ Dataset pushed successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")

    # Also upload README and stats
    print("\nUploading additional files...")
    api = HfApi()

    readme_file = dataset_dir / "README.md"
    stats_file = dataset_dir / "dataset_stats.json"
    token_dist_file = dataset_dir / "token_distribution.json"

    if readme_file.exists():
        api.upload_file(
            path_or_fileobj=str(readme_file),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  ✓ Uploaded README.md")

    if stats_file.exists():
        api.upload_file(
            path_or_fileobj=str(stats_file),
            path_in_repo="dataset_stats.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  ✓ Uploaded dataset_stats.json")

    if token_dist_file.exists():
        api.upload_file(
            path_or_fileobj=str(token_dist_file),
            path_in_repo="token_distribution.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  ✓ Uploaded token_distribution.json")

    print(f"\n🎉 All done! Visit: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python push_to_hf.py <repo_id>")
        print("Example: python push_to_hf.py archit11/deepwiki-16k")
        sys.exit(1)

    repo_id = sys.argv[1]
    dataset_dir = Path(__file__).parent / "token_aware_dataset_output"

    push_to_hub(dataset_dir, repo_id)
