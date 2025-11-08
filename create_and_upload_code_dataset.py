#!/usr/bin/env python3
"""
Create code dataset and upload to Hugging Face in one go.
Creates a simplified, unified code dataset ready for CPT training.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleCodeDatasetCreator:
    """Simplified code dataset creator - one unified dataset"""

    def __init__(self, repo_dir: Path, output_dir: Path):
        self.repo_dir = repo_dir
        self.output_dir = output_dir
        self.tokenizer = None

    def load_tokenizer(self):
        """Load tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Kwaipilot/KAT-Dev", trust_remote_code=True)
            logger.info("✓ Tokenizer loaded")
        except Exception as e:
            logger.warning(f"Using estimation: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    def is_excluded(self, path: Path) -> bool:
        """Check if should exclude"""
        path_str = str(path)
        return any(p in path_str for p in ['/target/', '/generated/', '.gen.rs', '/build/', '/tests/'])

    def extract_module_path(self, rel_path: Path) -> str:
        """Get module path"""
        parts = list(rel_path.parts)
        if parts[-1].endswith('.rs'):
            parts[-1] = parts[-1][:-3]
        if 'crates' in parts:
            idx = parts.index('crates')
            parts = parts[idx + 1:]
        return "::".join(parts)

    def create_sample(self, file_path: Path) -> Dict:
        """Create a simple code sample"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = file_path.relative_to(self.repo_dir)

            if self.is_excluded(file_path):
                return None

            module_path = self.extract_module_path(rel_path)

            # Simple format - just file path header + code
            text = f"// File: {rel_path}\n// Module: {module_path}\n\n{content}"

            return {
                'text': text,
                'file_path': str(rel_path),
                'module': module_path,
                'token_count': self.count_tokens(text),
                'has_source_code': True  # All samples are code
            }

        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None

    def create_dataset(self) -> List[Dict]:
        """Create dataset from all Rust files"""
        logger.info(f"Collecting Rust files from {self.repo_dir}")
        rust_files = list(self.repo_dir.rglob("*.rs"))
        logger.info(f"✓ Found {len(rust_files)} Rust files")

        samples = []
        logger.info("Processing files...")
        for file_path in tqdm(rust_files, desc="Processing"):
            sample = self.create_sample(file_path)
            if sample:
                samples.append(sample)

        logger.info(f"✓ Created {len(samples)} code samples")
        return samples

    def save_and_upload(self, samples: List[Dict], repo_id: str):
        """Save locally and upload to HuggingFace"""
        self.output_dir.mkdir(exist_ok=True)

        # Save as JSONL locally
        jsonl_path = self.output_dir / 'code_dataset.jsonl'
        logger.info(f"Saving to {jsonl_path}")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Create HuggingFace dataset
        logger.info("Creating HuggingFace dataset...")
        dataset = Dataset.from_list(samples)
        dataset_dict = DatasetDict({'train': dataset})

        logger.info(f"Dataset: {dataset}")
        logger.info(f"  Samples: {len(dataset)}")
        logger.info(f"  Columns: {dataset.column_names}")

        # Upload to HuggingFace
        logger.info(f"\nUploading to {repo_id}...")
        dataset_dict.push_to_hub(repo_id, private=False)

        logger.info(f"\n✅ Dataset uploaded!")
        logger.info(f"   View at: https://huggingface.co/datasets/{repo_id}")

        # Save stats
        token_counts = [s['token_count'] for s in samples]
        stats = {
            'total_samples': len(samples),
            'total_tokens': sum(token_counts),
            'mean_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'unique_modules': len(set(s['module'] for s in samples))
        }

        stats_path = self.output_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"\n📊 Statistics:")
        logger.info(f"   Samples: {stats['total_samples']:,}")
        logger.info(f"   Total tokens: {stats['total_tokens']:,}")
        logger.info(f"   Mean tokens: {stats['mean_tokens']:.0f}")
        logger.info(f"   Range: {stats['min_tokens']:,} - {stats['max_tokens']:,}")
        logger.info(f"   Modules: {stats['unique_modules']}")

        return dataset_dict


def main():
    """Main pipeline"""
    script_dir = Path(__file__).parent.resolve()
    repo_dir = script_dir / 'hyperswitch'
    output_dir = script_dir / 'simple_code_output'
    repo_url = 'https://github.com/juspay/hyperswitch.git'
    hf_repo_id = 'archit11/hyperswitch-code'

    logger.info("=" * 70)
    logger.info("Hyperswitch Code Dataset Creator & Uploader")
    logger.info("=" * 70)

    # Clone repo if needed
    if not repo_dir.exists():
        logger.info(f"\nCloning {repo_url}...")
        subprocess.run(['git', 'clone', repo_url, str(repo_dir)], check=True)
        logger.info("✓ Repository cloned")
    else:
        logger.info(f"\n✓ Repository exists at {repo_dir}")

    # Create dataset
    creator = SimpleCodeDatasetCreator(repo_dir, output_dir)
    creator.load_tokenizer()
    samples = creator.create_dataset()

    if not samples:
        logger.error("No samples created!")
        return

    # Save and upload
    logger.info("\nSaving and uploading...")
    creator.save_and_upload(samples, hf_repo_id)

    logger.info("\n" + "=" * 70)
    logger.info("✅ Complete! Dataset available at:")
    logger.info(f"   https://huggingface.co/datasets/{hf_repo_id}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()