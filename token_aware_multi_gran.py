#!/usr/bin/env python3
"""
TOKEN-AWARE MULTI-GRANULARITY CPT DATASET
Uses tokenizer to measure samples, ensures BIG meaningful chunks
Target: 2000-16000 tokens per sample
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set
from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "Kwaipilot/KAT-Dev"):
    """Load the tokenizer for measuring tokens."""
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer loaded (vocab: {tokenizer.vocab_size:,})")
        return tokenizer
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return None


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tokenizer."""
    if not tokenizer:
        return len(text) // 4  # Fallback estimate
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except:
        return len(text) // 4


class TokenAwareDatasetGenerator:
    def __init__(self, repo_path: Path, tokenizer):
        self.repo_path = repo_path
        self.tokenizer = tokenizer
        self.samples = []

        # Token limits
        self.MIN_TOKENS = 2000   # Minimum for meaningful context
        self.MAX_TOKENS = 16000  # Max context length

        # File analysis
        self.file_data = {}  # path -> (content, token_count)
        self.crate_files = defaultdict(list)

    def analyze_files(self):
        """Scan and analyze all Rust files with token counts."""
        print("📊 Analyzing repository files...")

        rs_files = list(self.repo_path.rglob('*.rs'))
        print(f"   Found {len(rs_files)} Rust files")

        for rs_file in rs_files:
            try:
                with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Count tokens
                token_count = count_tokens(content, self.tokenizer)

                rel_path = rs_file.relative_to(self.repo_path)
                self.file_data[str(rel_path)] = (content, token_count)

                # Group by crate
                crate_name = self._extract_crate(rel_path)
                self.crate_files[crate_name].append((rs_file, content, token_count))

            except Exception as e:
                continue

        print(f"   Processed {len(self.file_data)} files")
        print(f"   Found {len(self.crate_files)} crates")

    def _extract_crate(self, rel_path: Path) -> str:
        """Extract crate name from path."""
        parts = rel_path.parts
        if 'crates' in parts:
            idx = parts.index('crates')
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return parts[0] if parts else 'unknown'

    def generate_samples(self):
        """Generate token-aware samples at different granularities."""
        print("\n📝 Generating TOKEN-AWARE samples...\n")

        self._generate_large_files()
        self._generate_combined_small_files()
        self._generate_module_clusters()
        self._generate_crate_samples()

        print(f"\n✅ Total samples generated: {len(self.samples)}")
        return self.samples

    def _generate_large_files(self):
        """Strategy 1: Large files (2k-16k tokens each)"""
        print("[1/4] Large file samples...")
        count = 0

        for file_path, (content, token_count) in self.file_data.items():
            # Only include files with good token count
            if self.MIN_TOKENS <= token_count <= self.MAX_TOKENS:
                rel_path = Path(file_path)
                crate_name = self._extract_crate(rel_path)

                sample = {
                    'id': f'large_file_{hash(file_path)}',
                    'type': 'clm',
                    'granularity': 'file',
                    'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
File: {file_path}
</path>

<file>
{content}
</file>
''',
                    'metadata': {
                        'crate': crate_name,
                        'file': file_path,
                        'token_count': token_count
                    }
                }

                self.samples.append(sample)
                count += 1

        print(f"   Generated {count} large file samples")

    def _generate_combined_small_files(self):
        """Strategy 2: Combine small files from same crate to reach 2k-16k tokens"""
        print("[2/4] Combined small files...")
        count = 0

        for crate_name, files in self.crate_files.items():
            # Get small files (under MIN_TOKENS)
            small_files = [(f, content, tokens) for f, content, tokens in files
                          if tokens < self.MIN_TOKENS]

            if not small_files:
                continue

            # Sort by size descending
            small_files.sort(key=lambda x: x[2], reverse=True)

            # Combine files until we reach MIN_TOKENS
            current_batch = []
            current_tokens = 0

            for rs_file, content, tokens in small_files:
                rel_path = rs_file.relative_to(self.repo_path)

                # Check if adding this would exceed MAX_TOKENS
                if current_tokens + tokens > self.MAX_TOKENS and current_batch:
                    # Flush current batch
                    combined_content = self._create_combined_sample(
                        crate_name, current_batch, current_tokens
                    )
                    self.samples.append(combined_content)
                    count += 1

                    # Start new batch
                    current_batch = [(rel_path, content, tokens)]
                    current_tokens = tokens
                else:
                    current_batch.append((rel_path, content, tokens))
                    current_tokens += tokens

                    # If we reached good size, flush
                    if current_tokens >= self.MIN_TOKENS * 2:  # Target mid-range
                        combined_content = self._create_combined_sample(
                            crate_name, current_batch, current_tokens
                        )
                        self.samples.append(combined_content)
                        count += 1

                        current_batch = []
                        current_tokens = 0

            # Flush remaining if it meets minimum
            if current_batch and current_tokens >= self.MIN_TOKENS:
                combined_content = self._create_combined_sample(
                    crate_name, current_batch, current_tokens
                )
                self.samples.append(combined_content)
                count += 1

        print(f"   Generated {count} combined file samples")

    def _create_combined_sample(self, crate_name: str, files: List, token_count: int) -> Dict:
        """Create a sample from multiple combined files."""
        file_list = [str(f[0]) for f in files]
        combined_content = []

        for rel_path, content, _ in files:
            # Add proper separation between files
            combined_content.append(f"// File: {rel_path}\n\n{content}")

        return {
            'id': f'combined_{crate_name}_{hash(tuple(file_list))}',
            'type': 'clm',
            'granularity': 'combined_files',
            'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
Files: {len(files)}
</path>

<files>
{chr(10) + chr(10).join(combined_content)}
</files>
''',
            'metadata': {
                'crate': crate_name,
                'num_files': len(files),
                'files': file_list,
                'token_count': token_count
            }
        }

    def _generate_module_clusters(self):
        """Strategy 3: Group files by module path"""
        print("[3/4] Module cluster samples...")
        count = 0

        # Group files by module (directory structure)
        module_groups = defaultdict(list)

        for file_path, (content, tokens) in self.file_data.items():
            rel_path = Path(file_path)
            # Get module path (directory without filename)
            if len(rel_path.parts) > 1:
                module_path = '/'.join(rel_path.parts[:-1])
                module_groups[module_path].append((file_path, content, tokens))

        # Create samples for modules with good total size
        for module_path, files in module_groups.items():
            total_tokens = sum(f[2] for f in files)

            # Only create if total is in good range
            if self.MIN_TOKENS * 1.5 <= total_tokens <= self.MAX_TOKENS:
                crate_name = self._extract_crate(Path(files[0][0]))
                file_list = [f[0] for f in files]

                combined_content = []
                for file_path, content, _ in files:
                    combined_content.append(f"// File: {file_path}\n\n{content}")

                sample = {
                    'id': f'module_{hash(module_path)}',
                    'type': 'clm',
                    'granularity': 'module',
                    'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
Module: {module_path}
Files: {len(files)}
</path>

<module>
{chr(10) + chr(10).join(combined_content)}
</module>
''',
                    'metadata': {
                        'crate': crate_name,
                        'module': module_path,
                        'num_files': len(files),
                        'files': file_list,
                        'token_count': total_tokens
                    }
                }

                self.samples.append(sample)
                count += 1

        print(f"   Generated {count} module cluster samples")

    def _generate_crate_samples(self):
        """Strategy 4: Small crates as single samples"""
        print("[4/4] Crate-level samples...")
        count = 0

        for crate_name, files in self.crate_files.items():
            # Calculate total tokens for crate
            total_tokens = sum(f[2] for f in files)

            # Only include crates that fit in context window
            if total_tokens <= self.MAX_TOKENS and total_tokens >= self.MIN_TOKENS:
                combined_content = []
                file_list = []

                for rs_file, content, tokens in files:
                    rel_path = rs_file.relative_to(self.repo_path)
                    file_list.append(str(rel_path))
                    combined_content.append(f"// File: {rel_path}\n\n{content}")

                sample = {
                    'id': f'crate_{hash(crate_name)}',
                    'type': 'clm',
                    'granularity': 'crate',
                    'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
Files: {len(files)}
</path>

<crate>
{chr(10) + chr(10).join(combined_content)}
</crate>
''',
                    'metadata': {
                        'crate': crate_name,
                        'num_files': len(files),
                        'files': file_list,
                        'token_count': total_tokens
                    }
                }

                self.samples.append(sample)
                count += 1

        print(f"   Generated {count} crate-level samples")

    def save_dataset(self, output_file: Path):
        """Save dataset to JSONL with statistics."""
        print(f"\n💾 Saving dataset to: {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Calculate statistics
        token_counts = [s['metadata']['token_count'] for s in self.samples]
        granularity_counts = Counter(s['granularity'] for s in self.samples)
        crate_counts = Counter(s['metadata']['crate'] for s in self.samples)

        # Token distribution
        small = len([t for t in token_counts if t < 4000])
        medium = len([t for t in token_counts if 4000 <= t < 10000])
        large = len([t for t in token_counts if t >= 10000])

        print("\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(self.samples)}")
        print(f"\n   Token distribution:")
        print(f"     < 4k tokens: {small} ({small/len(self.samples)*100:.1f}%)")
        print(f"     4k-10k tokens: {medium} ({medium/len(self.samples)*100:.1f}%)")
        print(f"     10k+ tokens: {large} ({large/len(self.samples)*100:.1f}%)")
        print(f"\n   Token stats:")
        print(f"     Min: {min(token_counts):,} tokens")
        print(f"     Max: {max(token_counts):,} tokens")
        print(f"     Mean: {sum(token_counts)//len(token_counts):,} tokens")
        print(f"     Total: {sum(token_counts):,} tokens")

        print(f"\n   By granularity:")
        for gran, count in granularity_counts.most_common():
            print(f"     {gran}: {count}")

        print(f"\n   Top 15 crates:")
        for crate, count in crate_counts.most_common(15):
            print(f"     {crate}: {count} samples")


def main():
    repo_path = Path('/Users/architsinghai/code/deepwiki-scripts/hyperswitch')
    output_file = Path('/Users/architsinghai/code/repo_cpt_dataset_clean/token_aware_dataset.jsonl')

    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return

    print("=" * 80)
    print("TOKEN-AWARE MULTI-GRANULARITY CPT DATASET GENERATOR")
    print("Target: 2000-16000 tokens per sample")
    print("=" * 80)

    # Load tokenizer
    tokenizer = load_tokenizer()
    if not tokenizer:
        print("❌ Cannot proceed without tokenizer")
        return

    # Generate dataset
    generator = TokenAwareDatasetGenerator(repo_path, tokenizer)
    generator.analyze_files()
    generator.generate_samples()
    generator.save_dataset(output_file)

    print(f"\n✅ Done! Dataset saved to: {output_file}")


if __name__ == '__main__':
    main()
