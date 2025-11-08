#!/usr/bin/env python3
"""
FILE-LEVEL CPT DATASET GENERATOR
Simple: <path> metadata + <file> full content
"""

import json
from pathlib import Path
from collections import defaultdict


def generate_file_level_dataset(repo_path: Path, output_file: Path, min_size=100, max_size=500000):
    """
    Generate a file-level CPT dataset
    Each sample: <path> + <file>
    """

    print(f"📁 Scanning repository: {repo_path}")

    # Find all .rs files
    rs_files = list(repo_path.rglob('*.rs'))
    print(f"   Found {len(rs_files)} Rust files")

    samples = []
    stats = defaultdict(int)

    for rs_file in rs_files:
        try:
            # Get file size
            file_size = rs_file.stat().st_size

            # Filter by size
            if file_size < min_size or file_size > max_size:
                stats['skipped_size'] += 1
                continue

            # Read file content
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Get relative path
            try:
                rel_path = rs_file.relative_to(repo_path)
            except:
                rel_path = rs_file.name

            # Determine crate (if in crates/ directory structure)
            path_parts = str(rel_path).split('/')
            crate_name = 'unknown'
            if 'crates' in path_parts:
                crate_idx = path_parts.index('crates')
                if len(path_parts) > crate_idx + 1:
                    crate_name = path_parts[crate_idx + 1]
            elif len(path_parts) > 1:
                crate_name = path_parts[0]

            # Create sample with <path> and <file> format
            sample = {
                'id': f'file_{hash(str(rel_path))}',
                'type': 'clm',
                'granularity': 'file',
                'path': str(rel_path),
                'crate': crate_name,
                'file_size': file_size,
                'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
File: {rel_path}
Size: {file_size} bytes

<file>
{content}
'''
            }

            samples.append(sample)
            stats['included'] += 1
            stats[f'crate_{crate_name}'] += 1

        except Exception as e:
            stats['errors'] += 1
            continue

    # Save dataset
    print(f"\n💾 Saving dataset to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ Dataset created!")
    print(f"   Total samples: {len(samples)}")
    print(f"   Skipped (size): {stats['skipped_size']}")
    print(f"   Errors: {stats['errors']}")

    # Show top crates
    print(f"\n📊 Top crates by file count:")
    crate_stats = [(k.replace('crate_', ''), v) for k, v in stats.items() if k.startswith('crate_')]
    crate_stats.sort(key=lambda x: x[1], reverse=True)
    for crate, count in crate_stats[:15]:
        print(f"   {crate}: {count} files")

    return samples


def main():
    repo_path = Path('/Users/architsinghai/code/deepwiki-scripts/hyperswitch')
    output_file = Path('/Users/architsinghai/code/repo_cpt_dataset_clean/file_level_dataset.jsonl')

    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return

    print("=" * 80)
    print("FILE-LEVEL CPT DATASET GENERATOR")
    print("Format: <path> metadata + <file> full content")
    print("=" * 80)

    samples = generate_file_level_dataset(
        repo_path=repo_path,
        output_file=output_file,
        min_size=100,      # Skip tiny files
        max_size=500000    # Skip huge files (500KB limit)
    )

    print(f"\n✅ Done! Dataset saved to: {output_file}")


if __name__ == '__main__':
    main()
