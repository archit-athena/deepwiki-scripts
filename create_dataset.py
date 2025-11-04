#!/usr/bin/env python3
"""
Process scraped documentation files and create a chunked dataset for Hugging Face.
Extracts markdown content, chunks by sections, and creates a structured dataset.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def extract_markdown_content(file_path: Path) -> str:
    """Extract markdown content from file, skipping React SSR boilerplate."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the first markdown heading (# or ##)
    match = re.search(r'^#\s+.+$', content, re.MULTILINE)
    if match:
        # Extract everything from the first heading onwards
        markdown_content = content[match.start():]
        return markdown_content.strip()

    return ""


def chunk_by_sections(markdown_text: str, source_file: str) -> List[Dict[str, Any]]:
    """
    Chunk markdown content by sections (## headings).
    Returns a list of chunks with metadata.
    """
    chunks = []

    # Split by level 1 and level 2 headings
    # Pattern: Match lines starting with # or ##
    lines = markdown_text.split('\n')

    current_chunk = []
    current_heading = None
    current_level = None
    chunk_id = 0

    for line in lines:
        # Check if line is a heading
        heading_match = re.match(r'^(#{1,3})\s+(.+)$', line)

        if heading_match:
            # Save previous chunk if it exists
            if current_chunk and current_heading:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:  # Only add non-empty chunks
                    chunks.append({
                        'id': f"{source_file}_{chunk_id}",
                        'source_file': source_file,
                        'heading': current_heading,
                        'heading_level': current_level,
                        'content': chunk_text,
                        'chunk_index': chunk_id
                    })
                    chunk_id += 1

            # Start new chunk
            current_level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_chunk = [line]
        else:
            current_chunk.append(line)

    # Add the last chunk
    if current_chunk and current_heading:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append({
                'id': f"{source_file}_{chunk_id}",
                'source_file': source_file,
                'heading': current_heading,
                'heading_level': current_level,
                'content': chunk_text,
                'chunk_index': chunk_id
            })

    return chunks


def process_directory(input_dir: Path) -> List[Dict[str, Any]]:
    """Process all .md files in the directory and create dataset."""
    all_chunks = []

    # Get all .md files (not .raw.txt)
    md_files = sorted(input_dir.glob('*.md'))

    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        print(f"Processing {md_file.name}...")

        # Extract markdown content
        markdown_content = extract_markdown_content(md_file)

        if not markdown_content:
            print(f"  ⚠️  No markdown content found in {md_file.name}")
            continue

        # Chunk the content
        chunks = chunk_by_sections(markdown_content, md_file.stem)
        all_chunks.extend(chunks)

        print(f"  ✓ Extracted {len(chunks)} chunks")

    return all_chunks


def save_dataset(chunks: List[Dict[str, Any]], output_dir: Path):
    """Save dataset in multiple formats."""
    output_dir.mkdir(exist_ok=True)

    # Save as JSON Lines
    jsonl_path = output_dir / 'dataset.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"\n✓ Saved {len(chunks)} chunks to {jsonl_path}")

    # Save as Parquet
    df = pd.DataFrame(chunks)
    parquet_path = output_dir / 'dataset.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved {len(chunks)} chunks to {parquet_path}")

    # Save summary statistics
    stats = {
        'total_chunks': len(chunks),
        'unique_source_files': len(set(c['source_file'] for c in chunks)),
        'chunks_by_heading_level': {
            level: len([c for c in chunks if c['heading_level'] == level])
            for level in sorted(set(c['heading_level'] for c in chunks))
        },
        'average_chunk_length': sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
        'source_files': sorted(set(c['source_file'] for c in chunks))
    }

    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dumps(stats, indent=2, ensure_ascii=False, default=str)
        f.write(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"✓ Saved statistics to {stats_path}")

    return stats


def create_readme(stats: Dict[str, Any], output_dir: Path):
    """Create a README.md for the dataset."""
    readme_content = f"""# DeepWiki Dataset

This dataset contains documentation extracted from the Hyperswitch payment router project,
chunked by sections for easy consumption.

## Dataset Statistics

- **Total Chunks**: {stats['total_chunks']}
- **Source Files**: {stats['unique_source_files']}
- **Average Chunk Length**: {stats['average_chunk_length']:.0f} characters

### Chunks by Heading Level

{chr(10).join(f"- Level {level}: {count} chunks" for level, count in stats['chunks_by_heading_level'].items())}

## Dataset Structure

Each row contains:
- `id`: Unique identifier for the chunk
- `source_file`: Source filename (without extension)
- `heading`: Section heading
- `heading_level`: Markdown heading level (1-3)
- `content`: The actual markdown content including the heading
- `chunk_index`: Index of chunk within the source file

## Files

- `dataset.jsonl`: Dataset in JSON Lines format
- `dataset.parquet`: Dataset in Parquet format
- `dataset_stats.json`: Detailed statistics

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("archit11/deepwiki", split="train")

# Access a sample
print(dataset[0])
```

## Source

Documentation extracted from: juspay/hyperswitch project documentation
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created {readme_path}")


def main():
    # Paths
    input_dir = Path('/home/dumball/out')
    output_dir = Path('/home/dumball/dataset_output')

    print("=" * 60)
    print("DeepWiki Dataset Creator")
    print("=" * 60)

    # Process files
    chunks = process_directory(input_dir)

    if not chunks:
        print("\n❌ No chunks extracted!")
        return

    # Save dataset
    stats = save_dataset(chunks, output_dir)

    # Create README
    create_readme(stats, output_dir)

    print("\n" + "=" * 60)
    print(f"✅ Dataset creation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
