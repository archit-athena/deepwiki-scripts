#!/usr/bin/env python3
"""
Enhanced dataset creator with semantic chunking.
Creates larger, more meaningful chunks based on logical content boundaries.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def clone_or_update_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone or update the hyperswitch repository."""
    if target_dir.exists():
        print(f"Repository already exists at {target_dir}")
        try:
            subprocess.run(['git', '-C', str(target_dir), 'status'],
                         check=True, capture_output=True)
            print("✓ Repository is ready for use")
            return True
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Directory exists but is not a valid git repository")
            return False
    else:
        print(f"Cloning repository to {target_dir}...")
        try:
            subprocess.run(['git', 'clone', repo_url, str(target_dir)],
                         check=True, capture_output=True)
            print("✓ Repository cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Failed to clone: {e}")
            return False


def extract_source_references(content: str) -> List[Dict[str, Any]]:
    """Extract source references from content."""
    references = []
    pattern = r'\[([\w\-\/\._]+):(\d+)(?:-(\d+))?\]\(\)'

    for match in re.finditer(pattern, content):
        file_path = match.group(1)
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line

        references.append({
            'file_path': file_path,
            'start_line': start_line,
            'end_line': end_line
        })

    return references


def extract_code_from_file(repo_dir: Path, file_path: str,
                          start_line: int, end_line: int) -> Optional[str]:
    """Extract specific lines from a file in the repository."""
    full_path = repo_dir / file_path

    if not full_path.exists():
        return None

    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        extracted_lines = lines[start_idx:end_idx]
        return ''.join(extracted_lines)
    except Exception as e:
        return None


def extract_markdown_content(file_path: Path) -> str:
    """Extract markdown content from file, skipping React SSR boilerplate."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r'^#\s+.+$', content, re.MULTILINE)
    if match:
        markdown_content = content[match.start():]
        return markdown_content.strip()

    return ""


def chunk_semantically(markdown_text: str, source_file: str,
                      repo_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Create semantic chunks based on logical content boundaries.
    Groups content between '---' separators or large heading blocks.
    """
    chunks = []
    chunk_id = 0

    # Split by horizontal rules (---) which often separate logical sections
    sections = re.split(r'\n---\n', markdown_text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 100:  # Skip tiny sections
            continue

        # Extract the main heading for this section
        heading_match = re.search(r'^(#{1,3})\s+(.+)$', section, re.MULTILINE)
        if heading_match:
            heading = heading_match.group(2).strip()
            heading_level = len(heading_match.group(1))
        else:
            heading = f"Section {chunk_id}"
            heading_level = 2

        # Count subsections to understand content richness
        subsection_count = len(re.findall(r'^#{2,6}\s+', section, re.MULTILINE))

        # Extract all code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', section)

        # Calculate semantic metrics
        has_mermaid = 'mermaid' in section
        has_code = len(code_blocks) > 0
        has_tables = '|' in section and '---' in section
        word_count = len(section.split())

        chunk_data = {
            'id': f"{source_file}_{chunk_id}",
            'source_file': source_file,
            'heading': heading,
            'heading_level': heading_level,
            'content': section,
            'chunk_index': chunk_id,
            'source_references': [],
            'source_code_snippets': [],
            'metadata': {
                'word_count': word_count,
                'subsection_count': subsection_count,
                'code_block_count': len(code_blocks),
                'has_mermaid_diagram': has_mermaid,
                'has_code_examples': has_code,
                'has_tables': has_tables
            }
        }

        # Extract source references
        refs = extract_source_references(section)
        chunk_data['source_references'] = refs

        # Mine the actual code if repo is available
        if repo_dir and refs:
            code_snippets = []
            for ref in refs:
                code = extract_code_from_file(
                    repo_dir,
                    ref['file_path'],
                    ref['start_line'],
                    ref['end_line']
                )
                if code:
                    code_snippets.append({
                        'file_path': ref['file_path'],
                        'start_line': ref['start_line'],
                        'end_line': ref['end_line'],
                        'code': code
                    })
            chunk_data['source_code_snippets'] = code_snippets

        chunks.append(chunk_data)
        chunk_id += 1

    return chunks


def process_directory(input_dir: Path, repo_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Process all .md files in the directory and create dataset."""
    all_chunks = []
    md_files = sorted(input_dir.glob('*.md'))

    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        print(f"Processing {md_file.name}...")

        markdown_content = extract_markdown_content(md_file)

        if not markdown_content:
            print(f"  ⚠️  No markdown content found in {md_file.name}")
            continue

        chunks = chunk_semantically(markdown_content, md_file.stem, repo_dir)

        # Calculate statistics
        chunks_with_code = sum(1 for c in chunks if c['source_code_snippets'])
        avg_words = sum(c['metadata']['word_count'] for c in chunks) / len(chunks) if chunks else 0

        all_chunks.extend(chunks)

        print(f"  ✓ Extracted {len(chunks)} chunks ({chunks_with_code} with source code)")
        print(f"    Average words per chunk: {avg_words:.0f}")

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

    # Calculate statistics
    chunks_with_code = [c for c in chunks if c['source_code_snippets']]
    total_code_snippets = sum(len(c['source_code_snippets']) for c in chunks)
    avg_word_count = sum(c['metadata']['word_count'] for c in chunks) / len(chunks) if chunks else 0
    chunks_with_diagrams = sum(1 for c in chunks if c['metadata']['has_mermaid_diagram'])
    chunks_with_tables = sum(1 for c in chunks if c['metadata']['has_tables'])

    stats = {
        'total_chunks': len(chunks),
        'chunks_with_source_code': len(chunks_with_code),
        'total_code_snippets': total_code_snippets,
        'unique_source_files': len(set(c['source_file'] for c in chunks)),
        'average_word_count': avg_word_count,
        'chunks_with_mermaid_diagrams': chunks_with_diagrams,
        'chunks_with_tables': chunks_with_tables,
        'chunks_by_heading_level': {
            level: len([c for c in chunks if c['heading_level'] == level])
            for level in sorted(set(c['heading_level'] for c in chunks))
        },
        'source_files': sorted(set(c['source_file'] for c in chunks))
    }

    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"✓ Saved statistics to {stats_path}")

    return stats


def create_readme(stats: Dict[str, Any], output_dir: Path):
    """Create a README.md for the dataset."""
    readme_content = f"""# DeepWiki Semantic Dataset

This dataset contains semantically chunked documentation from the Hyperswitch payment router project,
enhanced with source code references mined directly from the repository.

## Key Features

- **Semantic Chunking**: Content is split by logical boundaries for better context
- **Source Code Integration**: Actual code snippets from the repository included
- **Rich Metadata**: Word counts, subsection counts, diagram/table indicators
- **Larger Chunks**: More complete semantic units vs line-by-line splitting

## Dataset Statistics

- **Total Chunks**: {stats['total_chunks']:,}
- **Chunks with Source Code**: {stats['chunks_with_source_code']:,}
- **Total Code Snippets**: {stats['total_code_snippets']:,}
- **Average Words per Chunk**: {stats['average_word_count']:.0f}
- **Chunks with Mermaid Diagrams**: {stats['chunks_with_mermaid_diagrams']:,}
- **Chunks with Tables**: {stats['chunks_with_tables']:,}
- **Source Files**: {stats['unique_source_files']}

### Chunks by Heading Level

{chr(10).join(f"- Level {level}: {count:,} chunks" for level, count in stats['chunks_by_heading_level'].items())}

## Dataset Structure

Each row contains:
- `id`: Unique identifier for the chunk
- `source_file`: Source filename (without extension)
- `heading`: Main heading for this semantic section
- `heading_level`: Markdown heading level (1-3)
- `content`: Complete semantic section with subsections
- `chunk_index`: Index of chunk within the source file
- `source_references`: List of source file references mentioned in the chunk
- `source_code_snippets`: Actual code extracted from the referenced files
- `metadata`: Rich metadata about the chunk
  - `word_count`: Number of words in the chunk
  - `subsection_count`: Number of subsections
  - `code_block_count`: Number of code examples
  - `has_mermaid_diagram`: Boolean indicator for diagrams
  - `has_code_examples`: Boolean indicator for code
  - `has_tables`: Boolean indicator for tables

### Source Code Snippets Structure

Each snippet contains:
- `file_path`: Path to the source file in the repository
- `start_line`: Starting line number
- `end_line`: Ending line number
- `code`: The actual code content

## Files

- `dataset.jsonl`: Dataset in JSON Lines format
- `dataset.parquet`: Dataset in Parquet format
- `dataset_stats.json`: Detailed statistics

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("archit11/deepwiki3", split="train")

# Access a sample with rich metadata
sample = dataset[0]
print(f"Heading: {{sample['heading']}}")
print(f"Words: {{sample['metadata']['word_count']}}")
print(f"Subsections: {{sample['metadata']['subsection_count']}}")

# Check if it has source code
if sample['source_code_snippets']:
    for snippet in sample['source_code_snippets']:
        print(f"\\nCode from {{snippet['file_path']}} (lines {{snippet['start_line']}}-{{snippet['end_line']}}):")
        print(snippet['code'])
```

## Chunking Strategy

This dataset uses semantic chunking based on logical content boundaries (typically marked by `---` separators
in the original documentation), which creates larger, more coherent chunks compared to simple heading-based
splitting. This preserves context and makes the chunks more suitable for:

- RAG (Retrieval Augmented Generation) applications
- Documentation question-answering systems
- Code understanding and learning tasks
- Multi-modal doc + code training

## Source

- **Documentation**: Extracted from juspay/hyperswitch wiki
- **Source Code**: Mined from https://github.com/juspay/hyperswitch (commit 820f1831)
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created {readme_path}")


def main():
    # Paths
    input_dir = Path('/home/dumball/out')
    output_dir = Path('/home/dumball/semantic_dataset_output')
    repo_dir = Path('/home/dumball/hyperswitch')
    repo_url = 'https://github.com/juspay/hyperswitch.git'

    print("=" * 60)
    print("DeepWiki Semantic Dataset Creator")
    print("(Larger chunks with rich semantic context)")
    print("=" * 60)

    # Clone or update repository
    print("\n[1/3] Setting up hyperswitch repository...")
    if clone_or_update_repo(repo_url, repo_dir):
        print("✓ Repository ready")
    else:
        print("⚠️  Proceeding without source code mining")
        repo_dir = None

    # Process files
    print("\n[2/3] Processing documentation files...")
    chunks = process_directory(input_dir, repo_dir)

    if not chunks:
        print("\n❌ No chunks extracted!")
        return

    # Save dataset
    print("\n[3/3] Saving dataset...")
    stats = save_dataset(chunks, output_dir)

    # Create README
    create_readme(stats, output_dir)

    print("\n" + "=" * 60)
    print(f"✅ Semantic dataset creation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Chunks with source code: {stats['chunks_with_source_code']:,}")
    print(f"Average words per chunk: {stats['average_word_count']:.0f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
