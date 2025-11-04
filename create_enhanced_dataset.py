#!/usr/bin/env python3
"""
Enhanced dataset creator that extracts source code references from documentation
and mines the actual code from the hyperswitch repository.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


def clone_or_update_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone or update the hyperswitch repository."""
    if target_dir.exists():
        print(f"Repository already exists at {target_dir}")
        # Check if it's a valid git repo
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
    """
    Extract source references from content.
    Format: Sources:** [filename:start-end]() or [filename:line]()
    Example: Sources:** [CHANGELOG.md:1-50](), [crates/router/Cargo.toml:1-40]()
    """
    references = []

    # Pattern to match [filename:line_range]()
    # Matches: [path/to/file.ext:1-50]() or [file.ext:123]()
    # Allow for word characters, hyphens, slashes, dots, and underscores in filenames
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

        # Adjust for 0-based indexing
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        extracted_lines = lines[start_idx:end_idx]
        return ''.join(extracted_lines)
    except Exception as e:
        print(f"  ⚠️  Error reading {file_path}: {e}")
        return None


def extract_markdown_content(file_path: Path) -> str:
    """Extract markdown content from file, skipping React SSR boilerplate."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the first markdown heading (# or ##)
    match = re.search(r'^#\s+.+$', content, re.MULTILINE)
    if match:
        markdown_content = content[match.start():]
        return markdown_content.strip()

    return ""


def chunk_by_sections(markdown_text: str, source_file: str,
                     repo_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Chunk markdown content by sections and extract source code references.
    Only chunks by level 2 headings (##) to keep larger context.
    """
    chunks = []
    lines = markdown_text.split('\n')

    current_chunk = []
    current_heading = None
    current_level = None
    chunk_id = 0

    for line in lines:
        # Only chunk on level 2 headings (##) to avoid over-chunking
        heading_match = re.match(r'^(##)\s+(.+)$', line)

        if heading_match:
            # Save previous chunk
            if current_chunk and current_heading:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunk_data = {
                        'id': f"{source_file}_{chunk_id}",
                        'source_file': source_file,
                        'heading': current_heading,
                        'heading_level': current_level,
                        'content': chunk_text,
                        'chunk_index': chunk_id,
                        'source_references': [],
                        'source_code_snippets': []
                    }

                    # Always extract source references
                    refs = extract_source_references(chunk_text)
                    chunk_data['source_references'] = refs

                    # Mine the actual code only if repo is available
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
            chunk_data = {
                'id': f"{source_file}_{chunk_id}",
                'source_file': source_file,
                'heading': current_heading,
                'heading_level': current_level,
                'content': chunk_text,
                'chunk_index': chunk_id,
                'source_references': [],
                'source_code_snippets': []
            }

            # Always extract source references
            refs = extract_source_references(chunk_text)
            chunk_data['source_references'] = refs

            # Mine the actual code only if repo is available
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

        chunks = chunk_by_sections(markdown_content, md_file.stem, repo_dir)

        # Count chunks with source code
        chunks_with_code = sum(1 for c in chunks if c['source_code_snippets'])
        all_chunks.extend(chunks)

        print(f"  ✓ Extracted {len(chunks)} chunks ({chunks_with_code} with source code)")

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

    stats = {
        'total_chunks': len(chunks),
        'chunks_with_source_code': len(chunks_with_code),
        'total_code_snippets': total_code_snippets,
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
        f.write(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"✓ Saved statistics to {stats_path}")

    return stats


def create_readme(stats: Dict[str, Any], output_dir: Path):
    """Create a README.md for the dataset."""
    readme_content = f"""# DeepWiki Enhanced Dataset

This dataset contains documentation from the Hyperswitch payment router project,
enhanced with source code references mined directly from the repository.

## Dataset Statistics

- **Total Chunks**: {stats['total_chunks']:,}
- **Chunks with Source Code**: {stats['chunks_with_source_code']:,}
- **Total Code Snippets**: {stats['total_code_snippets']:,}
- **Source Files**: {stats['unique_source_files']}
- **Average Chunk Length**: {stats['average_chunk_length']:.0f} characters

### Chunks by Heading Level

{chr(10).join(f"- Level {level}: {count:,} chunks" for level, count in stats['chunks_by_heading_level'].items())}

## Dataset Structure

Each row contains:
- `id`: Unique identifier for the chunk
- `source_file`: Source filename (without extension)
- `heading`: Section heading
- `heading_level`: Markdown heading level (1-3)
- `content`: The actual markdown content including the heading
- `chunk_index`: Index of chunk within the source file
- `source_references`: List of source file references mentioned in the chunk
- `source_code_snippets`: Actual code extracted from the referenced files

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
dataset = load_dataset("archit11/deepwiki2", split="train")

# Access a sample with source code
sample = dataset[0]
print(f"Heading: {{sample['heading']}}")
print(f"Content: {{sample['content'][:200]}}...")

# Check if it has source code
if sample['source_code_snippets']:
    for snippet in sample['source_code_snippets']:
        print(f"\\nCode from {{snippet['file_path']}}:")
        print(snippet['code'])
```

## Source

- **Documentation**: Extracted from juspay/hyperswitch wiki
- **Source Code**: Mined from https://github.com/juspay/hyperswitch
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created {readme_path}")


def main():
    # Paths
    input_dir = Path('/home/dumball/out')
    output_dir = Path('/home/dumball/enhanced_dataset_output')
    repo_dir = Path('/home/dumball/hyperswitch')
    repo_url = 'https://github.com/juspay/hyperswitch.git'

    print("=" * 60)
    print("DeepWiki Enhanced Dataset Creator")
    print("(with Source Code Mining)")
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
    print(f"✅ Enhanced dataset creation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Chunks with source code: {stats['chunks_with_source_code']:,} / {stats['total_chunks']:,}")
    print("=" * 60)


if __name__ == '__main__':
    main()
