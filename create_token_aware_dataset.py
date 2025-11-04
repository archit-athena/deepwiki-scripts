#!/usr/bin/env python3
"""
Token-aware dataset creator with analysis for Kwaipilot/KAT-Dev model.
Analyzes token distribution and optimizes chunks for the target model.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from transformers import AutoTokenizer
import numpy as np


def load_tokenizer(model_name: str = "Kwaipilot/KAT-Dev"):
    """Load the tokenizer for the target model."""
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")
        if hasattr(tokenizer, 'model_max_length'):
            print(f"  Max length: {tokenizer.model_max_length:,}")
        return tokenizer
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return None


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


def analyze_token_distribution(text: str, tokenizer, include_special: bool = True) -> Dict[str, Any]:
    """Analyze token distribution for a given text."""
    if not tokenizer:
        return {}

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=include_special)

    # Calculate statistics
    token_count = len(tokens)

    # Decode back to check compression ratio
    decoded = tokenizer.decode(tokens, skip_special_tokens=not include_special)
    char_count = len(text)
    compression_ratio = char_count / token_count if token_count > 0 else 0

    return {
        'token_count': token_count,
        'char_count': char_count,
        'compression_ratio': compression_ratio,
        'tokens_per_word': token_count / len(text.split()) if text.split() else 0
    }


def get_overlap_prefix(previous_sections: List[str], overlap_tokens: int, tokenizer) -> str:
    """
    Get overlap text from previous sections that fits within overlap_tokens limit.
    Returns empty string if no previous sections or tokenizer unavailable.
    """
    if not previous_sections or not tokenizer:
        return ""

    # Start with the most recent section and work backwards
    overlap_text = ""
    for section in reversed(previous_sections):
        test_text = section + ("\n---\n" + overlap_text if overlap_text else "")
        token_analysis = analyze_token_distribution(test_text, tokenizer)

        if token_analysis['token_count'] <= overlap_tokens:
            overlap_text = test_text
        else:
            break

    return overlap_text + "\n---\n" if overlap_text else ""


def chunk_with_token_awareness(markdown_text: str, source_file: str,
                               tokenizer, repo_dir: Optional[Path] = None,
                               max_tokens: int = 8192,
                               overlap_tokens: int = 200,
                               target_distribution: Dict[str, tuple] = None) -> List[Dict[str, Any]]:
    """
    Create chunks with well-distributed token counts and minimal overlap.
    Distribution targets: small (<2k), medium (2k-5k), large (5k-8k)
    Aims for varied chunk sizes while respecting semantic boundaries.
    Adds ~200 token overlap between adjacent chunks for context preservation.
    """
    if target_distribution is None:
        target_distribution = {
            'small': (512, 2000),    # 25% target
            'medium': (2000, 5000),  # 50% target
            'large': (5000, 8000)    # 25% target
        }

    chunks = []
    chunk_id = 0

    # Track distribution
    size_counts = {'small': 0, 'medium': 0, 'large': 0}

    # Store last section for overlap
    previous_sections = []

    # Split by horizontal rules (---) which separate logical sections
    sections = re.split(r'\n---\n', markdown_text)

    current_buffer = []
    current_tokens = 0

    # Determine target size based on current distribution
    def get_target_size():
        total = sum(size_counts.values())
        if total == 0:
            return target_distribution['medium'][1]  # Start with medium

        # Calculate current percentages
        small_pct = size_counts['small'] / total
        medium_pct = size_counts['medium'] / total
        large_pct = size_counts['large'] / total

        # Prefer sizes that are underrepresented
        if small_pct < 0.25:
            return target_distribution['small'][1]
        elif medium_pct < 0.50:
            return target_distribution['medium'][1]
        elif large_pct < 0.25:
            return target_distribution['large'][1]
        else:
            return target_distribution['medium'][1]

    for section in sections:
        section = section.strip()
        if not section or len(section) < 100:
            continue

        # Analyze tokens for this section
        section_token_analysis = analyze_token_distribution(section, tokenizer)
        section_tokens = section_token_analysis['token_count']

        # If this section alone exceeds max_tokens, split it further
        if section_tokens > max_tokens:
            # Flush current buffer first
            if current_buffer:
                # Add overlap from previous if available
                overlap_prefix = get_overlap_prefix(previous_sections, overlap_tokens, tokenizer)
                chunk_content = overlap_prefix + '\n---\n'.join(current_buffer)
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir)
                chunk_data['metadata']['has_overlap'] = len(overlap_prefix) > 0
                chunks.append(chunk_data)

                # Update distribution tracking
                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 2000:
                    size_counts['small'] += 1
                elif tokens < 5000:
                    size_counts['medium'] += 1
                else:
                    size_counts['large'] += 1

                # Store sections for next overlap
                previous_sections = current_buffer[-2:] if len(current_buffer) >= 2 else current_buffer

                chunk_id += 1
                current_buffer = []
                current_tokens = 0

            # Split large section by paragraphs
            paragraphs = section.split('\n\n')
            para_buffer = []
            para_tokens = 0

            for para in paragraphs:
                para_analysis = analyze_token_distribution(para, tokenizer)
                para_token_count = para_analysis['token_count']

                if para_tokens + para_token_count > max_tokens and para_buffer:
                    # Flush paragraph buffer
                    chunk_content = '\n\n'.join(para_buffer)
                    chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir)
                    chunks.append(chunk_data)

                    tokens = chunk_data['token_stats']['total_tokens']
                    if tokens < 2000:
                        size_counts['small'] += 1
                    elif tokens < 5000:
                        size_counts['medium'] += 1
                    else:
                        size_counts['large'] += 1

                    chunk_id += 1
                    para_buffer = [para]
                    para_tokens = para_token_count
                else:
                    para_buffer.append(para)
                    para_tokens += para_token_count

            # Flush remaining paragraphs
            if para_buffer:
                chunk_content = '\n\n'.join(para_buffer)
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir)
                chunks.append(chunk_data)

                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 2000:
                    size_counts['small'] += 1
                elif tokens < 5000:
                    size_counts['medium'] += 1
                else:
                    size_counts['large'] += 1

                chunk_id += 1

        # Section fits in token limit
        elif current_tokens + section_tokens > max_tokens:
            # Flush current buffer and start new chunk
            if current_buffer:
                overlap_prefix = get_overlap_prefix(previous_sections, overlap_tokens, tokenizer)
                chunk_content = overlap_prefix + '\n---\n'.join(current_buffer)
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir)
                chunk_data['metadata']['has_overlap'] = len(overlap_prefix) > 0
                chunks.append(chunk_data)

                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 2000:
                    size_counts['small'] += 1
                elif tokens < 5000:
                    size_counts['medium'] += 1
                else:
                    size_counts['large'] += 1

                previous_sections = current_buffer[-2:] if len(current_buffer) >= 2 else current_buffer
                chunk_id += 1

            current_buffer = [section]
            current_tokens = section_tokens

        else:
            # Add to current buffer
            current_buffer.append(section)
            current_tokens += section_tokens

            # Get dynamic target based on distribution
            target_size = get_target_size()

            # If we've reached target size, flush
            if current_tokens >= target_size:
                overlap_prefix = get_overlap_prefix(previous_sections, overlap_tokens, tokenizer)
                chunk_content = overlap_prefix + '\n---\n'.join(current_buffer)
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir)
                chunk_data['metadata']['has_overlap'] = len(overlap_prefix) > 0
                chunks.append(chunk_data)

                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 2000:
                    size_counts['small'] += 1
                elif tokens < 5000:
                    size_counts['medium'] += 1
                else:
                    size_counts['large'] += 1

                previous_sections = current_buffer[-2:] if len(current_buffer) >= 2 else current_buffer
                chunk_id += 1
                current_buffer = []
                current_tokens = 0

    # Flush remaining buffer (no overlap for last chunk)
    if current_buffer:
        chunk_content = '\n---\n'.join(current_buffer)
        chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir)
        chunk_data['metadata']['has_overlap'] = False  # Last chunk has no overlap
        chunks.append(chunk_data)

        tokens = chunk_data['token_stats']['total_tokens']
        if tokens < 2000:
            size_counts['small'] += 1
        elif tokens < 5000:
            size_counts['medium'] += 1
        else:
            size_counts['large'] += 1

    return chunks


def create_chunk(content: str, source_file: str, chunk_id: int,
                tokenizer, repo_dir: Optional[Path]) -> Dict[str, Any]:
    """Create a chunk with full metadata and token analysis."""

    # Extract heading
    heading_match = re.search(r'^(#{1,3})\s+(.+)$', content, re.MULTILINE)
    if heading_match:
        heading = heading_match.group(2).strip()
        heading_level = len(heading_match.group(1))
    else:
        heading = f"Section {chunk_id}"
        heading_level = 2

    # Count subsections and other features
    subsection_count = len(re.findall(r'^#{2,6}\s+', content, re.MULTILINE))
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    has_mermaid = 'mermaid' in content
    has_code = len(code_blocks) > 0
    has_tables = '|' in content and '---' in content
    word_count = len(content.split())

    # Token analysis
    token_analysis = analyze_token_distribution(content, tokenizer)

    # Analyze content vs code token distribution
    content_only = re.sub(r'```[\s\S]*?```', '', content)
    code_only = '\n'.join(code_blocks)

    content_tokens = analyze_token_distribution(content_only, tokenizer) if tokenizer else {}
    code_tokens = analyze_token_distribution(code_only, tokenizer) if tokenizer and code_only else {}

    chunk_data = {
        'id': f"{source_file}_{chunk_id}",
        'source_file': source_file,
        'heading': heading,
        'heading_level': heading_level,
        'content': content,
        'chunk_index': chunk_id,
        'source_references': [],
        'source_code_snippets': [],
        'metadata': {
            'word_count': word_count,
            'subsection_count': subsection_count,
            'code_block_count': len(code_blocks),
            'has_mermaid_diagram': has_mermaid,
            'has_code_examples': has_code,
            'has_tables': has_tables,
            'has_overlap': False  # Will be set to True if chunk has overlap
        },
        'token_stats': {
            'total_tokens': token_analysis.get('token_count', 0),
            'total_chars': token_analysis.get('char_count', 0),
            'compression_ratio': token_analysis.get('compression_ratio', 0),
            'tokens_per_word': token_analysis.get('tokens_per_word', 0),
            'content_tokens': content_tokens.get('token_count', 0),
            'code_tokens': code_tokens.get('token_count', 0),
            'code_token_percentage': (code_tokens.get('token_count', 0) / token_analysis.get('token_count', 1)) * 100 if token_analysis.get('token_count', 0) > 0 else 0
        }
    }

    # Extract source references
    refs = extract_source_references(content)
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
                # Analyze tokens for this code snippet
                code_snippet_tokens = analyze_token_distribution(code, tokenizer)

                code_snippets.append({
                    'file_path': ref['file_path'],
                    'start_line': ref['start_line'],
                    'end_line': ref['end_line'],
                    'code': code,
                    'token_count': code_snippet_tokens.get('token_count', 0)
                })
        chunk_data['source_code_snippets'] = code_snippets

    return chunk_data


def process_directory(input_dir: Path, tokenizer, repo_dir: Optional[Path] = None,
                     max_tokens: int = 8192) -> List[Dict[str, Any]]:
    """Process all .md files in the directory and create dataset."""
    all_chunks = []
    md_files = sorted(input_dir.glob('*.md'))

    print(f"Found {len(md_files)} markdown files")
    print(f"Max tokens: {max_tokens}, Distribution: Small (<2k): 25%, Medium (2k-5k): 50%, Large (5k-8k): 25%\n")

    for md_file in md_files:
        print(f"Processing {md_file.name}...")

        markdown_content = extract_markdown_content(md_file)

        if not markdown_content:
            print(f"  ⚠️  No markdown content found in {md_file.name}")
            continue

        chunks = chunk_with_token_awareness(
            markdown_content,
            md_file.stem,
            tokenizer,
            repo_dir,
            max_tokens
        )

        # Calculate statistics
        chunks_with_code = sum(1 for c in chunks if c['source_code_snippets'])
        avg_tokens = sum(c['token_stats']['total_tokens'] for c in chunks) / len(chunks) if chunks else 0
        avg_words = sum(c['metadata']['word_count'] for c in chunks) / len(chunks) if chunks else 0

        all_chunks.extend(chunks)

        print(f"  ✓ Extracted {len(chunks)} chunks ({chunks_with_code} with source code)")
        print(f"    Average tokens: {avg_tokens:.0f}, Average words: {avg_words:.0f}")

    return all_chunks


def save_dataset(chunks: List[Dict[str, Any]], output_dir: Path):
    """Save dataset in multiple formats with token distribution analysis."""
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

    # Calculate comprehensive statistics
    chunks_with_code = [c for c in chunks if c['source_code_snippets']]
    total_code_snippets = sum(len(c['source_code_snippets']) for c in chunks)

    token_counts = [c['token_stats']['total_tokens'] for c in chunks]
    word_counts = [c['metadata']['word_count'] for c in chunks]

    # Size distribution
    small_chunks = [c for c in chunks if c['token_stats']['total_tokens'] < 2000]
    medium_chunks = [c for c in chunks if 2000 <= c['token_stats']['total_tokens'] < 5000]
    large_chunks = [c for c in chunks if c['token_stats']['total_tokens'] >= 5000]

    chunks_with_diagrams = sum(1 for c in chunks if c['metadata']['has_mermaid_diagram'])
    chunks_with_tables = sum(1 for c in chunks if c['metadata']['has_tables'])

    stats = {
        'total_chunks': len(chunks),
        'chunks_with_source_code': len(chunks_with_code),
        'total_code_snippets': total_code_snippets,
        'unique_source_files': len(set(c['source_file'] for c in chunks)),
        'chunks_with_mermaid_diagrams': chunks_with_diagrams,
        'chunks_with_tables': chunks_with_tables,
        'size_distribution': {
            'small_chunks': {
                'count': len(small_chunks),
                'percentage': (len(small_chunks) / len(chunks) * 100) if chunks else 0,
                'range': '< 2000 tokens'
            },
            'medium_chunks': {
                'count': len(medium_chunks),
                'percentage': (len(medium_chunks) / len(chunks) * 100) if chunks else 0,
                'range': '2000-5000 tokens'
            },
            'large_chunks': {
                'count': len(large_chunks),
                'percentage': (len(large_chunks) / len(chunks) * 100) if chunks else 0,
                'range': '5000-8000 tokens'
            }
        },
        'chunks_by_heading_level': {
            level: len([c for c in chunks if c['heading_level'] == level])
            for level in sorted(set(c['heading_level'] for c in chunks))
        },
        'token_distribution': {
            'mean': float(np.mean(token_counts)),
            'median': float(np.median(token_counts)),
            'std': float(np.std(token_counts)),
            'min': int(np.min(token_counts)),
            'max': int(np.max(token_counts)),
            'percentiles': {
                '25th': float(np.percentile(token_counts, 25)),
                '50th': float(np.percentile(token_counts, 50)),
                '75th': float(np.percentile(token_counts, 75)),
                '90th': float(np.percentile(token_counts, 90)),
                '95th': float(np.percentile(token_counts, 95)),
                '99th': float(np.percentile(token_counts, 99))
            }
        },
        'word_distribution': {
            'mean': float(np.mean(word_counts)),
            'median': float(np.median(word_counts)),
            'std': float(np.std(word_counts))
        },
        'source_files': sorted(set(c['source_file'] for c in chunks))
    }

    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"✓ Saved statistics to {stats_path}")

    # Create token distribution visualization data
    token_dist_path = output_dir / 'token_distribution.json'
    with open(token_dist_path, 'w', encoding='utf-8') as f:
        json.dump({
            'token_counts': token_counts,
            'bins': list(range(0, max(token_counts) + 100, 100))
        }, f, indent=2)
    print(f"✓ Saved token distribution data to {token_dist_path}")

    return stats


def create_readme(stats: Dict[str, Any], output_dir: Path, model_name: str):
    """Create a README.md for the dataset."""
    readme_content = f"""# DeepWiki Token-Optimized Dataset (KAT-Dev)

This dataset contains token-optimized documentation from the Hyperswitch payment router project,
specifically optimized for the **{model_name}** model tokenizer.

## Key Features

- **Token-Aware Chunking**: Chunks optimized for {model_name} tokenizer
- **Source Code Integration**: Actual code snippets with token counts
- **Rich Token Metadata**: Detailed token distribution analysis
- **Well-Distributed Sizes**: Small (<2k), Medium (2k-5k), Large (5k-8k) chunks for varied context
- **Minimal Overlap**: ~200 token overlap between adjacent chunks for context continuity

## Dataset Statistics

- **Total Chunks**: {stats['total_chunks']:,}
- **Chunks with Source Code**: {stats['chunks_with_source_code']:,}
- **Total Code Snippets**: {stats['total_code_snippets']:,}
- **Chunks with Mermaid Diagrams**: {stats['chunks_with_mermaid_diagrams']:,}
- **Chunks with Tables**: {stats['chunks_with_tables']:,}
- **Source Files**: {stats['unique_source_files']}

### Size Distribution (Target: 25% Small, 50% Medium, 25% Large)

- **Small Chunks** (< 2k tokens): {stats['size_distribution']['small_chunks']['count']:,} ({stats['size_distribution']['small_chunks']['percentage']:.1f}%)
- **Medium Chunks** (2k-5k tokens): {stats['size_distribution']['medium_chunks']['count']:,} ({stats['size_distribution']['medium_chunks']['percentage']:.1f}%)
- **Large Chunks** (5k-8k tokens): {stats['size_distribution']['large_chunks']['count']:,} ({stats['size_distribution']['large_chunks']['percentage']:.1f}%)

### Token Distribution

- **Mean Tokens**: {stats['token_distribution']['mean']:.0f}
- **Median Tokens**: {stats['token_distribution']['median']:.0f}
- **Std Dev**: {stats['token_distribution']['std']:.0f}
- **Range**: {stats['token_distribution']['min']:,} - {stats['token_distribution']['max']:,} tokens

#### Percentiles
- 25th: {stats['token_distribution']['percentiles']['25th']:.0f} tokens
- 50th (Median): {stats['token_distribution']['percentiles']['50th']:.0f} tokens
- 75th: {stats['token_distribution']['percentiles']['75th']:.0f} tokens
- 90th: {stats['token_distribution']['percentiles']['90th']:.0f} tokens
- 95th: {stats['token_distribution']['percentiles']['95th']:.0f} tokens
- 99th: {stats['token_distribution']['percentiles']['99th']:.0f} tokens

### Word Distribution

- **Mean Words**: {stats['word_distribution']['mean']:.0f}
- **Median Words**: {stats['word_distribution']['median']:.0f}

### Chunks by Heading Level

{chr(10).join(f"- Level {level}: {count:,} chunks" for level, count in stats['chunks_by_heading_level'].items())}

## Dataset Structure

Each row contains:
- `id`: Unique identifier
- `source_file`: Source filename
- `heading`: Main heading
- `heading_level`: Heading level (1-3)
- `content`: Complete content
- `chunk_index`: Chunk index within file
- `source_references`: Source file references
- `source_code_snippets`: Code snippets with token counts
- `metadata`: Content metadata (word count, subsections, diagrams, etc.)
- `token_stats`: Comprehensive token analysis
  - `total_tokens`: Total tokens in chunk
  - `total_chars`: Total characters
  - `compression_ratio`: Chars per token
  - `tokens_per_word`: Average tokens per word
  - `content_tokens`: Tokens in documentation
  - `code_tokens`: Tokens in code blocks
  - `code_token_percentage`: % of tokens in code

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("archit11/deepwiki4", split="train")

# Access token statistics
sample = dataset[0]
print(f"Tokens: {{sample['token_stats']['total_tokens']}}")
print(f"Words: {{sample['metadata']['word_count']}}")
print(f"Code %: {{sample['token_stats']['code_token_percentage']:.1f}}%")

# Filter by token count
efficient_chunks = dataset.filter(lambda x: x['token_stats']['total_tokens'] <= 1024)
```

## Tokenizer

Optimized for: `{model_name}`

## Source

- **Documentation**: juspay/hyperswitch wiki
- **Source Code**: https://github.com/juspay/hyperswitch (commit 820f1831)
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created {readme_path}")


def main():
    # Paths
    input_dir = Path('/home/dumball/out')
    output_dir = Path('/home/dumball/token_aware_dataset_output')
    repo_dir = Path('/home/dumball/hyperswitch')
    repo_url = 'https://github.com/juspay/hyperswitch.git'

    # Token settings
    model_name = "Kwaipilot/KAT-Dev"
    max_tokens = 8192  # KAT-Dev supports 8k context

    print("=" * 60)
    print("DeepWiki Token-Aware Dataset Creator")
    print(f"Optimized for: {model_name}")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    if not tokenizer:
        print("❌ Cannot proceed without tokenizer")
        return

    # Setup repository
    print("\n[2/4] Setting up hyperswitch repository...")
    if clone_or_update_repo(repo_url, repo_dir):
        print("✓ Repository ready")
    else:
        print("⚠️  Proceeding without source code mining")
        repo_dir = None

    # Process files
    print("\n[3/4] Processing documentation files...")
    chunks = process_directory(input_dir, tokenizer, repo_dir, max_tokens)

    if not chunks:
        print("\n❌ No chunks extracted!")
        return

    # Save dataset
    print("\n[4/4] Saving dataset...")
    stats = save_dataset(chunks, output_dir)

    # Create README
    create_readme(stats, output_dir, model_name)

    print("\n" + "=" * 60)
    print(f"✅ Token-aware dataset creation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Chunks with source code: {stats['chunks_with_source_code']:,}")
    print(f"Mean tokens per chunk: {stats['token_distribution']['mean']:.0f}")
    print(f"Median tokens per chunk: {stats['token_distribution']['median']:.0f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
