#!/usr/bin/env python3
"""
Create code-only dataset from Hyperswitch Rust repository.
Extracts code at multiple granularities: file, module, function, struct/impl levels.
Batches into 4k, 8k, 16k token chunks.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
import pandas as pd
import numpy as np


def clone_or_update_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone or update the hyperswitch repository."""
    if target_dir.exists():
        print(f"Repository already exists at {target_dir}")
        try:
            subprocess.run(['git', '-C', str(target_dir), 'status'],
                         check=True, capture_output=True)
            print("✓ Repository is ready")
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


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=True))
    return int(len(text.split()) * 1.3)  # Rough estimate


def find_rust_files(repo_dir: Path) -> List[Path]:
    """Find all Rust source files in the repository."""
    rust_files = []

    # Look in crates directory (main Rust code)
    crates_dir = repo_dir / "crates"
    if crates_dir.exists():
        rust_files.extend(crates_dir.rglob("*.rs"))

    # Also check root src if it exists
    src_dir = repo_dir / "src"
    if src_dir.exists():
        rust_files.extend(src_dir.rglob("*.rs"))

    return sorted(rust_files)


def extract_functions(code: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract function definitions from Rust code."""
    functions = []

    # Pattern for Rust functions (including pub, async, unsafe, const, etc.)
    pattern = r'(?:(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{)'

    for match in re.finditer(pattern, code):
        fn_name = match.group(1)
        start_pos = match.start()

        # Find the matching closing brace
        brace_count = 0
        in_function = False
        end_pos = start_pos

        for i, char in enumerate(code[start_pos:], start=start_pos):
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos > start_pos:
            fn_code = code[start_pos:end_pos]
            functions.append({
                'name': fn_name,
                'code': fn_code,
                'file_path': file_path,
                'type': 'function'
            })

    return functions


def extract_structs_and_impls(code: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract struct definitions and their implementations."""
    items = []

    # Extract struct definitions
    struct_pattern = r'(?:pub(?:\([^)]+\))?\s+)?struct\s+(\w+)(?:<[^>]+>)?(?:\s*\{[^}]*\}|;)'
    for match in re.finditer(struct_pattern, code):
        struct_name = match.group(1)
        struct_code = match.group(0)

        items.append({
            'name': struct_name,
            'code': struct_code,
            'file_path': file_path,
            'type': 'struct'
        })

    # Extract impl blocks
    impl_pattern = r'impl(?:<[^>]+>)?\s+(?:(\w+)(?:<[^>]+>)?)\s*(?:for\s+(\w+)(?:<[^>]+>)?)?\s*\{'

    for match in re.finditer(impl_pattern, code):
        impl_target = match.group(1) or match.group(2)
        start_pos = match.start()

        # Find matching closing brace
        brace_count = 0
        end_pos = start_pos

        for i, char in enumerate(code[start_pos:], start=start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos > start_pos:
            impl_code = code[start_pos:end_pos]
            items.append({
                'name': impl_target,
                'code': impl_code,
                'file_path': file_path,
                'type': 'impl'
            })

    return items


def get_module_path(file_path: Path, repo_dir: Path) -> str:
    """Get the module path from file path."""
    try:
        relative = file_path.relative_to(repo_dir)
        parts = list(relative.parts)

        # Remove .rs extension from last part
        if parts[-1].endswith('.rs'):
            parts[-1] = parts[-1][:-3]

        # Remove 'src' or 'crates' from path
        if parts[0] in ['src', 'crates']:
            parts = parts[1:]

        return "::".join(parts)
    except ValueError:
        return str(file_path.name)


def chunk_code(code_items: List[Dict[str, Any]], tokenizer, max_tokens: int,
               granularity: str) -> List[Dict[str, Any]]:
    """
    Chunk code items into batches that fit within max_tokens.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0

    for item in code_items:
        item_tokens = count_tokens(item['code'], tokenizer)

        # If single item exceeds max_tokens, split it
        if item_tokens > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append({
                    'id': f"{granularity}_{chunk_id}",
                    'code': "\n\n".join([i['code'] for i in current_chunk]),
                    'token_count': current_tokens,
                    'granularity': granularity,
                    'items': current_chunk,
                    'item_count': len(current_chunk)
                })
                chunk_id += 1
                current_chunk = []
                current_tokens = 0

            # Split large item by lines
            lines = item['code'].split('\n')
            temp_code = ""
            temp_tokens = 0

            for line in lines:
                test_code = temp_code + "\n" + line if temp_code else line
                test_tokens = count_tokens(test_code, tokenizer)

                if test_tokens > max_tokens and temp_code:
                    # Save current temp
                    chunks.append({
                        'id': f"{granularity}_{chunk_id}",
                        'code': temp_code,
                        'token_count': temp_tokens,
                        'granularity': granularity,
                        'items': [{'code': temp_code, 'file_path': item['file_path'], 'type': item['type'], 'name': item.get('name', 'split')}],
                        'item_count': 1
                    })
                    chunk_id += 1
                    temp_code = line
                    temp_tokens = count_tokens(line, tokenizer)
                else:
                    temp_code = test_code
                    temp_tokens = test_tokens

            if temp_code:
                chunks.append({
                    'id': f"{granularity}_{chunk_id}",
                    'code': temp_code,
                    'token_count': temp_tokens,
                    'granularity': granularity,
                    'items': [{'code': temp_code, 'file_path': item['file_path'], 'type': item['type'], 'name': item.get('name', 'split')}],
                    'item_count': 1
                })
                chunk_id += 1

        # Item fits, check if adding it exceeds limit
        elif current_tokens + item_tokens > max_tokens:
            # Save current chunk
            if current_chunk:
                chunks.append({
                    'id': f"{granularity}_{chunk_id}",
                    'code': "\n\n".join([i['code'] for i in current_chunk]),
                    'token_count': current_tokens,
                    'granularity': granularity,
                    'items': current_chunk,
                    'item_count': len(current_chunk)
                })
                chunk_id += 1

            # Start new chunk with this item
            current_chunk = [item]
            current_tokens = item_tokens

        else:
            # Add to current chunk
            current_chunk.append(item)
            current_tokens += item_tokens

    # Flush remaining chunk
    if current_chunk:
        chunks.append({
            'id': f"{granularity}_{chunk_id}",
            'code': "\n\n".join([i['code'] for i in current_chunk]),
            'token_count': current_tokens,
            'granularity': granularity,
            'items': current_chunk,
            'item_count': len(current_chunk)
        })

    return chunks


def process_repository(repo_dir: Path, tokenizer,
                       token_sizes: List[int] = [4096, 8192, 16384]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process repository at multiple granularities and token sizes.
    Returns dict with keys like 'file_4k', 'function_8k', etc.
    """
    results = {}

    print(f"\nFinding Rust files...")
    rust_files = find_rust_files(repo_dir)
    print(f"  Found {len(rust_files)} Rust files")

    # Collect all code items by granularity
    file_level_items = []
    function_level_items = []
    struct_impl_items = []
    module_groups = {}

    for rust_file in rust_files:
        try:
            with open(rust_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            rel_path = str(rust_file.relative_to(repo_dir))
            module_path = get_module_path(rust_file, repo_dir)

            # File level
            file_level_items.append({
                'code': code,
                'file_path': rel_path,
                'module_path': module_path,
                'type': 'file',
                'name': rust_file.name
            })

            # Function level
            functions = extract_functions(code, rel_path)
            function_level_items.extend(functions)

            # Struct/Impl level
            structs_impls = extract_structs_and_impls(code, rel_path)
            struct_impl_items.extend(structs_impls)

            # Module level grouping
            # Group by crate/module (e.g., router::core, api_models, etc.)
            module_key = "::".join(module_path.split("::")[:2]) if "::" in module_path else module_path
            if module_key not in module_groups:
                module_groups[module_key] = []
            module_groups[module_key].append({
                'code': code,
                'file_path': rel_path,
                'module_path': module_path,
                'type': 'module_file',
                'name': module_key
            })

        except Exception as e:
            print(f"  ⚠️  Error processing {rust_file}: {e}")
            continue

    print(f"\n  Extracted:")
    print(f"    Files: {len(file_level_items)}")
    print(f"    Functions: {len(function_level_items)}")
    print(f"    Structs/Impls: {len(struct_impl_items)}")
    print(f"    Modules: {len(module_groups)}")

    # Create chunks for each granularity and token size
    granularities = [
        ('file', file_level_items),
        ('function', function_level_items),
        ('struct_impl', struct_impl_items),
    ]

    # Add module level
    module_items = []
    for module_key, files in module_groups.items():
        combined_code = "\n\n".join([f['code'] for f in files])
        module_items.append({
            'code': combined_code,
            'file_path': f"module::{module_key}",
            'module_path': module_key,
            'type': 'module',
            'name': module_key
        })
    granularities.append(('module', module_items))

    for gran_name, items in granularities:
        for max_tokens in token_sizes:
            size_name = f"{max_tokens // 1024}k"
            key = f"{gran_name}_{size_name}"

            print(f"\n  Chunking {gran_name} at {size_name} tokens...")
            chunks = chunk_code(items, tokenizer, max_tokens, gran_name)
            results[key] = chunks
            print(f"    Created {len(chunks)} chunks")

    return results


def save_datasets(all_chunks: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Save all datasets with statistics."""
    output_dir.mkdir(exist_ok=True)

    all_stats = {}

    for dataset_key, chunks in all_chunks.items():
        # Simplify items for saving (remove nested dicts)
        simplified_chunks = []
        for chunk in chunks:
            simplified = {
                'id': chunk['id'],
                'code': chunk['code'],
                'token_count': chunk['token_count'],
                'granularity': chunk['granularity'],
                'item_count': chunk['item_count'],
                'file_paths': list(set([item['file_path'] for item in chunk['items']])),
                'item_types': list(set([item['type'] for item in chunk['items']])),
                'item_names': [item.get('name', 'unknown') for item in chunk['items'][:10]]  # First 10 names
            }
            simplified_chunks.append(simplified)

        # Save as JSONL
        jsonl_path = output_dir / f'{dataset_key}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in simplified_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        # Save as Parquet
        df = pd.DataFrame(simplified_chunks)
        parquet_path = output_dir / f'{dataset_key}.parquet'
        df.to_parquet(parquet_path, index=False)

        # Calculate stats
        token_counts = [c['token_count'] for c in simplified_chunks]
        stats = {
            'dataset_key': dataset_key,
            'total_chunks': len(simplified_chunks),
            'total_tokens': sum(token_counts),
            'mean_tokens': np.mean(token_counts),
            'median_tokens': np.median(token_counts),
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0
        }
        all_stats[dataset_key] = stats

        print(f"\n  ✓ Saved {dataset_key}: {len(simplified_chunks)} chunks")

    # Save overall stats
    stats_path = output_dir / 'all_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Saved statistics to {stats_path}")

    return all_stats


def create_readme(stats: Dict[str, Any], output_dir: Path):
    """Create comprehensive README."""
    readme_content = """# Hyperswitch Code-Only Dataset

Multi-granularity Rust code dataset from the Hyperswitch payment router repository.

## Dataset Variants

This dataset provides code at multiple granularities and token sizes:

### Granularity Levels
- **file**: Complete Rust source files
- **module**: Grouped by crate/module (e.g., router::core)
- **function**: Individual function definitions
- **struct_impl**: Struct definitions and impl blocks

### Token Sizes
- **4k**: Max 4,096 tokens per chunk
- **8k**: Max 8,192 tokens per chunk
- **16k**: Max 16,384 tokens per chunk

## Available Datasets

"""

    for key, stat in stats.items():
        readme_content += f"""### {key}
- Chunks: {stat['total_chunks']:,}
- Total Tokens: {stat['total_tokens']:,}
- Avg Tokens: {stat['mean_tokens']:.0f}
- Range: {stat['min_tokens']:,} - {stat['max_tokens']:,}

"""

    readme_content += """
## Dataset Structure

Each example contains:
- `id`: Unique identifier
- `code`: Rust code content
- `token_count`: Token count (Kwaipilot/KAT-Dev tokenizer)
- `granularity`: Level (file/module/function/struct_impl)
- `item_count`: Number of code items in this chunk
- `file_paths`: Source file paths
- `item_types`: Types of items included
- `item_names`: Names of items (functions, structs, modules)

## Usage

```python
from datasets import load_dataset

# Load specific granularity/size
dataset = load_dataset("parquet", data_files="function_8k.parquet")

# Access code
sample = dataset['train'][0]
print(sample['code'])
print(f"Tokens: {sample['token_count']}")
```

## Source

- Repository: https://github.com/juspay/hyperswitch
- License: Apache 2.0
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created {readme_path}")


def main():
    # Configuration
    script_dir = Path(__file__).parent.resolve()
    repo_dir = script_dir / 'hyperswitch'
    output_dir = script_dir / 'code_dataset_output'
    repo_url = 'https://github.com/juspay/hyperswitch.git'
    model_name = "Kwaipilot/KAT-Dev"
    token_sizes = [4096, 8192, 16384]

    print("=" * 60)
    print("Hyperswitch Code-Only Dataset Creator")
    print("=" * 60)

    # Clone/update repo
    print("\n[1/4] Setting up repository...")
    if not clone_or_update_repo(repo_url, repo_dir):
        print("❌ Failed to setup repository")
        return

    # Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Loaded {model_name} tokenizer")
    except Exception as e:
        print(f"⚠️  Failed to load tokenizer: {e}")
        tokenizer = None

    # Process repository
    print("\n[3/4] Processing repository...")
    all_chunks = process_repository(repo_dir, tokenizer, token_sizes)

    # Save datasets
    print("\n[4/4] Saving datasets...")
    stats = save_datasets(all_chunks, output_dir)

    # Create README
    create_readme(stats, output_dir)

    print("\n" + "=" * 60)
    print("✅ Code dataset creation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total datasets: {len(all_chunks)}")
    print("=" * 60)


if __name__ == '__main__':
    main()