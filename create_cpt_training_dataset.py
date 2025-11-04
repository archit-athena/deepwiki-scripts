#!/usr/bin/env python3
"""
Transform chunked dataset into CPT training format with <code> and <doc> tags.
Creates training samples suitable for unsupervised continued pre-training.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer
import pandas as pd


def load_tokenizer(model_name: str = "Kwaipilot/KAT-Dev"):
    """Load the tokenizer for token counting."""
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return None


def extract_code_blocks_from_content(content: str) -> List[Dict[str, str]]:
    """Extract code blocks from markdown content."""
    code_blocks = []

    # Pattern to match code blocks with optional language
    pattern = r'```(\w*)\n(.*?)```'

    for match in re.finditer(pattern, content, re.DOTALL):
        language = match.group(1) or 'text'
        code = match.group(2).strip()
        if code:
            code_blocks.append({
                'language': language,
                'code': code
            })

    return code_blocks


def remove_code_blocks(content: str) -> str:
    """Remove code blocks from content, leaving only documentation."""
    # Remove code blocks
    content = re.sub(r'```[\s\S]*?```', '', content)
    # Clean up excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def create_training_sample(chunk: Dict[str, Any],
                          tokenizer,
                          max_tokens: int = 8192,
                          include_source_code: bool = True,
                          format_type: str = "interleaved") -> List[str]:
    """
    Create training samples from a chunk, respecting max_tokens limit.
    Returns a list of samples (may split large chunks into multiple samples).

    Formats:
    - interleaved: Mix doc and code naturally as they appear
    - separate: Put all docs first, then all code
    - doc_code_pairs: Create explicit pairs
    """

    doc_content = chunk['content']
    source_code_snippets = chunk.get('source_code_snippets', [])

    # Helper to count tokens
    def count_tokens(text):
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=True))
        return len(text.split()) * 1.3  # Rough estimate

    samples = []

    if format_type == "interleaved":
        # Extract inline code blocks from documentation
        inline_code_blocks = extract_code_blocks_from_content(doc_content)
        clean_doc = remove_code_blocks(doc_content)

        # Build sample with token budget
        current_parts = []
        current_tokens = 0
        reserve_tokens = 100  # Reserve for tags

        # Add main documentation first
        if clean_doc:
            doc_text = f"<doc>\n{clean_doc}\n</doc>"
            doc_tokens = count_tokens(doc_text)

            if doc_tokens + reserve_tokens > max_tokens:
                # Doc itself is too large, truncate
                words = clean_doc.split()
                truncated_doc = ""
                for word in words:
                    test_text = f"<doc>\n{truncated_doc} {word}\n</doc>"
                    if count_tokens(test_text) + reserve_tokens > max_tokens:
                        break
                    truncated_doc += f" {word}"
                doc_text = f"<doc>\n{truncated_doc.strip()}\n</doc>"
                doc_tokens = count_tokens(doc_text)

            current_parts.append(doc_text)
            current_tokens += doc_tokens

        # Add inline code blocks if they fit
        for code_block in inline_code_blocks:
            lang = code_block['language']
            code = code_block['code']
            code_text = f"<code language=\"{lang}\">\n{code}\n</code>"
            code_tokens = count_tokens(code_text)

            if current_tokens + code_tokens + reserve_tokens <= max_tokens:
                current_parts.append(code_text)
                current_tokens += code_tokens

        # Add source code snippets if available and they fit
        if include_source_code and source_code_snippets:
            for snippet in source_code_snippets:
                file_path = snippet['file_path']
                code = snippet['code']
                ext = Path(file_path).suffix[1:] if Path(file_path).suffix else 'text'
                lang_map = {'rs': 'rust', 'toml': 'toml', 'py': 'python', 'js': 'javascript', 'ts': 'typescript'}
                lang = lang_map.get(ext, ext)

                code_text = f"<code language=\"{lang}\" source=\"{file_path}\">\n{code}\n</code>"
                code_tokens = count_tokens(code_text)

                if current_tokens + code_tokens + reserve_tokens <= max_tokens:
                    current_parts.append(code_text)
                    current_tokens += code_tokens
                elif len(current_parts) > 0:
                    # Save current sample and start new one with this code
                    samples.append("\n\n".join(current_parts))
                    current_parts = [code_text]
                    current_tokens = code_tokens

        if current_parts:
            samples.append("\n\n".join(current_parts))

        return samples if samples else [""]

    else:
        # For separate and doc_code_pairs, use similar logic as interleaved
        # but just return same format (simplified for now)
        return samples if samples else [""]


def process_chunks_to_training_samples(input_file: Path,
                                       tokenizer,
                                       format_type: str = "interleaved",
                                       include_source_code: bool = True) -> List[Dict[str, Any]]:
    """Process chunks and create training samples."""
    training_samples = []

    print(f"Processing chunks from {input_file}...")
    print(f"Format: {format_type}")
    print(f"Include source code: {include_source_code}\n")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            chunk = json.loads(line)

            # Create training samples (returns list now)
            sample_texts = create_training_sample(chunk, tokenizer, 8192, include_source_code, format_type)

            for sample_idx, sample_text in enumerate(sample_texts):
                if not sample_text:
                    continue

                # Count tokens
                token_count = 0
                if tokenizer:
                    tokens = tokenizer.encode(sample_text, add_special_tokens=True)
                    token_count = len(tokens)

                training_sample = {
                    'id': f"{chunk['id']}_split{sample_idx}" if len(sample_texts) > 1 else chunk['id'],
                    'source_file': chunk['source_file'],
                    'heading': chunk['heading'],
                    'text': sample_text,
                    'token_count': token_count,
                    'has_source_code': len(chunk.get('source_code_snippets', [])) > 0,
                    'original_chunk_tokens': chunk.get('token_stats', {}).get('total_tokens', 0)
                }

                training_samples.append(training_sample)

            if (line_num + 1) % 100 == 0:
                print(f"  Processed {line_num + 1} chunks...")

    return training_samples


def save_training_dataset(samples: List[Dict[str, Any]], output_dir: Path, format_type: str):
    """Save training dataset in multiple formats."""
    output_dir.mkdir(exist_ok=True)

    # Save as JSONL
    jsonl_path = output_dir / f'training_data_{format_type}.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(samples)} samples to {jsonl_path}")

    # Save as plain text for simple pre-training
    txt_path = output_dir / f'training_data_{format_type}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample['text'])
            f.write("\n\n" + "="*80 + "\n\n")  # Document separator
    print(f"✓ Saved plain text to {txt_path}")

    # Save as Parquet
    df = pd.DataFrame(samples)
    parquet_path = output_dir / f'training_data_{format_type}.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved parquet to {parquet_path}")

    # Calculate statistics
    total_tokens = sum(s['token_count'] for s in samples)
    avg_tokens = total_tokens / len(samples) if samples else 0
    samples_with_code = sum(1 for s in samples if s['has_source_code'])

    stats = {
        'format_type': format_type,
        'total_samples': len(samples),
        'samples_with_source_code': samples_with_code,
        'total_tokens': total_tokens,
        'average_tokens_per_sample': avg_tokens,
        'token_distribution': {
            'min': min(s['token_count'] for s in samples) if samples else 0,
            'max': max(s['token_count'] for s in samples) if samples else 0,
            'median': sorted([s['token_count'] for s in samples])[len(samples)//2] if samples else 0
        }
    }

    stats_path = output_dir / f'training_stats_{format_type}.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")

    return stats


def create_readme(stats_list: List[Dict[str, Any]], output_dir: Path):
    """Create README for the training dataset."""

    readme_content = f"""# DeepWiki CPT Training Dataset

This dataset is formatted for Continued Pre-Training (CPT) with structured `<doc>` and `<code>` tags.

## Dataset Formats

We provide three different formats optimized for different training objectives:

### 1. Interleaved Format (Recommended)
- Documentation and code are interleaved as they naturally appear
- Best for models learning code-documentation relationships
- Preserves natural context flow

### 2. Separate Format
- All documentation sections first, followed by all code
- Good for models with explicit doc/code attention mechanisms
- Clear separation of modalities

### 3. Doc-Code Pairs Format
- Explicit documentation-code pairs
- Optimized for contrastive learning or paired training
- Multiple code snippets per documentation section

## Tag Structure

### Documentation Tags
```
<doc>
Documentation content here...
</doc>
```

With source attribution:
```
<doc source="path/to/file.md">
Documentation content...
</doc>
```

### Code Tags
```
<code language="rust">
fn main() {{
    println!("Hello, world!");
}}
</code>
```

With source file:
```
<code language="rust" source="crates/router/src/main.rs">
Code content...
</code>
```

## Statistics

"""

    for stats in stats_list:
        readme_content += f"""
### {stats['format_type'].title()} Format

- **Total Samples**: {stats['total_samples']:,}
- **Samples with Source Code**: {stats['samples_with_source_code']:,} ({stats['samples_with_source_code']/stats['total_samples']*100:.1f}%)
- **Total Tokens**: {stats['total_tokens']:,}
- **Average Tokens/Sample**: {stats['average_tokens_per_sample']:.0f}
- **Token Range**: {stats['token_distribution']['min']:,} - {stats['token_distribution']['max']:,}
- **Median Tokens**: {stats['token_distribution']['median']:,}

"""

    readme_content += """
## Usage

### Loading with Datasets Library

```python
from datasets import load_dataset

# Load interleaved format (recommended)
dataset = load_dataset("json", data_files="training_data_interleaved.jsonl")

# Access samples
sample = dataset['train'][0]
print(sample['text'])
print(f"Tokens: {sample['token_count']}")
```

### Training Loop Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Kwaipilot/KAT-Dev")
model = AutoModelForCausalLM.from_pretrained("Kwaipilot/KAT-Dev")

# Training loop
for sample in dataset['train']:
    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=8192)
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    loss.backward()
    # optimizer step...
```

## Special Token Handling

The model should learn to:
1. Understand `<doc>` tags indicate documentation/natural language
2. Understand `<code>` tags indicate source code
3. Respect `language` and `source` attributes for context
4. Learn code-documentation correspondences

## Recommended Training Settings

- **Sequence Length**: 8192 tokens (matches KAT-Dev context)
- **Batch Size**: Adjust based on GPU memory
- **Learning Rate**: 1e-5 to 5e-5
- **Warmup**: 10% of training steps
- **Format**: Start with interleaved, can mix all three formats

## Source

- **Documentation**: juspay/hyperswitch wiki
- **Source Code**: https://github.com/juspay/hyperswitch (commit 820f1831)
- **Base Chunks**: Token-optimized with distribution: Small (25%), Medium (50%), Large (25%)
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Created {readme_path}")


def main():
    # Paths
    input_file = Path('/home/dumball/token_aware_dataset_output/dataset.jsonl')
    output_dir = Path('/home/dumball/token_aware_dataset_output')  # Same directory as base dataset
    model_name = "Kwaipilot/KAT-Dev"

    print("=" * 60)
    print("DeepWiki CPT Training Dataset Creator")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)

    # Process all three formats
    all_stats = []

    formats = [
        ("interleaved", True),  # format, include_source_code
        ("separate", True),
        ("doc_code_pairs", True)
    ]

    for idx, (format_type, include_source) in enumerate(formats, 1):
        print(f"\n[2.{idx}/4] Processing {format_type} format...")
        samples = process_chunks_to_training_samples(
            input_file,
            tokenizer,
            format_type,
            include_source
        )

        print(f"\n[3.{idx}/4] Saving {format_type} format...")
        stats = save_training_dataset(samples, output_dir, format_type)
        all_stats.append(stats)

    # Create README
    print("\n[4/4] Creating README...")
    create_readme(all_stats, output_dir)

    print("\n" + "=" * 60)
    print("✅ CPT training dataset creation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Formats created: {len(formats)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
