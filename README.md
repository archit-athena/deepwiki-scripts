# DeepWiki Dataset Creation Scripts

Collection of Python scripts for creating various documentation + code datasets from web-scraped documentation and source code repositories.

## Overview

This repository contains scripts that progressively build more sophisticated datasets from raw documentation, culminating in a token-aware, CPT-ready training dataset optimized for code-aware language models.

## Scripts

### 1. `deepwiki.py` - Documentation Scraper
Scrapes documentation from websites and saves as markdown files.

**Usage:**
```bash
python3 deepwiki.py
```

### 2. `create_dataset.py` - Basic Chunking
Creates a basic dataset by chunking markdown files by headings.

**Features:**
- Chunks by level 2 headings (`##`)
- Extracts metadata (word count, subsections, etc.)
- Outputs JSONL and Parquet formats

**Usage:**
```bash
python3 create_dataset.py
```

**Output:** `dataset_output/`

### 3. `create_enhanced_dataset.py` - With Source Code Mining
Enhances basic chunks by extracting and mining source code references.

**Features:**
- Parses source references from docs (e.g., `[file.rs:10-20]()`)
- Clones target repository
- Extracts actual source code snippets
- Links documentation to code

**Usage:**
```bash
python3 create_enhanced_dataset.py
```

**Output:** `enhanced_dataset_output/`

### 4. `create_semantic_dataset.py` - Semantic Chunking
Creates larger, semantically meaningful chunks based on logical boundaries.

**Features:**
- Chunks by `---` separators (semantic boundaries)
- Preserves complete context
- Larger chunks (~299 words average)
- Rich metadata (diagrams, tables, code blocks)

**Usage:**
```bash
python3 create_semantic_dataset.py
```

**Output:** `semantic_dataset_output/`

### 5. `create_token_aware_dataset.py` - Token-Optimized Chunking
Creates chunks optimized for a specific tokenizer with well-distributed sizes.

**Features:**
- Loads target model tokenizer (Kwaipilot/KAT-Dev)
- Token-aware chunking with size distribution:
  - Small (<2k tokens): 25%
  - Medium (2k-5k tokens): 50%
  - Large (5k-8k tokens): 25%
- ~200 token overlap between chunks
- Comprehensive token statistics
- Max 8192 tokens per chunk

**Usage:**
```bash
python3 create_token_aware_dataset.py
```

**Output:** `token_aware_dataset_output/`

**Statistics provided:**
- Token count distribution (mean, median, percentiles)
- Size distribution breakdown
- Compression ratios
- Code vs documentation token ratios

### 6. `create_cpt_training_dataset.py` - CPT Training Format
Transforms token-aware chunks into CPT (Continued Pre-Training) format with `<doc>` and `<code>` tags.

**Features:**
- Three format variants:
  1. **Interleaved**: Natural doc + code mixing (recommended)
  2. **Separate**: All docs first, then all code
  3. **Doc-Code Pairs**: Explicit pairing for contrastive learning
- Respects 8k token limit per sample
- Structured tags with metadata:
  - `<doc>` for documentation
  - `<code language="rust" source="path/to/file.rs">` for code
- Splits large chunks into multiple samples if needed

**Usage:**
```bash
python3 create_cpt_training_dataset.py
```

**Output:** `token_aware_dataset_output/training_data_*.{jsonl,txt,parquet}`

**Training formats:**
- `training_data_interleaved.*` - Mixed doc/code
- `training_data_separate.*` - Separated doc/code
- `training_data_doc_code_pairs.*` - Explicit pairs

## Pipeline Flow

```
Raw Docs → Scraper → Basic Chunks → Enhanced (+ Code) → Semantic → Token-Aware → CPT Format
                                                                                      ↓
                                                                              Training-Ready
```

## Dependencies

```bash
pip install pandas transformers
```

## Dataset Outputs

Each script creates its output in a separate directory with:
- `dataset.jsonl` - JSON Lines format
- `dataset.parquet` - Parquet format
- `dataset_stats.json` - Statistics
- `README.md` - Dataset documentation

## Configuration

Key parameters to adjust:

### Token-Aware Dataset
- `max_tokens`: 8192 (adjust for your model's context window)
- `overlap_tokens`: 200 (adjust overlap amount)
- Target distribution: Modify in `target_distribution` dict

### CPT Training Dataset
- `max_tokens`: 8192 (max tokens per training sample)
- Format types: Enable/disable formats in `main()`

## Use Cases

1. **RAG Applications**: Use semantic or token-aware datasets
2. **Code Generation Training**: Use CPT format with interleaved mode
3. **Contrastive Learning**: Use doc-code pairs format
4. **General Pre-training**: Use separate format

## Example: Full Pipeline

```bash
# 1. Scrape documentation
python3 deepwiki.py

# 2. Create token-aware base dataset
python3 create_token_aware_dataset.py

# 3. Transform to CPT format
python3 create_cpt_training_dataset.py

# 4. Upload to HuggingFace
huggingface-cli repo create your-username/dataset-name --type dataset
cd token_aware_dataset_output
huggingface-cli upload your-username/dataset-name . . --repo-type dataset
```

## Dataset Structure

### Base Chunk Format
```json
{
  "id": "file_0",
  "source_file": "1-overview",
  "heading": "Environment Configuration",
  "content": "...",
  "source_references": [...],
  "source_code_snippets": [...],
  "token_stats": {
    "total_tokens": 3017,
    "content_tokens": 2500,
    "code_tokens": 517
  },
  "metadata": {
    "word_count": 1061,
    "has_mermaid_diagram": true,
    "has_code_examples": true
  }
}
```

### CPT Training Format
```
<doc>
Documentation content explaining the feature...
</doc>

<code language="rust">
fn example() {
    println!("Hello");
}
</code>

<code language="rust" source="crates/router/src/main.rs">
// Source code from repository
fn main() { ... }
</code>
```

## License

[Add your license here]

## Citation

If you use these scripts, please cite:

```bibtex
@misc{deepwiki-scripts,
  title={DeepWiki Dataset Creation Scripts},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/deepwiki-scripts}
}
```

## Target Dataset

Originally created for: [juspay/hyperswitch](https://github.com/juspay/hyperswitch) documentation and source code.

Optimized for: [Kwaipilot/KAT-Dev](https://huggingface.co/Kwaipilot/KAT-Dev) tokenizer (8k context window).
