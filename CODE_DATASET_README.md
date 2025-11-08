# Hyperswitch Code Dataset Creation Guide

This directory contains scripts to create comprehensive code datasets from the Hyperswitch Rust repository.

## Available Scripts

### 1. **Enhanced Code Dataset** (`create_enhanced_code_dataset.py`)
Creates multi-granularity code datasets with proper AST parsing.

**Features:**
- ✅ Tree-sitter AST parsing (with regex fallback)
- ✅ Multiple granularities: file, function, struct, trait, impl
- ✅ Doc comment extraction
- ✅ Filters test and generated files
- ✅ Multiple token sizes: 4k, 8k, 16k
- ✅ Rich metadata for each sample

**Outputs 15 datasets:**
- `file_4k.parquet`, `file_8k.parquet`, `file_16k.parquet`
- `function_4k.parquet`, `function_8k.parquet`, `function_16k.parquet`
- `struct_4k.parquet`, `struct_8k.parquet`, `struct_16k.parquet`
- `trait_4k.parquet`, `trait_8k.parquet`, `trait_16k.parquet`
- `impl_4k.parquet`, `impl_8k.parquet`, `impl_16k.parquet`

### 2. **CPT Training Script** (`train_cpt_lora.py`)
LoRA fine-tuning with on-the-fly CPT formatting.

**Features:**
- ✅ Loads `archit11/deepwiki-16k` from Hugging Face
- ✅ Formats with `<doc>` and `<code>` tags dynamically
- ✅ LoRA configuration optimized for 4x H200
- ✅ Supports 16k context
- ✅ bf16 + gradient checkpointing

### 3. **Dataset Upload** (`push_to_hf.py`)
Pushes datasets to Hugging Face Hub.

## Quick Start

### Option 1: Using Tree-sitter (Recommended)

```bash
# Setup tree-sitter
chmod +x setup_treesitter.sh
./setup_treesitter.sh

# Create enhanced code datasets
uv run python create_enhanced_code_dataset.py
```

### Option 2: Without Tree-sitter (Regex fallback)

```bash
# Just run the script - it will use regex parsing
uv run python create_enhanced_code_dataset.py
```

## Setup Details

### Install Dependencies

```bash
# Using uv (recommended)
uv pip install transformers datasets pandas pyarrow tqdm

# Optional: for tree-sitter
uv pip install tree-sitter
```

### Tree-sitter Setup (Optional but Better)

Tree-sitter provides more accurate AST parsing than regex:

```bash
# Install tree-sitter
uv pip install tree-sitter

# Clone Rust grammar
git clone https://github.com/tree-sitter/tree-sitter-rust

# Build language library
python3 -c "from tree_sitter import Language; Language.build_library('build/rust.so', ['tree-sitter-rust'])"
```

Or use the provided script:

```bash
chmod +x setup_treesitter.sh
./setup_treesitter.sh
```

## Usage Examples

### Create Code Datasets

```bash
# Create all granularities at all token sizes
uv run python create_enhanced_code_dataset.py
```

Output: `code_dataset_output/` directory with 15 parquet files.

### Train with CPT Formatting

```bash
# Start LoRA training
uv run python train_cpt_lora.py
```

### Push Dataset to Hugging Face

```bash
# Upload to HF Hub
uv run python push_to_hf.py archit11/hyperswitch-code-16k
```

## Dataset Structure

Each dataset chunk contains:

```json
{
  "id": "function_8k_42",
  "text": "// Function: process_payment\n// File: crates/router/src/core/payments.rs\n// Module: router\n\npub async fn process_payment(...) { ... }",
  "token_count": 2048,
  "granularity": "function",
  "max_tokens": 8192,
  "item_count": 3
}
```

## Granularity Levels

### File Level
Complete Rust source files with module context.

**Use case:** Understanding full file structure, learning file-level patterns.

### Function Level
Individual function definitions with documentation.

**Use case:** Learning function signatures, API patterns, function-level code understanding.

### Struct Level
Struct definitions with impl block counts and trait info.

**Use case:** Understanding data structures, type relationships.

### Trait Level
Trait definitions with bounds and requirements.

**Use case:** Learning Rust traits, interface patterns.

### Impl Level
Implementation blocks with method counts.

**Use case:** Understanding type implementations, method organization.

## Token Size Variants

- **4k**: Best for quick iterations, lower memory usage
- **8k**: Balanced for most use cases
- **16k**: Maximum context, captures larger code units

## Training Workflow

1. **Create code datasets:**
   ```bash
   uv run python create_enhanced_code_dataset.py
   ```

2. **Upload to Hugging Face:**
   ```bash
   uv run python push_to_hf.py archit11/hyperswitch-code-8k
   ```

3. **Train with LoRA:**
   ```bash
   uv run python train_cpt_lora.py
   ```

## Customization

### Change Token Sizes

Edit `create_enhanced_code_dataset.py`:

```python
self.token_sizes = [2048, 4096, 8192]  # Add/remove sizes
```

### Filter Specific File Types

Edit exclusion patterns:

```python
self.exclude_patterns = ['/target/', '/tests/', '/benches/']
```

### Include Test Files

Modify the `is_test_file` check or skip the test filter in `process_file`.

## Performance Tips

- **Tree-sitter**: 2-3x faster than regex for large repos
- **Token estimation**: Use tokenizer for accuracy, but it's slower
- **Parallel processing**: Can be added with multiprocessing if needed

## Troubleshooting

### Tree-sitter Build Fails

Fall back to regex parsing - the script handles this automatically.

### Out of Memory

- Reduce `token_sizes` to fewer variants
- Process in batches by modifying the script

### Missing Dependencies

```bash
uv pip install transformers datasets pandas pyarrow tqdm tree-sitter
```

## Output Statistics

The script generates:
- Individual parquet files for each granularity/size combo
- `all_stats.json` with token distribution stats
- `README.md` with dataset overview

## Next Steps

After creating datasets:

1. Upload to Hugging Face
2. Use in training scripts
3. Experiment with different granularities
4. Mix granularities for varied training data

## Questions?

Check the code comments in each script for detailed implementation notes.