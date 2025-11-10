# DeepWiki Scripts - File Documentation

Comprehensive documentation explaining what each file does, how it generates datasets, and key differences.

---

## Table of Contents

1. [Data Collection Scripts](#1-data-collection-scripts)
2. [Documentation Dataset Creators](#2-documentation-dataset-creators)
3. [Code Dataset Creators](#3-code-dataset-creators)
4. [Training Dataset Formatters](#4-training-dataset-formatters)
5. [Training Scripts](#5-training-scripts)
6. [Upload Utilities](#6-upload-utilities)
7. [Advanced Pattern Mining Scripts](#7-advanced-pattern-mining-scripts)
8. [Setup Scripts](#8-setup-scripts)
9. [Documentation Files](#9-documentation-files)
10. [Comparison Matrix](#10-comparison-matrix)

---

> **Note**: For key implementation differences between scripts, see [README.md](README.md)

---

## 1. Data Collection Scripts

### `deepwiki.py`
**Purpose**: Scrapes documentation from DeepWiki websites using Playwright browser automation.

**How it works**:
- Opens a DeepWiki URL in a headless browser
- Hovers over sidebar links to trigger React Server Component (RSC) fetches
- Captures `?rsc=` requests containing documentation content
- Extracts markdown from `text/x-component` responses
- Saves each unique document as a separate `.md` file

**Output**:
- Raw markdown files in `./out/` directory
- Example: `hyperswitch-0.md`, `hyperswitch-1.md`, etc.

**Usage**:
```bash
python3 deepwiki.py https://deepwiki.com/juspay/hyperswitch/ ./out --verbose
```

---

## 2. Documentation Dataset Creators

### `create_dataset.py`
**Purpose**: Basic dataset creator that chunks documentation by markdown headings.

**Chunking strategy**:
- Splits content by level 1-3 headings (`#`, `##`, `###`)
- Each chunk = one heading section
- Includes metadata: heading, heading_level, chunk_index

**Output format**:
- `dataset.jsonl` - JSON Lines format
- `dataset.parquet` - Parquet format
- `dataset_stats.json` - Statistics

**Key difference**: Simple heading-based chunking, no code extraction.

---

### `create_enhanced_dataset.py`
**Purpose**: Enhanced dataset with source code mining from the repository.

**Chunking strategy**:
- Chunks by level 2 headings only (`##`) for larger context
- Extracts source references in format `[filename:start-end]()`
- Mines actual code from the hyperswitch repository
- Includes both documentation and referenced source code

**Key features**:
- Clones/updates hyperswitch repository
- Extracts code snippets with line ranges
- Links documentation to actual implementation

**Output structure**:
```json
{
  "id": "source_file_0",
  "heading": "Payment Processing",
  "content": "markdown content...",
  "source_references": [{"file_path": "router/src/core.rs", "start_line": 10, "end_line": 50}],
  "source_code_snippets": [{"file_path": "...", "code": "actual code..."}]
}
```

**Key difference**: Includes actual source code linked to documentation.

---

### `create_semantic_dataset.py`
**Purpose**: Semantic chunking with larger, more coherent sections.

**Chunking strategy**:
- Splits by horizontal rules (`---`) which mark logical boundaries
- Preserves complete semantic sections
- Larger chunks (more context per sample)
- Rich metadata: word count, subsections, diagrams, tables

**Key features**:
- Counts subsections, code blocks, Mermaid diagrams
- Calculates content richness metrics
- Optimized for RAG and QA applications

**Output structure**:
```json
{
  "metadata": {
    "word_count": 1500,
    "subsection_count": 5,
    "code_block_count": 3,
    "has_mermaid_diagram": true,
    "has_tables": false
  }
}
```

**Key difference**: Larger semantic chunks vs. heading-based chunks.

---

### `create_token_aware_dataset.py`
**Purpose**: Most advanced documentation dataset with token optimization and call graphs.

**Chunking strategy**:
- Token-aware: Uses Kwaipilot/KAT-Dev tokenizer
- Target distribution: 25% small (<4k), 50% medium (4k-10k), 25% large (10k-16k)
- ~200 token overlap between adjacent chunks
- Semantic boundaries (---) respected

**Key features**:
1. **Token optimization**: Precise token counting with actual tokenizer
2. **Call graph extraction**: Tree-sitter AST parsing for function calls
3. **Bidirectional graphs**: Shows what function calls AND what calls it
4. **ASCII tree visualizations**: Visual call graph representations
5. **Natural language flow descriptions**: Plain text function relationships
6. **Crate/module context**: Full Rust workspace structure

**Output structure**:
```json
{
  "token_stats": {
    "total_tokens": 8192,
    "content_tokens": 5000,
    "code_tokens": 3192,
    "compression_ratio": 3.2
  },
  "call_graph_data": [{
    "function_name": "process_payment",
    "calls": [...],
    "called_by": [...],
    "ascii_tree": "process_payment\nCalls:\n├─→ validate_request\n└─→ execute_payment",
    "flow_description": "process_payment is a public async function..."
  }],
  "metadata": {
    "has_overlap": true
  }
}
```

**Key difference**: Token-optimized with advanced call graph analysis and context overlap.

---

## 3. Code Dataset Creators

### `create_code_dataset.py`
**Purpose**: Code-only dataset at multiple granularities using regex parsing.

**Granularities**:
- **File level**: Complete source files
- **Module level**: Grouped by crate/module
- **Function level**: Individual functions
- **Struct/Impl level**: Struct definitions and impl blocks

**Token sizes**: 4k, 8k, 16k variants

**Parsing method**: Regex patterns

**Output**: Multiple datasets like `file_4k.jsonl`, `function_8k.jsonl`, etc.

**Key difference**: Multiple granularity levels, regex-based extraction.

---

### `create_enhanced_code_dataset.py`
**Purpose**: Enhanced code dataset with tree-sitter AST parsing.

**Key improvements over create_code_dataset.py**:
- **Tree-sitter AST parsing**: Proper syntax understanding
- **Doc comment extraction**: `///` style documentation
- **Attribute detection**: `#[derive(...)]`, `#[test]`, etc.
- **Impl context**: Knows if function is in trait impl
- **Visibility tracking**: pub, pub(crate), private
- **Function modifiers**: async, unsafe, const detection

**Granularities**:
- File, function, struct, trait, impl blocks

**Output structure**:
```json
{
  "text": "// Function: process_payment\n// File: router/core.rs\n// Module: router::core\n// Doc: Processes payment requests\n\npub async fn process_payment(...) { ... }",
  "type": "function",
  "name": "process_payment",
  "is_public": true,
  "is_async": true,
  "doc": "Processes payment requests"
}
```

**Key difference**: AST-based parsing with rich metadata vs. regex.

---

### `create_and_upload_code_dataset.py`
**Purpose**: Simplified code dataset creator that uploads directly to Hugging Face.

**Strategy**:
- File-level only (no granularity variants)
- Simple format: `<path>` metadata + full file content
- Excludes tests, target, generated files
- Immediate upload to HF Hub

**Format**:
```
// File: crates/router/src/core.rs
// Module: router::core

[full file content]
```

**Key difference**: Simplified, single-granularity, with auto-upload.

---

### `file_level_cpt.py`
**Purpose**: File-level CPT dataset with structured tags.

**Format**:
```
<path>
Repository: hyperswitch
Crate: router
File: crates/router/src/core.rs
Size: 15000 bytes
</path>

<file>
[file content]
</file>
```

**Filters**:
- Min size: 100 bytes
- Max size: 500KB
- Excludes very small and very large files

**Key difference**: Structured `<path>` and `<file>` tags for CPT training.

---

### `multi_granularity_cpt.py`
**Purpose**: Most comprehensive multi-granularity CPT dataset.

**Granularities**:
1. **File**: Large files (5KB+)
2. **Module**: All files in a module combined
3. **Crate**: Entire crate (if <2MB)
4. **Impl collection**: All implementations for a type
5. **Trait ecosystem**: Trait + all implementors
6. **Cross-crate**: Dependency usage patterns

**Key features**:
- Analyzes repository structure
- Tracks imports, exports, dependencies
- Groups related code intelligently

**Example outputs**:
```
<impl_collection>
Type: PaymentRequest
Implementations: 5
[all impl blocks for PaymentRequest]
</impl_collection>

<trait_ecosystem>
Trait: Processor
Implementors: 12
[all trait implementations]
</trait_ecosystem>
```

**Key difference**: 6 different abstraction levels, relationship-aware.

---

### `token_aware_multi_gran.py`
**Purpose**: Token-aware version of multi-granularity dataset.

**Strategy**:
- Uses tokenizer to measure samples (2k-16k tokens)
- 4 strategies:
  1. Large files (2k-16k tokens each)
  2. Combined small files (batched to reach 2k+)
  3. Module clusters (grouped by directory)
  4. Small crates (entire crate if fits)

**Key features**:
- Precise token counting
- Smart file batching
- Token distribution statistics

**Key difference**: Token-optimized granularity selection vs. fixed strategies.

---

## 4. Training Dataset Formatters

### `create_cpt_training_dataset.py`
**Purpose**: Transforms chunked datasets into CPT training format.

**Formats supported**:
1. **Interleaved**: Doc and code naturally mixed
2. **Separate**: All docs first, then all code
3. **Doc-code pairs**: Explicit pairing

**Tag structure**:
```
<doc>
Documentation content...
</doc>

<code language="rust">
fn example() { ... }
</code>

<code language="rust" source="crates/router/src/core.rs">
[source code with file reference]
</code>
```

**Features**:
- Extracts inline code blocks from markdown
- Adds source code snippets
- Token limit: 8192 per sample
- May split large chunks

**Outputs**:
- `training_data_interleaved.jsonl`
- `training_data_separate.jsonl`
- `training_data_doc_code_pairs.jsonl`
- `.txt` versions for simple training
- `.parquet` versions

**Key difference**: Adds `<doc>` and `<code>` tags for structured CPT.

---

## 5. Training Scripts

### `train_cpt_lora.py`
**Purpose**: LoRA fine-tuning script for documentation dataset.

**Configuration**:
- Model: Kwaipilot/KAT-Dev
- Dataset: archit11/deepwiki-16k
- Max sequence length: 16384
- LoRA r=16, alpha=32, dropout=0.05

**Key features**:
- Adds special tokens: `<doc>`, `</doc>`, `<code>`, `</code>`
- On-the-fly CPT formatting
- Gradient checkpointing
- BF16 precision
- Multi-GPU support

**Key difference**: Documentation-only training.

---

### `train_combined_cpt.py`
**Purpose**: Combined training with documentation AND code datasets.

**Datasets**:
- Documentation: archit11/deepwiki-16k (50%)
- Code: archit11/hyperswitch-code (50%)

**Key features**:
- Interleaves both datasets
- Formats docs with `<doc>` tags
- Formats code with `<code>` tags
- Balanced training on both modalities

**Key difference**: Multi-modal training (docs + code) vs. single-modal.

---

## 6. Upload Utilities

### `push_to_hf.py`
**Purpose**: Upload token-aware dataset to Hugging Face Hub.

**Features**:
- Uploads from `token_aware_dataset_output/`
- Includes README.md, dataset_stats.json, token_distribution.json
- Public dataset

**Usage**:
```bash
python push_to_hf.py archit11/deepwiki-16k
```

---

### `upload_to_hf.py`
**Purpose**: Upload token-aware multi-granularity dataset.

**Features**:
- Generates comprehensive README
- Includes token statistics
- Uploads to archit11/hyperswitch-token-aware-cpt-fixed

---

## 7. Advanced Pattern Mining Scripts

### `gen_data.py`
**Purpose**: **MOST COMPREHENSIVE** - Repository-level CPT dataset generator with tree-sitter AST parsing.

**Key features**:
1. **Full Repository Analysis**:
   - Discovers all crates with Cargo.toml parsing
   - Builds crate dependency DAG
   - Analyzes module structure
   - Extracts 14+ crate purposes from DeepWiki knowledge

2. **Tree-sitter AST Parsing**:
   - Proper syntax understanding (not regex)
   - Extracts functions with full context (visibility, async, unsafe, const)
   - Extracts traits with super traits and methods
   - Extracts impl blocks (both trait and inherent)
   - Doc comment extraction (`///`, `//!`)
   - Attribute detection (`#[...]`)

3. **Multiple Sample Types**:
   - **CLM** (Causal Language Modeling): Standard next-token prediction
   - **FIM** (Fill-in-Middle): `<fim_prefix>...<fim_suffix>...<fim_middle>`
   - **Graph-Augmented**: With call graph prefix showing relationships
   - **Instruction**: Instruction-tuning format with input/output
   - **Contrastive**: Positive/negative pairs (structure defined)
   - **Chain-of-Thought**: Reasoning traces (structure defined)

4. **Granularities**:
   - Repository overview (crate dependency graph)
   - Crate-level (purpose, dependencies, public API)
   - Module-level (imports, exports, types)
   - Function-level (with hierarchical context)
   - Trait-level (with implementations)
   - Pattern-level (connector integrations)

5. **Call Graph Analysis**:
   - Forward call graph (what function calls)
   - Reverse call graph (who calls this function)
   - Call graph visualization in samples

**Output structure**:
```json
{
  "id": "clm_router_process_payment",
  "type": "clm",
  "granularity": "function",
  "hierarchy": {
    "repository": "hyperswitch",
    "crate": "router",
    "module": "core::payments",
    "function": "process_payment"
  },
  "content": "// REPO: hyperswitch\n// CRATE: router\n// CRATE_PURPOSE: Main application server...\n\n[function code]",
  "metadata": {
    "is_pub": true,
    "is_async": true,
    "impl_context": {...}
  }
}
```

**Key difference**: Most sophisticated - tree-sitter AST parsing, multiple sample formats (CLM/FIM/Graph/Instruction), hierarchical context, full repository understanding.

---

### `pattern_miner.py`
**Purpose**: Advanced pattern mining with NO truncation - mines ALL patterns from codebase.

**Pattern types mined**:
1. **Serde Patterns**:
   - Derive combinations (#[derive(...)])
   - Serde attributes (#[serde(...)])
   - Full struct/enum definitions with attributes

2. **Diesel Patterns**:
   - `diesel::table!` macro schemas
   - `#[diesel_enum]` patterns with storage types

3. **Macro Usage**:
   - All macro invocations across codebase
   - Frequency counts
   - Top 50 macros with 10 examples each

4. **Validation Patterns**:
   - All functions with 'validate' or 'check' in name
   - Full function bodies

5. **Async Patterns**:
   - Categorized by purpose: database_query, database_mutation, connector_call, redis_operation
   - Full async function implementations

6. **Error Handling**:
   - All Result return types
   - Error type frequencies
   - Full error handling implementations

7. **Type Conversions**:
   - From/Into/TryFrom/TryInto implementations
   - Layer transitions (api → domain → database)

8. **Impl Patterns**:
   - All trait implementations
   - Grouped by trait name

9. **Routing Patterns**:
   - Connector implementations
   - Routing function patterns

**Sample generation**:
- Generates samples for each mined pattern
- Both CLM and FIM formats
- Full code (NO truncation!)
- Top 1000 important functions
- Pattern metadata preserved

**Key difference**: Pattern-based approach, mines actual usage patterns from codebase, full samples with no truncation.

---

### `clean_pattern_miner.py`
**Purpose**: Clean version of pattern mining with NATURAL samples and HuggingFace auto-upload.

**Key features**:
1. **Natural Samples** (vs. artificial pattern comments):
   - No "PATTERN: xxx" metadata in content
   - Real code comments only
   - Looks like natural repository code

2. **BIG Chunks**:
   - Full file samples (not just patterns)
   - Surrounding context (100 lines before/after functions)
   - Module-level samples with multiple functions

3. **Sample strategies**:
   - **Type samples**: Full files with all structs/enums
   - **Diesel samples**: Full files with table schemas
   - **Function samples**: BIG chunks with surrounding code
   - **Async samples**: Full context (100 lines before/after)
   - **Conversion samples**: Full files with conversions
   - **Large file samples**: Important crates (router, api_models, etc.)

4. **Auto-upload to HuggingFace**:
   - Integrated upload after generation
   - Uploads to `archit11/hyperswitch-code-dataset`

**Key difference**: Natural-looking code (no pattern metadata), large context windows, auto-upload integration.

---

## 8. Setup Scripts

### `setup_treesitter.sh`
**Purpose**: Setup script for tree-sitter Rust grammar.

**What it does**:
1. Installs tree-sitter Python package via uv
2. Clones tree-sitter-rust grammar repository
3. Builds Rust language library (`build/rust.so`)

**Usage**:
```bash
chmod +x setup_treesitter.sh
./setup_treesitter.sh
```

**Required for**:
- `create_enhanced_code_dataset.py`
- `gen_data.py`
- `pattern_miner.py`
- `clean_pattern_miner.py`
- `create_token_aware_dataset.py` (call graph features)

---

## 9. Documentation Files

### `CODE_DATASET_README.md`
**Purpose**: Comprehensive guide for code dataset creation.

**Contents**:
- Available scripts overview
- Quick start guides (with/without tree-sitter)
- Setup instructions
- Usage examples
- Granularity level explanations
- Token size variants
- Training workflow
- Customization options
- Troubleshooting

**Key sections**:
- Setup tree-sitter for better AST parsing
- Granularity levels (file, function, struct, trait, impl)
- Token sizes (4k, 8k, 16k)
- Training workflow from creation → upload → training

---

## 10. Comparison Matrix

| Script | Input | Chunking Strategy | Token-Aware | Code Extraction | Call Graphs | Output Granularities |
|--------|-------|-------------------|-------------|-----------------|-------------|---------------------|
| `create_dataset.py` | Docs | Headings (1-3) | ❌ | ❌ | ❌ | 1 |
| `create_enhanced_dataset.py` | Docs | Headings (2 only) | ❌ | ✅ (basic) | ❌ | 1 |
| `create_semantic_dataset.py` | Docs | Semantic (---) | ❌ | ✅ (basic) | ❌ | 1 |
| `create_token_aware_dataset.py` | Docs | Token-optimized | ✅ | ✅ (advanced) | ✅ | 1 with overlap |
| `create_code_dataset.py` | Code | Multiple | ✅ | N/A | ❌ | 12 (4 levels × 3 sizes) |
| `create_enhanced_code_dataset.py` | Code | AST-based | ✅ | N/A | ❌ | 15 (5 types × 3 sizes) |
| `create_and_upload_code_dataset.py` | Code | File-level | ✅ | N/A | ❌ | 1 |
| `file_level_cpt.py` | Code | File-level | ❌ | N/A | ❌ | 1 |
| `multi_granularity_cpt.py` | Code | 6 strategies | ❌ | N/A | ❌ | 6 |
| `token_aware_multi_gran.py` | Code | 4 strategies | ✅ | N/A | ❌ | 4 |

---

## Key Differentiators

### Documentation Dataset Evolution:
1. **Basic** (`create_dataset.py`): Simple heading-based chunking
2. **Enhanced** (`create_enhanced_dataset.py`): + Source code references
3. **Semantic** (`create_semantic_dataset.py`): + Larger semantic chunks
4. **Token-Aware** (`create_token_aware_dataset.py`): + Token optimization + Call graphs + Overlap

### Code Dataset Evolution:
1. **Basic Multi-Granularity** (`create_code_dataset.py`): Regex parsing, 4 granularities
2. **AST-Enhanced** (`create_enhanced_code_dataset.py`): Tree-sitter parsing, rich metadata
3. **Simple Upload** (`create_and_upload_code_dataset.py`): File-level only, auto-upload
4. **CPT File-Level** (`file_level_cpt.py`): Structured tags
5. **CPT Multi-Granularity** (`multi_granularity_cpt.py`): 6 abstraction levels
6. **Token-Aware CPT** (`token_aware_multi_gran.py`): Token-optimized batching

### Training Dataset Types:
1. **CPT Formatter** (`create_cpt_training_dataset.py`): Adds `<doc>` and `<code>` tags
2. **Doc-Only Training** (`train_cpt_lora.py`): Documentation dataset
3. **Combined Training** (`train_combined_cpt.py`): Docs + Code interleaved

---

## Recommended Usage

### For RAG/QA Applications:
→ Use `create_token_aware_dataset.py`
- Token-optimized chunks
- Call graph context
- Overlap for continuity

### For Code-Only CPT:
→ Use `token_aware_multi_gran.py` or `multi_granularity_cpt.py`
- Multiple abstraction levels
- Token-aware batching
- Relationship preservation

### For Combined Doc+Code CPT:
→ Use `create_cpt_training_dataset.py` + `train_combined_cpt.py`
- Structured tags
- Multi-modal training
- Balanced datasets

### For Quick Upload to HF:
→ Use `create_and_upload_code_dataset.py`
- Simplified process
- Auto-upload
- No local storage needed

---

## Pipeline Examples

### Full Documentation Dataset Pipeline:
```bash
# 1. Scrape documentation
python deepwiki.py https://deepwiki.com/juspay/hyperswitch/ ./out

# 2. Create token-aware dataset
python create_token_aware_dataset.py

# 3. Upload to HuggingFace
python push_to_hf.py archit11/deepwiki-16k
```

### Full Code Dataset Pipeline:
```bash
# 1. Create token-aware multi-granularity dataset
python token_aware_multi_gran.py

# 2. Upload to HuggingFace
python upload_to_hf.py
```

### Combined Training Pipeline:
```bash
# 1. Create doc dataset
python create_token_aware_dataset.py

# 2. Create code dataset
python create_and_upload_code_dataset.py

# 3. Train with both
python train_combined_cpt.py
```

---

## Summary

This project provides a comprehensive suite of tools for creating, formatting, and training on documentation and code datasets. The evolution from basic chunking to token-aware, call-graph-enhanced datasets shows progressive sophistication in dataset quality and training effectiveness.

Choose the appropriate script based on your specific needs:
- **Simplicity**: `create_dataset.py`, `file_level_cpt.py`
- **Quality**: `create_token_aware_dataset.py`, `multi_granularity_cpt.py`
- **Training**: `train_cpt_lora.py`, `train_combined_cpt.py`
- **Deployment**: `push_to_hf.py`, `upload_to_hf.py`
