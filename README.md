# DeepWiki Scripts - Dataset Creation Pipeline

Tools for creating high-quality training datasets from DeepWiki documentation and Rust codebases.

---

## Key Implementation Differences

### Documentation Dataset Creators

**`create_dataset.py`** (Basic):
- Chunking: Splits by H1/H2/H3 headings using regex
- Context: Single heading section only
- No code extraction
- Simple metadata (heading, level, index)

**`create_enhanced_dataset.py`** (+ Code Mining):
- Chunking: Splits by H2 only (larger chunks)
- Extracts source references: `[filename:start-end]()`
- Clones hyperswitch repo to mine actual code
- Links documentation to implementation
- Rich metadata: source_references, code_snippets

**`create_semantic_dataset.py`** (Larger Semantic):
- Chunking: Splits by `---` (horizontal rules) = logical boundaries
- Preserves complete semantic sections
- Counts: subsections, code blocks, Mermaid diagrams, tables
- Better for RAG (more context per chunk)

**`create_token_aware_dataset.py`** (Most Advanced):
- Uses actual tokenizer (Kwaipilot/KAT-Dev) for precise token counting
- Target distribution: 25% small (<4k), 50% medium (4k-10k), 25% large (10k-16k)
- ~200 token overlap between chunks for continuity
- Tree-sitter AST parsing for call graphs
- Extracts: forward calls, reverse calls, ASCII tree viz, flow descriptions
- Full crate/module context

### Code Dataset Creators

**`create_code_dataset.py`** (Regex-based):
- Parsing: Regex patterns for functions/structs
- 4 granularities: File, Module, Function, Struct/Impl
- 3 token sizes: 4k, 8k, 16k = 12 output files
- No doc comments, no attributes
- Fast but less accurate

**`create_enhanced_code_dataset.py`** (AST-based):
- Parsing: Tree-sitter for proper syntax understanding
- Extracts doc comments (`///` style)
- Detects attributes (`#[derive(...)]`, `#[test]`)
- Tracks impl context (trait impl vs inherent)
- Visibility: pub, pub(crate), private
- Modifiers: async, unsafe, const
- 5 types × 3 sizes = 15 variants

**`file_level_cpt.py`** (Simple CPT):
- File-level only, no granularity variants
- Structured tags: `<path>` metadata + `<file>` content
- Filters: 100 bytes min, 500KB max
- Simple and fast

**`multi_granularity_cpt.py`** (6 Abstraction Levels):
- **File**: Large files (5KB+)
- **Module**: All files in a module combined
- **Crate**: Entire crate if <2MB
- **Impl collection**: All implementations for a type
- **Trait ecosystem**: Trait + all implementors
- **Cross-crate**: Dependency usage patterns
- Analyzes imports, exports, dependencies
- Groups related code intelligently

**`token_aware_multi_gran.py`** (Token-optimized):
- Uses tokenizer for precise 2k-16k token measurement
- 4 strategies:
  1. Large files (2k-16k tokens each)
  2. Combined small files (batched to reach 2k+)
  3. Module clusters (grouped by directory)
  4. Small crates (entire crate if fits)
- Token distribution statistics
- Smart file batching

### Advanced Pattern Mining

**`gen_data.py`** (Most Comprehensive):
- Full repository analysis with Cargo.toml parsing
- Builds crate dependency DAG
- 14+ crate purposes from DeepWiki knowledge
- Tree-sitter AST with full dataclasses
- 6 sample types:
  - **CLM**: Standard next-token prediction
  - **FIM**: `<fim_prefix>...<fim_suffix>...<fim_middle>`
  - **Graph-Augmented**: With call graph prefix
  - **Instruction**: Instruction-tuning format
  - **Contrastive**: Positive/negative pairs
  - **Chain-of-Thought**: Reasoning traces
- Hierarchical context: repo → crate → module → function
- Forward and reverse call graphs

**`pattern_miner.py`** (Pattern-focused):
- Mines 9 specific pattern types:
  1. Serde (derive + attributes)
  2. Diesel (table schemas, enums)
  3. Macros (top 50 with examples)
  4. Validation (validate/check functions)
  5. Async (categorized by purpose)
  6. Error handling (Result types)
  7. Type conversions (From/Into/TryFrom)
  8. Impl patterns (grouped by trait)
  9. Routing (connector implementations)
- NO truncation - preserves full code
- Both CLM and FIM formats
- Top 1000 important functions

**`clean_pattern_miner.py`** (Natural + Auto-upload):
- Natural samples: No "PATTERN: xxx" metadata
- BIG chunks strategy:
  - Full file samples (not just patterns)
  - 100 lines before/after functions for context
  - Module-level samples with multiple functions
- 8 sample generation strategies
- Auto-upload to HuggingFace
- Looks like real repository code

---

## Which to Use?

**For RAG/QA Applications**:
→ `create_token_aware_dataset.py` (token-optimized + call graphs + overlap)

**For Code-Only CPT**:
→ `token_aware_multi_gran.py` (best) or `multi_granularity_cpt.py` (good)

**For Combined Doc+Code CPT**:
→ `create_cpt_training_dataset.py` + `train_combined_cpt.py`

**For Comprehensive Code Understanding**:
→ `gen_data.py` (6 sample types, hierarchical context, call graphs)

**For Pattern-Based Learning**:
→ `clean_pattern_miner.py` (natural samples, big context, auto-upload)

---

## Quick Start

### Scrape DeepWiki Documentation
```bash
python deepwiki.py https://deepwiki.com/juspay/hyperswitch/ ./out --verbose
```

### Create Token-Aware Dataset
```bash
python create_token_aware_dataset.py
```

### Create Multi-Granularity Code Dataset
```bash
python token_aware_multi_gran.py
```

### Train Combined Model
```bash
python train_combined_cpt.py
```

---

See [FILE_DOCUMENTATION.md](FILE_DOCUMENTATION.md) for complete details on all scripts.
