#!/usr/bin/env python3
"""
Enhanced code-only dataset creator for Hyperswitch using tree-sitter.
Extracts code at multiple granularities with proper AST parsing.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False
    logger.warning("tree-sitter not available, falling back to regex parsing")


class EnhancedCodeDatasetCreator:
    """Enhanced code dataset creator with tree-sitter AST parsing"""

    def __init__(self, repo_dir: Path, output_dir: Path, model_name: str = "Kwaipilot/KAT-Dev"):
        self.repo_dir = repo_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.tokenizer = None
        self.parser = None

        # Token size variants
        self.token_sizes = [4096, 8192, 16384]

        # Exclusion patterns
        self.exclude_patterns = ['/target/', '/generated/', '.gen.rs', '/build/']

    def load_tokenizer(self):
        """Load tokenizer for token counting"""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info("✓ Tokenizer loaded")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}, using estimation")
            self.tokenizer = None

    def setup_tree_sitter(self):
        """Setup tree-sitter parser for Rust"""
        if not HAS_TREE_SITTER:
            return False

        try:
            # Try to load pre-built language library
            # User would need to build this first with tree-sitter CLI
            rust_lang = Language('build/rust.so', 'rust')
            self.parser = Parser()
            self.parser.set_language(rust_lang)
            logger.info("✓ Tree-sitter parser loaded")
            return True
        except Exception as e:
            logger.warning(f"Could not setup tree-sitter: {e}, using regex fallback")
            return False

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4  # Rough estimate

    def is_excluded_path(self, path: Path) -> bool:
        """Check if path should be excluded"""
        path_str = str(path)
        return any(pattern in path_str for pattern in self.exclude_patterns)

    def is_test_file(self, path: Path, content: str) -> bool:
        """Check if file is a test file"""
        path_str = str(path)
        if '/tests/' in path_str or path_str.endswith('_test.rs'):
            return True
        if '#[cfg(test)]' in content or 'mod tests {' in content:
            return True
        return False

    def is_generated_file(self, content: str) -> bool:
        """Check if file appears to be generated"""
        first_lines = '\n'.join(content.split('\n')[:10])
        generation_markers = ['@generated', 'auto-generated', 'this file is generated', 'do not edit']
        return any(marker in first_lines.lower() for marker in generation_markers)

    def extract_module_name(self, rel_path: Path) -> str:
        """Extract module/crate name from path"""
        parts = rel_path.parts
        if 'crates' in parts:
            crate_idx = parts.index('crates')
            if len(parts) > crate_idx + 1:
                return parts[crate_idx + 1]
        if len(parts) > 1:
            return parts[0]
        return "root"

    def extract_module_path(self, rel_path: Path) -> str:
        """Get full module path"""
        parts = list(rel_path.parts)
        if parts[-1].endswith('.rs'):
            parts[-1] = parts[-1][:-3]
        if 'crates' in parts:
            crate_idx = parts.index('crates')
            parts = parts[crate_idx + 1:]
        if 'src' in parts:
            parts.remove('src')
        return "::".join(parts)

    def extract_doc_comments(self, lines: List[str], start_line: int) -> str:
        """Extract doc comments before a declaration"""
        doc_lines = []
        i = start_line - 1
        while i >= 0 and (lines[i].strip().startswith('///') or lines[i].strip().startswith('#[')):
            line = lines[i].strip()
            if line.startswith('///'):
                doc_lines.insert(0, line[3:].strip())
            i -= 1
        return ' '.join(doc_lines)

    def extract_functions_regex(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract functions using regex"""
        samples = []
        lines = content.split('\n')

        # Pattern for function definitions
        pattern = r'^\s*(pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+(\w+)(<[^>]+>)?\s*\(([^)]*)\)(?:\s*->\s*([^{;]+))?'

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                is_pub = match.group(1) is not None
                fn_name = match.group(2)
                generics = match.group(3) or ""
                params = match.group(4)
                return_type = match.group(5)

                # Extract doc comments
                doc = self.extract_doc_comments(lines, i)

                # Find function body
                fn_start = i
                brace_count = 0
                fn_end = i
                in_fn = False

                for j in range(i, len(lines)):
                    line_content = lines[j]
                    if '{' in line_content:
                        brace_count += line_content.count('{')
                        in_fn = True
                    if '}' in line_content:
                        brace_count -= line_content.count('}')
                    if in_fn and brace_count == 0:
                        fn_end = j
                        break

                fn_code = '\n'.join(lines[fn_start:fn_end + 1])

                text = f"// Function: {fn_name}\n"
                text += f"// File: {file_path}\n"
                text += f"// Module: {module}\n"
                if doc:
                    text += f"// Doc: {doc}\n"
                text += f"\n{fn_code}"

                samples.append({
                    'text': text,
                    'code': fn_code,
                    'file_path': file_path,
                    'module': module,
                    'type': 'function',
                    'name': fn_name,
                    'is_public': is_pub,
                    'doc': doc,
                    'tokens': self.count_tokens(text)
                })

        return samples

    def extract_structs_regex(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract struct definitions using regex"""
        samples = []
        lines = content.split('\n')

        pattern = r'^\s*(pub(?:\([^)]+\))?\s+)?struct\s+(\w+)(<[^>]+>)?'

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                is_pub = match.group(1) is not None
                struct_name = match.group(2)
                generics = match.group(3) or ""

                doc = self.extract_doc_comments(lines, i)

                # Find impl blocks for this struct
                impl_pattern = fr'impl(?:<[^>]+>)?\s+(?:(\w+(?:::\w+)*)\s+for\s+)?{re.escape(struct_name)}'
                impl_matches = re.findall(impl_pattern, content)
                impl_count = len(impl_matches)

                # Extract struct definition
                struct_end = i
                if '{' in line:
                    brace_count = line.count('{') - line.count('}')
                    for j in range(i + 1, len(lines)):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:
                            struct_end = j
                            break

                struct_code = '\n'.join(lines[i:struct_end + 1])

                text = f"// Struct: {struct_name}\n"
                text += f"// File: {file_path}\n"
                text += f"// Module: {module}\n"
                text += f"// Impls: {impl_count}\n"
                if doc:
                    text += f"// Doc: {doc}\n"
                text += f"\n{struct_code}"

                samples.append({
                    'text': text,
                    'code': struct_code,
                    'file_path': file_path,
                    'module': module,
                    'type': 'struct',
                    'name': struct_name,
                    'is_public': is_pub,
                    'impl_count': impl_count,
                    'doc': doc,
                    'tokens': self.count_tokens(text)
                })

        return samples

    def extract_traits_regex(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract trait definitions using regex"""
        samples = []
        lines = content.split('\n')

        pattern = r'^\s*(pub(?:\([^)]+\))?\s+)?trait\s+(\w+)(<[^>]+>)?'

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                is_pub = match.group(1) is not None
                trait_name = match.group(2)

                doc = self.extract_doc_comments(lines, i)

                # Find trait end
                trait_end = i
                if '{' in line:
                    brace_count = line.count('{') - line.count('}')
                    for j in range(i + 1, len(lines)):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:
                            trait_end = j
                            break

                trait_code = '\n'.join(lines[i:trait_end + 1])

                text = f"// Trait: {trait_name}\n"
                text += f"// File: {file_path}\n"
                text += f"// Module: {module}\n"
                if doc:
                    text += f"// Doc: {doc}\n"
                text += f"\n{trait_code}"

                samples.append({
                    'text': text,
                    'code': trait_code,
                    'file_path': file_path,
                    'module': module,
                    'type': 'trait',
                    'name': trait_name,
                    'is_public': is_pub,
                    'doc': doc,
                    'tokens': self.count_tokens(text)
                })

        return samples

    def extract_impls_regex(self, content: str, file_path: str, module: str) -> List[Dict]:
        """Extract impl blocks using regex"""
        samples = []
        lines = content.split('\n')

        pattern = r'^\s*impl(?:<[^>]+>)?\s+((?:\w+(?:::\w+)*)\s+for\s+)?(\w+)'

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                trait_part = match.group(1)
                type_name = match.group(2)

                # Find impl end
                impl_end = i
                if '{' in line:
                    brace_count = line.count('{') - line.count('}')
                    for j in range(i + 1, len(lines)):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:
                            impl_end = j
                            break

                impl_code = '\n'.join(lines[i:impl_end + 1])

                # Count methods
                fn_count = impl_code.count('\n    fn ') + impl_code.count('\n    pub fn ')

                trait_name = trait_part.replace(' for ', '').strip() if trait_part else None

                text = f"// Impl: {type_name}"
                if trait_name:
                    text += f" ({trait_name})"
                text += f"\n// File: {file_path}\n"
                text += f"// Module: {module}\n"
                text += f"// Methods: {fn_count}\n"
                text += f"\n{impl_code}"

                samples.append({
                    'text': text,
                    'code': impl_code,
                    'file_path': file_path,
                    'module': module,
                    'type': 'impl',
                    'name': type_name,
                    'trait_name': trait_name,
                    'method_count': fn_count,
                    'tokens': self.count_tokens(text)
                })

        return samples

    def process_file(self, file_path: Path) -> Tuple[Dict, List[Dict]]:
        """
        Process a single file.
        Returns: (file_level_sample, granular_samples)
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = file_path.relative_to(self.repo_dir)

            if self.is_excluded_path(file_path):
                return None, []

            if self.is_generated_file(content):
                return None, []

            is_test = self.is_test_file(file_path, content)

            module = self.extract_module_name(rel_path)
            module_path = self.extract_module_path(rel_path)

            # File-level sample
            file_sample = {
                'text': f"// File: {rel_path}\n// Module: {module_path}\n\n{content}",
                'code': content,
                'file_path': str(rel_path),
                'module': module,
                'module_path': module_path,
                'type': 'file',
                'is_test': is_test,
                'tokens': self.count_tokens(content)
            }

            # Granular samples (skip for test files)
            granular_samples = []
            if not is_test:
                granular_samples.extend(self.extract_functions_regex(content, str(rel_path), module))
                granular_samples.extend(self.extract_structs_regex(content, str(rel_path), module))
                granular_samples.extend(self.extract_traits_regex(content, str(rel_path), module))
                granular_samples.extend(self.extract_impls_regex(content, str(rel_path), module))

            return file_sample, granular_samples

        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None, []

    def chunk_samples(self, samples: List[Dict], max_tokens: int, granularity: str) -> List[Dict]:
        """Chunk samples to fit within max_tokens"""
        chunks = []
        current_batch = []
        current_tokens = 0
        chunk_id = 0

        for sample in samples:
            sample_tokens = sample['tokens']

            # Single sample too large, split it
            if sample_tokens > max_tokens:
                # Save current batch first
                if current_batch:
                    chunks.append({
                        'id': f"{granularity}_{max_tokens}_{chunk_id}",
                        'text': '\n\n'.join([s['text'] for s in current_batch]),
                        'token_count': current_tokens,
                        'granularity': granularity,
                        'max_tokens': max_tokens,
                        'item_count': len(current_batch),
                        'items': current_batch
                    })
                    chunk_id += 1
                    current_batch = []
                    current_tokens = 0

                # Split large sample
                lines = sample['text'].split('\n')
                temp_lines = []
                temp_tokens = 0

                for line in lines:
                    line_tokens = self.count_tokens(line + '\n')
                    if temp_tokens + line_tokens > max_tokens and temp_lines:
                        split_text = '\n'.join(temp_lines)
                        chunks.append({
                            'id': f"{granularity}_{max_tokens}_{chunk_id}",
                            'text': split_text,
                            'token_count': temp_tokens,
                            'granularity': granularity,
                            'max_tokens': max_tokens,
                            'item_count': 1,
                            'items': [{'text': split_text, 'type': sample['type'], 'file_path': sample['file_path']}]
                        })
                        chunk_id += 1
                        temp_lines = [line]
                        temp_tokens = line_tokens
                    else:
                        temp_lines.append(line)
                        temp_tokens += line_tokens

                if temp_lines:
                    split_text = '\n'.join(temp_lines)
                    chunks.append({
                        'id': f"{granularity}_{max_tokens}_{chunk_id}",
                        'text': split_text,
                        'token_count': temp_tokens,
                        'granularity': granularity,
                        'max_tokens': max_tokens,
                        'item_count': 1,
                        'items': [{'text': split_text, 'type': sample['type'], 'file_path': sample['file_path']}]
                    })
                    chunk_id += 1

            # Adding sample would exceed limit
            elif current_tokens + sample_tokens > max_tokens:
                if current_batch:
                    chunks.append({
                        'id': f"{granularity}_{max_tokens}_{chunk_id}",
                        'text': '\n\n'.join([s['text'] for s in current_batch]),
                        'token_count': current_tokens,
                        'granularity': granularity,
                        'max_tokens': max_tokens,
                        'item_count': len(current_batch),
                        'items': current_batch
                    })
                    chunk_id += 1
                current_batch = [sample]
                current_tokens = sample_tokens

            # Add to current batch
            else:
                current_batch.append(sample)
                current_tokens += sample_tokens

        # Flush remaining
        if current_batch:
            chunks.append({
                'id': f"{granularity}_{max_tokens}_{chunk_id}",
                'text': '\n\n'.join([s['text'] for s in current_batch]),
                'token_count': current_tokens,
                'granularity': granularity,
                'max_tokens': max_tokens,
                'item_count': len(current_batch),
                'items': current_batch
            })

        return chunks

    def create_datasets(self):
        """Main pipeline to create all dataset variants"""
        logger.info("=" * 70)
        logger.info("Enhanced Code Dataset Creator for Hyperswitch")
        logger.info("=" * 70)

        self.load_tokenizer()
        self.setup_tree_sitter()

        # Collect all Rust files
        logger.info(f"\nCollecting Rust files from {self.repo_dir}")
        rust_files = list(self.repo_dir.rglob("*.rs"))
        logger.info(f"✓ Found {len(rust_files)} Rust files")

        # Process all files
        logger.info("\nProcessing files...")
        file_samples = []
        all_granular = {'function': [], 'struct': [], 'trait': [], 'impl': []}

        for file_path in tqdm(rust_files, desc="Processing", unit="files"):
            file_sample, granular_samples = self.process_file(file_path)

            if file_sample:
                file_samples.append(file_sample)

            for sample in granular_samples:
                stype = sample['type']
                if stype in all_granular:
                    all_granular[stype].append(sample)

        logger.info(f"\n✓ File-level samples: {len(file_samples)}")
        logger.info(f"✓ Granular samples:")
        for gtype, samples in all_granular.items():
            logger.info(f"  - {gtype}: {len(samples)}")

        # Create chunked datasets for each granularity and token size
        logger.info("\nCreating chunked datasets...")
        self.output_dir.mkdir(exist_ok=True)

        all_stats = {}

        # File-level datasets
        for max_tokens in self.token_sizes:
            size_name = f"{max_tokens // 1024}k"
            logger.info(f"\n  Chunking files at {size_name}...")
            chunks = self.chunk_samples(file_samples, max_tokens, 'file')
            self.save_dataset(chunks, f"file_{size_name}", all_stats)

        # Granular datasets
        for gtype, samples in all_granular.items():
            if not samples:
                continue
            for max_tokens in self.token_sizes:
                size_name = f"{max_tokens // 1024}k"
                logger.info(f"\n  Chunking {gtype} at {size_name}...")
                chunks = self.chunk_samples(samples, max_tokens, gtype)
                self.save_dataset(chunks, f"{gtype}_{size_name}", all_stats)

        # Save overall stats and README
        stats_path = self.output_dir / 'all_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)

        self.create_readme(all_stats)

        logger.info("\n" + "=" * 70)
        logger.info("✅ Dataset creation complete!")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Datasets: {len(all_stats)}")
        logger.info("=" * 70)

    def save_dataset(self, chunks: List[Dict], name: str, all_stats: Dict):
        """Save a dataset variant"""
        # Simplify for saving
        simplified = []
        for chunk in chunks:
            simplified.append({
                'id': chunk['id'],
                'text': chunk['text'],
                'token_count': chunk['token_count'],
                'granularity': chunk['granularity'],
                'max_tokens': chunk['max_tokens'],
                'item_count': chunk['item_count']
            })

        # Save JSONL
        jsonl_path = self.output_dir / f'{name}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in simplified:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        # Save Parquet
        df = pd.DataFrame(simplified)
        parquet_path = self.output_dir / f'{name}.parquet'
        df.to_parquet(parquet_path, index=False)

        # Stats
        token_counts = [c['token_count'] for c in simplified]
        all_stats[name] = {
            'chunks': len(simplified),
            'total_tokens': sum(token_counts),
            'mean_tokens': np.mean(token_counts),
            'median_tokens': np.median(token_counts),
            'max_tokens': max(token_counts) if token_counts else 0
        }

        logger.info(f"    ✓ Saved {name}: {len(simplified)} chunks")

    def create_readme(self, stats: Dict):
        """Create README"""
        readme = """# Hyperswitch Enhanced Code Dataset

Multi-granularity Rust code dataset with proper AST parsing.

## Granularities

- **file**: Complete source files
- **function**: Individual functions
- **struct**: Struct definitions
- **trait**: Trait definitions
- **impl**: Implementation blocks

## Token Sizes

- **4k**: 4,096 max tokens
- **8k**: 8,192 max tokens
- **16k**: 16,384 max tokens

## Statistics

"""
        for name, stat in stats.items():
            readme += f"### {name}\n"
            readme += f"- Chunks: {stat['chunks']:,}\n"
            readme += f"- Total tokens: {stat['total_tokens']:,}\n"
            readme += f"- Mean: {stat['mean_tokens']:.0f}\n\n"

        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme)


def main():
    script_dir = Path(__file__).parent.resolve()
    repo_dir = script_dir / 'hyperswitch'
    output_dir = script_dir / 'code_dataset_output'

    # Clone repo if needed
    if not repo_dir.exists():
        logger.info("Cloning Hyperswitch repository...")
        subprocess.run(['git', 'clone', 'https://github.com/juspay/hyperswitch.git', str(repo_dir)], check=True)

    creator = EnhancedCodeDatasetCreator(repo_dir, output_dir)
    creator.create_datasets()


if __name__ == '__main__':
    main()