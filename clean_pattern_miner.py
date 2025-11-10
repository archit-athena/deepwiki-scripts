#!/usr/bin/env python3
"""
CLEAN PATTERN MINING: Mine patterns, generate NATURAL samples
NO artificial pattern comments - only natural code context
NO truncation - full samples
INCLUDES: Hugging Face upload
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

from tree_sitter import Language, Parser, Node
import tree_sitter_rust

import sys
sys.path.insert(0, str(Path(__file__).parent))
from gen_data import RepositoryAnalyzer


# ============================================================================
# PATTERN MINERS (Same as before)
# ============================================================================

class SerdePatternMiner:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        print("  Mining serde patterns...")
        for crate_name, crate_info in self.analyzer.crates.items():
            rs_files = list(crate_info.path.rglob('*.rs'))
            for rs_file in rs_files:
                root, code = self.analyzer.analyzer.parse_file(rs_file)
                if not root:
                    continue
                self._mine_structs(root, code, crate_name, str(rs_file.relative_to(self.analyzer.repo_path)))
                self._mine_enums(root, code, crate_name, str(rs_file.relative_to(self.analyzer.repo_path)))
        self._categorize_patterns()
        return self.patterns

    def _mine_structs(self, root: Node, code: str, crate: str, file_path: str):
        for struct_node in self.analyzer.analyzer.find_nodes_by_type(root, 'struct_item'):
            # Get struct name
            struct_name = None
            for child in struct_node.children:
                if child.type == 'type_identifier':
                    struct_name = self.analyzer.analyzer.get_text(child, code)
                    break

            pattern_info = self._extract_derive_and_attributes(struct_node, code)
            if pattern_info['serde_attrs'] or pattern_info['derives']:
                pattern_info['type'] = 'struct'
                pattern_info['name'] = struct_name
                pattern_info['crate'] = crate
                pattern_info['file'] = file_path
                pattern_info['code'] = self.analyzer.analyzer.get_text(struct_node, code)
                self.patterns['struct_patterns'].append(pattern_info)

    def _mine_enums(self, root: Node, code: str, crate: str, file_path: str):
        for enum_node in self.analyzer.analyzer.find_nodes_by_type(root, 'enum_item'):
            enum_name = None
            for child in enum_node.children:
                if child.type == 'type_identifier':
                    enum_name = self.analyzer.analyzer.get_text(child, code)
                    break

            pattern_info = self._extract_derive_and_attributes(enum_node, code)
            if pattern_info['serde_attrs'] or pattern_info['derives']:
                pattern_info['type'] = 'enum'
                pattern_info['name'] = enum_name
                pattern_info['crate'] = crate
                pattern_info['file'] = file_path
                pattern_info['code'] = self.analyzer.analyzer.get_text(enum_node, code)
                self.patterns['enum_patterns'].append(pattern_info)

    def _extract_derive_and_attributes(self, node: Node, code: str) -> Dict[str, Any]:
        derives = []
        serde_attrs = []
        other_attrs = []
        prev = node.prev_sibling
        while prev and prev.type == 'attribute_item':
            attr_text = self.analyzer.analyzer.get_text(prev, code)
            if '#[derive(' in attr_text:
                derive_match = re.search(r'#\[derive\((.*?)\)\]', attr_text, re.DOTALL)
                if derive_match:
                    derive_list = [d.strip() for d in derive_match.group(1).split(',')]
                    derives.extend(derive_list)
            elif '#[serde(' in attr_text:
                serde_match = re.search(r'#\[serde\((.*?)\)\]', attr_text, re.DOTALL)
                if serde_match:
                    serde_attrs.append(serde_match.group(1))
            else:
                other_attrs.append(attr_text)
            prev = prev.prev_sibling
        return {'derives': derives, 'serde_attrs': serde_attrs, 'other_attrs': other_attrs}

    def _categorize_patterns(self):
        derive_combos = Counter()
        serde_attr_combos = Counter()
        for pattern in self.patterns['struct_patterns'] + self.patterns['enum_patterns']:
            if pattern['derives']:
                combo = tuple(sorted(pattern['derives']))
                derive_combos[combo] += 1
            for serde_attr in pattern['serde_attrs']:
                serde_attr_combos[serde_attr] += 1
        self.patterns['derive_combinations'] = [
            {'combo': list(combo), 'count': count}
            for combo, count in derive_combos.most_common(50)
        ]
        self.patterns['serde_attributes'] = [
            {'attr': attr, 'count': count}
            for attr, count in serde_attr_combos.most_common(50)
        ]


class DieselPatternMiner:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        print("  Mining diesel patterns...")
        diesel_crate = self.analyzer.crates.get('diesel_models')
        if not diesel_crate:
            return self.patterns
        rs_files = list(diesel_crate.path.rglob('*.rs'))
        for rs_file in rs_files:
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            table_matches = re.finditer(r'diesel::table!\s*\{(.*?)\}', code, re.DOTALL)
            for match in table_matches:
                table_code = match.group(0)
                self.patterns['table_schemas'].append({
                    'code': table_code,
                    'file': str(rs_file.relative_to(self.analyzer.repo_path))
                })
            root, parsed_code = self.analyzer.analyzer.parse_file(rs_file)
            if root:
                self._mine_diesel_enums(root, code, str(rs_file.relative_to(self.analyzer.repo_path)))
        return self.patterns

    def _mine_diesel_enums(self, root: Node, code: str, file_path: str):
        for enum_node in self.analyzer.analyzer.find_nodes_by_type(root, 'enum_item'):
            prev = enum_node.prev_sibling
            has_diesel_enum = False
            storage_type = None
            while prev and prev.type == 'attribute_item':
                attr_text = self.analyzer.analyzer.get_text(prev, code)
                if 'diesel_enum' in attr_text:
                    has_diesel_enum = True
                    storage_match = re.search(r'storage_type\s*=\s*"(\w+)"', attr_text)
                    if storage_match:
                        storage_type = storage_match.group(1)
                    break
                prev = prev.prev_sibling
            if has_diesel_enum:
                enum_code = self.analyzer.analyzer.get_text(enum_node, code)
                enum_name = None
                for child in enum_node.children:
                    if child.type == 'type_identifier':
                        enum_name = self.analyzer.analyzer.get_text(child, code)
                        break
                self.patterns['diesel_enums'].append({
                    'name': enum_name,
                    'storage_type': storage_type,
                    'code': enum_code,
                    'file': file_path
                })


class AsyncPatternMiner:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        print("  Mining async patterns...")
        async_functions = [fn for fn in self.analyzer.functions.values() if fn.is_async]
        for fn_info in async_functions:
            category = self._categorize_async_function(fn_info)
            self.patterns[category].append(fn_info)
        return dict(self.patterns)

    def _categorize_async_function(self, fn_info):
        name_lower = fn_info.name.lower()
        if 'find' in name_lower or 'get' in name_lower or 'fetch' in name_lower:
            return 'database_query'
        elif 'insert' in name_lower or 'update' in name_lower or 'delete' in name_lower:
            return 'database_mutation'
        elif 'call' in name_lower or 'request' in name_lower or 'connector' in name_lower:
            return 'connector_call'
        elif 'redis' in name_lower:
            return 'redis_operation'
        else:
            return 'other_async'


class TypeConversionMiner:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = []

    def mine_patterns(self):
        print("  Mining type conversion patterns...")
        for impl in self.analyzer.impls:
            trait_name = impl.trait_name
            if trait_name and trait_name in ('From', 'Into', 'TryFrom', 'TryInto'):
                source_layer = self._get_layer(impl.crate)
                target_layer = self._infer_target_layer(impl.for_type)
                self.patterns.append({
                    'trait': trait_name,
                    'for_type': impl.for_type,
                    'crate': impl.crate,
                    'file': impl.file_path,
                    'methods': impl.methods,
                    'source_layer': source_layer,
                    'target_layer': target_layer
                })
        return {'type_conversions': self.patterns}

    def _get_layer(self, crate_name):
        if 'api_models' in crate_name:
            return 'api'
        elif 'domain_models' in crate_name or 'hyperswitch_domain' in crate_name:
            return 'domain'
        elif 'diesel_models' in crate_name:
            return 'database'
        else:
            return 'other'

    def _infer_target_layer(self, type_name):
        if '::api_models::' in type_name:
            return 'api'
        elif '::domain' in type_name or '::hyperswitch_domain' in type_name:
            return 'domain'
        elif '::diesel_models::' in type_name:
            return 'database'
        else:
            return 'unknown'


# ============================================================================
# CLEAN SAMPLE GENERATOR - No Pattern Metadata
# ============================================================================

class CleanSampleGenerator:
    """Generate NATURAL samples WITHOUT artificial pattern comments"""

    def __init__(self, repo_analyzer, tokenizer, mined_patterns):
        self.analyzer = repo_analyzer
        self.tokenizer = tokenizer
        self.mined_patterns = mined_patterns
        self.samples = []

    def generate_all_samples(self):
        print("\n📝 Generating CLEAN samples (no pattern metadata)...")

        print("\n[1/8] Struct and enum samples (FULL FILES)...")
        self._generate_type_samples()

        print("\n[2/8] Diesel patterns (FULL FILES)...")
        self._generate_diesel_samples()

        print("\n[3/8] Function samples (BIG CHUNKS)...")
        self._generate_function_samples()

        print("\n[4/8] Async patterns (BIG CHUNKS)...")
        self._generate_async_samples()

        print("\n[5/8] Type conversions (FULL FILES)...")
        self._generate_conversion_samples()

        print("\n[6/8] Additional large file samples...")
        self._generate_large_file_samples()

        print("\n[7/8] Repository overview...")
        self._generate_repo_sample()

        print("\n[8/8] Crate samples...")
        self._generate_crate_samples()

        print(f"\n✅ Generated {len(self.samples)} clean samples")
        return self.samples

    def _generate_type_samples(self):
        """Generate LARGE file-based samples with ALL types in each file"""
        serde_patterns = self.mined_patterns.get('serde', {})

        # Group patterns by file to create BIG chunks
        file_groups = defaultdict(lambda: {'structs': [], 'enums': [], 'crate': None})

        for pattern in serde_patterns.get('struct_patterns', []):
            file_key = f"{pattern['crate']}/{pattern['file']}"
            file_groups[file_key]['structs'].append(pattern)
            file_groups[file_key]['crate'] = pattern['crate']

        for pattern in serde_patterns.get('enum_patterns', []):
            file_key = f"{pattern['crate']}/{pattern['file']}"
            file_groups[file_key]['enums'].append(pattern)
            file_groups[file_key]['crate'] = pattern['crate']

        # Generate BIG samples with full file content or large sections
        for file_key, items in file_groups.items():
            if not items['structs'] and not items['enums']:
                continue

            # Try to read the entire file for maximum context
            crate_name = items['crate']
            file_path = file_key.split('/', 1)[1]
            full_path = self.analyzer.repo_path / file_path

            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        full_file_content = f.read()

                    # Create a BIG sample with the entire file
                    sample = {
                        'id': f'file_{crate_name}_{hash(file_path)}',
                        'type': 'clm',
                        'granularity': 'file',
                        'content': f'''// Repository: hyperswitch
// Crate: {crate_name}
// File: {file_path}
// Contains: {len(items['structs'])} structs, {len(items['enums'])} enums

{full_file_content}
''',
                        'metadata': {
                            'crate': crate_name,
                            'file': file_path,
                            'num_structs': len(items['structs']),
                            'num_enums': len(items['enums'])
                        }
                    }
                    self.samples.append(sample)
                except Exception as e:
                    # Fallback: combine just the extracted patterns
                    combined_code = []
                    for s in items['structs']:
                        combined_code.append(s['code'])
                    for e in items['enums']:
                        combined_code.append(e['code'])

                    sample = {
                        'id': f'types_{crate_name}_{hash(file_path)}',
                        'type': 'clm',
                        'granularity': 'file',
                        'content': f'''// Crate: {crate_name}
// File: {file_path}

{chr(10).join(combined_code)}
''',
                        'metadata': {'crate': crate_name}
                    }
                    self.samples.append(sample)

    def _generate_diesel_samples(self):
        """Diesel samples - FULL FILE context"""
        diesel_patterns = self.mined_patterns.get('diesel', {})

        # Group by file to get BIG chunks
        file_groups = defaultdict(lambda: {'tables': [], 'enums': []})

        for table_info in diesel_patterns.get('table_schemas', []):
            file_groups[table_info['file']]['tables'].append(table_info)

        for enum_info in diesel_patterns.get('diesel_enums', []):
            file_groups[enum_info['file']]['enums'].append(enum_info)

        # Generate BIG file-level samples
        diesel_crate = self.analyzer.crates.get('diesel_models')
        if diesel_crate:
            for file_path, items in file_groups.items():
                full_path = self.analyzer.repo_path / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            full_content = f.read()

                        sample = {
                            'id': f'diesel_file_{hash(file_path)}',
                            'type': 'clm',
                            'granularity': 'file',
                            'content': f'''// Repository: hyperswitch
// Crate: diesel_models
// File: {file_path}
// Contains: {len(items['tables'])} table schemas, {len(items['enums'])} diesel enums

{full_content}
''',
                            'metadata': {
                                'crate': 'diesel_models',
                                'file': file_path,
                                'num_tables': len(items['tables']),
                                'num_enums': len(items['enums'])
                            }
                        }
                        self.samples.append(sample)
                    except:
                        pass

    def _generate_function_samples(self):
        """Generate BIG function samples with surrounding context from files"""
        # Group functions by file to create LARGE chunks
        file_functions = defaultdict(list)

        for fn_key, fn_info in self.analyzer.functions.items():
            # Score functions to prioritize important ones
            score = 0
            score += len(fn_info.calls) * 2
            score += len(self.analyzer.reverse_call_graph.get(fn_info.name, [])) * 3
            if fn_info.is_pub: score += 10
            if fn_info.is_async: score += 5
            if fn_info.impl_context: score += 8

            file_key = f"{fn_info.crate}/{fn_info.module}"
            file_functions[file_key].append((fn_info, score))

        # Process each file - create BIG samples with multiple functions
        for file_key, functions in file_functions.items():
            # Sort by score and take top functions from each file
            functions.sort(key=lambda x: x[1], reverse=True)
            top_funcs = functions[:10]  # Top 10 functions per file

            if not top_funcs:
                continue

            # Try to read entire file for MAXIMUM context
            first_func = top_funcs[0][0]
            crate_path = self.analyzer.crates.get(first_func.crate)
            if crate_path:
                # Find the actual source file
                possible_paths = list(crate_path.path.rglob(f"*{first_func.module}.rs"))
                if not possible_paths:
                    # Try finding by module path
                    module_parts = first_func.module.split('::')
                    module_file = '/'.join(module_parts) + '.rs'
                    possible_paths = [crate_path.path / 'src' / module_file]

                for src_file in possible_paths:
                    if src_file.exists():
                        try:
                            with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                                file_content = f.read()

                            # Create BIG file-level sample
                            sample = {
                                'id': f'module_{hash(file_key)}',
                                'type': 'clm',
                                'granularity': 'module',
                                'content': f'''// Repository: hyperswitch
// Crate: {first_func.crate}
// Module: {first_func.module}
// Top functions in this module: {', '.join([f[0].name for f in top_funcs[:5]])}

{file_content}
''',
                                'metadata': {
                                    'crate': first_func.crate,
                                    'module': first_func.module,
                                    'num_functions': len(top_funcs)
                                }
                            }
                            self.samples.append(sample)
                            break
                        except:
                            pass

            # Also create combined function samples as backup
            for fn_info, score in top_funcs[:5]:  # Top 5 for individual samples
                context = self._build_natural_context(fn_info)

                # Get surrounding code context (try to get ~500 lines around the function)
                surrounding_context = self._get_surrounding_code(fn_info, lines_before=50, lines_after=50)

                if surrounding_context:
                    content = f'''{context}

{surrounding_context}
'''
                else:
                    content = f'''{context}

{fn_info.full_text}
'''

                clm_sample = {
                    'id': f'fn_clm_{fn_info.crate}_{fn_info.name}_{hash(fn_info.module)}',
                    'type': 'clm',
                    'granularity': 'function',
                    'content': content,
                    'metadata': {
                        'crate': fn_info.crate,
                        'is_async': fn_info.is_async,
                        'is_pub': fn_info.is_pub,
                        'score': score
                    }
                }
                self.samples.append(clm_sample)

    def _get_surrounding_code(self, fn_info, lines_before=50, lines_after=50):
        """Get code surrounding a function for maximum context"""
        try:
            crate_info = self.analyzer.crates.get(fn_info.crate)
            if not crate_info:
                return None

            # Find the source file
            module_parts = fn_info.module.split('::')
            possible_files = [
                crate_info.path / 'src' / f"{'/'.join(module_parts)}.rs",
                crate_info.path / 'src' / f"{module_parts[-1]}.rs",
            ]

            # Also try globbing
            possible_files.extend(list(crate_info.path.rglob(f"*{module_parts[-1]}.rs")))

            for src_file in possible_files:
                if src_file.exists():
                    with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                        all_lines = f.readlines()

                    # Find the function in the file
                    fn_signature_clean = fn_info.signature.strip()
                    for i, line in enumerate(all_lines):
                        if fn_info.name in line and ('fn ' in line or 'async fn' in line):
                            # Found the function, extract surrounding context
                            start = max(0, i - lines_before)
                            # Find the end of the function (look for closing brace at same indentation)
                            end = i
                            brace_count = 0
                            found_start = False
                            for j in range(i, len(all_lines)):
                                if '{' in all_lines[j]:
                                    found_start = True
                                    brace_count += all_lines[j].count('{')
                                if '}' in all_lines[j]:
                                    brace_count -= all_lines[j].count('}')
                                if found_start and brace_count == 0:
                                    end = min(len(all_lines), j + 1 + lines_after)
                                    break

                            if end > i:
                                surrounding = ''.join(all_lines[start:end])
                                return surrounding
            return None
        except Exception:
            return None

    def _build_natural_context(self, fn_info) -> str:
        """Build context that looks like REAL code comments"""
        lines = []
        lines.append(f"// Repository: hyperswitch")
        lines.append(f"// Crate: {fn_info.crate}")

        crate_info = self.analyzer.crates.get(fn_info.crate)
        if crate_info and crate_info.purpose:
            lines.append(f"// Purpose: {crate_info.purpose}")

        lines.append(f"// Module: {fn_info.module}")

        if fn_info.impl_context:
            impl_ctx = fn_info.impl_context
            if impl_ctx['type'] == 'trait_impl':
                lines.append(f"// Implementation of {impl_ctx['trait']} for {impl_ctx['for_type']}")
            else:
                lines.append(f"// Inherent implementation for {impl_ctx['for_type']}")

        if fn_info.doc_comments:
            lines.append("")
            lines.append(fn_info.doc_comments)

        return '\n'.join(lines)

    def _generate_async_samples(self):
        """Async samples - BIG chunks with surrounding context"""
        async_patterns = self.mined_patterns.get('async', {})

        for category, functions in async_patterns.items():
            for fn_info in functions[:50]:
                context = self._build_natural_context(fn_info)

                # Try to get BIG surrounding context
                surrounding_context = self._get_surrounding_code(fn_info, lines_before=100, lines_after=100)

                if surrounding_context:
                    content = f'''{context}

{surrounding_context}
'''
                else:
                    content = f'''{context}

{fn_info.full_text}
'''

                sample = {
                    'id': f'async_{category}_{fn_info.name}_{fn_info.crate}',
                    'type': 'clm',
                    'granularity': 'function',
                    'content': content,
                    'metadata': {'crate': fn_info.crate, 'category': category}
                }
                self.samples.append(sample)

    def _generate_conversion_samples(self):
        """Type conversion samples - BIG file context"""
        conversion_patterns = self.mined_patterns.get('conversions', {})

        # Group conversions by file for BIG samples
        file_conversions = defaultdict(list)

        for conv_info in conversion_patterns.get('type_conversions', []):
            for impl in self.analyzer.impls:
                if (impl.trait_name == conv_info['trait'] and
                    impl.for_type == conv_info['for_type'] and
                    impl.crate == conv_info['crate']):
                    file_conversions[impl.file_path].append(conv_info)
                    break

        # Generate BIG file-level samples with all conversions
        for file_path, conversions in file_conversions.items():
            full_path = self.analyzer.repo_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()

                    conversion_summary = ', '.join([f"{c['trait']}" for c in conversions[:5]])
                    sample = {
                        'id': f'conv_file_{hash(file_path)}',
                        'type': 'clm',
                        'granularity': 'file',
                        'content': f'''// Repository: hyperswitch
// Crate: {conversions[0]["crate"]}
// File: {file_path}
// Type conversions: {conversion_summary}
// Total conversions in file: {len(conversions)}

{file_content}
''',
                        'metadata': {
                            'crate': conversions[0]['crate'],
                            'file': file_path,
                            'num_conversions': len(conversions)
                        }
                    }
                    self.samples.append(sample)
                except:
                    pass

    def _generate_large_file_samples(self):
        """Generate additional BIG samples from important .rs files"""
        import random

        important_crates = [
            'api_models', 'diesel_models', 'hyperswitch_domain_models',
            'router', 'core', 'services', 'storage'
        ]

        files_added = 0
        max_files_per_crate = 20

        for crate_name in important_crates:
            crate_info = self.analyzer.crates.get(crate_name)
            if not crate_info:
                continue

            # Get all .rs files in this crate
            rs_files = list(crate_info.path.rglob('*.rs'))

            # Sort by file size (larger files = more code = more useful)
            rs_files_with_size = []
            for rs_file in rs_files:
                try:
                    size = rs_file.stat().st_size
                    # Only include files between 1KB and 100KB (good size for training)
                    if 1000 < size < 100000:
                        rs_files_with_size.append((rs_file, size))
                except:
                    pass

            # Sort by size descending and take largest files
            rs_files_with_size.sort(key=lambda x: x[1], reverse=True)
            selected_files = rs_files_with_size[:max_files_per_crate]

            for rs_file, size in selected_files:
                try:
                    with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    rel_path = rs_file.relative_to(self.analyzer.repo_path)

                    sample = {
                        'id': f'large_file_{crate_name}_{hash(str(rel_path))}',
                        'type': 'clm',
                        'granularity': 'file',
                        'content': f'''// Repository: hyperswitch
// Crate: {crate_name}
// File: {rel_path}
// File size: {size} bytes

{content}
''',
                        'metadata': {
                            'crate': crate_name,
                            'file': str(rel_path),
                            'file_size': size
                        }
                    }
                    self.samples.append(sample)
                    files_added += 1
                except Exception as e:
                    pass

        print(f"   Added {files_added} large file samples")

    def _generate_repo_sample(self):
        """Repository overview"""
        graph_text = "# Hyperswitch Repository Structure\n\n"
        for crate_name in sorted(self.analyzer.crates.keys()):
            deps = self.analyzer.crate_graph.get(crate_name, [])
            purpose = self.analyzer.crates[crate_name].purpose
            graph_text += f"\n{crate_name}/\n"
            if purpose:
                graph_text += f"  Purpose: {purpose}\n"
            if deps:
                graph_text += f"  Dependencies: {', '.join(deps)}\n"

        sample = {
            'id': 'repo_structure',
            'type': 'clm',
            'granularity': 'repository',
            'content': graph_text,
            'metadata': {'total_crates': len(self.analyzer.crates)}
        }
        self.samples.append(sample)

    def _generate_crate_samples(self):
        """Crate-level samples"""
        for crate_name, crate_info in self.analyzer.crates.items():
            content = f"// CRATE: {crate_name}\n"
            if crate_info.purpose:
                content += f"// PURPOSE: {crate_info.purpose}\n"
            if crate_info.local_deps:
                content += f"// DEPENDENCIES: {', '.join(crate_info.local_deps)}\n"

            sample = {
                'id': f'crate_{crate_name}',
                'type': 'clm',
                'granularity': 'crate',
                'content': content,
                'metadata': {'crate': crate_name}
            }
            self.samples.append(sample)


# ============================================================================
# MAIN WITH HF UPLOAD
# ============================================================================

def upload_to_huggingface(dataset_path: Path, repo_name: str = "archit11/hyperswitch-code-dataset"):
    """Upload dataset to Hugging Face"""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi, create_repo

        print(f"\n📤 Uploading to Hugging Face: {repo_name}")

        # Load JSONL
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))

        # Create dataset
        dataset = Dataset.from_list(samples)

        print(f"   Dataset: {len(samples)} samples")

        # Create repo (if doesn't exist)
        try:
            create_repo(repo_name, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"   Repo exists or error: {e}")

        # Push to hub
        dataset.push_to_hub(repo_name, private=False)

        print(f"✅ Uploaded to https://huggingface.co/datasets/{repo_name}")

    except ImportError:
        print("⚠️  Install: pip install datasets huggingface_hub")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


def main():
    repo_path = Path('/Users/architsinghai/code/deepwiki-scripts/hyperswitch')
    output_dir = Path('/Users/architsinghai/code/repo_cpt_dataset_clean')

    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return

    print("=" * 80)
    print("CLEAN Pattern Mining & Dataset Generation")
    print("NO artificial pattern comments - Natural code only")
    print("=" * 80)

    # Phase 1: Analyze
    print("\n📊 Phase 1: Repository Analysis")
    analyzer = RepositoryAnalyzer(repo_path)
    analyzer.analyze()

    # Phase 2: Mine patterns
    print("\n⛏️  Phase 2: Pattern Mining")
    mined_patterns = {}

    serde_miner = SerdePatternMiner(analyzer)
    mined_patterns['serde'] = serde_miner.mine_patterns()

    diesel_miner = DieselPatternMiner(analyzer)
    mined_patterns['diesel'] = diesel_miner.mine_patterns()

    async_miner = AsyncPatternMiner(analyzer)
    mined_patterns['async'] = async_miner.mine_patterns()

    conversion_miner = TypeConversionMiner(analyzer)
    mined_patterns['conversions'] = conversion_miner.mine_patterns()

    # Save patterns
    output_dir.mkdir(exist_ok=True)
    patterns_file = output_dir / 'mined_patterns.json'
    with open(patterns_file, 'w') as f:
        json.dump(mined_patterns, f, indent=2, default=str)
    print(f"\n✅ Saved patterns to {patterns_file}")

    # Phase 3: Load tokenizer
    print("\n🔤 Phase 3: Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Kwaipilot/KAT-Dev", trust_remote_code=True)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠️  Continuing without tokenizer")
        tokenizer = None

    # Phase 4: Generate CLEAN samples
    print("\n📝 Phase 4: Clean Sample Generation")
    generator = CleanSampleGenerator(analyzer, tokenizer, mined_patterns)
    samples = generator.generate_all_samples()

    # Save dataset
    output_file = output_dir / 'dataset.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ Dataset saved to {output_file}")
    print(f"   Total samples: {len(samples)}")

    # Statistics
    type_counts = defaultdict(int)
    crate_counts = defaultdict(int)

    for sample in samples:
        type_counts[sample['type']] += 1
        if 'crate' in sample.get('metadata', {}):
            crate_counts[sample['metadata']['crate']] += 1

    print("\n📊 Dataset Statistics:")
    print(f"\n  By Type:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")

    print(f"\n  Top Crates:")
    for crate, count in sorted(crate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {crate}: {count}")

    # Phase 5: Upload to HF
    print("\n📤 Phase 5: Uploading to Hugging Face")
    upload_to_huggingface(output_file, "archit11/hyperswitch-code-dataset")


if __name__ == '__main__':
    main()
