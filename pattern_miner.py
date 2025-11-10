#!/usr/bin/env python3
"""
PATTERN MINING: Automatically discover and extract patterns from Hyperswitch
No hardcoding - all patterns mined from actual code
NO TRUNCATION - Full code samples
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from tree_sitter import Language, Parser, Node
import tree_sitter_rust


# Import the repository analyzer from gen_data.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gen_data import RepositoryAnalyzer


# ============================================================================
# PATTERN MINERS
# ============================================================================

class SerdePatternMiner:
    """Mine all serde attribute patterns from the codebase"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract all serde patterns from structs and enums"""
        print("  Mining serde patterns...")

        for crate_name, crate_info in self.analyzer.crates.items():
            rs_files = list(crate_info.path.rglob('*.rs'))

            for rs_file in rs_files:  # NO LIMIT
                root, code = self.analyzer.analyzer.parse_file(rs_file)
                if not root:
                    continue

                # Find all structs and enums with derive macros
                self._mine_structs(root, code, crate_name, str(rs_file.relative_to(self.analyzer.repo_path)))
                self._mine_enums(root, code, crate_name, str(rs_file.relative_to(self.analyzer.repo_path)))

        self._categorize_patterns()
        return self.patterns

    def _mine_structs(self, root: Node, code: str, crate: str, file_path: str):
        """Mine patterns from struct definitions"""
        for struct_node in self.analyzer.analyzer.find_nodes_by_type(root, 'struct_item'):
            pattern_info = self._extract_derive_and_attributes(struct_node, code)
            if pattern_info['serde_attrs'] or pattern_info['derives']:
                pattern_info['type'] = 'struct'
                pattern_info['crate'] = crate
                pattern_info['file'] = file_path
                pattern_info['code'] = self.analyzer.analyzer.get_text(struct_node, code)  # FULL CODE
                self.patterns['struct_patterns'].append(pattern_info)

    def _mine_enums(self, root: Node, code: str, crate: str, file_path: str):
        """Mine patterns from enum definitions"""
        for enum_node in self.analyzer.analyzer.find_nodes_by_type(root, 'enum_item'):
            pattern_info = self._extract_derive_and_attributes(enum_node, code)
            if pattern_info['serde_attrs'] or pattern_info['derives']:
                pattern_info['type'] = 'enum'
                pattern_info['crate'] = crate
                pattern_info['file'] = file_path
                pattern_info['code'] = self.analyzer.analyzer.get_text(enum_node, code)  # FULL CODE
                self.patterns['enum_patterns'].append(pattern_info)

    def _extract_derive_and_attributes(self, node: Node, code: str) -> Dict[str, Any]:
        """Extract all derive macros and attributes from a type"""
        derives = []
        serde_attrs = []
        other_attrs = []

        # Walk backwards to find attributes
        prev = node.prev_sibling
        while prev and prev.type == 'attribute_item':
            attr_text = self.analyzer.analyzer.get_text(prev, code)

            # Check if it's a derive attribute
            if '#[derive(' in attr_text:
                # Extract derive list
                derive_match = re.search(r'#\[derive\((.*?)\)\]', attr_text, re.DOTALL)
                if derive_match:
                    derive_list = [d.strip() for d in derive_match.group(1).split(',')]
                    derives.extend(derive_list)

            # Check for serde attributes
            elif '#[serde(' in attr_text:
                serde_match = re.search(r'#\[serde\((.*?)\)\]', attr_text, re.DOTALL)
                if serde_match:
                    serde_attrs.append(serde_match.group(1))

            # Other attributes
            else:
                other_attrs.append(attr_text)

            prev = prev.prev_sibling

        return {
            'derives': derives,
            'serde_attrs': serde_attrs,
            'other_attrs': other_attrs
        }

    def _categorize_patterns(self):
        """Group patterns by common combinations"""
        # Count derive combinations
        derive_combos = Counter()
        serde_attr_combos = Counter()

        for pattern in self.patterns['struct_patterns'] + self.patterns['enum_patterns']:
            if pattern['derives']:
                combo = tuple(sorted(pattern['derives']))
                derive_combos[combo] += 1

            for serde_attr in pattern['serde_attrs']:
                serde_attr_combos[serde_attr] += 1

        self.patterns['derive_combinations'] = [
            {'combo': list(combo), 'count': count, 'example': None}
            for combo, count in derive_combos.most_common(50)  # Increased from 30
        ]

        self.patterns['serde_attributes'] = [
            {'attr': attr, 'count': count, 'example': None}
            for attr, count in serde_attr_combos.most_common(50)  # Increased from 30
        ]


class DieselPatternMiner:
    """Mine diesel patterns (table!, enum patterns)"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract diesel patterns"""
        print("  Mining diesel patterns...")

        diesel_crate = self.analyzer.crates.get('diesel_models')
        if not diesel_crate:
            return self.patterns

        rs_files = list(diesel_crate.path.rglob('*.rs'))

        for rs_file in rs_files:  # NO LIMIT
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            # Find diesel::table! macros
            table_matches = re.finditer(
                r'diesel::table!\s*\{(.*?)\}',
                code,
                re.DOTALL
            )

            for match in table_matches:
                table_code = match.group(0)  # FULL TABLE CODE
                self.patterns['table_schemas'].append({
                    'code': table_code,
                    'file': str(rs_file.relative_to(self.analyzer.repo_path))
                })

            # Find router_derive::diesel_enum attributes
            root, parsed_code = self.analyzer.analyzer.parse_file(rs_file)
            if root:
                self._mine_diesel_enums(root, code, str(rs_file.relative_to(self.analyzer.repo_path)))

        return self.patterns

    def _mine_diesel_enums(self, root: Node, code: str, file_path: str):
        """Mine diesel enum patterns"""
        for enum_node in self.analyzer.analyzer.find_nodes_by_type(root, 'enum_item'):
            # Check for router_derive::diesel_enum attribute
            prev = enum_node.prev_sibling
            has_diesel_enum = False
            storage_type = None

            while prev and prev.type == 'attribute_item':
                attr_text = self.analyzer.analyzer.get_text(prev, code)
                if 'diesel_enum' in attr_text:
                    has_diesel_enum = True
                    # Extract storage_type
                    storage_match = re.search(r'storage_type\s*=\s*"(\w+)"', attr_text)
                    if storage_match:
                        storage_type = storage_match.group(1)
                    break
                prev = prev.prev_sibling

            if has_diesel_enum:
                enum_code = self.analyzer.analyzer.get_text(enum_node, code)  # FULL CODE
                # Get enum name
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


class MacroUsageMiner:
    """Mine macro usage patterns"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract macro usage patterns"""
        print("  Mining macro usage patterns...")

        macro_calls = defaultdict(list)

        for crate_name, crate_info in self.analyzer.crates.items():  # ALL CRATES
            rs_files = list(crate_info.path.rglob('*.rs'))

            for rs_file in rs_files:  # ALL FILES
                root, code = self.analyzer.analyzer.parse_file(rs_file)
                if not root:
                    continue

                # Find all macro invocations
                for macro_node in self.analyzer.analyzer.find_nodes_by_type(root, 'macro_invocation'):
                    macro_text = self.analyzer.analyzer.get_text(macro_node, code)  # FULL MACRO

                    # Get macro name
                    macro_name = macro_text.split('!')[0] if '!' in macro_text else ''

                    if macro_name:
                        macro_calls[macro_name].append({
                            'code': macro_text,  # NO TRUNCATION
                            'file': str(rs_file.relative_to(self.analyzer.repo_path)),
                            'crate': crate_name
                        })

        # Keep top macros by frequency
        for macro_name, calls in sorted(macro_calls.items(), key=lambda x: len(x[1]), reverse=True)[:50]:  # Top 50
            self.patterns['macro_usage'].append({
                'macro': macro_name,
                'count': len(calls),
                'examples': calls[:10]  # Keep 10 examples
            })

        return self.patterns


class ValidationPatternMiner:
    """Mine validation patterns"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = []

    def mine_patterns(self):
        """Extract validation patterns"""
        print("  Mining validation patterns...")

        # Look for functions with 'validate' in name
        validation_functions = [
            fn for fn in self.analyzer.functions.values()
            if 'validate' in fn.name.lower() or 'check' in fn.name.lower()
        ]

        for fn_info in validation_functions:  # ALL validation functions
            self.patterns.append({
                'name': fn_info.name,
                'signature': fn_info.signature,
                'body': fn_info.body,  # FULL BODY
                'crate': fn_info.crate,
                'file': fn_info.file_path,
                'return_type': fn_info.return_type
            })

        return {'validation_functions': self.patterns}


class AsyncPatternMiner:
    """Mine async/await patterns"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract async patterns"""
        print("  Mining async patterns...")

        async_functions = [fn for fn in self.analyzer.functions.values() if fn.is_async]

        # Categorize by purpose
        for fn_info in async_functions:  # ALL async functions
            category = self._categorize_async_function(fn_info)

            self.patterns[category].append({
                'name': fn_info.name,
                'signature': fn_info.signature,
                'body': fn_info.body,  # FULL BODY
                'crate': fn_info.crate,
                'calls': [c['name'] for c in fn_info.calls],  # ALL calls
                'return_type': fn_info.return_type
            })

        return dict(self.patterns)

    def _categorize_async_function(self, fn_info):
        """Categorize async function by what it does"""
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


class ErrorHandlingMiner:
    """Mine error handling patterns"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract error handling patterns"""
        print("  Mining error handling patterns...")

        # Find functions with Result return types
        result_functions = [
            fn for fn in self.analyzer.functions.values()
            if fn.return_type and 'Result' in fn.return_type
        ]

        # Analyze error types
        error_types = Counter()

        for fn_info in result_functions:  # ALL result functions
            if fn_info.return_type:
                error_types[fn_info.return_type] += 1

            self.patterns['result_functions'].append({
                'name': fn_info.name,
                'return_type': fn_info.return_type,
                'signature': fn_info.signature,
                'body': fn_info.body,  # FULL BODY
                'crate': fn_info.crate
            })

        self.patterns['error_types'] = [
            {'type': t, 'count': c}
            for t, c in error_types.most_common(50)  # Top 50
        ]

        return dict(self.patterns)


class TypeConversionMiner:
    """Mine type conversion patterns (From, Into, TryFrom)"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = []

    def mine_patterns(self):
        """Extract type conversion implementations"""
        print("  Mining type conversion patterns...")

        # Find From/Into/TryFrom implementations
        from_impls = []

        for impl in self.analyzer.impls:  # ALL impls
            trait_name = impl.trait_name

            if trait_name and trait_name in ('From', 'Into', 'TryFrom', 'TryInto'):
                from_impls.append({
                    'trait': trait_name,
                    'for_type': impl.for_type,
                    'crate': impl.crate,
                    'file': impl.file_path,
                    'methods': impl.methods,
                    'generic_params': impl.generic_params
                })

        # Categorize by layer transitions
        for impl_info in from_impls:
            source_layer = self._get_layer(impl_info['crate'])
            # Try to infer target layer from for_type
            target_layer = self._infer_target_layer(impl_info['for_type'])

            impl_info['source_layer'] = source_layer
            impl_info['target_layer'] = target_layer

            self.patterns.append(impl_info)

        return {'type_conversions': self.patterns}

    def _get_layer(self, crate_name):
        """Determine architectural layer of crate"""
        if 'api_models' in crate_name:
            return 'api'
        elif 'domain_models' in crate_name or 'hyperswitch_domain' in crate_name:
            return 'domain'
        elif 'diesel_models' in crate_name:
            return 'database'
        else:
            return 'other'

    def _infer_target_layer(self, type_name):
        """Try to infer target layer from type name"""
        if '::api_models::' in type_name:
            return 'api'
        elif '::domain' in type_name or '::hyperswitch_domain' in type_name:
            return 'domain'
        elif '::diesel_models::' in type_name:
            return 'database'
        else:
            return 'unknown'


class ImplPatternMiner:
    """Mine trait implementation patterns"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract impl patterns"""
        print("  Mining impl patterns...")

        # Group by trait
        trait_impls = defaultdict(list)

        for impl in self.analyzer.impls:  # ALL impls
            if impl.trait_name:
                trait_impls[impl.trait_name].append({
                    'for_type': impl.for_type,
                    'crate': impl.crate,
                    'file': impl.file_path,
                    'methods': impl.methods,
                    'generic_params': impl.generic_params
                })

        # Keep common traits
        for trait_name, impls in sorted(trait_impls.items(), key=lambda x: len(x[1]), reverse=True)[:50]:  # Top 50
            self.patterns['by_trait'].append({
                'trait': trait_name,
                'impl_count': len(impls),
                'examples': impls[:10]  # Keep 10 examples
            })

        return dict(self.patterns)


class RoutingPatternMiner:
    """Mine routing and connector patterns"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.patterns = defaultdict(list)

    def mine_patterns(self):
        """Extract routing patterns"""
        print("  Mining routing patterns...")

        # Find connector implementations
        connector_impls = [
            impl for impl in self.analyzer.impls
            if impl.trait_name and 'Connector' in impl.trait_name
        ]

        # ALL connectors
        for impl in connector_impls:
            self.patterns['connector_impls'].append({
                'trait': impl.trait_name,
                'connector': impl.for_type,
                'methods': impl.methods,
                'crate': impl.crate
            })

        # Find routing functions
        routing_functions = [
            fn for fn in self.analyzer.functions.values()
            if 'rout' in fn.name.lower() and fn.is_pub
        ]

        for fn_info in routing_functions:  # ALL routing functions
            self.patterns['routing_functions'].append({
                'name': fn_info.name,
                'signature': fn_info.signature,
                'body': fn_info.body,  # FULL BODY
                'crate': fn_info.crate
            })

        return dict(self.patterns)


# ============================================================================
# PATTERN-BASED SAMPLE GENERATOR
# ============================================================================

class PatternBasedSampleGenerator:
    """Generate training samples from mined patterns"""

    def __init__(self, repo_analyzer, tokenizer, mined_patterns):
        self.analyzer = repo_analyzer
        self.tokenizer = tokenizer
        self.mined_patterns = mined_patterns
        self.samples = []

    def generate_all_samples(self):
        """Generate samples from all mined patterns"""
        print("\n📝 Generating samples from mined patterns...")

        print("\n[1/10] Serde patterns...")
        self._generate_serde_samples()

        print("\n[2/10] Diesel patterns...")
        self._generate_diesel_samples()

        print("\n[3/10] Macro usage patterns...")
        self._generate_macro_samples()

        print("\n[4/10] Validation patterns...")
        self._generate_validation_samples()

        print("\n[5/10] Async patterns...")
        self._generate_async_samples()

        print("\n[6/10] Error handling patterns...")
        self._generate_error_samples()

        print("\n[7/10] Type conversion patterns...")
        self._generate_conversion_samples()

        print("\n[8/10] Impl patterns...")
        self._generate_impl_samples()

        print("\n[9/10] Routing patterns...")
        self._generate_routing_samples()

        print("\n[10/10] Function samples with context...")
        self._generate_function_samples()

        print(f"\n✅ Generated {len(self.samples)} samples from mined patterns")
        return self.samples

    def _generate_serde_samples(self):
        """Generate samples from serde patterns"""
        serde_patterns = self.mined_patterns.get('serde', {})

        # Derive combinations
        for combo_info in serde_patterns.get('derive_combinations', []):
            sample = {
                'id': f'serde_derive_{hash(tuple(combo_info["combo"]))}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: common_derive_combination
// FREQUENCY: {combo_info["count"]} occurrences
// DERIVES: {", ".join(combo_info["combo"])}

#[derive({", ".join(combo_info["combo"])})]
pub struct ExampleType {{
    // Fields...
}}

// Common crates for these derives:
// - Debug, Clone: std
// - Serialize, Deserialize: serde
// - ToSchema: utoipa
// - Display, EnumString: strum
''',
                'metadata': {
                    'pattern_type': 'derive_combination',
                    'derives': combo_info['combo'],
                    'frequency': combo_info['count']
                }
            }
            self.samples.append(sample)

        # Serde attributes
        for attr_info in serde_patterns.get('serde_attributes', []):
            # Find real example
            example = None
            for pattern in serde_patterns.get('struct_patterns', []) + serde_patterns.get('enum_patterns', []):
                if attr_info['attr'] in pattern.get('serde_attrs', []):
                    example = pattern
                    break

            sample = {
                'id': f'serde_attr_{hash(attr_info["attr"])}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: serde_attribute
// ATTRIBUTE: #[serde({attr_info["attr"]})]
// FREQUENCY: {attr_info["count"]} occurrences

{example["code"] if example else f'#[serde({attr_info["attr"]})]'}

// Usage: {attr_info["attr"]}
''',
                'metadata': {
                    'pattern_type': 'serde_attribute',
                    'attribute': attr_info['attr'],
                    'frequency': attr_info['count']
                }
            }
            self.samples.append(sample)

    def _generate_diesel_samples(self):
        """Generate samples from diesel patterns"""
        diesel_patterns = self.mined_patterns.get('diesel', {})

        # ALL table schemas
        for table_info in diesel_patterns.get('table_schemas', []):
            sample = {
                'id': f'diesel_table_{hash(table_info["code"][:100])}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: diesel_table_schema
// FILE: {table_info["file"]}

{table_info["code"]}
''',
                'metadata': {
                    'pattern_type': 'diesel_table_schema'
                }
            }
            self.samples.append(sample)

        # ALL diesel enums
        for enum_info in diesel_patterns.get('diesel_enums', []):
            sample = {
                'id': f'diesel_enum_{enum_info.get("name", "unknown")}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: diesel_enum
// NAME: {enum_info.get("name")}
// STORAGE_TYPE: {enum_info.get("storage_type")}
// FILE: {enum_info["file"]}

{enum_info["code"]}
''',
                'metadata': {
                    'pattern_type': 'diesel_enum',
                    'storage_type': enum_info.get('storage_type')
                }
            }
            self.samples.append(sample)

    def _generate_macro_samples(self):
        """Generate samples from macro usage"""
        macro_patterns = self.mined_patterns.get('macros', {})

        for macro_info in macro_patterns.get('macro_usage', []):
            # Get ALL examples (no truncation)
            examples_text = '\n\n'.join([
                f'// Example {i+1} ({ex["crate"]}/{ex["file"]}):\n{ex["code"]}'
                for i, ex in enumerate(macro_info['examples'])
            ])

            sample = {
                'id': f'macro_{macro_info["macro"]}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: macro_usage
// MACRO: {macro_info["macro"]}!
// FREQUENCY: {macro_info["count"]} occurrences

{examples_text}
''',
                'metadata': {
                    'pattern_type': 'macro_usage',
                    'macro_name': macro_info['macro'],
                    'frequency': macro_info['count']
                }
            }
            self.samples.append(sample)

    def _generate_validation_samples(self):
        """Generate samples from validation patterns"""
        validation_patterns = self.mined_patterns.get('validation', {})

        for fn_info in validation_patterns.get('validation_functions', []):
            sample = {
                'id': f'validation_{fn_info["name"]}',
                'type': 'clm',
                'granularity': 'function',
                'content': f'''// PATTERN: validation_function
// FUNCTION: {fn_info["name"]}
// CRATE: {fn_info["crate"]}
// RETURN_TYPE: {fn_info["return_type"]}

{fn_info["signature"]} {{
{fn_info["body"]}
}}
''',
                'metadata': {
                    'pattern_type': 'validation',
                    'function_name': fn_info['name']
                }
            }
            self.samples.append(sample)

    def _generate_async_samples(self):
        """Generate samples from async patterns"""
        async_patterns = self.mined_patterns.get('async', {})

        for category, functions in async_patterns.items():
            for fn_info in functions:
                sample = {
                    'id': f'async_{category}_{fn_info["name"]}',
                    'type': 'clm',
                    'granularity': 'function',
                    'content': f'''// PATTERN: async_{category}
// FUNCTION: {fn_info["name"]}
// CRATE: {fn_info["crate"]}
// RETURN_TYPE: {fn_info["return_type"]}

{fn_info["signature"]} {{
{fn_info["body"]}
}}

// Calls: {", ".join(fn_info["calls"])}
''',
                    'metadata': {
                        'pattern_type': f'async_{category}',
                        'function_name': fn_info['name']
                    }
                }
                self.samples.append(sample)

    def _generate_error_samples(self):
        """Generate samples from error handling patterns"""
        error_patterns = self.mined_patterns.get('error_handling', {})

        # Error types
        for error_info in error_patterns.get('error_types', []):
            sample = {
                'id': f'error_type_{hash(error_info["type"])}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: error_return_type
// TYPE: {error_info["type"]}
// FREQUENCY: {error_info["count"]} occurrences
''',
                'metadata': {
                    'pattern_type': 'error_type',
                    'error_type': error_info['type'],
                    'frequency': error_info['count']
                }
            }
            self.samples.append(sample)

        # Result functions
        for fn_info in error_patterns.get('result_functions', []):
            sample = {
                'id': f'result_fn_{fn_info["name"]}',
                'type': 'clm',
                'granularity': 'function',
                'content': f'''// PATTERN: result_function
// FUNCTION: {fn_info["name"]}
// RETURN_TYPE: {fn_info["return_type"]}

{fn_info["signature"]} {{
{fn_info["body"]}
}}
''',
                'metadata': {
                    'pattern_type': 'result_function'
                }
            }
            self.samples.append(sample)

    def _generate_conversion_samples(self):
        """Generate samples from type conversion patterns"""
        conversion_patterns = self.mined_patterns.get('conversions', {})

        for conv_info in conversion_patterns.get('type_conversions', []):
            sample = {
                'id': f'conversion_{conv_info["trait"]}_{hash(conv_info["for_type"])}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: type_conversion
// TRAIT: {conv_info["trait"]}
// FOR_TYPE: {conv_info["for_type"]}
// CRATE: {conv_info["crate"]}
// LAYER_TRANSITION: {conv_info["source_layer"]} -> {conv_info["target_layer"]}
// FILE: {conv_info["file"]}

impl {conv_info["trait"]} for {conv_info["for_type"]} {{
    // Methods: {", ".join(conv_info["methods"])}
}}
''',
                'metadata': {
                    'pattern_type': 'type_conversion',
                    'trait': conv_info['trait'],
                    'source_layer': conv_info['source_layer'],
                    'target_layer': conv_info['target_layer']
                }
            }
            self.samples.append(sample)

    def _generate_impl_samples(self):
        """Generate samples from impl patterns"""
        impl_patterns = self.mined_patterns.get('impls', {})

        for trait_info in impl_patterns.get('by_trait', []):
            examples_text = '\n\n'.join([
                f'// impl {trait_info["trait"]} for {ex["for_type"]} ({ex["crate"]})\n// Methods: {", ".join(ex["methods"])}'
                for ex in trait_info['examples']
            ])

            sample = {
                'id': f'impl_trait_{trait_info["trait"]}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: trait_implementation
// TRAIT: {trait_info["trait"]}
// IMPLEMENTATIONS: {trait_info["impl_count"]} across codebase

{examples_text}
''',
                'metadata': {
                    'pattern_type': 'trait_impl',
                    'trait_name': trait_info['trait'],
                    'impl_count': trait_info['impl_count']
                }
            }
            self.samples.append(sample)

    def _generate_routing_samples(self):
        """Generate samples from routing patterns"""
        routing_patterns = self.mined_patterns.get('routing', {})

        for connector_info in routing_patterns.get('connector_impls', []):
            sample = {
                'id': f'connector_{connector_info["connector"]}',
                'type': 'clm',
                'granularity': 'pattern',
                'content': f'''// PATTERN: connector_implementation
// CONNECTOR: {connector_info["connector"]}
// TRAIT: {connector_info["trait"]}
// METHODS: {", ".join(connector_info["methods"])}
// CRATE: {connector_info["crate"]}
''',
                'metadata': {
                    'pattern_type': 'connector_impl',
                    'connector': connector_info['connector']
                }
            }
            self.samples.append(sample)

        # Routing functions
        for fn_info in routing_patterns.get('routing_functions', []):
            sample = {
                'id': f'routing_{fn_info["name"]}',
                'type': 'clm',
                'granularity': 'function',
                'content': f'''// PATTERN: routing_function
// FUNCTION: {fn_info["name"]}

{fn_info["signature"]} {{
{fn_info["body"]}
}}
''',
                'metadata': {
                    'pattern_type': 'routing_function'
                }
            }
            self.samples.append(sample)

    def _generate_function_samples(self):
        """Generate comprehensive function samples with all context"""
        # Top functions by importance
        function_scores = {}

        for fn_key, fn_info in self.analyzer.functions.items():
            score = 0
            score += len(fn_info.calls) * 2
            score += len(self.analyzer.reverse_call_graph.get(fn_info.name, [])) * 3
            if fn_info.is_pub:
                score += 10
            if fn_info.is_async:
                score += 5
            if fn_info.impl_context:
                score += 8

            function_scores[fn_key] = score

        top_functions = sorted(function_scores.items(), key=lambda x: x[1], reverse=True)[:1000]  # Top 1000

        for fn_key, score in top_functions:
            fn_info = self.analyzer.functions[fn_key]

            # Build context
            context = self._build_function_context(fn_info)

            # CLM sample
            clm_sample = {
                'id': f'function_clm_{fn_info.crate}_{fn_info.name}',
                'type': 'clm',
                'granularity': 'function',
                'content': f'''{context}

{fn_info.full_text}
''',
                'metadata': {
                    'function_name': fn_info.name,
                    'crate': fn_info.crate,
                    'is_async': fn_info.is_async,
                    'is_pub': fn_info.is_pub
                }
            }
            self.samples.append(clm_sample)

            # FIM sample
            if fn_info.body:
                fim_sample = {
                    'id': f'function_fim_{fn_info.crate}_{fn_info.name}',
                    'type': 'fim',
                    'granularity': 'function',
                    'content': f'<fim_prefix>{context}\n\n{fn_info.signature} {{\n<fim_suffix>\n}}<fim_middle>{fn_info.body}<|endoftext|>',
                    'metadata': {
                        'function_name': fn_info.name,
                        'crate': fn_info.crate
                    }
                }
                self.samples.append(fim_sample)

    def _build_function_context(self, fn_info) -> str:
        """Build rich context for a function"""
        lines = [
            f"// REPO: hyperswitch",
            f"// CRATE: {fn_info.crate}",
        ]

        crate_info = self.analyzer.crates.get(fn_info.crate)
        if crate_info and crate_info.purpose:
            lines.append(f"// CRATE_PURPOSE: {crate_info.purpose}")

        lines.append(f"// MODULE: {fn_info.module}")
        lines.append(f"// FUNCTION: {fn_info.name}")

        if fn_info.visibility != 'private':
            lines.append(f"// VISIBILITY: {fn_info.visibility}")

        if fn_info.is_async:
            lines.append("// ASYNC: true")

        if fn_info.impl_context:
            impl_ctx = fn_info.impl_context
            if impl_ctx['type'] == 'trait_impl':
                lines.append(f"// TRAIT_IMPL: {impl_ctx['trait']} for {impl_ctx['for_type']}")
            else:
                lines.append(f"// INHERENT_IMPL: {impl_ctx['for_type']}")

        if fn_info.calls:
            calls_str = ', '.join([c['name'] for c in fn_info.calls[:10]])
            if len(fn_info.calls) > 10:
                calls_str += f' ... and {len(fn_info.calls) - 10} more'
            lines.append(f"// CALLS: {calls_str}")

        callers = self.analyzer.reverse_call_graph.get(fn_info.name, [])
        if callers:
            callers_str = ', '.join([c['name'] for c in callers[:10]])
            if len(callers) > 10:
                callers_str += f' ... and {len(callers) - 10} more'
            lines.append(f"// CALLED_BY: {callers_str}")

        return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    repo_path = Path('/Users/architsinghai/code/deepwiki-scripts/hyperswitch')
    output_dir = Path('/Users/architsinghai/code/repo_cpt_dataset_mined')

    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return

    print("=" * 80)
    print("Pattern Mining & Dataset Generation (Tree-sitter)")
    print("NO TRUNCATION - FULL SAMPLES")
    print("=" * 80)

    # Analyze repository
    print("\n📊 Phase 1: Repository Analysis")
    analyzer = RepositoryAnalyzer(repo_path)
    analyzer.analyze()

    # Mine patterns
    print("\n⛏️  Phase 2: Pattern Mining")
    mined_patterns = {}

    serde_miner = SerdePatternMiner(analyzer)
    mined_patterns['serde'] = serde_miner.mine_patterns()

    diesel_miner = DieselPatternMiner(analyzer)
    mined_patterns['diesel'] = diesel_miner.mine_patterns()

    macro_miner = MacroUsageMiner(analyzer)
    mined_patterns['macros'] = macro_miner.mine_patterns()

    validation_miner = ValidationPatternMiner(analyzer)
    mined_patterns['validation'] = validation_miner.mine_patterns()

    async_miner = AsyncPatternMiner(analyzer)
    mined_patterns['async'] = async_miner.mine_patterns()

    error_miner = ErrorHandlingMiner(analyzer)
    mined_patterns['error_handling'] = error_miner.mine_patterns()

    conversion_miner = TypeConversionMiner(analyzer)
    mined_patterns['conversions'] = conversion_miner.mine_patterns()

    impl_miner = ImplPatternMiner(analyzer)
    mined_patterns['impls'] = impl_miner.mine_patterns()

    routing_miner = RoutingPatternMiner(analyzer)
    mined_patterns['routing'] = routing_miner.mine_patterns()

    # Save mined patterns
    output_dir.mkdir(exist_ok=True)
    patterns_file = output_dir / 'mined_patterns.json'
    with open(patterns_file, 'w') as f:
        json.dump(mined_patterns, f, indent=2, default=str)
    print(f"\n✅ Saved mined patterns to {patterns_file}")

    # Load tokenizer
    print("\n🔤 Phase 3: Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Kwaipilot/KAT-Dev", trust_remote_code=True)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠️  Continuing without tokenizer")
        tokenizer = None

    # Generate samples
    print("\n📝 Phase 4: Sample Generation")
    generator = PatternBasedSampleGenerator(analyzer, tokenizer, mined_patterns)
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
    pattern_counts = defaultdict(int)

    for sample in samples:
        type_counts[sample['type']] += 1
        if 'pattern_type' in sample.get('metadata', {}):
            pattern_counts[sample['metadata']['pattern_type']] += 1

    print("\n📊 Dataset Statistics:")
    print(f"\n  By Type:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")

    print(f"\n  Top Patterns:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pattern}: {count}")


if __name__ == '__main__':
    main()
