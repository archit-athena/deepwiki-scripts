#!/usr/bin/env python3
"""
Token-aware dataset creator with analysis for Kwaipilot/KAT-Dev model.
Analyzes token distribution and optimizes chunks for the target model.
Enhanced with bidirectional call graph extraction and visualization.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict
from tree_sitter import Language, Parser
from tree_sitter_rust import language as rust_language


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


# ============================================================================
# RUST CALL GRAPH EXTRACTION (Tree-sitter based)
# ============================================================================

# Initialize tree-sitter parser for Rust
_RUST_PARSER = None

def get_rust_parser() -> Parser:
    """Get or initialize the Rust parser."""
    global _RUST_PARSER
    if _RUST_PARSER is None:
        _RUST_PARSER = Parser(Language(rust_language()))
    return _RUST_PARSER


def parse_rust_crate_info(repo_dir: Path, file_path: str) -> Dict[str, Any]:
    """
    Parse crate and module information from file path.
    Returns crate name, module path, and file location.
    """
    full_path = Path(file_path)
    parts = full_path.parts

    crate_info = {
        'crate': None,
        'module_path': [],
        'file_name': full_path.name,
        'relative_path': file_path
    }

    # Check if path contains 'crates/' directory
    if 'crates' in parts:
        crate_idx = parts.index('crates')
        if crate_idx + 1 < len(parts):
            crate_info['crate'] = parts[crate_idx + 1]

            # Build module path from src/ onwards
            if 'src' in parts:
                src_idx = parts.index('src')
                module_parts = parts[src_idx + 1:]

                # Remove .rs extension and convert to module path
                module_path = []
                for part in module_parts[:-1]:  # Exclude file name
                    module_path.append(part)

                # Add file name without extension (unless it's mod.rs or lib.rs)
                file_stem = full_path.stem
                if file_stem not in ('mod', 'lib'):
                    module_path.append(file_stem)

                crate_info['module_path'] = module_path

    return crate_info


def extract_rust_functions(code: str, file_path: str = "") -> List[Dict[str, Any]]:
    """
    Extract function definitions from Rust code using tree-sitter AST.
    Handles: free functions, methods, trait implementations, async functions.
    """
    parser = get_rust_parser()
    tree = parser.parse(bytes(code, 'utf8'))
    root = tree.root_node

    functions = []
    code_bytes = bytes(code, 'utf8')

    def get_text(node):
        """Extract text from a tree-sitter node."""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    def get_line_number(node):
        """Get line number (1-indexed) from node."""
        return node.start_point[0] + 1

    def find_impl_context(function_node):
        """Find the impl block context for a function."""
        parent = function_node.parent
        while parent:
            if parent.type == 'impl_item':
                # Extract trait and type info
                impl_info = {'type': 'inherent_impl', 'for_type': None, 'trait': None}

                for child in parent.children:
                    if child.type == 'type_identifier':
                        impl_info['for_type'] = get_text(child)
                    elif child.type == 'scoped_type_identifier':
                        impl_info['for_type'] = get_text(child)

                # Check if it's a trait impl (has 'for' keyword)
                impl_text = get_text(parent)
                if ' for ' in impl_text:
                    impl_info['type'] = 'trait_impl'
                    # Try to extract trait name
                    for child in parent.children:
                        if child.type == 'type_identifier' and impl_info['trait'] is None:
                            impl_info['trait'] = get_text(child)
                        elif child.type == 'generic_type' and impl_info['trait'] is None:
                            trait_node = child.child_by_field_name('type')
                            if trait_node:
                                impl_info['trait'] = get_text(trait_node)

                return impl_info
            parent = parent.parent
        return None

    def extract_attributes(function_node):
        """Extract attributes from function node."""
        attrs = []
        prev_sibling = function_node.prev_sibling
        while prev_sibling and prev_sibling.type == 'attribute_item':
            attr_text = get_text(prev_sibling)
            # Remove outer #[ ] and extract content
            attr_content = attr_text.strip('#[]').strip()
            attrs.insert(0, attr_content)
            prev_sibling = prev_sibling.prev_sibling
        return attrs

    def extract_function_info(fn_node):
        """Extract complete information from a function node."""
        fn_info = {
            'name': None,
            'line': get_line_number(fn_node),
            'visibility': 'private',
            'is_async': False,
            'is_unsafe': False,
            'is_const': False,
            'params': [],
            'attributes': [],
            'impl_context': None,
            'body_start_byte': None,
            'body_end_byte': None
        }

        # Extract attributes
        fn_info['attributes'] = extract_attributes(fn_node)

        for child in fn_node.children:
            # Visibility
            if child.type == 'visibility_modifier':
                fn_info['visibility'] = get_text(child)

            # Modifiers
            elif child.type in ('async', 'unsafe', 'const'):
                if child.type == 'async':
                    fn_info['is_async'] = True
                elif child.type == 'unsafe':
                    fn_info['is_unsafe'] = True
                elif child.type == 'const':
                    fn_info['is_const'] = True

            # Function name
            elif child.type == 'identifier':
                fn_info['name'] = get_text(child)

            # Parameters
            elif child.type == 'parameters':
                for param in child.children:
                    if param.type == 'parameter':
                        param_text = get_text(param)
                        fn_info['params'].append(param_text)
                    elif param.type == 'self_parameter':
                        fn_info['params'].append(get_text(param))

            # Function body
            elif child.type == 'block':
                fn_info['body_start_byte'] = child.start_byte
                fn_info['body_end_byte'] = child.end_byte

        # Find impl context
        fn_info['impl_context'] = find_impl_context(fn_node)

        return fn_info

    # Query for function_item nodes (free functions and methods)
    def walk_tree(node):
        if node.type == 'function_item':
            fn_info = extract_function_info(node)
            if fn_info['name']:  # Only add if we found a name
                functions.append(fn_info)

        for child in node.children:
            walk_tree(child)

    walk_tree(root)

    return functions


def extract_function_calls(code: str, body_start_byte: int = None,
                          body_end_byte: int = None) -> List[Dict[str, Any]]:
    """
    Extract function calls from Rust code using tree-sitter AST.
    Handles: direct calls, method calls, trait method calls, qualified paths.
    If body_start/end_byte provided, only extracts calls within that range.
    """
    parser = get_rust_parser()
    tree = parser.parse(bytes(code, 'utf8'))
    root = tree.root_node

    calls = []
    code_bytes = bytes(code, 'utf8')

    def get_text(node):
        """Extract text from a tree-sitter node."""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    def get_line_number(node):
        """Get line number (1-indexed) from node."""
        return node.start_point[0] + 1

    def in_range(node):
        """Check if node is within the specified byte range."""
        if body_start_byte is None or body_end_byte is None:
            return True
        return body_start_byte <= node.start_byte < body_end_byte

    def extract_call_expression(node):
        """Extract information from a call_expression node."""
        call_info = {
            'name': None,
            'qualified_name': None,
            'type': 'direct_call',
            'line': get_line_number(node)
        }

        # Get the function being called (first child)
        function_node = node.child_by_field_name('function')
        if function_node:
            if function_node.type == 'identifier':
                # Simple function call: foo()
                call_info['name'] = get_text(function_node)
                call_info['qualified_name'] = call_info['name']

            elif function_node.type == 'scoped_identifier':
                # Qualified path: module::function()
                full_path = get_text(function_node)
                call_info['qualified_name'] = full_path
                # Extract just the function name (last part)
                name_node = function_node.child_by_field_name('name')
                if name_node:
                    call_info['name'] = get_text(name_node)
                else:
                    call_info['name'] = full_path.split('::')[-1]

            elif function_node.type == 'field_expression':
                # Method call: obj.method()
                call_info['type'] = 'method_call'
                field_node = function_node.child_by_field_name('field')
                if field_node:
                    call_info['name'] = get_text(field_node)
                    # Get full expression for qualified name
                    call_info['qualified_name'] = get_text(function_node)

            elif function_node.type == 'generic_function':
                # Generic function call: func::<T>()
                func_name_node = function_node.child_by_field_name('function')
                if func_name_node:
                    call_info['name'] = get_text(func_name_node)
                    call_info['qualified_name'] = get_text(function_node)

        return call_info if call_info['name'] else None

    def walk_tree(node):
        """Walk the tree and collect call expressions."""
        if not in_range(node):
            return

        if node.type == 'call_expression':
            call_info = extract_call_expression(node)
            if call_info and call_info['name']:
                # Filter out common macro-like calls and keywords
                if not call_info['name'].endswith('!') and call_info['name'] not in (
                    'println', 'print', 'vec', 'format', 'panic', 'assert',
                    'unwrap', 'expect', 'Some', 'None', 'Ok', 'Err'
                ):
                    calls.append(call_info)

        for child in node.children:
            walk_tree(child)

    walk_tree(root)

    return calls


def build_call_graph(code: str, file_path: str = "") -> Dict[str, Any]:
    """
    Build a call graph for the given Rust code using tree-sitter.
    Returns functions and their call relationships with proper scoping.
    """
    functions = extract_rust_functions(code, file_path)
    crate_info = parse_rust_crate_info(Path('.'), file_path)

    # Build call graph for each function
    for fn_info in functions:
        # Extract calls made by this function from its body only
        if fn_info.get('body_start_byte') and fn_info.get('body_end_byte'):
            calls = extract_function_calls(
                code,
                fn_info['body_start_byte'],
                fn_info['body_end_byte']
            )
            fn_info['calls'] = calls
        else:
            fn_info['calls'] = []

    call_graph = {
        'crate_info': crate_info,
        'functions': functions,
        'total_functions': len(functions),
        'total_calls': sum(len(fn['calls']) for fn in functions)
    }

    return call_graph


def build_global_call_graph(repo_dir: Path) -> Dict[str, Any]:
    """
    Build a global bidirectional call graph for the entire repository.
    Returns a mapping of function names to their callers and callees.
    """
    print("  Building global call graph index...")

    global_graph = {
        'functions': {},  # function_name -> {file, line, calls, impl_context}
        'reverse_index': defaultdict(list),  # callee_name -> list of callers
        'files_processed': 0,
        'total_functions': 0
    }

    # Find all .rs files in the repository
    rust_files = list(repo_dir.rglob('*.rs'))
    print(f"    Found {len(rust_files)} Rust files")

    for rs_file in rust_files[:100]:  # Limit to first 100 files for performance
        try:
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            # Extract functions and their calls
            call_graph = build_call_graph(code, str(rs_file.relative_to(repo_dir)))

            for fn_info in call_graph['functions']:
                fn_name = fn_info['name']
                file_path = str(rs_file.relative_to(repo_dir))

                # Store function info
                fn_key = f"{file_path}::{fn_name}"
                global_graph['functions'][fn_key] = {
                    'name': fn_name,
                    'file': file_path,
                    'line': fn_info['line'],
                    'crate': call_graph['crate_info'].get('crate'),
                    'impl_context': fn_info.get('impl_context'),
                    'calls': [c['name'] for c in fn_info.get('calls', [])]
                }

                # Build reverse index (who calls this function)
                for call in fn_info.get('calls', []):
                    callee_name = call['name']
                    global_graph['reverse_index'][callee_name].append({
                        'name': fn_name,
                        'file': file_path,
                        'line': call['line'],
                        'from_function': fn_name
                    })

                global_graph['total_functions'] += 1

            global_graph['files_processed'] += 1

            if global_graph['files_processed'] % 20 == 0:
                print(f"    Processed {global_graph['files_processed']} files, found {global_graph['total_functions']} functions")

        except Exception as e:
            # Skip files that can't be parsed
            continue

    print(f"  ✓ Global call graph built: {global_graph['total_functions']} functions across {global_graph['files_processed']} files")
    print(f"    Reverse index size: {len(global_graph['reverse_index'])} unique callees")

    return global_graph


def generate_ascii_call_tree(function_name: str, calls: List[Dict[str, Any]],
                             callers: List[Dict[str, Any]] = None,
                             max_depth: int = 3, indent: str = "") -> str:
    """
    Generate ASCII tree representation of function call graph.
    Shows both outgoing calls (what it calls) and incoming calls (what calls it).
    """
    tree_lines = []

    # Header with function name
    tree_lines.append(f"{function_name}")

    # Outgoing calls (what this function calls)
    if calls:
        tree_lines.append("")
        tree_lines.append("Calls:")
        for i, call in enumerate(calls[:10]):  # Limit to 10 for readability
            is_last = i == len(calls) - 1
            prefix = "└─→" if is_last else "├─→"
            call_name = call.get('qualified_name', call.get('name', 'unknown'))
            call_type = call.get('type', 'unknown')
            line = call.get('line', '?')
            tree_lines.append(f"{prefix} {call_name} ({call_type}, line {line})")

        if len(calls) > 10:
            tree_lines.append(f"    ... and {len(calls) - 10} more calls")

    # Incoming calls (what calls this function)
    if callers:
        tree_lines.append("")
        tree_lines.append("Called by:")
        for i, caller in enumerate(callers[:10]):
            is_last = i == len(callers) - 1
            prefix = "└─←" if is_last else "├─←"
            caller_name = caller.get('name', 'unknown')
            caller_file = caller.get('file', 'unknown')
            line = caller.get('line', '?')
            tree_lines.append(f"{prefix} {caller_name} ({caller_file}:{line})")

        if len(callers) > 10:
            tree_lines.append(f"    ... and {len(callers) - 10} more callers")

    return '\n'.join(tree_lines)


def generate_call_flow_description(function_name: str, fn_info: Dict[str, Any],
                                   calls: List[Dict[str, Any]],
                                   callers: List[Dict[str, Any]] = None) -> str:
    """
    Generate natural language description of function's call relationships.
    """
    parts = []

    # Function identity
    visibility = fn_info.get('visibility', 'private')
    is_async = fn_info.get('is_async', False)
    impl_ctx = fn_info.get('impl_context')

    # Opening description
    fn_type = "async function" if is_async else "function"
    if impl_ctx:
        if impl_ctx['type'] == 'trait_impl':
            parts.append(f"{function_name} is a {visibility} {fn_type} implementing {impl_ctx['trait']} for {impl_ctx['for_type']}.")
        else:
            parts.append(f"{function_name} is a {visibility} {fn_type} defined in the implementation of {impl_ctx['for_type']}.")
    else:
        parts.append(f"{function_name} is a {visibility} {fn_type}.")

    # What it does (based on calls)
    if calls:
        # Group calls by type
        direct_calls = [c for c in calls if c.get('type') == 'direct_call']
        method_calls = [c for c in calls if c.get('type') == 'method_call']

        call_desc = []
        if direct_calls:
            call_names = ', '.join([c.get('name', 'unknown') for c in direct_calls[:5]])
            if len(direct_calls) > 5:
                call_names += f", and {len(direct_calls) - 5} others"
            call_desc.append(f"calls {call_names}")

        if method_calls:
            method_names = ', '.join([c.get('name', 'unknown') for c in method_calls[:5]])
            if len(method_calls) > 5:
                method_names += f", and {len(method_calls) - 5} others"
            call_desc.append(f"invokes methods {method_names}")

        if call_desc:
            parts.append(f"It {' and '.join(call_desc)}.")

    # Who calls it
    if callers:
        caller_names = ', '.join([c.get('name', 'unknown') for c in callers[:3]])
        if len(callers) > 3:
            caller_names += f", and {len(callers) - 3} others"
        parts.append(f"This function is called by {caller_names}.")

    return ' '.join(parts)


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
                               max_tokens: int = 16384,
                               overlap_tokens: int = 200,
                               target_distribution: Dict[str, tuple] = None,
                               global_call_graph: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Create chunks with well-distributed token counts and minimal overlap.
    Distribution targets: small (<4k), medium (4k-10k), large (10k-16k)
    Aims for varied chunk sizes while respecting semantic boundaries.
    Adds ~200 token overlap between adjacent chunks for context preservation.
    """
    if target_distribution is None:
        target_distribution = {
            'small': (1000, 4000),    # 25% target
            'medium': (4000, 10000),  # 50% target
            'large': (10000, 16000)   # 25% target
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
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir, global_call_graph)
                chunk_data['metadata']['has_overlap'] = len(overlap_prefix) > 0
                chunks.append(chunk_data)

                # Update distribution tracking
                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 4000:
                    size_counts['small'] += 1
                elif tokens < 10000:
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
                    chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir, global_call_graph)
                    chunks.append(chunk_data)

                    tokens = chunk_data['token_stats']['total_tokens']
                    if tokens < 4000:
                        size_counts['small'] += 1
                    elif tokens < 10000:
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
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir, global_call_graph)
                chunks.append(chunk_data)

                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 4000:
                    size_counts['small'] += 1
                elif tokens < 10000:
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
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir, global_call_graph)
                chunk_data['metadata']['has_overlap'] = len(overlap_prefix) > 0
                chunks.append(chunk_data)

                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 4000:
                    size_counts['small'] += 1
                elif tokens < 10000:
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
                chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir, global_call_graph)
                chunk_data['metadata']['has_overlap'] = len(overlap_prefix) > 0
                chunks.append(chunk_data)

                tokens = chunk_data['token_stats']['total_tokens']
                if tokens < 4000:
                    size_counts['small'] += 1
                elif tokens < 10000:
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
        chunk_data = create_chunk(chunk_content, source_file, chunk_id, tokenizer, repo_dir, global_call_graph)
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
                tokenizer, repo_dir: Optional[Path],
                global_call_graph: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a chunk with full metadata, token analysis, and call graph information."""

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
        'call_graph_data': [],  # NEW: Will contain call graph info for each code snippet
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
        call_graph_data = []

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

                # NEW: Extract call graph for this code snippet
                try:
                    call_graph = build_call_graph(code, ref['file_path'])

                    # For each function in this snippet, generate visualizations
                    for fn_info in call_graph['functions']:
                        fn_name = fn_info['name']
                        calls = fn_info.get('calls', [])

                        # Look up callers from global call graph if available
                        callers = []
                        if global_call_graph and fn_name in global_call_graph.get('reverse_index', {}):
                            callers = global_call_graph['reverse_index'][fn_name]

                        # Generate ASCII tree
                        ascii_tree = generate_ascii_call_tree(fn_name, calls, callers)

                        # Generate flow description
                        flow_desc = generate_call_flow_description(fn_name, fn_info, calls, callers)

                        call_graph_data.append({
                            'function_name': fn_name,
                            'file_path': ref['file_path'],
                            'line': fn_info['line'],
                            'crate_info': call_graph['crate_info'],
                            'function_info': fn_info,
                            'calls': calls,
                            'called_by': callers,
                            'ascii_tree': ascii_tree,
                            'flow_description': flow_desc,
                            'call_count': len(calls),
                            'caller_count': len(callers)
                        })
                except Exception as e:
                    # If call graph extraction fails, continue without it
                    pass

        chunk_data['source_code_snippets'] = code_snippets
        chunk_data['call_graph_data'] = call_graph_data

    return chunk_data


def process_directory(input_dir: Path, tokenizer, repo_dir: Optional[Path] = None,
                     max_tokens: int = 8192,
                     global_call_graph: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Process all .md files in the directory and create dataset."""
    all_chunks = []
    md_files = sorted(input_dir.glob('*.md'))

    print(f"Found {len(md_files)} markdown files")
    print(f"Max tokens: {max_tokens}, Distribution: Small (<4k): 25%, Medium (4k-10k): 50%, Large (10k-16k): 25%\n")

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
            max_tokens,
            global_call_graph=global_call_graph
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
    small_chunks = [c for c in chunks if c['token_stats']['total_tokens'] < 4000]
    medium_chunks = [c for c in chunks if 4000 <= c['token_stats']['total_tokens'] < 10000]
    large_chunks = [c for c in chunks if c['token_stats']['total_tokens'] >= 10000]

    chunks_with_diagrams = sum(1 for c in chunks if c['metadata']['has_mermaid_diagram'])
    chunks_with_tables = sum(1 for c in chunks if c['metadata']['has_tables'])

    # Call graph statistics
    chunks_with_call_graphs = [c for c in chunks if c.get('call_graph_data')]
    total_functions_analyzed = sum(len(c.get('call_graph_data', [])) for c in chunks)
    total_calls = sum(sum(fn.get('call_count', 0) for fn in c.get('call_graph_data', [])) for c in chunks)
    total_callers = sum(sum(fn.get('caller_count', 0) for fn in c.get('call_graph_data', [])) for c in chunks)

    stats = {
        'total_chunks': len(chunks),
        'chunks_with_source_code': len(chunks_with_code),
        'total_code_snippets': total_code_snippets,
        'unique_source_files': len(set(c['source_file'] for c in chunks)),
        'chunks_with_mermaid_diagrams': chunks_with_diagrams,
        'chunks_with_tables': chunks_with_tables,
        'call_graph_stats': {
            'chunks_with_call_graphs': len(chunks_with_call_graphs),
            'total_functions_analyzed': total_functions_analyzed,
            'total_outgoing_calls': total_calls,
            'total_incoming_calls': total_callers,
            'avg_calls_per_function': total_calls / total_functions_analyzed if total_functions_analyzed > 0 else 0,
            'avg_callers_per_function': total_callers / total_functions_analyzed if total_functions_analyzed > 0 else 0
        },
        'size_distribution': {
            'small_chunks': {
                'count': len(small_chunks),
                'percentage': (len(small_chunks) / len(chunks) * 100) if chunks else 0,
                'range': '< 4000 tokens'
            },
            'medium_chunks': {
                'count': len(medium_chunks),
                'percentage': (len(medium_chunks) / len(chunks) * 100) if chunks else 0,
                'range': '4000-10000 tokens'
            },
            'large_chunks': {
                'count': len(large_chunks),
                'percentage': (len(large_chunks) / len(chunks) * 100) if chunks else 0,
                'range': '10000-16000 tokens'
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
- **Bidirectional Call Graphs**: Tree-sitter based AST analysis showing what each function calls and what calls it
- **ASCII Tree Visualizations**: Visual call graph representations for each function
- **Natural Language Flow Descriptions**: Plain text descriptions of function relationships
- **Crate/Module Context**: Full Rust workspace structure (crate → module → function)
- **Rich Token Metadata**: Detailed token distribution analysis
- **Well-Distributed Sizes**: Small (<4k), Medium (4k-10k), Large (10k-16k) chunks for varied context
- **Minimal Overlap**: ~200 token overlap between adjacent chunks for context continuity

## Dataset Statistics

- **Total Chunks**: {stats['total_chunks']:,}
- **Chunks with Source Code**: {stats['chunks_with_source_code']:,}
- **Total Code Snippets**: {stats['total_code_snippets']:,}
- **Chunks with Mermaid Diagrams**: {stats['chunks_with_mermaid_diagrams']:,}
- **Chunks with Tables**: {stats['chunks_with_tables']:,}
- **Source Files**: {stats['unique_source_files']}

### Call Graph Statistics

- **Chunks with Call Graphs**: {stats['call_graph_stats']['chunks_with_call_graphs']:,}
- **Functions Analyzed**: {stats['call_graph_stats']['total_functions_analyzed']:,}
- **Total Function Calls**: {stats['call_graph_stats']['total_outgoing_calls']:,}
- **Total Callers Tracked**: {stats['call_graph_stats']['total_incoming_calls']:,}
- **Avg Calls per Function**: {stats['call_graph_stats']['avg_calls_per_function']:.1f}
- **Avg Callers per Function**: {stats['call_graph_stats']['avg_callers_per_function']:.1f}

### Size Distribution (Target: 25% Small, 50% Medium, 25% Large)

- **Small Chunks** (< 4k tokens): {stats['size_distribution']['small_chunks']['count']:,} ({stats['size_distribution']['small_chunks']['percentage']:.1f}%)
- **Medium Chunks** (4k-10k tokens): {stats['size_distribution']['medium_chunks']['count']:,} ({stats['size_distribution']['medium_chunks']['percentage']:.1f}%)
- **Large Chunks** (10k-16k tokens): {stats['size_distribution']['large_chunks']['count']:,} ({stats['size_distribution']['large_chunks']['percentage']:.1f}%)

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
- `call_graph_data`: **NEW** - Bidirectional call graph for each function
  - `function_name`: Function name
  - `file_path`: Source file path
  - `line`: Line number
  - `crate_info`: Crate and module context
  - `function_info`: Visibility, async/unsafe/const modifiers, parameters, impl context
  - `calls`: List of outgoing calls (what this function calls)
  - `called_by`: List of incoming calls (what calls this function)
  - `ascii_tree`: ASCII art call graph visualization
  - `flow_description`: Natural language description of call relationships
  - `call_count`: Number of outgoing calls
  - `caller_count`: Number of incoming calls
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

# Access call graph data
if sample['call_graph_data']:
    fn = sample['call_graph_data'][0]
    print(f"\\nFunction: {{fn['function_name']}}")
    print(f"Calls: {{fn['call_count']}}, Called by: {{fn['caller_count']}}")
    print(f"\\nCall Graph:")
    print(fn['ascii_tree'])
    print(f"\\nDescription:")
    print(fn['flow_description'])

# Filter by token count
efficient_chunks = dataset.filter(lambda x: x['token_stats']['total_tokens'] <= 1024)

# Find chunks with specific function calls
chunks_with_async = dataset.filter(
    lambda x: any(fn['function_info'].get('is_async', False)
                  for fn in x.get('call_graph_data', []))
)
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
    script_dir = Path(__file__).parent.resolve()
    input_dir = script_dir / 'out'
    output_dir = script_dir / 'token_aware_dataset_output'
    repo_dir = script_dir / 'hyperswitch'
    repo_url = 'https://github.com/juspay/hyperswitch.git'

    # Token settings
    model_name = "Kwaipilot/KAT-Dev"
    max_tokens = 16384  # 16k context for LoRA training

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
    print("\n[2/5] Setting up hyperswitch repository...")
    if clone_or_update_repo(repo_url, repo_dir):
        print("✓ Repository ready")
    else:
        print("⚠️  Proceeding without source code mining")
        repo_dir = None

    # Build global call graph
    global_call_graph = None
    if repo_dir:
        print("\n[3/5] Building global call graph...")
        try:
            global_call_graph = build_global_call_graph(repo_dir)
        except Exception as e:
            print(f"⚠️  Failed to build call graph: {e}")
            print("  Proceeding without bidirectional call graph")

    # Process files
    print("\n[4/5] Processing documentation files...")
    chunks = process_directory(input_dir, tokenizer, repo_dir, max_tokens, global_call_graph)

    if not chunks:
        print("\n❌ No chunks extracted!")
        return

    # Save dataset
    print("\n[5/5] Saving dataset...")
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
