#!/usr/bin/env python3
"""
Repository-Level CPT Dataset Generator with Tree-sitter
Implements multi-granular training data with FIM, Instruction, Contrastive, and Graph-augmented formats.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import toml

from transformers import AutoTokenizer
import numpy as np
from tree_sitter import Language, Parser, Node
import tree_sitter_rust


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SampleType(Enum):
    CLM = "clm"  # Causal Language Modeling
    FIM = "fim"  # Fill-in-Middle
    INSTRUCTION = "instruction"  # Instruction tuning
    CONTRASTIVE = "contrastive"  # Positive/negative pairs
    GRAPH_AUGMENTED = "graph_augmented"  # With call graph prefix
    COT = "chain_of_thought"  # Reasoning traces


@dataclass
class CrateInfo:
    name: str
    path: Path
    purpose: str = ""
    local_deps: List[str] = field(default_factory=list)
    external_deps: List[str] = field(default_factory=list)
    features: Dict[str, List[str]] = field(default_factory=dict)
    public_api: List[str] = field(default_factory=list)  # pub use statements


@dataclass
class ModuleInfo:
    path: str
    crate: str
    imports: List[Dict[str, Any]] = field(default_factory=list)
    pub_use: List[str] = field(default_factory=list)
    structs: List[str] = field(default_factory=list)
    enums: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    impls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FunctionInfo:
    name: str
    signature: str
    visibility: str
    file_path: str
    crate: str
    module: str
    start_line: int
    end_line: int
    body: str
    full_text: str
    doc_comments: str = ""
    is_async: bool = False
    is_unsafe: bool = False
    is_const: bool = False
    is_pub: bool = False
    generic_params: List[str] = field(default_factory=list)
    where_clause: Optional[str] = None
    params: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    impl_context: Optional[Dict[str, Any]] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)
    called_by: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TraitInfo:
    name: str
    file_path: str
    crate: str
    module: str
    visibility: str
    super_traits: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    associated_types: List[str] = field(default_factory=list)
    generic_params: List[str] = field(default_factory=list)
    doc_comments: str = ""


@dataclass
class ImplInfo:
    trait_name: Optional[str]  # None for inherent impl
    for_type: str
    file_path: str
    crate: str
    module: str
    generic_params: List[str] = field(default_factory=list)
    where_clause: Optional[str] = None
    methods: List[str] = field(default_factory=list)


# ============================================================================
# TREE-SITTER RUST ANALYZER
# ============================================================================

class RustAnalyzer:
    """Tree-sitter based Rust code analyzer"""
    
    def __init__(self):
        self.parser = Parser(Language(tree_sitter_rust.language()))
    
    def parse_file(self, file_path: Path) -> Optional[Node]:
        """Parse a Rust file and return AST root"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            tree = self.parser.parse(bytes(code, 'utf8'))
            return tree.root_node, code
        except Exception as e:
            print(f"  Warning: Failed to parse {file_path}: {e}")
            return None, None
    
    def get_text(self, node: Node, code: str) -> str:
        """Extract text from a tree-sitter node"""
        return code[node.start_byte:node.end_byte]
    
    def find_nodes_by_type(self, node: Node, node_type: str) -> List[Node]:
        """Recursively find all nodes of a given type"""
        results = []
        if node.type == node_type:
            results.append(node)
        for child in node.children:
            results.extend(self.find_nodes_by_type(child, node_type))
        return results
    
    def extract_functions(self, root: Node, code: str, file_path: str, 
                         crate: str, module: str) -> List[FunctionInfo]:
        """Extract all function definitions"""
        functions = []
        
        for fn_node in self.find_nodes_by_type(root, 'function_item'):
            fn_info = self._extract_function_info(fn_node, code, file_path, crate, module)
            if fn_info:
                functions.append(fn_info)
        
        return functions
    
    def _extract_function_info(self, fn_node: Node, code: str, file_path: str,
                               crate: str, module: str) -> Optional[FunctionInfo]:
        """Extract detailed function information"""
        fn_name = None
        visibility = 'private'
        is_async = False
        is_unsafe = False
        is_const = False
        is_pub = False
        params = []
        generic_params = []
        where_clause = None
        return_type = None
        attributes = []
        body_start = None
        body_end = None
        
        # Extract doc comments (/// or //!)
        doc_comments = self._extract_doc_comments(fn_node, code)
        
        # Extract attributes (#[...])
        attributes = self._extract_attributes(fn_node, code)
        
        for child in fn_node.children:
            if child.type == 'visibility_modifier':
                visibility = self.get_text(child, code)
                is_pub = 'pub' in visibility
            
            elif child.type in ('async', 'unsafe', 'const'):
                if child.type == 'async':
                    is_async = True
                elif child.type == 'unsafe':
                    is_unsafe = True
                elif child.type == 'const':
                    is_const = True
            
            elif child.type == 'identifier':
                fn_name = self.get_text(child, code)
            
            elif child.type == 'type_parameters':
                generic_params = self._extract_generic_params(child, code)
            
            elif child.type == 'parameters':
                params = self._extract_params(child, code)
            
            elif child.type == 'where_clause':
                where_clause = self.get_text(child, code)
            
            elif child.type in ('type_identifier', 'generic_type', 'primitive_type', 
                               'reference_type', 'scoped_type_identifier'):
                # Check if this is return type (after ->)
                prev_sibling = child.prev_sibling
                if prev_sibling and self.get_text(prev_sibling, code) == '->':
                    return_type = self.get_text(child, code)
            
            elif child.type == 'block':
                body_start = child.start_byte
                body_end = child.end_byte
        
        if not fn_name:
            return None
        
        # Extract signature and body
        signature = self.get_text(fn_node, code)
        if body_start and body_end:
            # Get signature without body
            signature = code[fn_node.start_byte:body_start].strip()
            body = code[body_start:body_end]
        else:
            body = ""
        
        full_text = self.get_text(fn_node, code)
        
        # Extract calls from body
        calls = []
        if body_start and body_end:
            calls = self._extract_calls(fn_node, code, body_start, body_end)
        
        # Find impl context
        impl_context = self._find_impl_context(fn_node, code)
        
        return FunctionInfo(
            name=fn_name,
            signature=signature,
            visibility=visibility,
            file_path=file_path,
            crate=crate,
            module=module,
            start_line=fn_node.start_point[0] + 1,
            end_line=fn_node.end_point[0] + 1,
            body=body,
            full_text=full_text,
            doc_comments=doc_comments,
            is_async=is_async,
            is_unsafe=is_unsafe,
            is_const=is_const,
            is_pub=is_pub,
            generic_params=generic_params,
            where_clause=where_clause,
            params=params,
            return_type=return_type,
            attributes=attributes,
            impl_context=impl_context,
            calls=calls
        )
    
    def _extract_doc_comments(self, node: Node, code: str) -> str:
        """Extract doc comments (/// or //!) before a node"""
        doc_lines = []
        prev = node.prev_sibling
        
        while prev and prev.type in ('line_comment', 'block_comment'):
            comment_text = self.get_text(prev, code)
            if comment_text.startswith('///') or comment_text.startswith('//!'):
                doc_lines.insert(0, comment_text)
            prev = prev.prev_sibling
        
        return '\n'.join(doc_lines)
    
    def _extract_attributes(self, node: Node, code: str) -> List[str]:
        """Extract attributes (#[...]) before a node"""
        attrs = []
        prev = node.prev_sibling
        
        while prev and prev.type == 'attribute_item':
            attr_text = self.get_text(prev, code)
            attrs.insert(0, attr_text)
            prev = prev.prev_sibling
        
        return attrs
    
    def _extract_generic_params(self, type_params_node: Node, code: str) -> List[str]:
        """Extract generic type parameters"""
        params = []
        for child in type_params_node.children:
            if child.type in ('type_identifier', 'lifetime', 'const_parameter'):
                params.append(self.get_text(child, code))
        return params
    
    def _extract_params(self, params_node: Node, code: str) -> List[str]:
        """Extract function parameters"""
        params = []
        for child in params_node.children:
            if child.type in ('parameter', 'self_parameter'):
                params.append(self.get_text(child, code))
        return params
    
    def _extract_calls(self, fn_node: Node, code: str, 
                      body_start: int, body_end: int) -> List[Dict[str, Any]]:
        """Extract function calls within a function body"""
        calls = []
        
        for node in self.find_nodes_by_type(fn_node, 'call_expression'):
            # Only include calls within the body
            if node.start_byte < body_start or node.end_byte > body_end:
                continue
            
            call_info = self._extract_call_info(node, code)
            if call_info:
                calls.append(call_info)
        
        return calls
    
    def _extract_call_info(self, call_node: Node, code: str) -> Optional[Dict[str, Any]]:
        """Extract information about a function call"""
        function_node = call_node.child_by_field_name('function')
        if not function_node:
            return None
        
        call_info = {
            'line': call_node.start_point[0] + 1,
            'name': None,
            'qualified_name': None,
            'type': 'direct_call'
        }
        
        if function_node.type == 'identifier':
            call_info['name'] = self.get_text(function_node, code)
            call_info['qualified_name'] = call_info['name']
        
        elif function_node.type == 'scoped_identifier':
            call_info['qualified_name'] = self.get_text(function_node, code)
            name_node = function_node.child_by_field_name('name')
            if name_node:
                call_info['name'] = self.get_text(name_node, code)
            else:
                call_info['name'] = call_info['qualified_name'].split('::')[-1]
        
        elif function_node.type == 'field_expression':
            call_info['type'] = 'method_call'
            field_node = function_node.child_by_field_name('field')
            if field_node:
                call_info['name'] = self.get_text(field_node, code)
                call_info['qualified_name'] = self.get_text(function_node, code)
        
        # Filter out macros and common constructors
        if call_info['name'] and not call_info['name'].endswith('!'):
            if call_info['name'] not in ('Some', 'None', 'Ok', 'Err', 'println', 'print', 
                                          'format', 'vec', 'panic', 'assert', 'unwrap', 'expect'):
                return call_info
        
        return None
    
    def _find_impl_context(self, fn_node: Node, code: str) -> Optional[Dict[str, Any]]:
        """Find if function is inside an impl block"""
        parent = fn_node.parent
        while parent:
            if parent.type == 'impl_item':
                impl_info = {'type': 'inherent_impl', 'for_type': None, 'trait': None}
                
                for child in parent.children:
                    if child.type in ('type_identifier', 'generic_type', 'scoped_type_identifier'):
                        # This might be the trait or the type
                        text = self.get_text(child, code)
                        if impl_info['for_type'] is None:
                            impl_info['for_type'] = text
                        elif impl_info['trait'] is None:
                            impl_info['trait'] = text
                
                # Check if it's a trait impl
                impl_text = self.get_text(parent, code)
                if ' for ' in impl_text:
                    impl_info['type'] = 'trait_impl'
                
                return impl_info
            parent = parent.parent
        
        return None
    
    def extract_traits(self, root: Node, code: str, file_path: str,
                      crate: str, module: str) -> List[TraitInfo]:
        """Extract trait definitions"""
        traits = []
        
        for trait_node in self.find_nodes_by_type(root, 'trait_item'):
            trait_info = self._extract_trait_info(trait_node, code, file_path, crate, module)
            if trait_info:
                traits.append(trait_info)
        
        return traits
    
    def _extract_trait_info(self, trait_node: Node, code: str, file_path: str,
                           crate: str, module: str) -> Optional[TraitInfo]:
        """Extract detailed trait information"""
        trait_name = None
        visibility = 'private'
        super_traits = []
        methods = []
        associated_types = []
        generic_params = []
        
        doc_comments = self._extract_doc_comments(trait_node, code)
        
        for child in trait_node.children:
            if child.type == 'visibility_modifier':
                visibility = self.get_text(child, code)
            
            elif child.type == 'identifier':
                trait_name = self.get_text(child, code)
            
            elif child.type == 'type_parameters':
                generic_params = self._extract_generic_params(child, code)
            
            elif child.type == 'trait_bounds':
                super_traits = self._extract_trait_bounds(child, code)
            
            elif child.type == 'declaration_list':
                for item in child.children:
                    if item.type == 'function_signature_item':
                        name_node = item.child_by_field_name('name')
                        if name_node:
                            methods.append(self.get_text(name_node, code))
                    elif item.type == 'associated_type':
                        name_node = item.child_by_field_name('name')
                        if name_node:
                            associated_types.append(self.get_text(name_node, code))
        
        if not trait_name:
            return None
        
        return TraitInfo(
            name=trait_name,
            file_path=file_path,
            crate=crate,
            module=module,
            visibility=visibility,
            super_traits=super_traits,
            methods=methods,
            associated_types=associated_types,
            generic_params=generic_params,
            doc_comments=doc_comments
        )
    
    def _extract_trait_bounds(self, bounds_node: Node, code: str) -> List[str]:
        """Extract trait bounds from trait_bounds node"""
        bounds = []
        for child in bounds_node.children:
            if child.type in ('type_identifier', 'scoped_type_identifier', 'generic_type'):
                bounds.append(self.get_text(child, code))
        return bounds
    
    def extract_impls(self, root: Node, code: str, file_path: str,
                     crate: str, module: str) -> List[ImplInfo]:
        """Extract impl blocks"""
        impls = []
        
        for impl_node in self.find_nodes_by_type(root, 'impl_item'):
            impl_info = self._extract_impl_info(impl_node, code, file_path, crate, module)
            if impl_info:
                impls.append(impl_info)
        
        return impls
    
    def _extract_impl_info(self, impl_node: Node, code: str, file_path: str,
                          crate: str, module: str) -> Optional[ImplInfo]:
        """Extract impl block information"""
        trait_name = None
        for_type = None
        generic_params = []
        where_clause = None
        methods = []
        
        # Check if it's a trait impl or inherent impl
        impl_text = self.get_text(impl_node, code)
        is_trait_impl = ' for ' in impl_text
        
        for child in impl_node.children:
            if child.type == 'type_parameters':
                generic_params = self._extract_generic_params(child, code)
            
            elif child.type == 'where_clause':
                where_clause = self.get_text(child, code)
            
            elif child.type in ('type_identifier', 'generic_type', 'scoped_type_identifier'):
                if is_trait_impl:
                    if trait_name is None:
                        trait_name = self.get_text(child, code)
                    elif for_type is None:
                        for_type = self.get_text(child, code)
                else:
                    for_type = self.get_text(child, code)
            
            elif child.type == 'declaration_list':
                for item in child.children:
                    if item.type == 'function_item':
                        name_node = None
                        for fn_child in item.children:
                            if fn_child.type == 'identifier':
                                name_node = fn_child
                                break
                        if name_node:
                            methods.append(self.get_text(name_node, code))
        
        if not for_type:
            return None
        
        return ImplInfo(
            trait_name=trait_name,
            for_type=for_type,
            file_path=file_path,
            crate=crate,
            module=module,
            generic_params=generic_params,
            where_clause=where_clause,
            methods=methods
        )
    
    def extract_imports(self, root: Node, code: str) -> List[Dict[str, Any]]:
        """Extract use declarations (imports)"""
        imports = []
        
        for use_node in self.find_nodes_by_type(root, 'use_declaration'):
            import_text = self.get_text(use_node, code)
            
            # Parse the import
            import_info = {
                'raw': import_text,
                'is_pub': 'pub use' in import_text,
                'path': None,
                'items': []
            }
            
            # Extract the main path
            scoped_nodes = self.find_nodes_by_type(use_node, 'scoped_identifier')
            if scoped_nodes:
                import_info['path'] = self.get_text(scoped_nodes[0], code)
            else:
                identifiers = self.find_nodes_by_type(use_node, 'identifier')
                if identifiers:
                    import_info['path'] = self.get_text(identifiers[0], code)
            
            imports.append(import_info)
        
        return imports
    
    def extract_structs(self, root: Node, code: str) -> List[str]:
        """Extract struct names"""
        structs = []
        for struct_node in self.find_nodes_by_type(root, 'struct_item'):
            for child in struct_node.children:
                if child.type == 'type_identifier':
                    structs.append(self.get_text(child, code))
                    break
        return structs
    
    def extract_enums(self, root: Node, code: str) -> List[str]:
        """Extract enum names"""
        enums = []
        for enum_node in self.find_nodes_by_type(root, 'enum_item'):
            for child in enum_node.children:
                if child.type == 'type_identifier':
                    enums.append(self.get_text(child, code))
                    break
        return enums


# ============================================================================
# REPOSITORY ANALYZER
# ============================================================================

class RepositoryAnalyzer:
    """Analyzes entire Rust repository structure"""
    
    # Crate purposes from DeepWiki research
    CRATE_PURPOSES = {
        'router': 'Main application server handling HTTP requests, authentication, and business logic orchestration',
        'api_models': 'External API request/response types (what clients see)',
        'hyperswitch_domain_models': 'Business logic data models bridging API and database layers',
        'diesel_models': 'Database schema types directly mapping to PostgreSQL tables',
        'hyperswitch_connectors': 'Payment provider integrations (Stripe, PayPal, etc.)',
        'hyperswitch_interfaces': 'Trait definitions for connectors and services',
        'common_enums': 'Shared enumerations across request/response and database types',
        'common_utils': 'Utility functions shared across crates',
        'storage_impl': 'Storage backend implementations for database operations',
        'scheduler': 'Background task scheduling and execution',
        'drainer': 'Redis stream processing and database writing',
        'masking': 'PII protection and data masking',
        'redis_interface': 'User-friendly Redis interface',
        'analytics': 'Event logging with Kafka and ClickHouse',
    }
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.analyzer = RustAnalyzer()
        
        # Repository indices
        self.crates: Dict[str, CrateInfo] = {}
        self.modules: Dict[str, ModuleInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.traits: Dict[str, TraitInfo] = {}
        self.impls: List[ImplInfo] = []
        
        # Graphs
        self.crate_graph: Dict[str, List[str]] = defaultdict(list)
        self.module_graph: Dict[str, List[str]] = defaultdict(list)
        self.call_graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_call_graph: Dict[str, List[str]] = defaultdict(list)
        
    def analyze(self):
        """Main analysis entry point"""
        print("🔍 Analyzing Rust repository...")
        
        print("\n[1/6] Discovering crates...")
        self._discover_crates()
        
        print("\n[2/6] Building crate dependency graph...")
        self._build_crate_graph()
        
        print("\n[3/6] Analyzing modules...")
        self._analyze_modules()
        
        print("\n[4/6] Extracting functions...")
        self._extract_all_functions()
        
        print("\n[5/6] Building call graphs...")
        self._build_call_graphs()
        
        print("\n[6/6] Extracting traits and impls...")
        self._extract_traits_and_impls()
        
        self._print_stats()
    
    def _discover_crates(self):
        """Find all Cargo.toml files"""
        cargo_tomls = [p for p in self.repo_path.rglob('Cargo.toml') 
                      if 'target' not in p.parts]
        
        for cargo_path in cargo_tomls:
            crate_info = self._parse_cargo_toml(cargo_path)
            if crate_info:
                self.crates[crate_info.name] = crate_info
        
        print(f"   Found {len(self.crates)} crates")
    
    def _parse_cargo_toml(self, cargo_path: Path) -> Optional[CrateInfo]:
        """Parse Cargo.toml for crate metadata"""
        try:
            with open(cargo_path, 'r') as f:
                cargo_data = toml.load(f)
            
            package = cargo_data.get('package', {})
            name = package.get('name')
            
            if not name:
                return None
            
            # Extract dependencies
            local_deps = []
            external_deps = []
            
            for dep_section in ['dependencies', 'dev-dependencies']:
                if dep_section in cargo_data:
                    for dep_name in cargo_data[dep_section].keys():
                        if dep_name in self.CRATE_PURPOSES or dep_name.startswith('hyperswitch_'):
                            local_deps.append(dep_name)
                        else:
                            external_deps.append(dep_name)
            
            purpose = self.CRATE_PURPOSES.get(name, "")
            
            return CrateInfo(
                name=name,
                path=cargo_path.parent,
                purpose=purpose,
                local_deps=local_deps,
                external_deps=external_deps,
                features=cargo_data.get('features', {})
            )
        except Exception as e:
            return None
    
    def _build_crate_graph(self):
        """Build crate dependency DAG"""
        local_crate_names = set(self.crates.keys())
        
        for crate_name, crate_info in self.crates.items():
            for dep in crate_info.local_deps:
                if dep in local_crate_names:
                    self.crate_graph[crate_name].append(dep)
        
        print(f"   Built graph with {sum(len(deps) for deps in self.crate_graph.values())} edges")
    
    def _analyze_modules(self):
        """Analyze all .rs files as modules"""
        for crate_name, crate_info in self.crates.items():
            rs_files = [f for f in crate_info.path.rglob('*.rs') 
                       if 'target' not in f.parts]
            
            for rs_file in rs_files:
                self._analyze_module(rs_file, crate_name)
        
        print(f"   Analyzed {len(self.modules)} modules")
    
    def _analyze_module(self, file_path: Path, crate: str):
        """Analyze single module file"""
        root, code = self.analyzer.parse_file(file_path)
        if not root:
            return
        
        # Get module path
        rel_path = file_path.relative_to(self.repo_path)
        module_key = f"{crate}::{rel_path}"
        
        module_info = ModuleInfo(
            path=str(rel_path),
            crate=crate,
            imports=self.analyzer.extract_imports(root, code),
            structs=self.analyzer.extract_structs(root, code),
            enums=self.analyzer.extract_enums(root, code)
        )
        
        self.modules[module_key] = module_info
    
    def _extract_all_functions(self):
        """Extract all functions from all modules"""
        total = 0
        for crate_name, crate_info in self.crates.items():
            rs_files = [f for f in crate_info.path.rglob('*.rs') 
                       if 'target' not in f.parts]
            
            for rs_file in rs_files:
                root, code = self.analyzer.parse_file(rs_file)
                if not root:
                    continue
                
                # Determine module name
                rel_path = rs_file.relative_to(self.repo_path)
                module = str(rel_path.with_suffix(''))
                
                functions = self.analyzer.extract_functions(root, code, str(rel_path), 
                                                           crate_name, module)
                
                for fn in functions:
                    fn_key = f"{crate_name}::{module}::{fn.name}"
                    self.functions[fn_key] = fn
                    total += 1
        
        print(f"   Extracted {total} functions")
    
    def _build_call_graphs(self):
        """Build function call graphs"""
        # Forward graph (what each function calls)
        for fn_key, fn_info in self.functions.items():
            for call in fn_info.calls:
                callee_name = call['name']
                self.call_graph[fn_key].append(callee_name)
                
                # Also build reverse graph
                self.reverse_call_graph[callee_name].append({
                    'name': fn_info.name,
                    'file': fn_info.file_path,
                    'line': call['line'],
                    'crate': fn_info.crate
                })
        
        print(f"   Built call graph with {len(self.call_graph)} nodes")
    
    def _extract_traits_and_impls(self):
        """Extract all traits and impl blocks"""
        trait_count = 0
        impl_count = 0
        
        for crate_name, crate_info in self.crates.items():
            rs_files = [f for f in crate_info.path.rglob('*.rs') 
                       if 'target' not in f.parts]
            
            for rs_file in rs_files:
                root, code = self.analyzer.parse_file(rs_file)
                if not root:
                    continue
                
                rel_path = rs_file.relative_to(self.repo_path)
                module = str(rel_path.with_suffix(''))
                
                # Extract traits
                traits = self.analyzer.extract_traits(root, code, str(rel_path), 
                                                     crate_name, module)
                for trait in traits:
                    trait_key = f"{crate_name}::{module}::{trait.name}"
                    self.traits[trait_key] = trait
                    trait_count += 1
                
                # Extract impls
                impls = self.analyzer.extract_impls(root, code, str(rel_path), 
                                                   crate_name, module)
                self.impls.extend(impls)
                impl_count += len(impls)
        
        print(f"   Extracted {trait_count} traits and {impl_count} impl blocks")
    
    def _print_stats(self):
        """Print analysis statistics"""
        print("\n📊 Repository Analysis Complete:")
        print(f"   Crates: {len(self.crates)}")
        print(f"   Modules: {len(self.modules)}")
        print(f"   Functions: {len(self.functions)}")
        print(f"   Traits: {len(self.traits)}")
        print(f"   Impl blocks: {len(self.impls)}")
        print(f"   Call graph edges: {sum(len(calls) for calls in self.call_graph.values())}")


# ============================================================================
# SAMPLE GENERATORS
# ============================================================================

class SampleGenerator:
    """Generate training samples in different formats"""
    
    def __init__(self, repo_analyzer: RepositoryAnalyzer, tokenizer):
        self.analyzer = repo_analyzer
        self.tokenizer = tokenizer
        self.samples = []
    
    def generate_all_samples(self):
        """Generate all types of training samples"""
        print("\n📝 Generating training samples...")

        print("\n[1/7] Repository structure samples...")
        self._generate_repo_structure_samples()

        print("\n[2/7] Crate-level samples...")
        self._generate_crate_samples()

        print("\n[3/7] Module-level samples...")
        self._generate_module_samples()

        print("\n[4/7] Function samples (CLM, FIM, Graph-augmented)...")
        self._generate_function_samples()

        print("\n[5/7] Trait and impl samples...")
        self._generate_trait_samples()

        print("\n[6/7] Pattern samples (Connector, Type conversion)...")
        self._generate_pattern_samples()

        print("\n[7/7] Instruction tuning samples...")
        self._generate_instruction_samples()
        
        print(f"\n✅ Generated {len(self.samples)} total samples")
        return self.samples
    
    def _generate_repo_structure_samples(self):
        """Generate repository-level overview"""
        # Crate dependency graph
        graph_text = "# Hyperswitch Crate Dependency Graph\n\n"
        graph_text += "```\n"
        graph_text += "Repository Structure:\n"
        
        for crate_name in sorted(self.analyzer.crates.keys()):
            deps = self.analyzer.crate_graph.get(crate_name, [])
            purpose = self.analyzer.crates[crate_name].purpose
            
            graph_text += f"\n{crate_name}/"
            if purpose:
                graph_text += f"\n  Purpose: {purpose}"
            if deps:
                graph_text += f"\n  Depends on: {', '.join(deps)}"
        
        graph_text += "\n```\n"
        
        sample = {
            'id': 'repo_structure',
            'type': SampleType.CLM.value,
            'granularity': 'repository',
            'content': graph_text,
            'metadata': {
                'total_crates': len(self.analyzer.crates),
                'total_modules': len(self.analyzer.modules),
                'total_functions': len(self.analyzer.functions)
            }
        }
        
        self.samples.append(sample)
    
    def _generate_crate_samples(self):
        """Generate per-crate samples"""
        for crate_name, crate_info in self.analyzer.crates.items():
            content = self._build_crate_context(crate_name, crate_info)
            
            sample = {
                'id': f'crate_{crate_name}',
                'type': SampleType.CLM.value,
                'granularity': 'crate',
                'hierarchy': {
                    'repository': 'hyperswitch',
                    'crate': crate_name
                },
                'content': content,
                'metadata': {
                    'purpose': crate_info.purpose,
                    'local_deps': crate_info.local_deps,
                    'external_deps': crate_info.external_deps[:10]
                }
            }
            
            self.samples.append(sample)
    
    def _build_crate_context(self, crate_name: str, crate_info: CrateInfo) -> str:
        """Build detailed crate context"""
        lines = []
        
        lines.append(f"// CRATE: {crate_name}")
        if crate_info.purpose:
            lines.append(f"// PURPOSE: {crate_info.purpose}")
        
        if crate_info.local_deps:
            lines.append(f"// DEPENDS_ON: {', '.join(crate_info.local_deps)}")
        
        # Find who depends on this crate
        dependents = [name for name, deps in self.analyzer.crate_graph.items() 
                     if crate_name in deps]
        if dependents:
            lines.append(f"// DEPENDED_BY: {', '.join(dependents)}")
        
        lines.append("")
        lines.append("## Public API")
        
        # List key types and traits
        crate_modules = [m for m in self.analyzer.modules.values() 
                        if m.crate == crate_name]
        
        all_structs = set()
        all_traits = set()
        
        for module in crate_modules:
            all_structs.update(module.structs)
            all_traits.update(module.traits)
        
        if all_structs:
            lines.append(f"\nStructs: {', '.join(sorted(list(all_structs)[:20]))}")
        
        if all_traits:
            lines.append(f"\nTraits: {', '.join(sorted(list(all_traits)[:20]))}")
        
        return '\n'.join(lines)
    
    def _generate_function_samples(self):
        """Generate function-level samples with different formats"""
        # Select important functions (those with most calls or most callers)
        function_scores = {}
        
        for fn_key, fn_info in self.analyzer.functions.items():
            score = len(fn_info.calls) + len(self.analyzer.reverse_call_graph.get(fn_info.name, []))
            if fn_info.is_pub:
                score += 5
            if fn_info.impl_context:
                score += 3
            function_scores[fn_key] = score
        
        # Get top 500 functions (increased from 100)
        top_functions = sorted(function_scores.items(), key=lambda x: x[1], reverse=True)[:500]
        
        for fn_key, score in top_functions:
            fn_info = self.analyzer.functions[fn_key]
            
            # Generate CLM sample
            clm_sample = self._create_clm_sample(fn_info)
            self.samples.append(clm_sample)
            
            # Generate FIM sample
            fim_sample = self._create_fim_sample(fn_info)
            self.samples.append(fim_sample)
            
            # Generate graph-augmented sample
            graph_sample = self._create_graph_augmented_sample(fn_info)
            self.samples.append(graph_sample)
    
    def _create_clm_sample(self, fn_info: FunctionInfo) -> Dict[str, Any]:
        """Create Causal Language Modeling sample"""
        content = self._build_function_context(fn_info)
        content += "\n\n" + fn_info.full_text
        
        return {
            'id': f'clm_{fn_info.crate}_{fn_info.name}',
            'type': SampleType.CLM.value,
            'granularity': 'function',
            'hierarchy': {
                'repository': 'hyperswitch',
                'crate': fn_info.crate,
                'module': fn_info.module,
                'function': fn_info.name
            },
            'content': content,
            'metadata': {
                'is_pub': fn_info.is_pub,
                'is_async': fn_info.is_async,
                'impl_context': fn_info.impl_context
            }
        }
    
    def _create_fim_sample(self, fn_info: FunctionInfo) -> Dict[str, Any]:
        """Create Fill-in-Middle sample"""
        context = self._build_function_context(fn_info)
        
        prefix = f"{context}\n\n{fn_info.signature} {{"
        suffix = "\n}"
        middle = fn_info.body.strip()
        
        content = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}<|endoftext|>"
        
        return {
            'id': f'fim_{fn_info.crate}_{fn_info.name}',
            'type': SampleType.FIM.value,
            'granularity': 'function',
            'hierarchy': {
                'repository': 'hyperswitch',
                'crate': fn_info.crate,
                'module': fn_info.module,
                'function': fn_info.name
            },
            'content': content,
            'metadata': {
                'signature': fn_info.signature,
                'has_body': bool(fn_info.body)
            }
        }
    
    def _create_graph_augmented_sample(self, fn_info: FunctionInfo) -> Dict[str, Any]:
        """Create sample with call graph prefix"""
        graph_context = self._build_call_graph_context(fn_info)
        function_context = self._build_function_context(fn_info)
        
        content = f"{graph_context}\n\n{function_context}\n\n{fn_info.full_text}"
        
        return {
            'id': f'graph_{fn_info.crate}_{fn_info.name}',
            'type': SampleType.GRAPH_AUGMENTED.value,
            'granularity': 'function',
            'hierarchy': {
                'repository': 'hyperswitch',
                'crate': fn_info.crate,
                'module': fn_info.module,
                'function': fn_info.name
            },
            'content': content,
            'metadata': {
                'call_count': len(fn_info.calls),
                'caller_count': len(self.analyzer.reverse_call_graph.get(fn_info.name, []))
            }
        }
    
    def _build_function_context(self, fn_info: FunctionInfo) -> str:
        """Build hierarchical context for a function"""
        lines = []
        
        lines.append(f"// REPO: hyperswitch")
        lines.append(f"// CRATE: {fn_info.crate}")
        
        crate_info = self.analyzer.crates.get(fn_info.crate)
        if crate_info:
            if crate_info.purpose:
                lines.append(f"// CRATE_PURPOSE: {crate_info.purpose}")
            if crate_info.local_deps:
                lines.append(f"// CRATE_DEPS: {', '.join(crate_info.local_deps)}")
        
        lines.append(f"// MODULE: {fn_info.module}")
        lines.append(f"// FUNCTION: {fn_info.name}")
        
        if fn_info.visibility != 'private':
            lines.append(f"// VISIBILITY: {fn_info.visibility}")
        
        if fn_info.impl_context:
            impl_ctx = fn_info.impl_context
            if impl_ctx['type'] == 'trait_impl':
                lines.append(f"// IMPL: {impl_ctx['trait']} for {impl_ctx['for_type']}")
            else:
                lines.append(f"// IMPL: {impl_ctx['for_type']}")
        
        return '\n'.join(lines)
    
    def _build_call_graph_context(self, fn_info: FunctionInfo) -> str:
        """Build call graph visualization"""
        lines = []
        
        lines.append("[CALL_GRAPH]")
        lines.append(f"{fn_info.name} ({fn_info.file_path}:{fn_info.start_line})")
        
        # Outgoing calls
        if fn_info.calls:
            for call in fn_info.calls[:10]:
                lines.append(f"  ├─→ {call['name']} (line {call['line']})")
        
        # Incoming calls
        callers = self.analyzer.reverse_call_graph.get(fn_info.name, [])
        if callers:
            lines.append("")
            lines.append("[CALLED_BY]")
            for caller in callers[:10]:
                lines.append(f"  ←─┤ {caller['name']} ({caller['file']}:{caller['line']})")
        
        return '\n'.join(lines)
    
    def _generate_trait_samples(self):
        """Generate trait and impl samples"""
        # Process all traits
        for trait_key, trait_info in self.analyzer.traits.items():
            content = self._build_trait_context(trait_info)
            
            sample = {
                'id': f'trait_{trait_info.crate}_{trait_info.name}',
                'type': SampleType.CLM.value,
                'granularity': 'trait',
                'hierarchy': {
                    'repository': 'hyperswitch',
                    'crate': trait_info.crate,
                    'module': trait_info.module,
                    'trait': trait_info.name
                },
                'content': content,
                'metadata': {
                    'super_traits': trait_info.super_traits,
                    'methods': trait_info.methods
                }
            }
            
            self.samples.append(sample)
    
    def _build_trait_context(self, trait_info: TraitInfo) -> str:
        """Build trait context with implementations"""
        lines = []
        
        lines.append(f"// TRAIT: {trait_info.name}")
        lines.append(f"// CRATE: {trait_info.crate}")
        lines.append(f"// MODULE: {trait_info.module}")
        
        if trait_info.super_traits:
            lines.append(f"// SUPER_TRAITS: {', '.join(trait_info.super_traits)}")
        
        # Find implementations
        impls = [impl for impl in self.analyzer.impls 
                if impl.trait_name == trait_info.name]
        
        if impls:
            lines.append(f"// IMPLEMENTATIONS: {', '.join([impl.for_type for impl in impls[:10]])}")
        
        if trait_info.doc_comments:
            lines.append("")
            lines.append(trait_info.doc_comments)
        
        lines.append("")
        lines.append(f"pub trait {trait_info.name} {{")
        for method in trait_info.methods:
            lines.append(f"    fn {method}(...);")
        lines.append("}")
        
        return '\n'.join(lines)
    
    def _generate_module_samples(self):
        """Generate module-level samples"""
        # Generate samples for important modules (those with many exports)
        for module_key, module_info in list(self.analyzer.modules.items())[:100]:
            if not module_info.structs and not module_info.enums and not module_info.functions:
                continue

            content = f"""// MODULE: {module_info.path}
// CRATE: {module_info.crate}

// Structs: {', '.join(module_info.structs[:20])}
// Enums: {', '.join(module_info.enums[:20])}
// Functions: {len(module_info.functions)}

// Imports:
{chr(10).join(f"// {imp['raw']}" for imp in module_info.imports[:10])}
"""

            sample = {
                'id': f'module_{module_info.crate}_{module_info.path}',
                'type': SampleType.CLM.value,
                'granularity': 'module',
                'hierarchy': {
                    'repository': 'hyperswitch',
                    'crate': module_info.crate,
                    'module': module_info.path
                },
                'content': content,
                'metadata': {
                    'struct_count': len(module_info.structs),
                    'enum_count': len(module_info.enums)
                }
            }

            self.samples.append(sample)

    def _generate_pattern_samples(self):
        """Generate pattern-specific samples"""
        # Find connector implementations
        connector_impls = [impl for impl in self.analyzer.impls
                          if impl.trait_name and 'Connector' in impl.trait_name]

        # Process all connector impls
        for impl in connector_impls:
            content = f"""// PATTERN: connector_integration
// TRAIT: {impl.trait_name}
// FOR_TYPE: {impl.for_type}
// CRATE: {impl.crate}
// REFERENCE: See stripe/mod.rs, paypal/mod.rs

impl {impl.trait_name} for {impl.for_type} {{
    // Methods: {', '.join(impl.methods)}
}}
"""
            
            sample = {
                'id': f'pattern_connector_{impl.for_type}',
                'type': SampleType.CLM.value,
                'granularity': 'pattern',
                'content': content,
                'metadata': {
                    'pattern_type': 'connector_integration',
                    'trait': impl.trait_name,
                    'for_type': impl.for_type
                }
            }
            
            self.samples.append(sample)
    
    def _generate_instruction_samples(self):
        """Generate instruction tuning samples"""
        # Example: Type conversion instruction
        instruction_sample = {
            'id': 'instruction_type_conversion_1',
            'type': SampleType.INSTRUCTION.value,
            'granularity': 'instruction',
            'content': json.dumps({
                'instruction': 'Convert API model to domain model following Hyperswitch patterns',
                'input': '''// Context: api_models → hyperswitch_domain_models conversion
// Pattern: Implement From trait
// Architecture: api_models cannot depend on diesel_models

pub struct PaymentAttemptAmountDetails {
    pub net_amount: MinorUnit,
    pub amount_to_capture: Option<MinorUnit>,
}''',
                'output': '''impl From<api_models::payments::PaymentAttemptAmountDetails> 
    for hyperswitch_domain_models::payments::AttemptAmountDetails 
{
    fn from(api: api_models::payments::PaymentAttemptAmountDetails) -> Self {
        Self {
            net_amount: api.net_amount,
            amount_to_capture: api.amount_to_capture,
        }
    }
}''',
                'metadata': {
                    'pattern_type': 'type_conversion',
                    'source_crate': 'api_models',
                    'target_crate': 'hyperswitch_domain_models'
                }
            }),
            'metadata': {
                'task_type': 'type_conversion'
            }
        }
        
        self.samples.append(instruction_sample)


# ============================================================================
# MAIN
# ============================================================================

def main():
    repo_path = Path('/Users/architsinghai/code/deepwiki-scripts/hyperswitch')
    output_dir = Path('/Users/architsinghai/code/repo_cpt_dataset_treesitter')
    
    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return
    
    print("=" * 80)
    print("Repository-Level CPT Dataset Generator (Tree-sitter)")
    print("=" * 80)
    
    # Analyze repository
    analyzer = RepositoryAnalyzer(repo_path)
    analyzer.analyze()
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Kwaipilot/KAT-Dev", trust_remote_code=True)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠️  Failed to load tokenizer: {e}")
        print("  Continuing without tokenizer...")
        tokenizer = None
    
    # Generate samples
    generator = SampleGenerator(analyzer, tokenizer)
    samples = generator.generate_all_samples()
    
    # Save dataset
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'dataset.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Dataset saved to {output_file}")
    print(f"   Total samples: {len(samples)}")
    
    # Print distribution
    type_counts = defaultdict(int)
    for sample in samples:
        type_counts[sample['type']] += 1
    
    print("\n📊 Sample Distribution:")
    for sample_type, count in sorted(type_counts.items()):
        print(f"   {sample_type}: {count}")


if __name__ == '__main__':
    main()