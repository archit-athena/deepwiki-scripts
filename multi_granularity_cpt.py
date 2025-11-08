#!/usr/bin/env python3
"""
MULTI-GRANULARITY CPT DATASET GENERATOR
Generates samples at multiple abstraction levels:
- File level
- Module level (all files in a module)
- Crate level (entire crate)
- Impl collection (all impls for a type)
- Trait ecosystem (trait + all implementors)
- Cross-crate usage (dependency patterns)
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set


class MultiGranularityDatasetGenerator:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.samples = []

        # Analysis data
        self.crate_files = defaultdict(list)  # crate -> [files]
        self.module_files = defaultdict(list)  # module -> [files]
        self.impl_blocks = defaultdict(list)  # type -> [(file, impl_code)]
        self.trait_impls = defaultdict(list)  # trait -> [(type, file, code)]
        self.imports = defaultdict(set)  # file -> set of imports
        self.exports = defaultdict(set)  # file -> set of pub items
        self.crate_deps = defaultdict(set)  # crate -> set of dependent crates

    def analyze_repository(self):
        """Scan and analyze the repository structure"""
        print("📊 Analyzing repository structure...")

        rs_files = list(self.repo_path.rglob('*.rs'))
        print(f"   Found {len(rs_files)} Rust files")

        for rs_file in rs_files:
            self._analyze_file(rs_file)

        print(f"   Crates found: {len(self.crate_files)}")
        print(f"   Modules found: {len(self.module_files)}")
        print(f"   Types with impls: {len(self.impl_blocks)}")
        print(f"   Traits with impls: {len(self.trait_impls)}")

    def _analyze_file(self, rs_file: Path):
        """Analyze a single file"""
        try:
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            rel_path = rs_file.relative_to(self.repo_path)
            path_str = str(rel_path)

            # Determine crate
            crate_name = self._extract_crate(rel_path)
            self.crate_files[crate_name].append((rs_file, content))

            # Determine module
            module_name = self._extract_module(rel_path, crate_name)
            self.module_files[f"{crate_name}::{module_name}"].append((rs_file, content))

            # Extract impl blocks
            impl_pattern = r'impl(?:\s+<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)(?:\s+where[^{]*?)?\s*\{'
            for match in re.finditer(impl_pattern, content):
                trait_name = match.group(1)  # Can be None for inherent impls
                type_name = match.group(2)

                # Extract the full impl block
                start = match.start()
                impl_code = self._extract_block(content, start)

                if impl_code:
                    self.impl_blocks[type_name].append((path_str, impl_code))

                    if trait_name:
                        self.trait_impls[trait_name].append((type_name, path_str, impl_code))

            # Extract imports (use statements)
            use_pattern = r'use\s+([^;]+);'
            for match in re.finditer(use_pattern, content):
                import_path = match.group(1).strip()
                self.imports[path_str].add(import_path)

                # Track cross-crate dependencies
                if '::' in import_path:
                    potential_crate = import_path.split('::')[0]
                    if potential_crate != crate_name:
                        self.crate_deps[crate_name].add(potential_crate)

            # Extract public items
            pub_pattern = r'pub\s+(?:async\s+)?(?:fn|struct|enum|trait|type|const|static)\s+(\w+)'
            for match in re.finditer(pub_pattern, content):
                item_name = match.group(1)
                self.exports[path_str].add(item_name)

        except Exception as e:
            pass

    def _extract_crate(self, rel_path: Path) -> str:
        """Extract crate name from path"""
        parts = rel_path.parts
        if 'crates' in parts:
            idx = parts.index('crates')
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return parts[0] if parts else 'unknown'

    def _extract_module(self, rel_path: Path, crate_name: str) -> str:
        """Extract module path from file path"""
        parts = list(rel_path.parts)

        # Remove crates/crate_name prefix
        if 'crates' in parts:
            idx = parts.index('crates')
            parts = parts[idx + 2:]  # Skip 'crates' and crate name

        # Remove 'src' if present
        if parts and parts[0] == 'src':
            parts = parts[1:]

        # Remove .rs extension and handle mod.rs
        if parts:
            last = parts[-1].replace('.rs', '')
            if last == 'mod' or last == 'lib':
                parts = parts[:-1]
            else:
                parts[-1] = last

        return '::'.join(parts) if parts else 'root'

    def _extract_block(self, content: str, start: int) -> str:
        """Extract a code block starting from position"""
        brace_count = 0
        in_block = False
        end = start

        for i in range(start, len(content)):
            if content[i] == '{':
                brace_count += 1
                in_block = True
            elif content[i] == '}':
                brace_count -= 1
                if in_block and brace_count == 0:
                    end = i + 1
                    break

        if end > start:
            return content[start:end]
        return None

    def generate_samples(self):
        """Generate samples at all granularity levels"""
        print("\n📝 Generating multi-granularity samples...\n")

        self._generate_file_level()
        self._generate_module_level()
        self._generate_crate_level()
        self._generate_impl_collection()
        self._generate_trait_ecosystem()
        self._generate_cross_crate_patterns()

        print(f"\n✅ Total samples generated: {len(self.samples)}")
        return self.samples

    def _generate_file_level(self):
        """Generate file-level samples - ONLY LARGE FILES"""
        print("[1/6] File-level samples (BIG files only)...")
        count = 0

        for crate_name, files in self.crate_files.items():
            for rs_file, content in files:
                file_size = len(content)

                # Only include LARGE files: 5KB to 500KB
                if file_size < 5000 or file_size > 500000:
                    continue

                rel_path = rs_file.relative_to(self.repo_path)

                sample = {
                    'id': f'file_{hash(str(rel_path))}',
                    'type': 'clm',
                    'granularity': 'file',
                    'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
File: {rel_path}
Size: {file_size} bytes
</path>

<file>
{content}
</file>
''',
                    'metadata': {
                        'crate': crate_name,
                        'file': str(rel_path),
                        'size': file_size
                    }
                }

                self.samples.append(sample)
                count += 1

        print(f"   Generated {count} file-level samples (all 5KB+)")

    def _generate_module_level(self):
        """Generate module-level samples (all files in a module combined)"""
        print("[2/6] Module-level samples...")
        count = 0

        for module_name, files in self.module_files.items():
            if len(files) < 2:  # Skip single-file modules (already in file-level)
                continue

            # Combine all files in the module
            combined_content = []
            file_list = []
            total_size = 0

            for rs_file, content in files:
                rel_path = rs_file.relative_to(self.repo_path)
                file_list.append(str(rel_path))
                combined_content.append(f"// File: {rel_path}\n{content}")
                total_size += len(content)

            # Skip if too large
            if total_size > 1000000:  # 1MB limit for modules
                continue

            crate_name = module_name.split('::')[0]

            sample = {
                'id': f'module_{hash(module_name)}',
                'type': 'clm',
                'granularity': 'module',
                'content': f'''<path>
Repository: hyperswitch
Module: {module_name}
Files: {len(files)}
Total size: {total_size} bytes
</path>

<module>
{chr(10).join(combined_content)}
</module>
''',
                'metadata': {
                    'crate': crate_name,
                    'module': module_name,
                    'num_files': len(files),
                    'files': file_list,
                    'size': total_size
                }
            }

            self.samples.append(sample)
            count += 1

        print(f"   Generated {count} module-level samples")

    def _generate_crate_level(self):
        """Generate crate-level samples (entire crate)"""
        print("[3/6] Crate-level samples...")
        count = 0

        for crate_name, files in self.crate_files.items():
            # Combine all files
            combined_content = []
            total_size = 0

            for rs_file, content in files:
                rel_path = rs_file.relative_to(self.repo_path)
                combined_content.append(f"// File: {rel_path}\n{content}")
                total_size += len(content)

            # Skip if too large (limit to 2MB per crate)
            if total_size > 2000000:
                continue

            # Get dependencies
            deps = sorted(self.crate_deps.get(crate_name, []))

            sample = {
                'id': f'crate_{hash(crate_name)}',
                'type': 'clm',
                'granularity': 'crate',
                'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
Files: {len(files)}
Total size: {total_size} bytes
Dependencies: {', '.join(deps) if deps else 'none'}
</path>

<crate>
{chr(10).join(combined_content)}
</crate>
''',
                'metadata': {
                    'crate': crate_name,
                    'num_files': len(files),
                    'size': total_size,
                    'dependencies': deps
                }
            }

            self.samples.append(sample)
            count += 1

        print(f"   Generated {count} crate-level samples")

    def _generate_impl_collection(self):
        """Generate samples showing all implementations for a type"""
        print("[4/6] Impl-collection samples...")
        count = 0

        for type_name, impls in self.impl_blocks.items():
            if len(impls) < 2:  # Need at least 2 impls to be interesting
                continue

            # Combine all impls for this type
            impl_texts = []
            files = set()

            for file_path, impl_code in impls[:10]:  # Limit to 10 impls
                impl_texts.append(f"// From: {file_path}\n{impl_code}")
                files.add(file_path)

            combined = '\n\n'.join(impl_texts)

            sample = {
                'id': f'impl_collection_{hash(type_name)}',
                'type': 'clm',
                'granularity': 'impl_collection',
                'content': f'''<path>
Repository: hyperswitch
Type: {type_name}
Implementations: {len(impls)}
Files: {len(files)}
</path>

<impl_collection>
{combined}
</impl_collection>
''',
                'metadata': {
                    'type': type_name,
                    'num_impls': len(impls),
                    'files': sorted(files)
                }
            }

            self.samples.append(sample)
            count += 1

        print(f"   Generated {count} impl-collection samples")

    def _generate_trait_ecosystem(self):
        """Generate samples showing trait + all implementations"""
        print("[5/6] Trait-ecosystem samples...")
        count = 0

        for trait_name, implementations in self.trait_impls.items():
            if len(implementations) < 2:
                continue

            impl_texts = []
            types = []

            for type_name, file_path, impl_code in implementations[:15]:  # Limit to 15
                impl_texts.append(f"// {type_name} in {file_path}\n{impl_code}")
                types.append(type_name)

            combined = '\n\n'.join(impl_texts)

            sample = {
                'id': f'trait_ecosystem_{hash(trait_name)}',
                'type': 'clm',
                'granularity': 'trait_ecosystem',
                'content': f'''<path>
Repository: hyperswitch
Trait: {trait_name}
Implementors: {len(implementations)}
Types: {', '.join(types[:10])}
</path>

<trait_ecosystem>
{combined}
</trait_ecosystem>
''',
                'metadata': {
                    'trait': trait_name,
                    'num_implementors': len(implementations),
                    'types': types
                }
            }

            self.samples.append(sample)
            count += 1

        print(f"   Generated {count} trait-ecosystem samples")

    def _generate_cross_crate_patterns(self):
        """Generate samples showing cross-crate dependency patterns"""
        print("[6/6] Cross-crate usage patterns...")
        count = 0

        for crate_name, deps in self.crate_deps.items():
            if not deps:
                continue

            # Show how this crate uses its dependencies
            usage_examples = []

            for dep in sorted(deps)[:5]:  # Top 5 dependencies
                # Find files that import from this dep
                files_using_dep = []

                for file_path, imports in self.imports.items():
                    if any(dep in imp for imp in imports):
                        files_using_dep.append(file_path)

                if files_using_dep:
                    usage_examples.append(f"Dependency: {dep}\n  Used in {len(files_using_dep)} files: {', '.join(files_using_dep[:3])}")

            if usage_examples:
                sample = {
                    'id': f'cross_crate_{hash(crate_name)}',
                    'type': 'clm',
                    'granularity': 'cross_crate',
                    'content': f'''<path>
Repository: hyperswitch
Crate: {crate_name}
Dependencies: {len(deps)}
</path>

<cross_crate_usage>
# Cross-crate dependencies for {crate_name}

{chr(10).join(usage_examples)}
</cross_crate_usage>
''',
                    'metadata': {
                        'crate': crate_name,
                        'num_dependencies': len(deps),
                        'dependencies': sorted(deps)
                    }
                }

                self.samples.append(sample)
                count += 1

        print(f"   Generated {count} cross-crate usage samples")

    def save_dataset(self, output_file: Path):
        """Save dataset to JSONL"""
        print(f"\n💾 Saving dataset to: {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Print statistics
        granularity_counts = Counter(s['granularity'] for s in self.samples)

        print("\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(self.samples)}")
        print("\n   By granularity:")
        for granularity, count in granularity_counts.most_common():
            print(f"     {granularity}: {count}")

        # Crate distribution
        crate_counts = Counter()
        for s in self.samples:
            if 'crate' in s.get('metadata', {}):
                crate_counts[s['metadata']['crate']] += 1

        print("\n   Top crates:")
        for crate, count in crate_counts.most_common(15):
            print(f"     {crate}: {count} samples")


def main():
    repo_path = Path('/Users/architsinghai/code/deepwiki-scripts/hyperswitch')
    output_file = Path('/Users/architsinghai/code/repo_cpt_dataset_clean/multi_granularity_dataset.jsonl')

    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return

    print("=" * 80)
    print("MULTI-GRANULARITY CPT DATASET GENERATOR")
    print("Granularities: file, module, crate, impl-collection, trait-ecosystem, cross-crate")
    print("=" * 80)

    generator = MultiGranularityDatasetGenerator(repo_path)
    generator.analyze_repository()
    generator.generate_samples()
    generator.save_dataset(output_file)

    print(f"\n✅ Done! Dataset saved to: {output_file}")


if __name__ == '__main__':
    main()
