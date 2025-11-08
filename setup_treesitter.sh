#!/bin/bash
# Setup tree-sitter for Rust parsing

set -e

echo "Setting up tree-sitter for Rust..."

# Install tree-sitter Python package
echo "Installing tree-sitter Python package..."
uv pip install tree-sitter

# Clone tree-sitter-rust grammar
echo "Cloning tree-sitter-rust grammar..."
if [ ! -d "tree-sitter-rust" ]; then
    git clone https://github.com/tree-sitter/tree-sitter-rust
fi

# Build the Rust language library
echo "Building Rust language library..."
python3 << 'EOF'
from tree_sitter import Language

Language.build_library(
    'build/rust.so',
    ['tree-sitter-rust']
)
print("✓ Built build/rust.so")
EOF

echo ""
echo "✅ Tree-sitter setup complete!"
echo "You can now run: uv run python create_enhanced_code_dataset.py"