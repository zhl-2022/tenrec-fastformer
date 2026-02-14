#!/bin/bash
# =============================================================================
# Build Script for fast_pipeline Rust Module
# =============================================================================
# Prerequisites: Rust toolchain + maturin
#     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#     source $HOME/.cargo/env
#     pip install maturin
#
# Usage:
#     cd rust_pipeline/
#     bash build.sh            # development build (fast, debug symbols)
#     bash build.sh release    # production build (optimized, no debug symbols)
#     bash build.sh wheel      # build .whl for distribution
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check prerequisites
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust not installed. Run:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

if ! pip show maturin &> /dev/null 2>&1; then
    echo "Installing maturin..."
    pip install maturin
fi

MODE="${1:-dev}"

case "$MODE" in
    dev|debug)
        echo "=== Building fast_pipeline (debug) ==="
        maturin develop
        echo "=== Done! Use: import fast_pipeline ==="
        ;;
    release|prod)
        echo "=== Building fast_pipeline (release, optimized) ==="
        maturin develop --release
        echo "=== Done! Use: import fast_pipeline ==="
        ;;
    wheel)
        echo "=== Building wheel for distribution ==="
        maturin build --release
        echo "=== Wheel created in target/wheels/ ==="
        ;;
    *)
        echo "Usage: bash build.sh [dev|release|wheel]"
        exit 1
        ;;
esac

# Verify import works
echo ""
echo "=== Verifying import ==="
python -c "
import fast_pipeline
print('PostStore:', fast_pipeline.PostStore)
print('dedup_filter:', fast_pipeline.dedup_filter)
print('weighted_score:', fast_pipeline.weighted_score)
print('hash_embedding_indices:', fast_pipeline.hash_embedding_indices)
print()
print('All components loaded successfully!')
" && echo "✅ Verification passed!" || echo "❌ Verification failed!"
