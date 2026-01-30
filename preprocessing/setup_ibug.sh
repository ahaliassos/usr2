#!/usr/bin/env bash
# Installs ibug face_detection and face_alignment from source.
# These packages are NOT on PyPI and require git cloning.
#
# Prerequisites: git, git-lfs, pip, PyTorch
#
# Usage: bash preprocessing/setup_ibug.sh [install_dir]

set -euo pipefail

INSTALL_DIR="${1:-$(mktemp -d)}"
mkdir -p "$INSTALL_DIR"

echo "Installing ibug packages into: $INSTALL_DIR"

# 1. face_detection (must come first — face_alignment depends on it)
if ! python -c "from ibug.face_detection import RetinaFacePredictor" 2>/dev/null; then
    echo "Installing ibug.face_detection..."
    git clone https://github.com/hhj1897/face_detection.git "$INSTALL_DIR/face_detection"
    cd "$INSTALL_DIR/face_detection"
    git lfs pull
    pip install -e .
    cd -
else
    echo "ibug.face_detection already installed, skipping."
fi

# 2. face_alignment
if ! python -c "from ibug.face_alignment import FANPredictor" 2>/dev/null; then
    echo "Installing ibug.face_alignment..."
    git clone https://github.com/hhj1897/face_alignment.git "$INSTALL_DIR/face_alignment"
    cd "$INSTALL_DIR/face_alignment"
    pip install -e .
    cd -
else
    echo "ibug.face_alignment already installed, skipping."
fi

echo "Done. ibug packages installed successfully."
