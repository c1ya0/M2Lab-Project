#!/bin/bash

# AEGNN-M Conda Installation Script
# Using conda environment can better handle dependencies and GLIBC issues

echo "🚀 Installing AEGNN-M dependencies with Conda..."
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found, please install Anaconda or Miniconda first"
    exit 1
fi

ENV_NAME="aegnn_env"
PYTHON_VERSION="3.11"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment ${ENV_NAME} already exists"
    read -p "Delete and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment..."
        echo "Please run: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create new conda environment
echo "📦 Creating Conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment and install dependencies
echo "📥 Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch (CPU version) - Use stable version 2.4.0
echo "Installing PyTorch 2.4.0..."
pip install torch==2.4.0+cpu torchvision torchaudio==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install NumPy (must be < 2.0 for RDKit compatibility)
echo "Installing NumPy (< 2.0)..."
pip install "numpy<2.0"

# Install PyTorch Geometric and its dependencies
echo "Installing PyTorch Geometric..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

# Install other dependencies
echo "Installing other dependencies..."
pip install pandas scikit-learn matplotlib seaborn tqdm networkx tensorboard plotly pyyaml json5

# Install RDKit (must be installed after NumPy < 2.0)
echo "Installing RDKit..."
pip install rdkit-pypi || pip install rdkit

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not correctly installed"
python -c "import torch_geometric; print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')" 2>/dev/null || echo "❌ PyTorch Geometric not correctly installed"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy not correctly installed"
python -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')" 2>/dev/null || echo "❌ Pandas not correctly installed"
python -c "import rdkit; print(f'✅ RDKit: {rdkit.__version__}')" 2>/dev/null || echo "❌ RDKit not correctly installed"

echo ""
echo "=================================="
echo "✅ Installation completed!"
echo ""
echo "📋 Usage instructions:"
echo "1. Activate environment: conda activate ${ENV_NAME}"
echo "2. Run scripts: python3 scripts/train.py ..."
echo "3. Deactivate environment: conda deactivate"
echo ""

