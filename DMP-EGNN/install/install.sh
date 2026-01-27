#!/bin/bash

# AEGNN-M Installation Script

echo "Starting AEGNN-M dependency installation..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
python_major_minor=$(echo $python_version | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Check if Python 3.13 or higher
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 13 ]; then
    echo "⚠️  Detected Python 3.13+, using latest PyTorch version..."
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"
    TORCH_VERSION=""
else
    echo "Using standard PyTorch installation..."
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"
    TORCH_VERSION=""
fi

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip setuptools wheel

# Install basic dependencies
echo "Installing basic Python dependencies..."
pip3 install numpy pandas scikit-learn matplotlib seaborn tqdm

# Install PyTorch (CPU version)
echo "Installing PyTorch..."
if pip3 install torch torchvision torchaudio $TORCH_INDEX; then
    echo "✅ PyTorch installed successfully"
else
    echo "⚠️  PyTorch installation failed, trying default index..."
    pip3 install torch torchvision torchaudio
fi

# Get installed PyTorch version
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "Installed PyTorch version: $TORCH_VER"

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
# Get PyTorch major version number
TORCH_MAJOR=$(echo $TORCH_VER | cut -d'.' -f1)
TORCH_MINOR=$(echo $TORCH_VER | cut -d'.' -f2)
TORCH_FULL="${TORCH_MAJOR}.${TORCH_MINOR}"

if [[ "$TORCH_VER" == "unknown" ]]; then
    echo "⚠️  Cannot detect PyTorch version, using default installation..."
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
elif [[ "$TORCH_MAJOR" == "2" ]] && [[ "$TORCH_MINOR" -ge 8 ]]; then
    echo "Using PyTorch ${TORCH_FULL} compatible version..."
    # For PyTorch 2.8+, try using corresponding wheel files
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_FULL}+cpu.html || \
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
elif [[ "$TORCH_MAJOR" == "2" ]]; then
    echo "Using PyTorch 2.x compatible version..."
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html || \
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
else
    echo "Using PyTorch 1.x compatible version..."
    pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
fi

# Install RDKit
echo "Installing RDKit..."
if pip3 install rdkit-pypi; then
    echo "✅ RDKit installed successfully"
else
    echo "⚠️  RDKit installation failed, trying alternative..."
    pip3 install rdkit || echo "❌ RDKit installation failed, may need manual installation"
fi

# Install other dependencies
echo "Installing other dependencies..."
pip3 install networkx tensorboard plotly pyyaml json5

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not correctly installed"
python3 -c "import torch_geometric; print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')" 2>/dev/null || echo "❌ PyTorch Geometric not correctly installed"
python3 -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy not correctly installed"
python3 -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')" 2>/dev/null || echo "❌ Pandas not correctly installed"
python3 -c "from rdkit import Chem; print(f'✅ RDKit: {Chem.__version__}')" 2>/dev/null || python3 -c "import rdkit; print(f'✅ RDKit: {rdkit.__version__}')" 2>/dev/null || echo "❌ RDKit not correctly installed"

echo ""
echo "Installation completed!"
echo ""
echo "⚠️  Note: If PyTorch Geometric still has issues, it is recommended to use Conda environment:"
echo "    bash install_with_conda.sh"
echo ""
echo "You can run the following command to test the model:"
echo "python3 scripts/test_model.py"
