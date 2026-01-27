# AEGNN-M: Attention-Enhanced Graph Neural Network for Molecular Properties

## Overview

AEGNN-M (Attention-Enhanced Graph Neural Network for Molecular Properties) is a molecular property prediction model based on GAT-EGNN layers. The model follows the **molecular representation → Graph update (GAT-EGNN layer) → model training** architecture design, effectively learning structural features of molecular graphs and interactions between atoms.

## Model Architecture

AEGNN-M adopts a three-stage architecture:

1. **Molecular Representation**: Convert SMILES strings to molecular graph representation
2. **Graph Update**: Use GAT-EGNN layers for graph updates
3. **Model Training**: Train models based on graph-level representations

## Key Features

- **GAT-EGNN Core**: Combines Graph Attention Network (GAT) and Equivariant Graph Neural Network (EGNN)
- **Molecular Graph Processing**: Supports automatic conversion from SMILES strings to molecular graphs
- **Equivariance Support**: Supports equivariant processing of 3D molecular structures
- **Attention Mechanism**: Graph structure-based attention computation
- **Complete Pipeline**: Includes complete pipeline for data processing, training, and evaluation

### Core Components

#### 1. Molecular Representation
- **SMILES Parsing**: Use RDKit to convert SMILES strings to molecular graphs
- **Atom Feature Extraction**: Atomic number, hybridization type, formal charge, aromaticity, chirality, etc.
- **Bond Features**: Bond type, stereochemistry, aromaticity, etc.
- **Graph Construction**: Graph representation with nodes (atoms) and edges (bonds)

#### 2. Graph Update - GAT-EGNN Layers
- **GAT-EGNN Layer** (`GATEGNNLayer`) - **Core Layer**
  - Combines Graph Attention Network (GAT) and Equivariant Graph Neural Network (EGNN)
  - Supports multi-head attention mechanism
  - Supports equivariant updates
  - Supports edge feature processing
  - This is the most important layer of the AEGNN-M model

- **AEGNN Layer** (`AEGNNLayer`)
  - Uses GAT-EGNN as the core layer
  - Includes feed-forward network and residual connections
  - Supports position encoding

#### 3. Model Training
- **Graph Attention Pooling** (`GraphAttentionPooling`)
  - Graph structure-based attention pooling
  - Does not use Transformer architecture
  - Pooling mechanism suitable for graph data

- **Output Layer**
  - Maps graph-level representations to target properties
  - Supports regression and classification tasks

### Architecture Flow

```
SMILES String → Molecular Graph Construction → GAT-EGNN Layers → Graph Attention Pooling → Output Prediction
      ↓                    ↓                        ↓                      ↓                    ↓
Molecular Rep.      Node/Edge Features        Graph Update         Graph-level Rep.      Molecular Property
```

**Detailed Flow**:
1. **Input**: SMILES string (e.g., "CCO")
2. **Molecular Representation**: Parse into molecular graph, extract atom and bond features
3. **Graph Update**: Update node representations through multiple GAT-EGNN layers
4. **Pooling**: Use graph attention pooling to aggregate node representations into graph-level representation
5. **Output**: Predict molecular properties (e.g., boiling point, solubility, etc.)

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.1.0
- RDKit >= 2022.3.0

### Installation Steps

1. Clone the project
```bash
git clone <repository-url>
cd AEGNN-M
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Install PyTorch Geometric (if not already installed)
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

## Datasets

### Available Datasets

The AEGNN-M project includes the following molecular property prediction datasets:

#### Regression Task Datasets

1. **QM9 Dataset** (`data/qm9_dataset.csv`)
   - **Description**: QM9 molecular property prediction dataset
   - **Target Variable**: Molecular properties
   - **Sample Count**: 30
   - **Usage**: Classic benchmark dataset for molecular property prediction

2. **ESOL Dataset** (`data/esol_dataset.csv`)
   - **Description**: ESOL water solubility prediction dataset
   - **Target Variable**: Water solubility (log S)
   - **Sample Count**: 30
   - **Usage**: Predict molecular solubility in water

3. **Lipophilicity Dataset** (`data/lipophilicity_dataset.csv`)
   - **Description**: Lipophilicity lipophilicity prediction dataset
   - **Target Variable**: Lipophilicity (LogP)
   - **Sample Count**: 30
   - **Usage**: Predict molecular lipophilicity

4. **FreeSolv Dataset** (`data/freesolv_dataset.csv`)
   - **Description**: FreeSolv hydration free energy prediction dataset
   - **Target Variable**: Hydration free energy (kcal/mol)
   - **Sample Count**: 30
   - **Usage**: Predict molecular hydration free energy

#### Classification Task Datasets

5. **BBBP Dataset** (`data/bbbp_dataset.csv`)
   - **Description**: BBBP blood-brain barrier permeability classification dataset
   - **Target Variable**: Blood-brain barrier permeability (0/1)
   - **Sample Count**: 30
   - **Usage**: Predict whether molecules can penetrate the blood-brain barrier

### Data Format

All datasets use the same CSV format:

```csv
smiles,target
CCO,0.5
CC(=O)O,1.2
c1ccccc1,2.1
```

- `smiles`: SMILES string representation of the molecule
- `target`: Target property value (regression) or class label (classification)

### Data Statistics

| Dataset | Task Type | Sample Count | Target Range | Mean | Std Dev |
|---------|-----------|--------------|--------------|------|---------|
| QM9 | Regression | 30 | 0.5-6.8 | 3.39 | 1.49 |
| ESOL | Regression | 30 | -4.64 to -0.31 | -2.07 | 1.12 |
| Lipophilicity | Regression | 30 | -0.17 to 4.31 | 1.72 | 1.31 |
| FreeSolv | Regression | 30 | -7.8 to -0.9 | -4.80 | 1.82 |
| BBBP | Classification | 30 | 0/1 | 0.97 | 0.18 |

## Quick Start

### 🚀 One-Click Execution (Recommended)

#### Complete Pipeline Execution
```bash
# Execute the complete AEGNN-M pipeline (includes installation, testing, training)
./run_all.sh
```

#### Quick Start
```bash
# Most simplified execution flow
./quick_start.sh
```

#### Automatic Testing
```bash
# Run all tests
./test_all.sh
```

#### Automatic Training
```bash
# Train all available datasets
./train_all.sh
```

### 📋 Manual Execution Steps

#### 1. Prepare Data

Data format should be a CSV file containing the following fields:
- `smiles`: SMILES string of the molecule
- `target`: Target property value (regression) or class label (classification)

Example data:
```csv
smiles,target
CCO,0.5
CC(=O)O,1.2
c1ccccc1,2.1
```

#### 2. Train Model

#### Regression Task Examples

```bash
# QM9 molecular property prediction
python scripts/train.py \
    --data_path data/qm9_dataset.csv \
    --model_type regressor \
    --num_epochs 50 \
    --batch_size 16

# ESOL water solubility prediction
python scripts/train.py \
    --data_path data/esol_dataset.csv \
    --model_type regressor \
    --num_epochs 50 \
    --batch_size 16

# Lipophilicity lipophilicity prediction
python scripts/train.py \
    --data_path data/lipophilicity_dataset.csv \
    --model_type regressor \
    --num_epochs 50 \
    --batch_size 16

# FreeSolv hydration free energy prediction
python scripts/train.py \
    --data_path data/freesolv_dataset.csv \
    --model_type regressor \
    --num_epochs 50 \
    --batch_size 16
```

#### Classification Task Examples

```bash
# BBBP blood-brain barrier permeability classification
python scripts/train.py \
    --data_path data/bbbp_dataset.csv \
    --model_type classifier \
    --num_epochs 50 \
    --batch_size 16
```

### 3. Evaluate Model

#### Regression Task Evaluation

```bash
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_path data/qm9_dataset.csv \
    --model_type regressor \
    --plot
```

#### Classification Task Evaluation

```bash
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_path data/bbbp_dataset.csv \
    --model_type classifier \
    --plot
```

## Dataset Usage Guide

### Choosing the Right Dataset

Select the appropriate dataset based on your task requirements:

#### Regression Tasks
- **QM9**: General molecular property prediction
- **ESOL**: Water solubility prediction
- **Lipophilicity**: Lipophilicity prediction
- **FreeSolv**: Hydration free energy prediction

#### Classification Tasks
- **BBBP**: Blood-brain barrier permeability classification

### Quick Start Examples

```bash
# 1. Test model functionality
python scripts/test_model.py

# 2. Start training (select a dataset)
python scripts/train.py --data_path data/qm9_dataset.csv --model_type regressor
```

## Usage Examples

### Basic Usage

```python
from models.aegnn_model import create_aegnn_model
from utils.data_utils import MolecularDataset

# 1. Create AEGNN-M model (follows molecular representation → Graph update → model training)
model = create_aegnn_model(
    model_type='regressor',
    node_features=78,      # Atom feature dimension
    edge_features=4,        # Bond feature dimension
    hidden_dim=256,        # Hidden layer dimension
    num_layers=6,          # Number of GAT-EGNN layers
    num_heads=8,           # Number of attention heads
    use_equivariant=True   # Enable equivariant processing
)

# 2. Load molecular data (Molecular Representation)
dataset = MolecularDataset(
    data_path='data/qm9_dataset.csv',  # Use QM9 dataset
    target_column='target',
    smiles_column='smiles'
)

# 3. Process molecular graph data (SMILES → molecular graph)
graphs = dataset.process_graphs()

# 4. Get data loader
dataloader = dataset.get_dataloader(batch_size=32)

# 5. Model training (Graph Update + Model Training)
for batch in dataloader:
    # Forward propagation: molecular representation → GAT-EGNN → output
    output, attention_weights = model(
        batch.x,           # Atom features
        batch.edge_index,   # Bond indices
        batch.edge_attr,   # Bond features
        batch.batch,       # Batch information
        pos=batch.pos      # 3D positions (optional)
    )
```

### Custom Molecular Representation Construction

```python
from utils.data_utils import MolecularGraphBuilder

# Custom molecular graph builder (Molecular Representation)
graph_builder = MolecularGraphBuilder(
    use_atomic_number=True,    # Atomic number
    use_hybridization=True,    # Hybridization type
    use_formal_charge=True,    # Formal charge
    use_aromatic=True,         # Aromaticity
    use_chirality=True,        # Chirality
    use_hydrogen_bonds=True,   # Hydrogen bonds
    use_bond_type=True,        # Bond type
    use_bond_stereo=True       # Stereochemistry
)

# Single molecule conversion (SMILES → molecular graph)
smiles = "CCO"  # Ethanol
graph = graph_builder.smiles_to_graph(smiles)

print(f"Node feature shape: {graph.x.shape}")      # [num_atoms, node_features]
print(f"Edge index shape: {graph.edge_index.shape}") # [2, num_bonds]
print(f"Edge feature shape: {graph.edge_attr.shape}")  # [num_bonds, edge_features]
```

### GAT-EGNN Attention Weight Visualization

```python
# Get attention weights from GAT-EGNN layers
attention_weights = model.get_attention_weights(x, edge_index, edge_attr, batch, pos)

# Visualize attention weights for each GAT-EGNN layer
import matplotlib.pyplot as plt

for i, attn in enumerate(attention_weights):
    plt.figure(figsize=(12, 8))
    # Display weights for the first attention head
    attn_matrix = attn[0].cpu().numpy()  # [num_edges, heads]
    plt.imshow(attn_matrix.T, cmap='Blues', aspect='auto')
    plt.title(f'GAT-EGNN Layer {i+1} Attention Weights')
    plt.xlabel('Edge Index')
    plt.ylabel('Attention Head')
    plt.colorbar(label='Attention Weight')
    plt.show()

# Analyze attention weight statistics
for i, attn in enumerate(attention_weights):
    print(f"Layer {i+1} Attention Statistics:")
    print(f"  Mean: {attn.mean():.4f}")
    print(f"  Std: {attn.std():.4f}")
    print(f"  Max: {attn.max():.4f}")
    print(f"  Min: {attn.min():.4f}")
```

## Configuration Files

Use YAML configuration files for model configuration:

```yaml
# configs/default_config.yaml
model:
  type: "regressor"
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1

training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

## Model Parameters

### Molecular Representation Parameters

- `node_features`: Atom feature dimension (default: 78)
  - Includes: atomic number, hybridization type, formal charge, aromaticity, chirality, etc.
- `edge_features`: Bond feature dimension (default: 4)
  - Includes: bond type, stereochemistry, aromaticity, etc.

### Graph Update (GAT-EGNN) Parameters

- `hidden_dim`: Hidden layer dimension (default: 256)
- `num_layers`: Number of GAT-EGNN layers (default: 6)
- `num_heads`: Number of attention heads (default: 8)
- `use_equivariant`: Whether to enable equivariance (default: True)
- `alpha`: LeakyReLU negative slope (default: 0.2)
- `dropout`: Dropout rate (default: 0.1)

### Model Training Parameters

- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay (default: 1e-4)
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 100)

## Evaluation Metrics

### Regression Tasks
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Correlation Coefficient

### Classification Tasks
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Structure

```
AEGNN-M/
├── models/
│   └── aegnn_model.py          # Core model implementation
├── utils/
│   └── data_utils.py           # Data processing utilities
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── configs/
│   └── default_config.yaml     # Default configuration
├── data/                       # Data directory
├── checkpoints/               # Model checkpoints
├── logs/                      # Training logs
├── requirements.txt            # Dependencies
└── README.md                  # Documentation
```

## Advanced Features

### 1. Custom Loss Function

```python
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss
```

### 2. Data Augmentation

```python
# Add random noise to features during data processing
def add_noise_to_features(x, noise_std=0.01):
    noise = torch.randn_like(x) * noise_std
    return x + noise
```

### 3. Model Distillation

```python
# Use teacher model to guide student model
def distillation_loss(student_pred, teacher_pred, target, alpha=0.7, temperature=3.0):
    hard_loss = F.mse_loss(student_pred, target)
    soft_loss = F.mse_loss(student_pred, teacher_pred) / (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce model dimensions
   - Use gradient accumulation

2. **Graph Construction Failure**
   - Check SMILES string format
   - Ensure RDKit is correctly installed
   - Check if molecules are valid

3. **Training Not Converging**
   - Adjust learning rate
   - Check data preprocessing
   - Use learning rate scheduler

### Performance Optimization

1. **Use Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

2. **Parallel Data Loading**
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)
```

## Citation

If you use AEGNN-M in your research, please cite:

```bibtex
@article{aegnn_m_2024,
  title={AEGNN-M: Attention-Enhanced Graph Neural Network for Molecular Properties},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests to improve this project.

## Contact

For questions, please contact via:
- Email: your.email@example.com
- GitHub Issues: [Project Issues Page]

---

**Note**: This project is still under development, and the API may change. Please check the latest documentation before use.
