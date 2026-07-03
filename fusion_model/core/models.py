import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, global_max_pool, global_add_pool
# from nemo_chem.models.megamolbart import REGRegExTokenizer
from core.tokenizer import REGRegExTokenizer
from torch.cuda.amp import autocast
from typing import List, Optional

from core.edmpnn_model_new import AEGNNM as DMPEGNN  # 類別在 edmpnn_model_new.py 內原名為 AEGNNM

# =================== Configuration Constants ===================
DESC_DIM = 200
MMB_OUTPUT_DIM = 256

# =================== Utility Functions ===================
def get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh()
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "BatchNorm":
        return BatchNorm(dim)
    elif norm_type == "LayerNorm":
        return nn.LayerNorm(dim)
    else:
        return nn.Identity() # no normalization

# =================== Model Components ===================
class SharedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, activation="relu", dropout=0.2, norm_type="none"):
        super().__init__()
        layers = []

        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(get_norm_layer(norm_type, hidden_dim))
            # Create a fresh activation module for each layer to avoid
            # shared-module side effects in state_dict / parameters.
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim # used for task_head input dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# MegaMolBART (output dim: 256)
class MegaMolBART_Finetuned_Model(nn.Module):
    def __init__(self, pretrained_model: nn.Module, hidden_dim: int = 512):
        super().__init__()
        self.tokenizer = REGRegExTokenizer() # from paper
        self.encoder_embedding = pretrained_model.enc_dec_model.encoder_embedding
        self.encoder = pretrained_model.enc_dec_model.enc_dec_model.encoder
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, MMB_OUTPUT_DIM),
        )

    def forward(self, smiles: List[str]) -> torch.Tensor:
        token_ids, encoder_masks = self.tokenizer.tokenize(smiles)
        position_ids = torch.arange(token_ids.shape[1], device=token_ids.device)
        embeddings = self.encoder_embedding(token_ids, position_ids).to(dtype=torch.float16)
        encoder_masks = encoder_masks.to(dtype=torch.float16)

        with autocast(dtype=torch.float16):
            encoder_output = self.encoder(embeddings, encoder_masks) # [sequence_length, batch_size, 512]
            pooled_output = encoder_output[0, :, :]
            return self.head(pooled_output).squeeze(1)   

# GCN 
class GCN_Model(nn.Module):
    def __init__(self, 
                input_dim: int = 75,
                hidden_dim: int = 128, 
                output_dim: int = 1,  
                num_layers: int = 3,
                dropout: float = 0.2, 
                activation: str = "relu", 
                norm_type: str = "batch", 
                pooling: str = "mean"):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = get_activation(activation)
        self.pooling = pooling
        self.dropout = dropout

        # first layer: input_dim -> hidden_dim
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        self.norm_layers.append(get_norm_layer(norm_type, hidden_dim))

        # additional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norm_layers.append(get_norm_layer(norm_type, hidden_dim))
        
        # final fully connected layer: hidden_dim -> output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # GCN layers
        for conv, norm in zip(self.gcn_layers, self.norm_layers):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            

        # global pooling
        pool_fn = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "add": global_add_pool
        }.get(self.pooling, global_mean_pool)
        x = pool_fn(x, batch)
        
        # final fully connected layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


# MegaMolBART -> MLP
class MMB_Model(nn.Module):
    def __init__(self, 
                mmb_model: nn.Module, 
                task_output_dims: List[int],
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.mmb_model = mmb_model
        
        self.shared_mlp = SharedMLP(
            input_dim=MMB_OUTPUT_DIM,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(self, smiles: List[str], task_index: int) -> torch.Tensor:

        mmb_embeddings = self.mmb_model(smiles)
        
        # Process through MLP and task head
        shared_features = self.shared_mlp(mmb_embeddings.float()) # converts to float32 (MLP uses float32)
        return self.task_heads[task_index](shared_features)


# Descriptor -> MLP
class Desc_Model(nn.Module):
    def __init__(self, 
                task_output_dims: List[int],
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.shared_mlp = SharedMLP(
            input_dim=DESC_DIM,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(self, descriptors: torch.Tensor, task_index: int) -> torch.Tensor:
        
        descriptors = descriptors.contiguous().view(-1, DESC_DIM) # change the shape to (batch_size, 200)
        
        # Process through MLP and task head
        shared_features = self.shared_mlp(descriptors)
        return self.task_heads[task_index](shared_features)

# ---------------------------------------

# (GCN + MegaMolBART) -> MLP
class GCN_MMB_Model(nn.Module):
    def __init__(self, 
                gcn_model: nn.Module, 
                mmb_model: nn.Module, 
                gcn_output_dim: int,
                task_output_dims: List[int],
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.gcn_model = gcn_model
        self.mmb_model = mmb_model
        
        combined_dim = gcn_output_dim + MMB_OUTPUT_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])
        
    def forward(self, smiles: List[str], x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor, task_index: int) -> torch.Tensor:

        gcn_embeddings = self.gcn_model(x, edge_index, batch) # shape: (batch_size, gcn_output_dim)
        mmb_embeddings = self.mmb_model(smiles) # shape: (batch_size, megamolbart_output_dim)
        
        combined_embeddings = torch.cat([gcn_embeddings, mmb_embeddings], dim=1) # shape: (batch_size, combined_dim)
        
        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings.float())
        return self.task_heads[task_index](shared_features)


# (MegaMolBART + descriptor) -> MLP
class MMB_Desc_Model(nn.Module):
    def __init__(self, 
                mmb_model: nn.Module, 
                task_output_dims: List[int],
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.mmb_model = mmb_model
        
        combined_dim = MMB_OUTPUT_DIM + DESC_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])
        
    def forward(self, smiles: List[str], descriptors: torch.Tensor, task_index: int) -> torch.Tensor:
        
        mmb_embeddings = self.mmb_model(smiles) # shape: (batch_size, megamolbart_output_dim)
        descriptors = descriptors.view(mmb_embeddings.size(0), -1) # shape: (batch_size, combined_dim)
        
        combined_embeddings = torch.cat([mmb_embeddings, descriptors], dim=1) # shape: (batch_size, combined_dim)
        
        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings.float())
        return self.task_heads[task_index](shared_features)


# (GCN + descriptor) -> MLP
class GCN_Desc_Model(nn.Module):
    def __init__(self, 
                gcn_model: nn.Module, 
                gcn_output_dim: int,
                task_output_dims: List[int],
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.gcn_model = gcn_model
        
        combined_dim = gcn_output_dim + DESC_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, descriptors: torch.Tensor, 
               batch: torch.Tensor, task_index: int) -> torch.Tensor:
        
        gcn_embeddings = self.gcn_model(x, edge_index, batch) # shape: (batch_size, gcn_output_dim)
        descriptors = descriptors.view(gcn_embeddings.size(0), -1) # shape: (batch_size, combined_dim)
        
        combined_embeddings = torch.cat([gcn_embeddings, descriptors], dim=1) # shape: (batch_size, combined_dim)
        
        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings)
        return self.task_heads[task_index](shared_features)


# (GCN + MegaMolBART + descriptor) -> MLP
class GCN_MMB_Desc_Model(nn.Module):
    def __init__(self, 
                gcn_model: nn.Module, 
                mmb_model: nn.Module, 
                gcn_output_dim: int,
                task_output_dims: List[int],
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.gcn_model = gcn_model
        self.mmb_model = mmb_model
        
        combined_dim = gcn_output_dim + MMB_OUTPUT_DIM + DESC_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])
        
    def forward(self, smiles: List[str], x: torch.Tensor, edge_index: torch.Tensor, 
               descriptors: torch.Tensor, batch: torch.Tensor, task_index: int) -> torch.Tensor:
        
        gcn_embeddings= self.gcn_model(x, edge_index, batch) # shape: (batch_size, gcn_output_dim)
        mmb_embeddings = self.mmb_model(smiles) # shape: (batch_size, megamolbart_output_dim)
        descriptors = descriptors.view(gcn_embeddings.size(0), -1) # change the shape to (batch_size, 200)
        
        combined_embeddings = torch.cat([gcn_embeddings, mmb_embeddings, descriptors], dim=1) # (batch_size, combined_dim)
        
        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings.float())
        return self.task_heads[task_index](shared_features)

# ----------------------------------------

# (MPN + MegaMolBART + descriptor) -> MLP 
class MPN_MMB_Desc_Model(nn.Module):
    def __init__(self, 
                mpn_model: nn.Module, 
                mmb_model: nn.Module, 
                task_output_dims: List[int],
                mpn_output_dim: int,
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.mpn_model = mpn_model # Chemprop 的 MPN encoder
        self.mmb_model = mmb_model
        
        combined_dim = mpn_output_dim + MMB_OUTPUT_DIM + DESC_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(self, smiles: List[str], descriptors: torch.Tensor, task_index: int) -> torch.Tensor:
        # mpn_batch = list of SMILES (List[str]) or a Chemprop BatchGraph
        mpn_input = [[s] for s in smiles] # convert to List[List[str]]
        
        mpn_embeddings = self.mpn_model(mpn_input) # shape: (batch_size, mpn_output_dim)
        mmb_embeddings = self.mmb_model(smiles) # shape: (batch_size, megamolbart_output_dim)
        descriptors = descriptors.view(mpn_embeddings.size(0), -1) # change the shape tom(batch_size, 200)
        
        combined_embeddings = torch.cat([mpn_embeddings, mmb_embeddings, descriptors], dim=1)

        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings.float())
        return self.task_heads[task_index](shared_features)


# MPN -> MLP
class MPN_Model(nn.Module):
    def __init__(self, 
                mpn_model: nn.Module, 
                task_output_dims: List[int],
                mpn_output_dim: int,
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.mpn_model = mpn_model # Chemprop 的 MPN encoder
        
        self.shared_mlp = SharedMLP(
            input_dim=mpn_output_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])
        
    def forward(self, smiles: List[str], task_index: int) -> torch.Tensor:
        
        mpn_input = [[s] for s in smiles] # convert to List[List[str]]
        
        mpn_embeddings = self.mpn_model(mpn_input) # shape: (batch_size, mpn_output_dim)
        
        # process through MLP and task head
        shared_features = self.shared_mlp(mpn_embeddings)
        return self.task_heads[task_index](shared_features)

   
# (MPN + descriptor) -> MLP 
class MPN_Desc_Model(nn.Module):
    def __init__(self, 
                mpn_model: nn.Module, 
                task_output_dims: List[int],
                mpn_output_dim: int,
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.mpn_model = mpn_model # Chemprop 的 MPN encoder
        
        combined_dim = mpn_output_dim + DESC_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(self, smiles: List[str], descriptors: torch.Tensor, task_index: int) -> torch.Tensor:
        # mpn_batch = list of SMILES (List[str]) or a Chemprop BatchGraph
        mpn_input = [[s] for s in smiles] # convert to List[List[str]]
        
        mpn_embeddings = self.mpn_model(mpn_input) # shape: (batch_size, mpn_output_dim)

        descriptors = descriptors.view(mpn_embeddings.size(0), -1) # change the shape to (batch_size, 200)
        
        combined_embeddings = torch.cat([mpn_embeddings, descriptors], dim=1)

        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings.float())
        return self.task_heads[task_index](shared_features)
    

# (MPN + MegaMolBART) -> MLP 
class MPN_MMB_Model(nn.Module):
    def __init__(self, 
                mpn_model: nn.Module, 
                mmb_model: nn.Module, 
                task_output_dims: List[int],
                mpn_output_dim: int,
                mlp_hidden_dim: int = 128,
                mlp_num_layers: int = 3,
                mlp_activation: str = "relu",
                mlp_dropout: float = 0.2,
                mlp_norm_type: str = "none"):
        super().__init__()
        
        self.mpn_model = mpn_model # Chemprop 的 MPN encoder
        self.mmb_model = mmb_model
        
        combined_dim = mpn_output_dim + MMB_OUTPUT_DIM
        
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type
        )
        
        self.task_heads = nn.ModuleList([
            # e.g., task_output_dims = [1, 1, 1, 1, 1]
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(self, smiles: List[str], task_index: int) -> torch.Tensor:
        # mpn_batch = list of SMILES (List[str]) or a Chemprop BatchGraph
        mpn_input = [[s] for s in smiles] # convert to List[List[str]]
        
        mpn_embeddings = self.mpn_model(mpn_input) # shape: (batch_size, mpn_output_dim)
        mmb_embeddings = self.mmb_model(smiles) # shape: (batch_size, megamolbart_output_dim)
        
        combined_embeddings = torch.cat([mpn_embeddings, mmb_embeddings], dim=1)

        # process through MLP and task head
        shared_features = self.shared_mlp(combined_embeddings.float())
        return self.task_heads[task_index](shared_features)


# =================== DMPEGNN Fusion Wrapper ===================
# Uses local core/edmpnn_model_new.py (AEGNNM).
# Fusion data (OGB-style extended):
#   - x: 82-dim node features (48 atom type + 11 degree + 11 charge + 5 hybrid + 1 aromatic + 6 num_H)
#   - edge_attr: 9-dim edge features
#   - descriptor: 200-dim RDKit descriptors
# Wrapper returns only logits for compatibility with Fusion train_utils / loss.

class DMPEGNN_Fusion_Model(nn.Module):
    """Fusion wrapper for DMP-EGNN (edmpnn_model_new.AEGNNM). PyG batch: x, edge_index, edge_attr, batch, descriptor."""

    def __init__(
        self,
        node_features: int = 82,
        edge_features: int = 9,
        descriptor_dim: int = 200,
        output_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        dmp_steps: int = 2,
        pool_type: str = "mean",
        use_descriptor: bool = True,
        # Optional multi-task head configuration (Route B)
        task_output_dims: Optional[List[int]] = None,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.2,
        mlp_norm_type: str = "none",
        **kwargs
    ):
        super().__init__()
        self.backbone = DMPEGNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=output_dim,
            pool_type=pool_type,
            use_equivariant=True,
            use_fingerprint=False,
            use_descriptor=False,
            descriptor_dim=descriptor_dim,
            dmp_steps=dmp_steps,
            use_coord_branch=False,
            **kwargs
        )
        # SharedMLP + task_heads are always created (no descriptor in this model).
        assert hasattr(self.backbone, "graph_repr_dim"), \
            "DMPEGNN_Fusion_Model backbone 必須定義 graph_repr_dim 屬性。"
        dmpegnn_graph_dim = int(self.backbone.graph_repr_dim)
        task_dims = task_output_dims if task_output_dims is not None else [output_dim]
        self.task_output_dims = task_dims
        self.shared_mlp = SharedMLP(
            input_dim=dmpegnn_graph_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type,
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_dims
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        descriptor: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        task_index: int = 0,
        molecule_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # When backbone is equivariant, require valid 3D coordinates.
        if getattr(self.backbone, "use_equivariant", False):
            if pos is None or pos.dim() != 2 or pos.size(-1) != 3:
                raise ValueError(
                    "DMPEGNN_Fusion_Model: backbone.use_equivariant=True 時，"
                    "必須提供形狀為 [N, 3] 的 3D 座標張量 `pos`。"
                )

        # All paths: extract graph_features from backbone (no descriptor inside backbone).
        # descriptor argument is intentionally unused — descriptors are not part of this model.
        result = self.backbone(
            x, edge_index, edge_attr, batch=batch,
            pos=pos, fingerprint=None, descriptor=None,
            return_graph_features=True, b2revb=None, compute_logits=False,
        )
        graph_features = result[-1]
        # Multi-conformer aggregation: average conformer graph_features per molecule before MLP.
        if molecule_idx is not None and molecule_idx.numel() == graph_features.size(0):
            graph_features = global_mean_pool(graph_features, molecule_idx)
        shared_features = self.shared_mlp(graph_features.float())
        return self.task_heads[task_index](shared_features)


class DMPEGNN_Desc_Model(nn.Module):
    """DMPEGNN graph encoder + RDKit Descriptor (200-dim), fused by MLP. No MMB."""

    def __init__(
        self,
        dmpegnn_backbone: nn.Module,
        task_output_dims: List[int],
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.2,
        mlp_norm_type: str = "none",
    ):
        super().__init__()
        self.dmpegnn_backbone = dmpegnn_backbone

        graph_dim = int(dmpegnn_backbone.graph_repr_dim)
        combined_dim = graph_dim + int(DESC_DIM)
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type,
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        descriptors: torch.Tensor,
        batch: torch.Tensor,
        task_index: int,
        pos: Optional[torch.Tensor] = None,
        molecule_idx: Optional[torch.Tensor] = None,
        conformer_aggregation: str = "mean",
    ) -> torch.Tensor:
        if getattr(self.dmpegnn_backbone, "use_equivariant", False):
            if pos is None or pos.dim() != 2 or pos.size(-1) != 3:
                raise ValueError(
                    "DMPEGNN_Desc_Model: backbone.use_equivariant=True 時，"
                    "必須提供形狀為 [N, 3] 的 3D 座標張量 `pos`。"
                )
        _, _, graph_features = self.dmpegnn_backbone(
            x, edge_index, edge_attr, batch=batch,
            pos=pos, fingerprint=None, descriptor=None,
            return_graph_features=True, b2revb=None, compute_logits=False,
        )
        if molecule_idx is not None and molecule_idx.numel() == graph_features.size(0):
            if conformer_aggregation == "max":
                graph_features = global_max_pool(graph_features, molecule_idx)
            else:
                graph_features = global_mean_pool(graph_features, molecule_idx)

        desc = descriptors.contiguous().view(-1, DESC_DIM).float()
        combined = torch.cat([graph_features.float(), desc], dim=1)
        shared_features = self.shared_mlp(combined)
        return self.task_heads[task_index](shared_features)


class DMPEGNN_MMB_Desc_Model(nn.Module):
    """DMPEGNN graph encoder + MegaMolBART + Descriptor, fused by MLP."""

    def __init__(
        self,
        dmpegnn_backbone: nn.Module,
        mmb_model: nn.Module,
        task_output_dims: List[int],
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.2,
        mlp_norm_type: str = "none",
    ):
        super().__init__()
        self.dmpegnn_backbone = dmpegnn_backbone
        self.mmb_model = mmb_model

        # Automatically derive graph feature dimension from backbone to avoid manual mismatch.
        # backbone.graph_repr_dim accounts for all enabled branches (H, coord, fingerprint, descriptor).
        # Route A: descriptors are fused at MLP level (not inside backbone), so DESC_DIM is added here.
        graph_dim = int(dmpegnn_backbone.graph_repr_dim)
        combined_dim = graph_dim + int(MMB_OUTPUT_DIM) + int(DESC_DIM)
        self.shared_mlp = SharedMLP(
            input_dim=combined_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_activation,
            dropout=mlp_dropout,
            norm_type=mlp_norm_type,
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(self.shared_mlp.output_dim, out_dim) for out_dim in task_output_dims
        ])

    def forward(
        self,
        smiles: List[str],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        descriptors: torch.Tensor,
        batch: torch.Tensor,
        task_index: int,
        pos: Optional[torch.Tensor] = None,
        molecule_idx: Optional[torch.Tensor] = None,
        conformer_aggregation: str = "mean",
    ) -> torch.Tensor:
        # DMPEGNN backbone returns (logits, attention_weights, graph_features) when requested.
        if getattr(self.dmpegnn_backbone, "use_equivariant", False):
            if pos is None or pos.dim() != 2 or pos.size(-1) != 3:
                raise ValueError(
                    "DMPEGNN_MMB_Desc_Model: backbone.use_equivariant=True 時，"
                    "必須提供形狀為 [N, 3] 的 3D 座標張量 `pos`。"
                )

        # Safety: if backbone is configured to consume descriptors internally, require non-None descriptor input.
        if getattr(self.dmpegnn_backbone, "use_descriptor", False) and descriptors is None:
            raise ValueError(
                "DMPEGNN_MMB_Desc_Model: dmpegnn_backbone.use_descriptor=True 但 descriptors 為 None，"
                "請傳入對應的 descriptor 張量或在建立 backbone 時將 use_descriptor=False。"
            )

        # Prepare descriptor tensor for backbone when enabled.
        descriptor_for_backbone = None
        if getattr(self.dmpegnn_backbone, "use_descriptor", False):
            # Expect shape [batch_size, DESC_DIM]; fallback to view(-1, DESC_DIM) for safety.
            if descriptors.dim() == 2:
                descriptor_for_backbone = descriptors.float()
            else:
                descriptor_for_backbone = descriptors.view(-1, DESC_DIM).float()

        _, _, graph_features = self.dmpegnn_backbone(
            x, edge_index, edge_attr, batch=batch,
            pos=pos, fingerprint=None, descriptor=descriptor_for_backbone,
            return_graph_features=True, b2revb=None, compute_logits=False,
        )
        # Multi-conformer aggregation (per-molecule):
        # Use PyG's global pooling kernels for efficient GPU mean/max over conformers.
        if molecule_idx is not None and molecule_idx.numel() == graph_features.size(0):
            if conformer_aggregation == "max":
                graph_features = global_max_pool(graph_features, molecule_idx)
            else:
                graph_features = global_mean_pool(graph_features, molecule_idx)
        mmb_embeddings = self.mmb_model(smiles)  # (batch, 256)

        # Route A: descriptors fused at MLP level alongside graph features and MMB embeddings.
        desc = descriptors.contiguous().view(-1, DESC_DIM).float()
        combined = torch.cat([graph_features.float(), mmb_embeddings.float(), desc], dim=1)
        shared_features = self.shared_mlp(combined)
        return self.task_heads[task_index](shared_features)