"""
AEGNN-M (Attention-Enhanced Graph Neural Network for Molecular Properties)
Attention mechanism-based enhanced graph neural network model for molecular property prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math
import numpy as np

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output



class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    
    Focal Loss = -α(1-p)^γ * log(p)
    where p is the predicted probability for the true class
    
    Supports class-weighted focal loss for extremely imbalanced datasets
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0, 
                 class_weight=None):
        """
        Args:
            alpha: Balancing factor for class imbalance (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
                   Higher gamma focuses more on hard examples
            reduction: 'mean' or 'sum' (default: 'mean')
            label_smoothing: Label smoothing factor (0.0 to 1.0, default: 0.0)
            class_weight: Class weights tensor [num_classes] or None (default: None)
                         If provided, will multiply focal loss by class weights
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if class_weight is not None:
            self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None
    
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions [batch_size, num_classes] (logits, not probabilities)
            target: Ground truth labels [batch_size] (class indices)
        """
        # Convert to long if needed
        target = target.long()
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            num_classes = pred.size(1)
            # Create one-hot encoding
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            # Apply smoothing
            target_one_hot = (1.0 - self.label_smoothing) * target_one_hot + \
                           self.label_smoothing / num_classes
            # Compute cross entropy with smoothed labels
            log_probs = F.log_softmax(pred, dim=1)
            ce_loss = -(target_one_hot * log_probs).sum(dim=1)
        else:
            # Standard cross entropy
            ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Compute probability of true class
        probs = torch.exp(-ce_loss)  # p_t = exp(-CE_loss)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Apply alpha weighting (if alpha is a tensor, use class-specific alpha)
        if isinstance(self.alpha, (float, int)):
            if pred.size(1) == 2:  # Binary classification
                alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            else:  # Multi-class
                alpha_t = self.alpha
        elif isinstance(self.alpha, torch.Tensor):
            # Check if alpha is a scalar tensor (0-dimensional)
            if self.alpha.dim() == 0:
                # Scalar tensor: use directly
                alpha_t = self.alpha.to(target.device)
            else:
                # Alpha is a tensor with per-class weights
                # Ensure alpha is on the same device as target
                alpha = self.alpha.to(target.device)
                alpha_t = alpha[target]
        else:
            # Fallback for other types
            alpha_t = self.alpha
        
        # Apply class weight if provided (for class-weighted focal loss)
        if self.class_weight is not None:
            # Get class weights for each sample
            if pred.size(1) == 2:  # Binary classification
                # For binary: weight for positive class if target==1, else weight for negative class
                if len(self.class_weight) > 1:
                    class_weight_t = torch.where(
                        target == 1, 
                        self.class_weight[1].to(target.device),
                        self.class_weight[0].to(target.device)
                    )
                else:
                    class_weight_t = self.class_weight[0].to(target.device)
            else:  # Multi-class
                class_weight_t = self.class_weight[target].to(target.device)
            focal_loss = class_weight_t * alpha_t * focal_weight * ce_loss
        else:
            # Standard focal loss
            focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GATEGNNLayer(MessagePassing):
    """
    GAT-EGNN Layer: Combines Graph Attention Network (GAT) and Equivariant Graph Neural Network (EGNN)
    This is the core layer of the AEGNN-M model
    """
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1, 
                 alpha=0.2, concat=True, edge_dim=None):
        super(GATEGNNLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.edge_dim = edge_dim
        
        # Calculate output dimension for each head
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        
        # Linear transformation layers
        self.W = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, heads * self.head_dim, bias=False) if edge_dim else None
        
        # Attention mechanism parameters
        self.att = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))
        self.att_edge = nn.Parameter(torch.empty(1, heads, self.head_dim)) if edge_dim else None
        
        # EGNN parameters (φ_e, φ_x, φ_h)
        e_input_dim = 2 * out_channels + 1
        if edge_dim:
            self.edge_attr_proj = nn.Linear(edge_dim, out_channels)
            e_input_dim += out_channels
        else:
            self.edge_attr_proj = None
        
        self.phi_e = nn.Sequential(
            nn.Linear(e_input_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.phi_x = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, 1)
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Learnable gate for coordinate update stability 
        # Use learnable gate in _egnn_update to avoid instability from excessive coordinate updates
        self.coord_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()  # Output gating value in [0, 1]
        )
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W.weight)
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.att)
        if self.att_edge is not None:
            nn.init.xavier_uniform_(self.att_edge)
    
    def forward(self, x, edge_index, edge_attr=None, pos=None, b2revb=None):
        """
        Forward propagation
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]
            pos: Node positions [N, 3] (optional, for equivariance)
            b2revb: Pre-computed reverse edge mapping [E] (optional, for compatibility with edmpnn_model)
        """
        # Preserve original edge attributes for EGNN part
        edge_attr_original = edge_attr
        
        # Linear transformation
        h = self.W(x).view(-1, self.heads, self.head_dim)  # [N, heads, head_dim]
        
        # Edge feature processing
        if edge_attr is not None and self.W_edge is not None:
            edge_attr_transformed = self.W_edge(edge_attr).view(-1, self.heads, self.head_dim)
        else:
            edge_attr_transformed = None
        
        # Compute attention coefficients
        alpha = self._compute_attention(h, edge_index, edge_attr_transformed)
        
        # Apply attention weights
        out = self.propagate(edge_index, x=h, alpha=alpha, edge_attr=edge_attr_transformed)
        
        # Output processing
        if self.concat:
            out = out.view(-1, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)
        
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        # EGNN-style equivariant update (if position information is provided)
        if pos is not None:
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            # Check if pos has meaningful values (not all zeros)
            pos_norm = torch.norm(pos, dim=-1)
            has_valid_pos = torch.isfinite(pos_norm).all() and (pos_norm.sum() > 1e-6)
            
            if has_valid_pos:
                out, pos = self._egnn_update(out, pos, edge_index, edge_attr_original)
        
        return out, alpha, pos if pos is not None else None
    
    def _compute_attention(self, h, edge_index, edge_attr=None):
        """Compute attention coefficients"""
        # Prepare attention inputs
        h_i = h[edge_index[0]]  # [E, heads, head_dim]
        h_j = h[edge_index[1]]  # [E, heads, head_dim]
        
        # Concatenate features
        h_concat = torch.cat([h_i, h_j], dim=-1)  # [E, heads, 2*head_dim]
        
        # Compute attention scores
        e = (h_concat * self.att).sum(dim=-1)  # [E, heads]
        
        # Edge feature attention (if exists)
        if edge_attr is not None and self.att_edge is not None:
            e_edge = (edge_attr * self.att_edge).sum(dim=-1)  # [E, heads]
            e = e + e_edge
        
        # GELU activation (matches described GAT formulation)
        e = F.gelu(e)
        
        # Softmax normalization
        alpha = softmax(e, edge_index[0], num_nodes=h.size(0))
        
        return alpha
    
    def _egnn_update(self, h, pos, edge_index, edge_attr=None):
        """EGNN update following equations (4)-(7)"""
        device = h.device
        num_nodes = h.size(0)
        
        pos_i = pos[edge_index[0]]  # [E, 3]
        pos_j = pos[edge_index[1]]  # [E, 3]
        rel_pos = pos_i - pos_j
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        hi = h[edge_index[0]]
        hj = h[edge_index[1]]
        
        inputs = [hi, hj, dist_sq]
        if edge_attr is not None and self.edge_attr_proj is not None:
            edge_feat = self.edge_attr_proj(edge_attr)
            inputs.append(edge_feat)
        
        m_ij = self.phi_e(torch.cat(inputs, dim=-1))  # Messages calculation
        
        # Coordinate update with normalization constant C = 1/deg(i)
        deg = torch.zeros(num_nodes, device=device).scatter_add_(
            0, edge_index[0], torch.ones(edge_index.size(1), device=device)
        )
        deg = deg.clamp(min=1.0)
        coord_coeff = (1.0 / deg)[edge_index[0]].unsqueeze(-1)
        phi_x_val = torch.tanh(self.phi_x(m_ij))  # [E, 1], bounded for stability
        coord_contrib = coord_coeff * rel_pos * phi_x_val
        
        # Optimization suggestion 2: Use learnable gate to control coordinate update magnitude, avoid instability from excessive updates
        gate_value = self.coord_gate(m_ij)  # [E, 1], range [0, 1]
        coord_contrib = coord_contrib * gate_value  # Gate controls update magnitude
        coord_contrib = coord_contrib.to(pos.dtype)
        
        pos_update = torch.zeros_like(pos, dtype=pos.dtype)
        pos_update.index_add_(0, edge_index[0], coord_contrib)
        pos = pos + pos_update
        
        # Aggregate node messages m_i = Σ_j m_ij
        node_messages = torch.zeros(num_nodes, h.size(-1), device=device, dtype=h.dtype)
        node_messages.index_add_(0, edge_index[0], m_ij.to(h.dtype))
        
        h = self.phi_h(torch.cat([h, node_messages], dim=-1))
        return h, pos
    
    def message(self, x_j, alpha, edge_attr=None):
        """Message passing function"""
        # Apply attention weights
        out = x_j * alpha.unsqueeze(-1)  # [E, heads, head_dim]
        
        # Edge feature processing
        if edge_attr is not None:
            out = out + edge_attr
        
        return out


class GraphAttentionPooling(nn.Module):
    """Graph Attention Pooling Layer - Graph structure-based pooling, does not use Transformer architecture"""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1):
        super(GraphAttentionPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads
        
        # Graph attention pooling parameters
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, heads),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, batch=None):
        """
        Graph attention pooling
        Args:
            x: Node features [N, in_channels]
            batch: Batch information [N]
        """
        if batch is None:
            # Single graph case
            attention_scores = self.attention_mlp(x)  # [N, heads]
            attention_weights = F.softmax(attention_scores, dim=0)  # [N, heads]
            
            # Weighted aggregation
            weighted_features = x.unsqueeze(1) * attention_weights.unsqueeze(-1)  # [N, heads, in_channels]
            pooled = weighted_features.sum(dim=0)  # [heads, in_channels]
            pooled = pooled.mean(dim=0)  # [in_channels]
            
        else:
            # Batch processing
            pooled_features = []
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                if mask.sum() > 0:
                    graph_nodes = x[mask]  # [num_nodes_in_graph, in_channels]
                    
                    # Compute attention scores
                    attention_scores = self.attention_mlp(graph_nodes)  # [num_nodes, heads]
                    attention_weights = F.softmax(attention_scores, dim=0)  # [num_nodes, heads]
                    
                    # Weighted aggregation
                    weighted_features = graph_nodes.unsqueeze(1) * attention_weights.unsqueeze(-1)
                    graph_pooled = weighted_features.sum(dim=0).mean(dim=0)  # [in_channels]
                    pooled_features.append(graph_pooled)
            
            if pooled_features:
                pooled = torch.stack(pooled_features, dim=0)  # [batch_size, in_channels]
            else:
                pooled = x.mean(dim=0, keepdim=True)
        
        # Output projection
        output = self.output_proj(pooled)
        output = self.dropout(output)
        
        return output


class AttentionEnhancedGCN(MessagePassing):
    """Attention-enhanced graph convolution layer"""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1):
        super(AttentionEnhancedGCN, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Linear transformation layers
        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)  # Edge feature processing
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(out_channels, heads, dropout=dropout)
        
        # Output projection
        self.lin_out = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Prepare query, key, value
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)
        
        # Edge feature processing
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(q, k, v)
        
        # Graph convolution message passing
        out = self.propagate(edge_index, x=attn_out, edge_attr=edge_attr)
        
        # Residual connection and layer normalization
        out = self.norm(out + attn_out)
        out = self.dropout_layer(out)
        
        return out, attn_weights
    
    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            return x_j + edge_attr
        return x_j


class AEGNNLayer(nn.Module):
    """AEGNN Layer, using GAT-EGNN as the core layer"""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1, 
                 alpha=0.2, edge_dim=None, use_equivariant=True, ffn_expansion_factor=4,
                 drop_path=0.0, pre_norm=False, activation=None):
        super(AEGNNLayer, self).__init__()
        
        self.use_equivariant = use_equivariant
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pre_norm = pre_norm
        
        # Set activation function (default to SiLU if not provided)
        if activation is None:
            self.activation = nn.SiLU()
        elif isinstance(activation, str):
            activation_map = {
                'ReLU': nn.ReLU(),
                'LeakyReLU': nn.LeakyReLU(),
                'PReLU': nn.PReLU(),
                'tanh': nn.Tanh(),
                'SELU': nn.SELU(),
                'ELU': nn.ELU(),
                'SiLU': nn.SiLU()
            }
            self.activation = activation_map.get(activation, nn.SiLU())
        else:
            # activation is already a nn.Module instance
            self.activation = activation
        
        # GAT-EGNN core layer
        self.gat_egnn = GATEGNNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
            edge_dim=edge_dim
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * ffn_expansion_factor),
            self.activation,  # Use configurable activation function
            nn.Dropout(dropout),
            nn.Linear(out_channels * ffn_expansion_factor, out_channels)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection layer (used when input and output dimensions differ)
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
        
        # Position encoding (for equivariance)
        if use_equivariant:
            # Project coordinates to input dimension to avoid shape inconsistency when adding to output dimension
            self.pos_embedding = nn.Linear(3, in_channels)
        
    def forward(self, x, edge_index, edge_attr=None, pos=None, b2revb=None):
        """
        Forward propagation
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            pos: Node positions (optional, for equivariance)
            b2revb: Pre-computed reverse edge mapping [E] (optional, for compatibility with edmpnn_model)
        """
        # Position encoding (if position information is provided)
        # Check if pos contains valid (non-zero) coordinates to avoid encoding zero positions
        if pos is not None and self.use_equivariant:
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            # Check if pos has meaningful values (not all zeros)
            # This prevents encoding zero positions which would cause identical encodings
            pos_norm = torch.norm(pos, dim=-1)
            has_valid_pos = pos_norm.sum() > 1e-6  # At least some non-zero coordinates
            
            if has_valid_pos:
                pos_encoded = self.pos_embedding(pos)
                x = x + pos_encoded
            # If pos is all zeros, skip position encoding to avoid identical encodings
        
        # Save input for residual connection (needs to be projected to output dimension)
        x_residual = x
        
        # GAT-EGNN layer
        if self.pre_norm:
            # Pre-Norm: Norm -> Attention -> Add
            x_norm = self.norm1(x)
            gat_out, attn_weights, updated_pos = self.gat_egnn(x_norm, edge_index, edge_attr, pos, b2revb=b2revb)
            
            # If input and output dimensions differ, need to project residual connection
            if self.residual_proj is not None:
                x_residual = self.residual_proj(x_residual)
                
            # Residual connection
            x = x_residual + self.drop_path(gat_out)
            
            # Feed-forward network (Pre-Norm: Norm -> FFN -> Add)
            x_norm2 = self.norm2(x)
            ffn_out = self.ffn(x_norm2)
            out = x + self.drop_path(ffn_out)
            out = self.dropout(out)
            
        else:
            # Post-Norm (Original): Attention -> Add -> Norm
            gat_out, attn_weights, updated_pos = self.gat_egnn(x, edge_index, edge_attr, pos, b2revb=b2revb)
            
            # If input and output dimensions differ, need to project residual connection
            if self.residual_proj is not None:
                x_residual = self.residual_proj(x_residual)
            
            # Residual connection and layer normalization
            x = self.norm1(x_residual + self.drop_path(gat_out))
            
            # Feed-forward network
            ffn_out = self.ffn(x)
            
            # Residual connection and layer normalization
            out = self.norm2(x + self.drop_path(ffn_out))
            out = self.dropout(out)
        
        return out, attn_weights, updated_pos


class AEGNNM(nn.Module):
    """AEGNN-M Main Model, using GAT-EGNN layers"""
    
    def __init__(self, 
                 node_features=78,  # Atom feature dimension
                 edge_features=4,   # Edge feature dimension
                 hidden_dim=256,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.1,
                 output_dim=1,
                 pool_type='mean',
                 use_equivariant=True,
                 alpha=0.2,
                 ffn_expansion_factor=4,
                 coord_dim=3,
                 drop_path_rate=0.0,
                 pre_norm=False,
                 rotate_aug=False,
                 use_fingerprint=False,
                 fingerprint_dim=2048,
                 fingerprint_dropout=0.0,
                 use_fingerprint_gate=False,
                 use_descriptor=True,
                 descriptor_dim=217,
                 descriptor_dropout=0.0,
                 activation='SiLU'):
        super(AEGNNM, self).__init__()
        
        # Get activation function
        activation_map = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'PReLU': nn.PReLU(),
            'tanh': nn.Tanh(),
            'SELU': nn.SELU(),
            'ELU': nn.ELU(),
            'SiLU': nn.SiLU()
        }
        self.activation = activation_map.get(activation, nn.SiLU())
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        self.use_equivariant = use_equivariant
        self.coord_dim = coord_dim
        self.rotate_aug = rotate_aug
        self.use_fingerprint = use_fingerprint
        self.fingerprint_dropout = fingerprint_dropout
        self.use_fingerprint_gate = use_fingerprint_gate
        self.use_descriptor = use_descriptor
        self.descriptor_dim = descriptor_dim
        self.descriptor_dropout = descriptor_dropout
        
        # Input projection layers
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Fingerprint projection (Wide part)
        if use_fingerprint:
            layers = []
            if fingerprint_dropout > 0.0:
                layers.append(nn.Dropout(fingerprint_dropout))
                
            layers.extend([
                nn.Linear(fingerprint_dim, hidden_dim // 2),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            ])
            self.fingerprint_mlp = nn.Sequential(*layers)
            
            if use_fingerprint_gate:
                # Gating Mechanism: Use GNN features (hidden_dim) to gate Fingerprint features (hidden_dim // 2)
                self.fingerprint_gate_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    self.activation,
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.Sigmoid()
                )
        
        # Descriptor projection (similar to fingerprint)
        if use_descriptor:
            layers = []
            if descriptor_dropout > 0.0:
                layers.append(nn.Dropout(descriptor_dropout))
                
            layers.extend([
                nn.Linear(descriptor_dim, hidden_dim // 2),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            ])
            self.descriptor_mlp = nn.Sequential(*layers)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # GAT-EGNN layers
        self.aegnn_layers = nn.ModuleList([
            AEGNNLayer(
                in_channels=hidden_dim, 
                out_channels=hidden_dim, 
                heads=num_heads, 
                dropout=dropout,
                alpha=alpha,
                edge_dim=hidden_dim,
                use_equivariant=use_equivariant,
                ffn_expansion_factor=ffn_expansion_factor,
                drop_path=dpr[i],
                pre_norm=pre_norm,
                activation=self.activation
            )
            for i in range(num_layers)
        ])
        
        # Modality-specific MLPs for balanced fusion (optimization suggestion 1)
        # Process 2D mean H and 3D mean X separately through small MLPs before concat, to avoid single modality dominance
        self.h_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.x_mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Output layer (cat of processed H and X)
        self.graph_repr_dim = hidden_dim  # Default: hidden/2 (H) + hidden/2 (X)
        if use_fingerprint:
            self.graph_repr_dim += hidden_dim // 2  # Add fingerprint dimension
        if use_descriptor:
            self.graph_repr_dim += hidden_dim // 2  # Add descriptor dimension
            
        self.output_norm = nn.LayerNorm(self.graph_repr_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.graph_repr_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None, pos=None, fingerprint=None, descriptor=None, return_graph_features=False, b2revb=None):
        """
        Forward propagation
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch information
            pos: Node positions (optional, for equivariance)
            fingerprint: Molecular fingerprints (optional, for Deep & Wide)
            descriptor: Molecular descriptors (optional, RDKit normalized descriptors)
            b2revb: Pre-computed reverse edge mapping [E] (optional, for compatibility with edmpnn_model)
        """
        # Node and edge embedding
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply 3D rotation augmentation if enabled and in training mode
        if self.rotate_aug and self.training and pos is not None:
            pos = self._apply_random_rotation(pos, batch)
        
        # Store attention weights for visualization
        attention_weights = []
        
        # Pass through GAT-EGNN layers
        for layer in self.aegnn_layers:
            x, attn_weights, pos = layer(x, edge_index, edge_attr, pos, b2revb=b2revb)
            attention_weights.append(attn_weights)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if pos is not None:
                pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure batch vector exists (single-graph fallback)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        
        # Per-graph pooling for node embeddings (H) - support different aggregation methods
        if self.pool_type == 'mean':
            node_mean = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        elif self.pool_type == 'sum':
            node_mean = global_add_pool(x, batch)  # [batch_size, hidden_dim]
        elif self.pool_type == 'norm':
            # Normalized sum: sum / sqrt(num_nodes)
            node_sum = global_add_pool(x, batch)  # [batch_size, hidden_dim]
            # Count nodes per graph
            num_nodes_per_graph = global_add_pool(torch.ones(x.size(0), 1, device=x.device), batch)  # [batch_size, 1]
            node_mean = node_sum / (torch.sqrt(num_nodes_per_graph) + 1e-8)  # [batch_size, hidden_dim]
        else:
            node_mean = global_mean_pool(x, batch)  # Default to mean
        
        # Per-graph pooling for coordinates (X) - use same aggregation method
        if pos is not None:
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            if self.pool_type == 'mean':
                coord_mean = global_mean_pool(pos, batch)  # [batch_size, coord_dim]
            elif self.pool_type == 'sum':
                coord_mean = global_add_pool(pos, batch)  # [batch_size, coord_dim]
            elif self.pool_type == 'norm':
                coord_sum = global_add_pool(pos, batch)  # [batch_size, coord_dim]
                num_nodes_per_graph = global_add_pool(torch.ones(pos.size(0), 1, device=pos.device), batch)  # [batch_size, 1]
                coord_mean = coord_sum / (torch.sqrt(num_nodes_per_graph) + 1e-8)  # [batch_size, coord_dim]
            else:
                coord_mean = global_mean_pool(pos, batch)  # Default to mean
        else:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            coord_mean = torch.zeros(num_graphs, self.coord_dim, device=x.device)
        
        # Optimization suggestion 1: Process separately through small MLPs to avoid single modality dominance
        h_processed = self.h_mlp(node_mean)  # [batch_size, hidden_dim // 2]
        x_processed = self.x_mlp(coord_mean)  # [batch_size, hidden_dim // 2]
        
        # Concatenate processed features
        feature_list = [h_processed, x_processed]
        
        # Process fingerprint if available
        if self.use_fingerprint:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            if fingerprint is not None:
                # fingerprint shape: [batch_size, fingerprint_dim]
                fp_processed = self.fingerprint_mlp(fingerprint)
                
                if self.use_fingerprint_gate:
                    # Gating Mechanism
                    # GNN features: cat(h_processed, x_processed) -> [batch_size, hidden_dim]
                    gnn_features = torch.cat([h_processed, x_processed], dim=-1)
                    gate = self.fingerprint_gate_mlp(gnn_features)  # [batch_size, hidden_dim // 2]
                    fp_processed = fp_processed * gate
            else:
                fp_processed = x.new_zeros(num_graphs, self.hidden_dim // 2)
            feature_list.append(fp_processed)
        
        # Process descriptor if available
        if self.use_descriptor:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            if descriptor is not None:
                # descriptor shape: [batch_size, descriptor_dim]
                desc_processed = self.descriptor_mlp(descriptor)
            else:
                desc_processed = x.new_zeros(num_graphs, self.hidden_dim // 2)
            feature_list.append(desc_processed)
        
        graph_features = torch.cat(feature_list, dim=-1)
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Output projection
        logits = self.project_graph_features(graph_features)
        
        if return_graph_features:
            return logits, attention_weights, graph_features
        return logits, attention_weights
    
    def _apply_random_rotation(self, pos, batch=None):
        """Apply random 3D rotation to node positions"""
        device = pos.device
        dtype = pos.dtype
        
        if batch is None:
            # Single graph rotation
            rot_matrix = self._get_random_rotation_matrix(dtype=dtype, device=device)
            return (pos @ rot_matrix).to(dtype)
        else:
            # Batch rotation (rotate each graph independently)
            num_graphs = batch.max().item() + 1
            pos_rotated = torch.zeros_like(pos)
            
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.sum() > 0:
                    rot_matrix = self._get_random_rotation_matrix(dtype=dtype, device=device)
                    pos_rotated[mask] = (pos[mask] @ rot_matrix).to(dtype)
            
            return pos_rotated
            
    def _get_random_rotation_matrix(self, dtype=torch.float32, device=None):
        """Get a random 3D rotation matrix"""
        device = device or torch.device('cpu')
        # Random rotation axis
        axis = torch.randn(3, device=device, dtype=dtype)
        axis = axis / (torch.norm(axis) + torch.tensor(1e-8, device=device, dtype=dtype))
        
        # Random rotation angle
        theta = torch.rand(1, device=device, dtype=dtype) * (2 * math.pi)
        
        # Rodrigues' rotation formula
        # R = I + (sin(theta))K + (1-cos(theta))K^2
        # where K is the cross-product matrix of axis
        
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device, dtype=dtype)
        
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        return R

    def get_attention_weights(self, x, edge_index, edge_attr, batch=None, pos=None, fingerprint=None):
        """Get attention weights for visualization"""
        with torch.no_grad():
            _, attention_weights = self.forward(x, edge_index, edge_attr, batch, pos, fingerprint)
        return attention_weights
    
    def project_graph_features(self, graph_features):
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        logits_input = self.output_norm(graph_features)
        logits = self.output_proj(logits_input)
        return logits


class AEGNNMRegressor(AEGNNM):
    """AEGNN-M Regression Model"""
    
    def __init__(self, primary_metric='mae', **kwargs):
        """
        Args:
            primary_metric: Primary evaluation metric ('mae', 'spearman', etc.)
                           - 'spearman': uses SpearmanLoss
                           - 'mae' or others: uses L1Loss (MAE Loss)
            **kwargs: Other arguments passed to AEGNNM
        """
        super(AEGNNMRegressor, self).__init__(output_dim=1, **kwargs)
        
        # Select loss function based on primary metric
        if primary_metric == 'spearman':
            try:
                from utils.loss_utils import SpearmanLoss
                self.loss_fn = SpearmanLoss(temperature=1.0, reduction='mean')
                self.use_spearman_loss = True
            except ImportError:
                print("⚠️  Warning: Could not import SpearmanLoss, falling back to L1Loss")
                self.loss_fn = nn.L1Loss()
                self.use_spearman_loss = False
        else:
            # Use L1Loss (MAE Loss) for MAE-based metrics and as default
            self.loss_fn = nn.L1Loss()
            self.use_spearman_loss = False
    
    def compute_loss(self, pred, target):
        return self.loss_fn(pred.squeeze(), target)


class AEGNNMClassifier(AEGNNM):
    """AEGNN-M Classification Model"""
    
    def __init__(self, num_classes=2, class_weight=None, 
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
                 label_smoothing=0.0, use_bce_for_imbalanced=False, pos_weight=None, **kwargs):
        """
        Args:
            num_classes: Number of classes
            class_weight: Class weights for balancing (list, array, or tensor)
            use_focal_loss: Whether to use Focal Loss instead of CrossEntropyLoss
            focal_alpha: Alpha parameter for Focal Loss (default: 0.25)
            focal_gamma: Gamma parameter for Focal Loss (default: 2.0)
            label_smoothing: Label smoothing factor (0.0 to 1.0, default: 0.0)
                            Can be used with both CrossEntropyLoss and FocalLoss
            use_bce_for_imbalanced: Whether to use BCEWithLogitsLoss for highly imbalanced datasets
            pos_weight: Positive class weight for BCEWithLogitsLoss (auto-calculated if None and use_bce_for_imbalanced=True)
        """
        # Remove parameters that should not be passed to parent class
        kwargs.pop('use_class_balanced_focal_loss', None)
        kwargs.pop('class_balanced_beta', None)
        kwargs.pop('class_counts', None)
        
        super(AEGNNMClassifier, self).__init__(output_dim=num_classes, **kwargs)
        
        # Convert class_weight to tensor if provided
        if class_weight is not None:
            if isinstance(class_weight, (list, np.ndarray)):
                class_weight = torch.tensor(class_weight, dtype=torch.float32)
            elif not isinstance(class_weight, torch.Tensor):
                raise ValueError("class_weight must be a list, numpy array, or torch.Tensor")
        
        self.class_weight = class_weight
        self.use_focal_loss = use_focal_loss
        self.label_smoothing = label_smoothing
        self.use_bce_for_imbalanced = use_bce_for_imbalanced
        self.pos_weight = pos_weight
        
        # Initialize loss function
        if use_bce_for_imbalanced:
            # Use BCEWithLogitsLoss for highly imbalanced datasets (e.g., MUV, HIV)
            # This is better for binary classification with extreme imbalance
            if pos_weight is not None:
                if isinstance(pos_weight, (list, np.ndarray, float, int)):
                    # Make sure pos_weight is a tensor
                    if not isinstance(pos_weight, torch.Tensor):
                        pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
                    if pos_weight.dim() == 0:
                        pos_weight = pos_weight.unsqueeze(0)
                elif not isinstance(pos_weight, torch.Tensor):
                    raise ValueError("pos_weight must be a scalar, list, numpy array, or torch.Tensor")
            
            # Initialize bias if pos_weight is provided (Bias Initialization)
            if pos_weight is not None:
                self._init_bias(pos_weight)
                
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            # Note: label_smoothing is not used with BCE for imbalanced datasets
        elif use_focal_loss:
            # Use Focal Loss
            # For imbalanced datasets, adjust alpha to give more weight to positive class
            if class_weight is not None and len(class_weight) == 2:
                # If class_weight provided, use it to adjust focal alpha
                # Higher weight for positive class (index 1) means higher alpha
                focal_alpha_value = class_weight[1] / class_weight.sum() if len(class_weight) == 2 else focal_alpha
            else:
                focal_alpha_value = focal_alpha
            
            self.loss_fn = FocalLoss(
                alpha=focal_alpha_value,
                gamma=focal_gamma,
                reduction='mean',
                label_smoothing=0.0 if use_bce_for_imbalanced else label_smoothing  # Disable label smoothing for imbalanced
            )
        else:
            # Use CrossEntropyLoss with optional label smoothing
            # For imbalanced datasets, disable label smoothing
            effective_label_smoothing = 0.0 if use_bce_for_imbalanced else label_smoothing
            if effective_label_smoothing > 0.0:
                # PyTorch 1.10+ supports label_smoothing in CrossEntropyLoss
                self.loss_fn = nn.CrossEntropyLoss(
                    weight=class_weight,
                    label_smoothing=effective_label_smoothing
                )
            else:
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    
    def _init_bias(self, pos_weight):
        """
        Initialize the bias of the last layer to reflect class imbalance
        Formula: b = -log(num_neg/num_pos) = -log(1/pos_weight) = log(pos_weight)
        """
        if isinstance(self.output_proj, nn.Sequential):
            last_layer = self.output_proj[-1]
        else:
            last_layer = self.output_proj
            
        if isinstance(last_layer, nn.Linear):
            # For BCE with pos_weight > 1 (more negatives), we want initial probability p = pos / (pos + neg)
            # Logits should be log(p / (1-p)) = log(pos / neg) = -log(neg / pos) = -log(1/pos_weight if pos_weight = neg/pos)
            # Actually, pos_weight in BCE is usually num_neg / num_pos
            # So initial bias should be -log(pos_weight) to start with low probability for positive class
            
            with torch.no_grad():
                if isinstance(pos_weight, torch.Tensor):
                     bias_init = -torch.log(pos_weight)
                     # Make sure bias_init has correct shape for the layer
                     if bias_init.numel() == last_layer.bias.numel():
                         last_layer.bias.data = bias_init.to(last_layer.bias.device)
                     elif bias_init.numel() == 1:
                         last_layer.bias.data.fill_(bias_init.item())
    
    def compute_loss(self, pred, target):
        if self.use_bce_for_imbalanced:
            # For BCEWithLogitsLoss, we need binary targets
            # Convert multi-class to binary if needed
            if pred.size(1) == 2:
                # Binary classification: use positive class logits
                pred = pred[:, 1:2]  # [batch, 1]
                target = target.float().unsqueeze(1)  # [batch, 1]
            else:
                # Single output: already binary
                pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                target = target.float()
            return self.loss_fn(pred, target)
        else:
            return self.loss_fn(pred, target.long())


def create_aegnn_model(model_type='regressor', **kwargs):
    """Factory function to create AEGNN-M model"""
    if model_type == 'regressor':
        return AEGNNMRegressor(**kwargs)
    elif model_type == 'classifier':
        return AEGNNMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    model = create_aegnn_model(
        node_features=78,
        edge_features=4,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_equivariant=True
    )
    
    # Simulate input
    num_nodes = 10
    num_edges = 20
    x = torch.randn(num_nodes, 78)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)
    pos = torch.randn(num_nodes, 3)  # 3D positions
    
    # Forward propagation
    output, attn_weights = model(x, edge_index, edge_attr, pos=pos)
    print(f"Model output shape: {output.shape}")
    print(f"Number of attention weight layers: {len(attn_weights)}")
    print(f"Using GAT-EGNN layers: True")
    print(f"Supports equivariance: {model.use_equivariant}")
