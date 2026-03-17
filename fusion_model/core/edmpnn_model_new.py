"""
AEGNN-M (Attention-Enhanced Graph Neural Network for Molecular Properties)
Attention mechanism-based enhanced graph neural network model for molecular property prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math
import numpy as np


def check_valid_positions(pos, min_non_zero_ratio=0.1, min_mean_norm=1e-4):
    """
    Check if position tensor contains valid (non-zero) coordinates.
    
    This function uses a stricter validation than simply checking the sum of norms,
    which can be misleading when only a few nodes have valid positions or when
    all positions are very small.
    
    Args:
        pos: Position tensor [N, 3] or [N, coord_dim]
        min_non_zero_ratio: Minimum ratio of nodes that must have non-zero positions (default: 0.1)
        min_mean_norm: Minimum mean norm for valid positions (default: 1e-4)
    
    Returns:
        bool: True if positions are valid, False otherwise
    
    Examples:
        # Problem scenario 1: 99% nodes are zero, 1% has small value
        # Old check: pos_norm.sum() > 1e-6 -> might pass incorrectly
        # New check: non_zero_ratio > 0.1 -> will correctly fail
        
        # Problem scenario 2: All positions are very small (1e-7)
        # Old check: 100 * 1e-7 = 1e-5 > 1e-6 -> might pass incorrectly  
        # New check: mean_norm > 1e-4 -> will correctly fail
    """
    if pos is None:
        return False
    
    # Compute L2 norm for each node
    pos_norm = torch.norm(pos, dim=-1)
    
    # Check 1: All values must be finite (no inf or nan)
    if not torch.isfinite(pos_norm).all():
        return False
    
    # Check 2: At least min_non_zero_ratio of nodes must have non-zero positions
    # This prevents cases where only a few nodes have valid positions
    non_zero_mask = pos_norm > 1e-6
    non_zero_ratio = non_zero_mask.float().mean()
    if non_zero_ratio < min_non_zero_ratio:
        return False
    
    # Check 3: Mean norm must be above threshold
    # This prevents cases where all positions are very small but sum exceeds threshold
    mean_norm = pos_norm.mean()
    if mean_norm < min_mean_norm:
        return False
    
    return True


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
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Args:
            alpha: Balancing factor for class imbalance (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
                   Higher gamma focuses more on hard examples
            reduction: 'mean' or 'sum' (default: 'mean')
            label_smoothing: Label smoothing factor (0.0 to 1.0, default: 0.0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
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
        # probs = torch.exp(-ce_loss)  # p_t = exp(-CE_loss)
        probs = F.softmax(pred, dim=1)  # [batch_size, num_classes]
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Compute focal weight: (1 - p_t)^gamma
        # Only compute weight for the true class, not all classes
        focal_weight = (1 - p_t) ** self.gamma  # [batch_size]
        
        # Apply alpha weighting (if alpha is a tensor, use class-specific alpha)
        if isinstance(self.alpha, (float, int)):
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
        
        # Compute focal loss
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
                 alpha=0.2, concat=True, edge_dim=None, dmp_steps=2):
        super(GATEGNNLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.edge_dim = edge_dim
        self.dmp_steps = max(0, dmp_steps)
        
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
        
        self.mlp_edge_init = nn.Sequential(
            nn.Linear(e_input_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.mlp_edge_update = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
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
            b2revb: Pre-computed reverse edge mapping [E] (optional, Chemprop style)
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
            # Use stricter validation to ensure meaningful position information
            # This checks: (1) all values are finite, (2) sufficient nodes have non-zero positions,
            # and (3) mean position norm is above threshold
            has_valid_pos = check_valid_positions(pos)
            
            if has_valid_pos:
                out, pos = self._egnn_update(out, pos, edge_index, edge_attr_original, b2revb=b2revb)
        
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
        
        
        alpha = softmax(e, edge_index[0], num_nodes=h.size(0))
        
        return alpha
    
    def _egnn_update(self, h, pos, edge_index, edge_attr=None, b2revb=None):
        """EGNN update following equations (4)-(7)"""
        device = h.device
        num_nodes = h.size(0)
        
        pos_i = pos[edge_index[0]]  # [E, 3]
        pos_j = pos[edge_index[1]]  # [E, 3]
        rel_pos = pos_i - pos_j
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        
        # Check for abnormally small distances (numerical stability)
        # If two nodes have identical positions, dist_sq will be 0, which may cause issues
        if (dist_sq < 1e-10).any():
            # Clamp to minimum value to prevent numerical instability
            dist_sq = torch.clamp(dist_sq, min=1e-8)
        
        hi = h[edge_index[0]]
        hj = h[edge_index[1]]
        
        inputs = [hi, hj, dist_sq]
        if edge_attr is not None and self.edge_attr_proj is not None:
            edge_feat = self.edge_attr_proj(edge_attr)
            inputs.append(edge_feat)
        
        e_ij = self.mlp_edge_init(torch.cat(inputs, dim=-1))  # Initial edge messages e^0
        
        if self.dmp_steps > 0:
            e_ij = self._run_directed_mp(e_ij, edge_index, num_nodes, b2revb=b2revb)
        
        # Coordinate update with normalization constant C = 1/deg(i)
        deg = torch.zeros(num_nodes, device=device).scatter_add_(
            0, edge_index[0], torch.ones(edge_index.size(1), device=device)
        )
        deg = deg.clamp(min=1.0)
        coord_coeff = (1.0 / deg)[edge_index[0]].unsqueeze(-1)
        phi_x_val = torch.tanh(self.phi_x(e_ij))  # [E, 1], bounded for stability
        coord_contrib = coord_coeff * rel_pos * phi_x_val
        
        # Optimization suggestion 2: Use learnable gate to control coordinate update magnitude, avoid instability from excessive updates
        gate_value = self.coord_gate(e_ij)  # [E, 1], range [0, 1]
        coord_contrib = coord_contrib * gate_value  # Gate controls update magnitude
        coord_contrib = coord_contrib.to(pos.dtype)
        
        pos_update = torch.zeros_like(pos, dtype=pos.dtype)
        pos_update.index_add_(0, edge_index[0], coord_contrib)
        pos = pos + pos_update
        
        # Aggregate node messages m_i = Σ_j e_ij^T
        node_messages = torch.zeros(num_nodes, h.size(-1), device=device, dtype=h.dtype)
        node_messages.index_add_(0, edge_index[0], e_ij.to(h.dtype))
        
        h = self.phi_h(torch.cat([h, node_messages], dim=-1))
        return h, pos

    @staticmethod
    def build_b2revb(edge_index, num_nodes=None):
        """
        Args:
            edge_index: Edge indices [2, E], where edge_index[0] is target and edge_index[1] is source
            num_nodes: Number of nodes (optional, will be inferred if not provided)
        
        Returns:
            b2revb: [E] tensor, b2revb[i] gives the index of the reverse edge of edge i
        """
        target_nodes = edge_index[0]
        source_nodes = edge_index[1]
        num_edges = edge_index.size(1)
        device = edge_index.device
        
        if num_nodes is None:
            num_nodes = max(target_nodes.max().item(), source_nodes.max().item()) + 1
        
        # Create a mapping from (target, source) pairs to edge indices
        max_node_idx = max(num_nodes, target_nodes.max().item(), source_nodes.max().item()) + 1
        edge_keys = target_nodes * max_node_idx + source_nodes
        reverse_keys = source_nodes * max_node_idx + target_nodes
        
        # Build mapping from keys to edge indices
        key_to_idx = {edge_keys[idx].item(): idx for idx in range(num_edges)}
        
        # Build b2revb: for each edge, find its reverse edge index
        b2revb_list = [
            key_to_idx.get(reverse_keys[idx].item(), idx)  # Use self if reverse doesn't exist
            for idx in range(num_edges)
        ]
        b2revb = torch.tensor(b2revb_list, dtype=torch.long, device=device)
        
        return b2revb
    
    '''
    def _run_directed_mp(self, e_ij, edge_index, num_nodes, b2revb=None):
        # Directed 3D message passing iterations (EGNN + DMPNN hybrid).

        e = e_ij  # edge features e_ij [E, hidden_dim]
        device = e.device
        
        # Build reverse edge mapping if not provided (Chemprop style: pre-compute during graph construction (BatchMolGraph))
        if b2revb is None:
            b2revb = self.build_b2revb(edge_index, num_nodes).to(device)
        else:
            b2revb = b2revb.to(device)
        
        # b2a: source atom index for each edge (edge_index[1])
        # In Chemprop, b2a[b] gives the source atom of bond b
        b2a = edge_index[1]
        
        # Save initial edge features for potential residual connection (like Chemprop)
        e_init = e.clone()
        
        for step in range(self.dmp_steps):
            # Step 1: Aggregate incoming messages to each source atom
            # For edge e_{ji} (from j to i), we need to aggregate messages to source node j
            # atom_message_sum[j] = sum_{k->j} e_{kj}
            # This matches Chemprop's: atom_message_sum.index_add_(0, a2b_target_index, message)
            # Note: We aggregate to source atoms (edge_index[1]) because we need messages to source node j
            atom_message_sum = torch.zeros(num_nodes, e.size(-1), device=device, dtype=e.dtype)
            atom_message_sum.index_add_(0, edge_index[1], e)  # Aggregate to source atoms
            
            # Step 2: For each edge e_{ji} (from j to i), we need to compute:
            #   Σ_{k∈N(j)\{i}} e_{kj}^{t-1}
            # This means: sum of all incoming messages to node j (source), excluding e_{ij}
            # 
            # For edge (i, j) where edge_index[0]=i (target) and edge_index[1]=j (source):
            # - The edge is e_{ji} (from j to i)
            # - We need to aggregate incoming messages to source node j
            # - Exclude the reverse edge e_{ij} (from i to j)
            a_message = atom_message_sum[b2a]  # [E, hidden_dim] - sum of incoming to source node j
            
            # Step 3: Get reverse edge message e_{ij} (from i to j)
            # This is the edge we need to exclude from the sum
            rev_message = e[b2revb]  # [E, hidden_dim]
            
            # Step 4: Compute neighbor sum: a_message - rev_message
            # This gives: Σ_{k->j} e_{kj} - e_{ij} = Σ_{k∈N(j)\{i}} e_{kj}
            # This matches the formula: Σ_{k∈N(i)\{j}} e_ki for edge e_{ij}
            # Note: For edge e_{ji}, we compute Σ_{k∈N(j)\{i}} e_{kj}, which is equivalent
            neighbor_sum = a_message - rev_message  # [E, hidden_dim]
            
            # Step 5: Update edge features via MLP
            # Formula: e_ij^t = MLP(e_ij^{t-1}, Σ_{k∈N(i)\{j}} e_ki^{t-1})
            # For edge e_{ji}: e_ji^t = MLP(e_ji^{t-1}, Σ_{k∈N(j)\{i}} e_{kj}^{t-1})
            update_input = torch.cat([e, neighbor_sum], dim=-1)  # [E, 2*hidden_dim]
            e = self.mlp_edge_update(update_input)  # [E, hidden_dim]
            
            # Optional: Add residual connection (similar to Chemprop's: act_func(input + message))
            # Chemprop uses: message = self.act_func(input + self.W_h(message_input_t))
            # Uncomment the following line if you want residual connection like Chemprop
            # e = e + e_init
        
        return e
    '''
    
    def _run_directed_mp(self, e_ij, edge_index, num_nodes, b2revb=None):
        """Directed 3D message passing iterations (EGNN + DMPNN hybrid).
        
        Args:
            e_ij: Edge features [E, hidden_dim], where e_ij represents edge from source (edge_index[0]) to target (edge_index[1])
            edge_index: [2, E] tensor, where edge_index[0] is source node and edge_index[1] is target node
            num_nodes: Number of nodes in the graph
            b2revb: [E] tensor, b2revb[i] gives the index of the reverse edge of edge i (optional, will be built if None)
        """
        e = e_ij  # edge features e_ij: from source (edge_index[0]) to target (edge_index[1])
        device = e.device
        
        # Build reverse edge mapping if not provided
        if b2revb is None:
            # For edge_index where [0]=source, [1]=target, reverse edge of (i->j) is (j->i)
            # We need to find the index of edge (j->i) for each edge (i->j)
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            num_edges = edge_index.size(1)
            
            if num_nodes is None:
                num_nodes = max(source_nodes.max().item(), target_nodes.max().item()) + 1
            
            max_node_idx = max(num_nodes, source_nodes.max().item(), target_nodes.max().item()) + 1
            edge_keys = source_nodes * max_node_idx + target_nodes  # keys are used for compuation efficiency, while ensuring uniqueness (different edges have different keys)
            reverse_keys = target_nodes * max_node_idx + source_nodes
            
            key_to_idx = {edge_keys[idx].item(): idx for idx in range(num_edges)} # build a dict (also for computation efficiency purpose)
            b2revb_list = [
                key_to_idx.get(reverse_keys[idx].item(), idx)  # Use self if reverse doesn't exist
                for idx in range(num_edges)
            ]
            b2revb = torch.tensor(b2revb_list, dtype=torch.long, device=device)
        
        for step in range(self.dmp_steps):
            # Aggregate all incoming edges to each node (edges where target = node)
            # For DMPNN: e_{ij}^{t+1} = MLP(e_{ij}^t, Σ_{k∈N(i)\{j}} e_{ki}^t)
            # For edge e_{ij} (from i to j), we need to aggregate all edges pointing to source node i
            # All edges pointing to i have target = i, i.e., edge_index[1] == i
            incoming_sum = torch.zeros(num_nodes, e.size(-1), device=device, dtype=e.dtype)
            incoming_sum.index_add_(0, edge_index[1], e)  # Aggregate to target nodes (all edges pointing to each node)
            
            # For edge e_{ij} (from i to j):
            # - incoming_sum[i] contains sum of all edges pointing to i (all edges with target = i)
            # - We need to exclude the reverse edge e_{ji} (from j to i), which also points to i
            # - The reverse edge e_{ji} is at index b2revb[edge_idx] for edge e_{ij}
            neighbor_sum = incoming_sum[edge_index[0]] - e[b2revb]  # Exclude reverse edge e_{ji}
            update_input = torch.cat([e, neighbor_sum], dim=-1)
            e = self.mlp_edge_update(update_input)
        return e
    
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
            # Batch processing - Vectorized implementation for better GPU utilization
            # Compute attention scores for all nodes at once (batch processing)
            attention_scores = self.attention_mlp(x)  # [N, heads]
            
            # Use PyG's softmax to compute attention weights per graph (grouped by batch)
            # softmax(src, index, num_nodes=None, dim=0) applies softmax within each group
            num_graphs = batch.max().item() + 1
            attention_weights = softmax(attention_scores, batch, num_nodes=num_graphs)  # [N, heads]
            
            # Compute weighted features: [N, heads, in_channels]
            weighted_features = x.unsqueeze(1) * attention_weights.unsqueeze(-1)  # [N, heads, in_channels]
            
            # Aggregate weighted features per graph using global_add_pool (optimized for batch processing)
            # For each head, aggregate features across nodes in the same graph
            pooled_per_head = []
            for head_idx in range(self.heads):
                # For each head, aggregate weighted features per graph
                head_features = weighted_features[:, head_idx, :]  # [N, in_channels]
                # Use global_add_pool: optimized for batch processing, more efficient than scatter_add
                pooled_head = global_add_pool(head_features, batch)  # [num_graphs, in_channels]
                pooled_per_head.append(pooled_head)  # [num_graphs, in_channels]
            
            # Stack and average across heads: [num_graphs, heads, in_channels] -> [num_graphs, in_channels]
            pooled = torch.stack(pooled_per_head, dim=1).mean(dim=1)  # [num_graphs, in_channels]
        
        # Output projection
        output = self.output_proj(pooled)
        output = self.dropout(output)
        
        return output


class AEGNNLayer(nn.Module):
    """AEGNN Layer, using GAT-EGNN as the core layer"""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1, 
                 alpha=0.2, edge_dim=None, use_equivariant=True, ffn_expansion_factor=4,
                 drop_path=0.0, pre_norm=False, dmp_steps=2, activation='SiLU'):
        super(AEGNNLayer, self).__init__()
        
        self.use_equivariant = use_equivariant
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pre_norm = pre_norm
        
        # GAT-EGNN core layer
        self.gat_egnn = GATEGNNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
            edge_dim=edge_dim,
            dmp_steps=dmp_steps
        )
        
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
        act_fn = activation_map.get(activation, nn.SiLU())
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * ffn_expansion_factor),
            act_fn,
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
        
        # Position encoding (for equivariance) with gating mechanism (Solution 2, Option B)
        if use_equivariant:
            # Project coordinates to input dimension to avoid shape inconsistency when adding to output dimension
            self.pos_embedding = nn.Linear(3, in_channels)
            
            # Gating mechanism (Option B: Per-dimension gating) - Solution 2 from PROBLEM_ANALYSIS.md
            # Each feature dimension has an independent gate value for adaptive fusion
            self.pos_gate = nn.Sequential(
                nn.Linear(in_channels * 2, in_channels // 2),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),  # Smaller dropout for gate network
                nn.Linear(in_channels // 2, in_channels),
                nn.Sigmoid()  # Output [N, in_channels] gate values in [0, 1]
            )
            
            # Initialize gate network: start with conservative fusion (small initial gate values)
            self._init_pos_gate()
    
    def _init_pos_gate(self):
        """
        Initialize position gate network with conservative values.
        This ensures initial gate values are small (~0.12), allowing the model
        to start with conservative fusion and learn optimal fusion strategy during training.
        """
        def init_gate_weights(m):
            if isinstance(m, nn.Linear):
                # Find the last linear layer (before Sigmoid)
                if m == self.pos_gate[-2]:  # Second to last layer (last is Sigmoid)
                    # Small weights for conservative initial gate values
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    # Negative bias: Sigmoid(-2) ≈ 0.12, small initial gate value
                    nn.init.constant_(m.bias, -2.0)
                else:
                    # Standard initialization for other layers
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        
        self.pos_gate.apply(init_gate_weights)
        
    def forward(self, x, edge_index, edge_attr=None, pos=None, b2revb=None, 
                layer_idx=None, pos_scale_per_layer=None):
        """
        Forward propagation
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            pos: Node positions (optional, for equivariance)
            b2revb: Pre-computed reverse edge mapping [E] (optional, Chemprop style)
            layer_idx: Layer index for layer-wise scaling (Recommendation C)
            pos_scale_per_layer: Per-layer scaling factors for positional encoding (Recommendation C)
        """
        # Position encoding (if position information is provided)
        # Check if pos contains valid (non-zero) coordinates to avoid encoding zero positions
        if pos is not None and self.use_equivariant:
            # Use stricter validation to ensure meaningful position information
            # This checks: (1) all values are finite, (2) sufficient nodes have non-zero positions,
            # and (3) mean position norm is above threshold
            # This prevents encoding zero positions which would cause identical encodings
            has_valid_pos = check_valid_positions(pos)
            
            if has_valid_pos:
                pos_encoded = self.pos_embedding(pos)
                # Gating mechanism: learn adaptive fusion weights per dimension
                gate_input = torch.cat([x, pos_encoded], dim=-1)  # [N, 2*in_channels]
                gate = self.pos_gate(gate_input)  # [N, in_channels]
                
                # Recommendation C: Layer-wise scaling for positional encoding
                # Apply per-layer scaling factor if provided
                if pos_scale_per_layer is not None and layer_idx is not None:
                    layer_scale = pos_scale_per_layer[layer_idx]  # Get scaling factor for this layer
                    # Adaptive fusion with layer-wise scaling: x = x + layer_scale * gate * pos_encoded
                    x = x + layer_scale * gate * pos_encoded
                else:
                    # Fallback: use gating mechanism without layer-wise scaling
                    x = x + gate * pos_encoded
            # If pos is invalid, skip position encoding to avoid identical encodings
        
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
                 node_features=5,  # Atom feature dimension (default matches MolecularGraphBuilder: atomic_number, hybridization, formal_charge, aromatic, chirality)
                 edge_features=9,   # Edge feature dimension (OGB-style: 4 bond types + 4 stereo + 1 conjugated)
                 hidden_dim=256,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.1,
                 output_dim=1,
                 pool_type='mean',
                 use_equivariant=True,
                 alpha=0.2,
                 ffn_expansion_factor=4,
                 dmp_steps=2,
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
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            ])
            self.fingerprint_mlp = nn.Sequential(*layers)
            
            if use_fingerprint_gate:
                # Gating Mechanism: Use GNN features (hidden_dim) to gate Fingerprint features (hidden_dim // 2)
                self.fingerprint_gate_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
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
                nn.SiLU(),
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
                dmp_steps=dmp_steps,
                activation=activation
            )
            for i in range(num_layers)
        ])
        
        # Recommendation C: Layer-wise scaling for positional encoding
        # Different layers use different scaling factors
        # Shallow layers: positional information is more important (fusion just begins)
        # Deep layers: positional information is relatively less important (already fused)
        if use_equivariant:
            self.pos_scale_per_layer = nn.Parameter(
                torch.ones(num_layers) * 0.1  # Initial value: 0.1 for conservative fusion
            )
        else:
            self.pos_scale_per_layer = None
        
        # Modality-specific MLPs for balanced fusion (optimization suggestion 1)
        # Process 2D mean H and 3D mean X separately through small MLPs before concat, to avoid single modality dominance
        self.h_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.x_mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Output layer (cat of processed H and X)
        self.graph_repr_dim = hidden_dim  # Default: hidden/2 (H) + hidden/2 (X)
        if use_fingerprint:
            self.graph_repr_dim += hidden_dim // 2  # Add fingerprint dimension
        if use_descriptor:
            self.graph_repr_dim += hidden_dim // 2  # Add descriptor dimension
            
        # NEW VERSION (edmpnn_model_new.py): Remove output_norm, use simple Linear (similar to fusion_model)
        # This version removes output_norm and uses direct Linear projection like fusion_model's task_heads
        self.output_proj = nn.Linear(self.graph_repr_dim, output_dim)  # Simple Linear like fusion_model
        
    def forward(self, x, edge_index, edge_attr, batch=None, pos=None, fingerprint=None, descriptor=None, return_graph_features=False, b2revb=None, compute_logits: bool = True):
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
            b2revb: Pre-computed reverse edge mapping [E] (optional, Chemprop style)
        """
        # Node and edge embedding
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply 3D rotation augmentation if enabled and in training mode
        if self.rotate_aug and self.training and pos is not None:
            pos = self._apply_random_rotation(pos, batch)
        
        # Store attention weights for visualization
        attention_weights = []
        
        # Pass through GAT-EGNN layers
        for layer_idx, layer in enumerate(self.aegnn_layers):
            # Pass layer index for layer-wise scaling (Recommendation C)
            x, attn_weights, updated_pos = layer(x, edge_index, edge_attr, pos, b2revb=b2revb, 
                                                 layer_idx=layer_idx, pos_scale_per_layer=self.pos_scale_per_layer)
            
            # Only update position if the updated position is valid (maintains continuity of position updates)
            # GATEGNNLayer already validates positions before returning, so updated_pos is valid if not None
            if updated_pos is not None:
                pos = updated_pos
            
            attention_weights.append(attn_weights)
        
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
        
        # Output projection: optionally skip computing logits when caller only needs graph_features
        if compute_logits:
            logits = self.project_graph_features(graph_features)
        else:
            # Create a lightweight placeholder tensor to preserve API shape when logits are unused.
            num_graphs = graph_features.size(0)
            out_dim = self.output_proj.out_features
            logits = graph_features.new_zeros(num_graphs, out_dim)
        
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
        # Keep nan_to_num only at the final output stage to ensure numerical stability
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        # NEW VERSION (edmpnn_model_new.py): Direct projection without output_norm (similar to fusion_model)
        # This matches fusion_model's approach: task_heads = nn.Linear(shared_mlp.output_dim, out_dim)
        logits = self.output_proj(graph_features)  # Direct Linear projection
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
                 label_smoothing=0.0, use_bce_for_imbalanced=False, pos_weight=None,
                 use_class_balanced_focal_loss=False, class_balanced_beta=0.9999, class_counts=None, **kwargs):
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
            use_class_balanced_focal_loss: Whether to use Class-Balanced Focal Loss (for extremely imbalanced datasets)
            class_balanced_beta: Beta parameter for Class-Balanced Loss (default: 0.9999)
            class_counts: Number of samples per class [num_classes], required for Class-Balanced Focal Loss
        """
        # Remove parameters that should not be passed to parent class
        # Note: These parameters are used by AEGNNMClassifier itself, not by parent AEGNNM
        kwargs.pop('use_class_balanced_focal_loss', None)
        kwargs.pop('class_balanced_beta', None)
        kwargs.pop('class_counts', None)
        kwargs.pop('primary_metric', None)  # primary_metric is only for regressor
        
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
        self.use_class_balanced_focal_loss = use_class_balanced_focal_loss
        self.class_balanced_beta = class_balanced_beta
        self.class_counts = class_counts
        
        # Initialize loss function
        if use_class_balanced_focal_loss:
            # Use Class-Balanced Focal Loss for extremely imbalanced datasets
            from utils.loss_utils import ClassBalancedFocalLoss
            self.loss_fn = ClassBalancedFocalLoss(
                beta=class_balanced_beta,
                gamma=focal_gamma,
                alpha=focal_alpha,
                reduction='mean'
            )
        elif use_bce_for_imbalanced:
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
        if self.use_class_balanced_focal_loss:
            # Class-Balanced Focal Loss requires class_counts
            if self.class_counts is not None:
                if isinstance(self.class_counts, (list, np.ndarray)):
                    class_counts_tensor = torch.tensor(self.class_counts, dtype=torch.float32, device=pred.device)
                elif isinstance(self.class_counts, torch.Tensor):
                    class_counts_tensor = self.class_counts.to(pred.device)
                else:
                    raise ValueError("class_counts must be a list, numpy array, or torch.Tensor")
                return self.loss_fn(pred, target, class_counts=class_counts_tensor)
            else:
                # Fallback to standard Focal Loss if class_counts not provided
                return self.loss_fn(pred, target, class_counts=None)
        elif self.use_bce_for_imbalanced:
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
        # Filter out classifier-specific parameters that AEGNNM base class doesn't accept
        regressor_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['class_weight', 'use_focal_loss', 'focal_alpha', 
                                       'focal_gamma', 'label_smoothing', 'use_bce_for_imbalanced',
                                       'pos_weight', 'use_class_balanced_focal_loss',
                                       'class_balanced_beta', 'class_counts', 'num_classes']}
        return AEGNNMRegressor(**regressor_kwargs)
    elif model_type == 'classifier':
        # Filter out regressor-specific parameters that AEGNNM base class doesn't accept
        classifier_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in ['primary_metric']}
        return AEGNNMClassifier(**classifier_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    model = create_aegnn_model(
        node_features=78,
        edge_features=9,
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
