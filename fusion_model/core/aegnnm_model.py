"""
AEGNN-M (Attention-Enhanced Graph Neural Network for Molecular Properties)
Ported from AEGNN-M (TDC)/models/aegnn_model.py for use in the fusion_model framework.

NOTE: This is a DISTINCT model from DMPEGNN (core/edmpnn_model_new.py).
Key architectural differences vs DMPEGNN:
  - GATEGNNLayer uses a single phi_e MLP (original EGNN style); no dmp_steps, no geo_gate
  - AEGNNLayer adds a pos_embedding (Linear 3→hidden) that directly encodes 3D coords
    into node features before attention, whereas DMPEGNN keeps 3D only in EGNN update
  - graph_repr_dim = hidden_dim (= h_branch hidden/2 + x_branch hidden/2)
    vs DMPEGNN's graph_repr_dim = hidden_dim // 2 (H branch only)

Changes from the original AEGNN-M (TDC) version:
  1. node_features default changed 78 → 82  (matches fusion_model's dmpegnn_data_utils)
  2. edge_features default changed  4 → 9   (matches fusion_model's 9-dim OGB-style edges)
  3. AEGNNM.forward() accepts compute_logits=True keyword arg (for compatibility with
     DMPEGNN_Desc_Model / AEGNN_Desc_Model wrapper calling convention)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv,
    global_mean_pool, global_max_pool, global_add_pool,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Paper: https://arxiv.org/abs/1708.02002

    Focal Loss = -α(1-p)^γ * log(p)
    Supports class-weighted focal loss for extremely imbalanced datasets.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0,
                 class_weight=None):
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
        target = target.long()

        if self.label_smoothing > 0.0:
            num_classes = pred.size(1)
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            target_one_hot = (1.0 - self.label_smoothing) * target_one_hot + \
                             self.label_smoothing / num_classes
            log_probs = F.log_softmax(pred, dim=1)
            ce_loss = -(target_one_hot * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(pred, target, reduction='none')

        probs = torch.exp(-ce_loss)
        focal_weight = (1 - probs) ** self.gamma

        if isinstance(self.alpha, (float, int)):
            if pred.size(1) == 2:
                alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            else:
                alpha_t = self.alpha
        elif isinstance(self.alpha, torch.Tensor):
            if self.alpha.dim() == 0:
                alpha_t = self.alpha.to(target.device)
            else:
                alpha = self.alpha.to(target.device)
                alpha_t = alpha[target]
        else:
            alpha_t = self.alpha

        if self.class_weight is not None:
            if pred.size(1) == 2:
                if len(self.class_weight) > 1:
                    class_weight_t = torch.where(
                        target == 1,
                        self.class_weight[1].to(target.device),
                        self.class_weight[0].to(target.device)
                    )
                else:
                    class_weight_t = self.class_weight[0].to(target.device)
            else:
                class_weight_t = self.class_weight[target].to(target.device)
            focal_loss = class_weight_t * alpha_t * focal_weight * ce_loss
        else:
            focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------------------------------------------------------
# GATEGNNLayer — AEGNN-M version
# DISTINCT from edmpnn_model_new.GATEGNNLayer which has:
#   dmp_steps, mlp_edge_init/update, geo_gate
# This version uses the original EGNN single-MLP phi_e approach (no DMP).
# ---------------------------------------------------------------------------
class GATEGNNLayer(MessagePassing):
    """
    GAT-EGNN Layer (AEGNN-M version):
    Combines Graph Attention Network (GAT) with Equivariant GNN (EGNN).

    Differences vs DMPEGNN's GATEGNNLayer:
      - Uses single phi_e MLP (original EGNN message network)
      - No directed message passing (dmp_steps)
      - No geometric gate (geo_gate)
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

        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"

        self.W = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, heads * self.head_dim, bias=False) if edge_dim else None

        self.att = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))
        self.att_edge = nn.Parameter(torch.empty(1, heads, self.head_dim)) if edge_dim else None

        # EGNN parameters: φ_e (edge MLP), φ_x (coord update), φ_h (node update)
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
        self.coord_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.att)
        if self.att_edge is not None:
            nn.init.xavier_uniform_(self.att_edge)

    def forward(self, x, edge_index, edge_attr=None, pos=None, b2revb=None):
        edge_attr_original = edge_attr

        h = self.W(x).view(-1, self.heads, self.head_dim)

        if edge_attr is not None and self.W_edge is not None:
            edge_attr_transformed = self.W_edge(edge_attr).view(-1, self.heads, self.head_dim)
        else:
            edge_attr_transformed = None

        alpha = self._compute_attention(h, edge_index, edge_attr_transformed)
        out = self.propagate(edge_index, x=h, alpha=alpha, edge_attr=edge_attr_transformed)

        if self.concat:
            out = out.view(-1, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)

        out = F.dropout(out, p=self.dropout, training=self.training)

        if pos is not None:
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            pos_norm = torch.norm(pos, dim=-1)
            has_valid_pos = torch.isfinite(pos_norm).all() and (pos_norm.sum() > 1e-6)
            if has_valid_pos:
                out, pos = self._egnn_update(out, pos, edge_index, edge_attr_original)

        return out, alpha, pos if pos is not None else None

    def _compute_attention(self, h, edge_index, edge_attr=None):
        h_i = h[edge_index[0]]
        h_j = h[edge_index[1]]
        h_concat = torch.cat([h_i, h_j], dim=-1)
        e = (h_concat * self.att).sum(dim=-1)

        if edge_attr is not None and self.att_edge is not None:
            e_edge = (edge_attr * self.att_edge).sum(dim=-1)
            e = e + e_edge

        e = F.gelu(e)
        alpha = softmax(e, edge_index[0], num_nodes=h.size(0))
        return alpha

    def _egnn_update(self, h, pos, edge_index, edge_attr=None):
        """EGNN equivariant update (equations 4-7)."""
        device = h.device
        num_nodes = h.size(0)

        pos_i = pos[edge_index[0]]
        pos_j = pos[edge_index[1]]
        rel_pos = pos_i - pos_j
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        hi = h[edge_index[0]]
        hj = h[edge_index[1]]

        inputs = [hi, hj, dist_sq]
        if edge_attr is not None and self.edge_attr_proj is not None:
            edge_feat = self.edge_attr_proj(edge_attr)
            inputs.append(edge_feat)

        m_ij = self.phi_e(torch.cat(inputs, dim=-1))

        deg = torch.zeros(num_nodes, device=device).scatter_add_(
            0, edge_index[0], torch.ones(edge_index.size(1), device=device)
        )
        deg = deg.clamp(min=1.0)
        coord_coeff = (1.0 / deg)[edge_index[0]].unsqueeze(-1)
        phi_x_val = torch.tanh(self.phi_x(m_ij))
        coord_contrib = coord_coeff * rel_pos * phi_x_val

        gate_value = self.coord_gate(m_ij)
        coord_contrib = coord_contrib * gate_value
        coord_contrib = coord_contrib.to(pos.dtype)

        pos_update = torch.zeros_like(pos, dtype=pos.dtype)
        pos_update.index_add_(0, edge_index[0], coord_contrib)
        pos = pos + pos_update

        node_messages = torch.zeros(num_nodes, h.size(-1), device=device, dtype=h.dtype)
        node_messages.index_add_(0, edge_index[0], m_ij.to(h.dtype))

        h = self.phi_h(torch.cat([h, node_messages], dim=-1))
        return h, pos

    def message(self, x_j, alpha, edge_attr=None):
        out = x_j * alpha.unsqueeze(-1)
        if edge_attr is not None:
            out = out + edge_attr
        return out


# ---------------------------------------------------------------------------
# AEGNNLayer — AEGNN-M version
# DISTINCT from edmpnn_model_new.AEGNNLayer which has no pos_embedding.
# This version encodes 3D pos directly into node features via pos_embedding
# (Linear 3 → in_channels) before passing to attention.
# ---------------------------------------------------------------------------
class AEGNNLayer(nn.Module):
    """
    AEGNN Layer (AEGNN-M version), using GATEGNNLayer as the core.

    Key difference vs DMPEGNN's AEGNNLayer:
      - Has pos_embedding = Linear(3, in_channels) that adds encoded 3D
        coordinates to node features before attention (AEGNN-M style).
      - DMPEGNN has no such pos_embedding; 3D is handled purely inside EGNN.
    """

    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1,
                 alpha=0.2, edge_dim=None, use_equivariant=True, ffn_expansion_factor=4,
                 drop_path=0.0, pre_norm=False, activation=None):
        super(AEGNNLayer, self).__init__()

        self.use_equivariant = use_equivariant
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pre_norm = pre_norm

        if activation is None:
            self.activation = nn.SiLU()
        elif isinstance(activation, str):
            activation_map = {
                'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'PReLU': nn.PReLU(),
                'tanh': nn.Tanh(), 'SELU': nn.SELU(), 'ELU': nn.ELU(), 'SiLU': nn.SiLU()
            }
            self.activation = activation_map.get(activation, nn.SiLU())
        else:
            self.activation = activation

        self.gat_egnn = GATEGNNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
            edge_dim=edge_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * ffn_expansion_factor),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(out_channels * ffn_expansion_factor, out_channels)
        )

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None

        # AEGNN-M specific: encode 3D coords into node features before attention.
        # This is absent in DMPEGNN's AEGNNLayer.
        if use_equivariant:
            self.pos_embedding = nn.Linear(3, in_channels)

    def forward(self, x, edge_index, edge_attr=None, pos=None, b2revb=None):
        # AEGNN-M: add position encoding to node features before attention
        if pos is not None and self.use_equivariant:
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            pos_norm = torch.norm(pos, dim=-1)
            has_valid_pos = pos_norm.sum() > 1e-6
            if has_valid_pos:
                pos_encoded = self.pos_embedding(pos)
                x = x + pos_encoded

        x_residual = x

        if self.pre_norm:
            x_norm = self.norm1(x)
            gat_out, attn_weights, updated_pos = self.gat_egnn(
                x_norm, edge_index, edge_attr, pos, b2revb=b2revb
            )
            if self.residual_proj is not None:
                x_residual = self.residual_proj(x_residual)
            x = x_residual + self.drop_path(gat_out)
            x_norm2 = self.norm2(x)
            ffn_out = self.ffn(x_norm2)
            out = x + self.drop_path(ffn_out)
            out = self.dropout(out)
        else:
            gat_out, attn_weights, updated_pos = self.gat_egnn(
                x, edge_index, edge_attr, pos, b2revb=b2revb
            )
            if self.residual_proj is not None:
                x_residual = self.residual_proj(x_residual)
            x = self.norm1(x_residual + self.drop_path(gat_out))
            ffn_out = self.ffn(x)
            out = self.norm2(x + self.drop_path(ffn_out))
            out = self.dropout(out)

        return out, attn_weights, updated_pos


# ---------------------------------------------------------------------------
# AEGNNM — main backbone
# ---------------------------------------------------------------------------
class AEGNNM(nn.Module):
    """
    AEGNN-M Main Model (backbone).

    graph_repr_dim = hidden_dim  (= h_branch hidden/2 + x_branch hidden/2)
    This differs from DMPEGNN whose graph_repr_dim = hidden_dim // 2.

    Defaults adjusted for fusion_model compatibility:
      node_features=82, edge_features=9
    """

    def __init__(self,
                 node_features=82,   # changed from 78 to match fusion_model data pipeline
                 edge_features=9,    # changed from 4 to match 9-dim OGB-style edges
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
                 use_descriptor=False,
                 descriptor_dim=200,
                 descriptor_dropout=0.0,
                 activation='SiLU'):
        super(AEGNNM, self).__init__()

        activation_map = {
            'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'PReLU': nn.PReLU(),
            'tanh': nn.Tanh(), 'SELU': nn.SELU(), 'ELU': nn.ELU(), 'SiLU': nn.SiLU()
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

        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)

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
                self.fingerprint_gate_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    self.activation,
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.Sigmoid()
                )

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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

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

        # h_mlp: pools node embeddings (hidden_dim → hidden_dim//2)
        # x_mlp: pools 3D coordinates  (coord_dim  → hidden_dim//2)
        # Together they form the graph representation of size hidden_dim.
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

        # graph_repr_dim = hidden_dim (h_branch hidden/2 + x_branch hidden/2)
        # plus optional fingerprint / descriptor branches
        self.graph_repr_dim = hidden_dim
        if use_fingerprint:
            self.graph_repr_dim += hidden_dim // 2
        if use_descriptor:
            self.graph_repr_dim += hidden_dim // 2

        self.output_norm = nn.LayerNorm(self.graph_repr_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.graph_repr_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch=None, pos=None,
                fingerprint=None, descriptor=None,
                return_graph_features=False, b2revb=None,
                compute_logits=True):   # ← added for DMPEGNN wrapper compatibility
        """
        Args:
            compute_logits: kept for API compatibility with DMPEGNN wrapper classes;
                            the logits are always computed internally regardless of this flag.
        """
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        if self.rotate_aug and self.training and pos is not None:
            pos = self._apply_random_rotation(pos, batch)

        attention_weights = []
        for layer in self.aegnn_layers:
            x, attn_weights, pos = layer(x, edge_index, edge_attr, pos, b2revb=b2revb)
            attention_weights.append(attn_weights)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if pos is not None:
                pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Pool node embeddings (H branch)
        if self.pool_type == 'mean':
            node_mean = global_mean_pool(x, batch)
        elif self.pool_type == 'sum':
            node_mean = global_add_pool(x, batch)
        elif self.pool_type == 'norm':
            node_sum = global_add_pool(x, batch)
            num_nodes_per_graph = global_add_pool(
                torch.ones(x.size(0), 1, device=x.device), batch
            )
            node_mean = node_sum / (torch.sqrt(num_nodes_per_graph) + 1e-8)
        else:
            node_mean = global_mean_pool(x, batch)

        # Pool 3D coordinates (X branch)
        if pos is not None:
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            if self.pool_type == 'mean':
                coord_mean = global_mean_pool(pos, batch)
            elif self.pool_type == 'sum':
                coord_mean = global_add_pool(pos, batch)
            elif self.pool_type == 'norm':
                coord_sum = global_add_pool(pos, batch)
                num_nodes_per_graph = global_add_pool(
                    torch.ones(pos.size(0), 1, device=pos.device), batch
                )
                coord_mean = coord_sum / (torch.sqrt(num_nodes_per_graph) + 1e-8)
            else:
                coord_mean = global_mean_pool(pos, batch)
        else:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            coord_mean = torch.zeros(num_graphs, self.coord_dim, device=x.device)

        h_processed = self.h_mlp(node_mean)
        x_processed = self.x_mlp(coord_mean)

        feature_list = [h_processed, x_processed]

        if self.use_fingerprint:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            if fingerprint is not None:
                fp_processed = self.fingerprint_mlp(fingerprint)
                if self.use_fingerprint_gate:
                    gnn_features = torch.cat([h_processed, x_processed], dim=-1)
                    gate = self.fingerprint_gate_mlp(gnn_features)
                    fp_processed = fp_processed * gate
            else:
                fp_processed = x.new_zeros(num_graphs, self.hidden_dim // 2)
            feature_list.append(fp_processed)

        if self.use_descriptor:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            if descriptor is not None:
                desc_processed = self.descriptor_mlp(descriptor)
            else:
                desc_processed = x.new_zeros(num_graphs, self.hidden_dim // 2)
            feature_list.append(desc_processed)

        graph_features = torch.cat(feature_list, dim=-1)
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.project_graph_features(graph_features)

        if return_graph_features:
            return logits, attention_weights, graph_features
        return logits, attention_weights

    def project_graph_features(self, graph_features):
        graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        logits_input = self.output_norm(graph_features)
        return self.output_proj(logits_input)

    def _apply_random_rotation(self, pos, batch=None):
        device = pos.device
        dtype = pos.dtype
        if batch is None:
            rot_matrix = self._get_random_rotation_matrix(dtype=dtype, device=device)
            return (pos @ rot_matrix).to(dtype)
        else:
            num_graphs = batch.max().item() + 1
            pos_rotated = torch.zeros_like(pos)
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.sum() > 0:
                    rot_matrix = self._get_random_rotation_matrix(dtype=dtype, device=device)
                    pos_rotated[mask] = (pos[mask] @ rot_matrix).to(dtype)
            return pos_rotated

    def _get_random_rotation_matrix(self, dtype=torch.float32, device=None):
        device = device or torch.device('cpu')
        axis = torch.randn(3, device=device, dtype=dtype)
        axis = axis / (torch.norm(axis) + torch.tensor(1e-8, device=device, dtype=dtype))
        theta = torch.rand(1, device=device, dtype=dtype) * (2 * math.pi)
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device, dtype=dtype)
        I = torch.eye(3, device=device, dtype=dtype)
        return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    def get_attention_weights(self, x, edge_index, edge_attr, batch=None, pos=None,
                               fingerprint=None):
        with torch.no_grad():
            _, attention_weights = self.forward(x, edge_index, edge_attr, batch, pos,
                                                fingerprint)
        return attention_weights
