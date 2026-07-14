"""
Shared model construction logic used by both optuna_train.py and seed_train.py.
Single source of truth for get_model() to prevent divergence between training
and hyperparameter-search code paths.
"""

import torch

from core.models import (
    GCN_Model, MMB_Model, Desc_Model,
    GCN_MMB_Model, MMB_Desc_Model,
    GCN_Desc_Model, GCN_MMB_Desc_Model,
    MegaMolBART_Finetuned_Model, MPN_MMB_Desc_Model,
    MPN_Model, MPN_Desc_Model, MPN_MMB_Model,
    DMPEGNN, DMPEGNN_Fusion_Model, DMPEGNN_Desc_Model, DMPEGNN_MMB_Desc_Model,
    # AEGNN-M wrappers — backbone is core.aegnnm_model.AEGNNM (distinct from DMPEGNN)
    AEGNN, AEGNN_Fusion_Model, AEGNN_Desc_Model,
)
from chemprop.models import MPN
from chemprop.args import TrainArgs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(args, model_type, task_output_dims,
              gcn_model=None, megamolbart_model=None,
              gcn_output_dim=None, **mlp_kwargs):
    """Build and return the requested model.

    Parameters
    ----------
    args:               Namespace holding all hyperparameters.
    model_type:         String identifier (e.g. 'DMPEGNN_MMB_DESC').
    task_output_dims:   List of per-task output dimensions.
    gcn_model:          Pre-built GCN backbone (required for GCN-* models).
    megamolbart_model:  Pre-built MMB backbone (required for MMB-* models).
    gcn_output_dim:     Output dim of the GCN backbone.
    **mlp_kwargs:       MLP hyperparameters forwarded to the fusion head.
    """
    if model_type == 'GCN':
        return gcn_model

    if model_type == 'MMB':
        return MMB_Model(megamolbart_model, task_output_dims, **mlp_kwargs)

    if model_type == 'DESC':
        return Desc_Model(task_output_dims, **mlp_kwargs)

    if model_type == 'GCN_MMB':
        return GCN_MMB_Model(gcn_model, megamolbart_model, gcn_output_dim,
                             task_output_dims, **mlp_kwargs)

    if model_type == 'MMB_DESC':
        return MMB_Desc_Model(megamolbart_model, task_output_dims, **mlp_kwargs)

    if model_type == 'GCN_DESC':
        return GCN_Desc_Model(gcn_model, gcn_output_dim, task_output_dims, **mlp_kwargs)

    if model_type == 'GCN_MMB_DESC':
        return GCN_MMB_Desc_Model(gcn_model, megamolbart_model, gcn_output_dim,
                                  task_output_dims, **mlp_kwargs)

    if model_type == 'MPN':
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_Model(mpn_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == 'MPN_MMB_DESC':
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_MMB_Desc_Model(mpn_model, megamolbart_model, task_output_dims,
                                  mpn_args.hidden_size, **mlp_kwargs)

    if model_type == 'MPN_DESC':
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_Desc_Model(mpn_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == 'MPN_MMB':
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_MMB_Model(mpn_model, megamolbart_model, task_output_dims,
                             mpn_args.hidden_size, **mlp_kwargs)

    if model_type == 'DMPEGNN':
        return DMPEGNN_Fusion_Model(
            node_features=82,
            edge_features=9,
            output_dim=task_output_dims[0],
            hidden_dim=args.dmpegnn_hidden_dim,
            num_layers=args.dmpegnn_num_layers,
            num_heads=args.dmpegnn_num_heads,
            dropout=args.dmpegnn_dropout,
            dmp_steps=args.dmpegnn_dmp_steps,
            pool_type=args.dmpegnn_pool_type,
            task_output_dims=task_output_dims,
            **mlp_kwargs,
        )

    if model_type == 'DMPEGNN_DESC':
        dmpegnn_backbone = DMPEGNN(
            node_features=82,
            edge_features=9,
            hidden_dim=args.dmpegnn_hidden_dim,
            num_layers=args.dmpegnn_num_layers,
            num_heads=args.dmpegnn_num_heads,
            dropout=args.dmpegnn_dropout,
            output_dim=task_output_dims[0],
            pool_type=args.dmpegnn_pool_type,
            use_equivariant=True,
            use_fingerprint=False,
            use_descriptor=False,
            descriptor_dim=200,
            dmp_steps=args.dmpegnn_dmp_steps,
            use_coord_branch=False,
        ).to(DEVICE)
        return DMPEGNN_Desc_Model(
            dmpegnn_backbone=dmpegnn_backbone,
            task_output_dims=task_output_dims,
            **mlp_kwargs,
        )

    if model_type == 'DMPEGNN_MMB_DESC':
        dmpegnn_backbone = DMPEGNN(
            node_features=82,
            edge_features=9,
            hidden_dim=args.dmpegnn_hidden_dim,
            num_layers=args.dmpegnn_num_layers,
            num_heads=args.dmpegnn_num_heads,
            dropout=args.dmpegnn_dropout,
            output_dim=task_output_dims[0],
            pool_type=args.dmpegnn_pool_type,
            use_equivariant=True,
            use_fingerprint=False,
            use_descriptor=False,
            descriptor_dim=200,
            dmp_steps=args.dmpegnn_dmp_steps,
            use_coord_branch=False,
        ).to(DEVICE)
        return DMPEGNN_MMB_Desc_Model(
            dmpegnn_backbone=dmpegnn_backbone,
            mmb_model=megamolbart_model,
            task_output_dims=task_output_dims,
            **mlp_kwargs,
        )

    # ------------------------------------------------------------------
    # AEGNN-M models
    # Backbone: core.aegnnm_model.AEGNNM  (single phi_e, pos_embedding)
    # Distinct from DMPEGNN (no dmp_steps, no geo_gate, graph_repr_dim=hidden_dim)
    # ------------------------------------------------------------------
    if model_type == 'AEGNN':
        return AEGNN_Fusion_Model(
            node_features=82,
            edge_features=9,
            output_dim=task_output_dims[0],
            hidden_dim=args.aegnn_hidden_dim,
            num_layers=args.aegnn_num_layers,
            num_heads=args.aegnn_num_heads,
            dropout=args.aegnn_dropout,
            pool_type=args.aegnn_pool_type,
            task_output_dims=task_output_dims,
            **mlp_kwargs,
        )

    if model_type == 'AEGNN_DESC':
        aegnn_backbone = AEGNN(
            node_features=82,
            edge_features=9,
            hidden_dim=args.aegnn_hidden_dim,
            num_layers=args.aegnn_num_layers,
            num_heads=args.aegnn_num_heads,
            dropout=args.aegnn_dropout,
            output_dim=task_output_dims[0],
            pool_type=args.aegnn_pool_type,
            use_equivariant=True,
            use_fingerprint=False,
            use_descriptor=False,
        ).to(DEVICE)
        return AEGNN_Desc_Model(
            aegnn_backbone=aegnn_backbone,
            task_output_dims=task_output_dims,
            **mlp_kwargs,
        )

    raise ValueError(f"Unknown model type: {model_type}")
