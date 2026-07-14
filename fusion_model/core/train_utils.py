import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from typing import Tuple


def train(model, loader, loss_fn, optimizer, MODEL_TYPE, DEVICE, scheduler=None)-> float:
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader: 
        if batch.batch.size(0) < 2:
            continue  # skip too-small batch
        batch = batch.to(DEVICE)             
        # forward pass
        output = model_forward(model, batch, MODEL_TYPE)
        # calculating loss
        loss = loss_fn(output.view(-1), batch.y)
        
        # backpropagation
        optimizer.zero_grad()  # clear gradients to prevent accumulation
        loss.backward()  # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # update weights
        
        if scheduler is not None:
            scheduler.step()
        
        # total loss
        total_loss += loss.item()  # float
        num_batches += 1
    
    # average loss  
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def valid(model, loader, loss_fn, MODEL_TYPE, metric, TASK_TYPE, DEVICE, log_transform: bool = False) -> Tuple[float, float]:
    """Evaluate on validation set.

    log_transform: mirror of the flag in test().  When True (regression with
        log1p targets), apply expm1 to predictions and labels *before* computing
        the metric so that the reported validation score is on the original scale
        (e.g., original-unit MAE).  The loss is always computed in log space
        (consistent with training) and is not affected by this flag.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculation
        for batch in loader:
            if batch.batch.size(0) < 2:
                continue  # skip too-small batch
            batch = batch.to(DEVICE)
            # forward pass
            output = model_forward(model, batch, MODEL_TYPE)
            # calculating loss (always in log space when log_transform=True)
            loss = loss_fn(output.view(-1), batch.y)

            # total loss
            total_loss += loss.item()
            num_batches += 1

            # === metric ===
            if TASK_TYPE == 'classification':
                output = torch.sigmoid(output) # convert to a probability value between 0 and 1

            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(batch.y.detach().cpu().numpy())

    # average loss
    avg_loss = total_loss / max(num_batches, 1)
        
    # average metric
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    if log_transform and TASK_TYPE == 'regression':
        # Inverse log1p so the metric is on the original (interpretable) scale.
        # Early stopping and HP selection therefore operate on original-scale
        # MAE, making study.best_value directly comparable to test results.
        all_preds  = np.expm1(all_preds)
        all_labels = np.expm1(all_labels)

    metric_score = metric(all_labels, all_preds)
        
    return avg_loss, metric_score


def test(model, loader, metric, TASK_TYPE, MODEL_TYPE, DEVICE, log_transform: bool = False) -> float:
    """Run inference on the test set and return (metric_score, all_preds).

    log_transform: if True, apply torch.expm1 to both predictions and labels
        before computing the metric.  Use this when the model was trained on
        log1p-transformed regression targets so that the reported score is in
        the original (interpretable) scale.  For rank-based metrics such as
        Spearman, the result is identical regardless of this flag, but for MAE
        the original-scale value is required to be comparable with benchmarks.
    """
    model.eval()
    all_preds = []
    all_labels = []     
    
    with torch.no_grad():   # disable gradient calculation        
        for batch in loader:
            if batch.batch.size(0) < 2:
                continue  # skip too-small batch
            batch = batch.to(DEVICE)           
            # forward pass
            output = model_forward(model, batch, MODEL_TYPE)
            
            # predictions, labels
            all_preds.extend(output.view(-1).tolist())
            all_labels.extend(batch.y.tolist())
    
    # convert to tensor
    if TASK_TYPE == 'regression':
        all_preds = torch.tensor(all_preds, dtype=torch.float32)
        all_labels = torch.tensor(all_labels, dtype=torch.float32)
        if log_transform:
            # Inverse the log1p applied during training → original scale
            all_preds  = torch.expm1(all_preds)
            all_labels = torch.expm1(all_labels)
    elif TASK_TYPE == 'classification':
        all_preds = torch.tensor(all_preds, dtype=torch.float32)
        all_preds = torch.sigmoid(all_preds)  # convert to a probability value between 0 and 1
        all_labels = torch.tensor(all_labels, dtype=torch.int)  # convert to an integer
    
    # calculate metric
    test_score = metric(all_labels, all_preds)
    return test_score, all_preds

# --------------------------------------
def predict(model, loader, TASK_TYPE, MODEL_TYPE, DEVICE):
    model.eval()
    all_preds = []

    for batch in loader:
        batch = batch.to(DEVICE)
        output = model_forward(model, batch, MODEL_TYPE)

        if TASK_TYPE == 'classification':
            output = torch.sigmoid(output)

        all_preds.append(output.detach().float().cpu().view(-1))

    return torch.cat(all_preds, dim=0)

# --------------------------------------

def model_forward(model, data, model_type):
    if model_type == 'DESC':
        return model(data.descriptor, task_index=0)
    elif model_type == 'DMPEGNN':
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            # backward compat: old cached data without edge_attr
            n_edges = data.edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 9, device=data.x.device, dtype=data.x.dtype)
        pos = getattr(data, 'pos', None)
        molecule_idx = getattr(data, 'molecule_idx', None)
        return model(data.x, data.edge_index, edge_attr, data.batch, pos=pos, task_index=0, molecule_idx=molecule_idx)
    elif model_type == 'DMPEGNN_DESC':
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            n_edges = data.edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 9, device=data.x.device, dtype=data.x.dtype)
        pos = getattr(data, 'pos', None)
        molecule_idx = getattr(data, 'molecule_idx', None)
        return model(data.x, data.edge_index, edge_attr, data.descriptor, data.batch, task_index=0, pos=pos, molecule_idx=molecule_idx)
    elif model_type == 'DMPEGNN_MMB_DESC':
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            n_edges = data.edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 9, device=data.x.device, dtype=data.x.dtype)
        pos = getattr(data, 'pos', None)
        molecule_idx = getattr(data, 'molecule_idx', None)
        return model(data.smiles, data.x, data.edge_index, edge_attr, data.descriptor, data.batch, task_index=0, pos=pos, molecule_idx=molecule_idx)
    # ------------------------------------------------------------------
    # AEGNN-M routing — shares the DMPEGNN 3D data pipeline (same batch
    # fields: x[82-dim], edge_attr[9-dim], pos, molecule_idx).
    # Backbone: core.aegnnm_model.AEGNNM  ≠  DMPEGNN (edmpnn_model_new)
    # ------------------------------------------------------------------
    elif model_type == 'AEGNN':
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            n_edges = data.edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 9, device=data.x.device, dtype=data.x.dtype)
        pos = getattr(data, 'pos', None)
        molecule_idx = getattr(data, 'molecule_idx', None)
        return model(data.x, data.edge_index, edge_attr, data.batch, pos=pos, task_index=0, molecule_idx=molecule_idx)
    elif model_type == 'AEGNN_DESC':
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            n_edges = data.edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 9, device=data.x.device, dtype=data.x.dtype)
        pos = getattr(data, 'pos', None)
        molecule_idx = getattr(data, 'molecule_idx', None)
        return model(data.x, data.edge_index, edge_attr, data.descriptor, data.batch, task_index=0, pos=pos, molecule_idx=molecule_idx)
    elif model_type == 'GCN':
        return model(data.x, data.edge_index, data.batch)
    elif model_type == 'MMB':
        return model(data.smiles, task_index=0)
    elif model_type == 'GCN_DESC':
        return model(data.x, data.edge_index, data.descriptor, data.batch, task_index=0)
    elif model_type == 'MMB_DESC':
        return model(data.smiles, data.descriptor, task_index=0)
    elif model_type == 'GCN_MMB':
        return model(data.smiles, data.x, data.edge_index, data.batch, task_index=0)
    elif model_type == 'GCN_MMB_DESC':
        return model(data.smiles, data.x, data.edge_index, data.descriptor, data.batch, task_index=0)
    elif model_type == 'MPN_MMB_DESC':
        return model(data.smiles, data.descriptor, task_index=0)
    elif model_type == 'MPN':
        return model(data.smiles, task_index=0)
    elif model_type == 'MPN_DESC':
        return model(data.smiles, data.descriptor, task_index=0)
    elif model_type == 'MPN_MMB':
        return model(data.smiles, task_index=0)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
