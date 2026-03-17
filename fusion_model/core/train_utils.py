import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from typing import Tuple


def train(model, loader, loss_fn, optimizer, MODEL_TYPE, DEVICE, scheduler=None)-> float:
    model.train()
    total_loss = 0
    
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
        optimizer.step()  # update weights
        
        if scheduler is not None:
            scheduler.step()
        
        # total loss
        total_loss += loss.item()  # float
    
    # average loss  
    avg_loss = total_loss / len(loader)
    return avg_loss


def valid(model, loader, loss_fn, MODEL_TYPE, metric, TASK_TYPE, DEVICE) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # disable gradient calculation
        for batch in loader:
            if batch.batch.size(0) < 2:
                continue  # skip too-small batch
            batch = batch.to(DEVICE)      
            # forward pass
            output = model_forward(model, batch, MODEL_TYPE)  
            # calculating loss
            loss = loss_fn(output.view(-1), batch.y)
            
            # total loss
            total_loss += loss.item()
            
            # === metric ===
            if TASK_TYPE == 'classification':
                output = torch.sigmoid(output) # convert to a probability value between 0 and 1

            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(batch.y.detach().cpu().numpy())
                
    # average loss  
    avg_loss = total_loss / len(loader)
        
    # average metric
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metric_score = metric(all_labels, all_preds)
        
    return avg_loss, metric_score


def test(model, loader, metric, TASK_TYPE, MODEL_TYPE, DEVICE) -> float:           
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
        descriptor = data.descriptor if hasattr(data, 'descriptor') and data.descriptor is not None else torch.zeros(data.num_graphs if hasattr(data, 'num_graphs') else data.batch.max().item() + 1, 200, device=data.x.device, dtype=data.x.dtype)
        return model(data.x, data.edge_index, edge_attr, data.batch, descriptor, pos=pos, task_index=0)
    elif model_type == 'DMPEGNN_MMB_DESC':
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            n_edges = data.edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 9, device=data.x.device, dtype=data.x.dtype)
        pos = getattr(data, 'pos', None)
        molecule_idx = getattr(data, 'molecule_idx', None)
        return model(data.smiles, data.x, data.edge_index, edge_attr, data.descriptor, data.batch, task_index=0, pos=pos, molecule_idx=molecule_idx)
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
    
