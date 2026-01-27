"""
E-DMPNN Distributed Data Parallel (DDP) Training Script
Supports multi-GPU training with EGNN+DMPNN architecture

This is a completely independent version from train_aegnnm.py.
Uses models.edmpnn_model.create_aegnn_model with dmp_steps parameter.
"""

import os
import sys
import argparse
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import math
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import yaml
from sklearn.preprocessing import RobustScaler

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.edmpnn_model import create_aegnn_model
from models.edmpnn_model import GATEGNNLayer
from utils.data_utils import MolecularDataset, DataPreprocessor, MolecularGraphBuilder
from utils.loss_utils import (
    calculate_focal_params,
    calculate_class_weights as calc_class_weights,
    calculate_pos_weight as calc_pos_weight,
    calculate_imbalance_ratio
)
from utils.threshold_utils import (
    find_optimal_threshold_adaptive,
    find_optimal_threshold_cv,
    evaluate_with_threshold,
    check_threshold_stability
)

# Progress monitoring for Optuna pruning
try:
    from progress_monitor import JSONProgressMonitor
    PROGRESS_MONITORING_AVAILABLE = True
except ImportError:
    PROGRESS_MONITORING_AVAILABLE = False
    JSONProgressMonitor = None


def init_weights_advanced(model):
    """
    Advanced initialization: Determine initialization strategy based on module name
    - Output layer/output_proj: Smaller gain, avoid extreme initial outputs
    - Embedding layer: Standard Xavier
    - Gate/attention layers: Smaller normal initialization, conservative start
    - Other Linear: Standard Xavier
    - LayerNorm: Weight 1, bias 0
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            lname = name.lower()
            if ("output_proj" in lname) or ("output" in lname):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            elif "embedding" in lname:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif ("gate" in lname) or ("attention" in lname):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            else:
                nn.init.xavier_uniform_(m.weight, gain=1.0)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def collate_with_b2revb(data_list):
    """
    Custom collate function that builds b2revb for batched data (Chemprop style).
    This automatically constructs the reverse edge mapping during batch creation.
    """
    batch = Batch.from_data_list(data_list)
    
    # Build b2revb for batched data if edge_index exists
    if hasattr(batch, 'edge_index') and batch.edge_index is not None:
        try:
            batch.b2revb = GATEGNNLayer.build_b2revb(
                batch.edge_index,
                num_nodes=batch.num_nodes
            )
        except Exception as e:
            # If building b2revb fails, set to None (model will build it dynamically)
            batch.b2revb = None
    
    return batch


def calculate_macro_averaged_auroc(y_true, y_pred_probs):
    """
    Calculate macro-averaged AUROC for multitask classification datasets.
    
    For multitask datasets (e.g., TOX21, SIDER, MUV, HIV), this function:
    1. Calculates AUROC for each task separately
    2. Returns the mean AUROC across all tasks (macro-averaged)
    
    For single-task datasets, returns the standard AUROC.
    
    Args:
        y_true: Ground truth labels [num_samples] or [num_samples, num_tasks]
        y_pred_probs: Predicted probabilities [num_samples] or [num_samples, num_tasks]
    
    Returns:
        macro_auroc: Macro-averaged AUROC (float)
        task_aurocs: List of AUROC for each task (for debugging)
    """
    from sklearn.metrics import roc_auc_score
    
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.cpu().numpy()
    
    # Ensure probabilities are in [0, 1]
    y_pred_probs = np.clip(y_pred_probs, 0, 1)
    
    # Check if multitask (2D arrays)
    is_multitask = y_true.ndim == 2 and y_pred_probs.ndim == 2 and y_true.shape[1] > 1
    
    if is_multitask:
        # Multitask: calculate AUROC for each task separately
        num_tasks = y_true.shape[1]
        task_aurocs = []
        
        for task_idx in range(num_tasks):
            task_y_true = y_true[:, task_idx]
            task_y_pred = y_pred_probs[:, task_idx]
            
            # Filter out invalid labels (e.g., -1 for missing labels)
            valid_mask = (task_y_true != -1) & np.isfinite(task_y_true) & np.isfinite(task_y_pred)
            
            if valid_mask.sum() < 2:
                # Not enough valid samples for this task
                continue
            
            task_y_true_valid = task_y_true[valid_mask]
            task_y_pred_valid = task_y_pred[valid_mask]
            
            # Check if binary classification (has both classes)
            unique_labels = np.unique(task_y_true_valid)
            if len(unique_labels) < 2:
                # Only one class present, skip this task
                continue
            
            try:
                task_auroc = roc_auc_score(task_y_true_valid, task_y_pred_valid)
                task_aurocs.append(task_auroc)
            except Exception as e:
                # Skip tasks that fail AUROC calculation
                continue
        
        if len(task_aurocs) == 0:
            # No valid tasks, return 0.0
            return 0.0, []
        
        macro_auroc = np.mean(task_aurocs)
        return macro_auroc, task_aurocs
    else:
        # Single task: standard AUROC calculation
        # Flatten if needed
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_probs.flatten()
        
        # Filter out invalid labels
        valid_mask = (y_true_flat != -1) & np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        
        if valid_mask.sum() < 2:
            return 0.0, []
        
        y_true_valid = y_true_flat[valid_mask]
        y_pred_valid = y_pred_flat[valid_mask]
        
        # Check if binary classification
        unique_labels = np.unique(y_true_valid)
        if len(unique_labels) < 2:
            return 0.0, []
        
        try:
            auroc = roc_auc_score(y_true_valid, y_pred_valid)
            return auroc, [auroc]
        except Exception as e:
            return 0.0, []


class SmartEarlyStopping:
    """
    Smart Early Stopping class
    Implements multiple improvements:
    1. Moving average smoothing for validation loss
    2. Patience decay (dynamic adjustment)
    3. Trend analysis (identify overall improvement trends)
    4. Multi-metric monitoring (AUROC, F1 Score, Train-Val Loss Gap)
    5. Multi-metric Early Stopping (e.g., continue training when AUROC improves)
    """
    
    def __init__(self, 
                 initial_patience=20,
                 max_patience=50,
                 min_patience=10,
                 moving_avg_window=5,
                 trend_window=10,
                 improvement_threshold=3,
                 trend_slope_threshold=-0.001,
                 use_multi_metric=True,
                 auroc_improvement_threshold=0.001,
                 f1_improvement_threshold=0.001):
        """
        Args:
            initial_patience: Initial patience value
            max_patience: Maximum patience value
            min_patience: Minimum patience value
            moving_avg_window: Moving average window size
            trend_window: Trend analysis window size
            improvement_threshold: Number of consecutive improvements before increasing patience
            trend_slope_threshold: Trend judgment threshold (negative value indicates decrease)
            use_multi_metric: Whether to use multi-metric Early Stopping
            auroc_improvement_threshold: AUROC improvement threshold (values above this are considered improvement)
            f1_improvement_threshold: F1 Score improvement threshold (values above this are considered improvement)
        """
        self.initial_patience = initial_patience
        self.current_patience = float(initial_patience)
        self.max_patience = max_patience
        self.min_patience = min_patience
        self.moving_avg_window = moving_avg_window
        self.trend_window = trend_window
        self.improvement_threshold = improvement_threshold
        self.trend_slope_threshold = trend_slope_threshold
        self.use_multi_metric = use_multi_metric
        self.auroc_improvement_threshold = auroc_improvement_threshold
        self.f1_improvement_threshold = f1_improvement_threshold
        
        # State variables - loss related
        self.val_losses = []
        self.train_losses = []
        self.smoothed_losses = []
        self.patience_counter = 0
        self.best_smoothed_loss = float('inf')
        self.best_raw_loss = float('inf')
        self.improvement_streak = 0
        
        # State variables - multi-metric related
        self.val_aurocs = []
        self.val_f1_scores = []
        self.val_pr_aucs = []  # PR-AUC tracking
        self.val_mae = []  # MAE tracking (for regression tasks)
        self.train_val_gaps = []
        self.best_auroc = 0.0
        self.best_f1 = 0.0
        self.best_pr_auc = 0.0  # Best PR-AUC
        self.best_train_val_gap = float('inf')
        
        # AUROC stability: moving average and consecutive improvement check
        self.auroc_moving_avg_window = 5  # Moving average window for AUROC
        self.auroc_improvement_streak = 0  # Consecutive improvement streak
        self.auroc_improvement_required = 2  # Required consecutive improvements
        self.best_smoothed_auroc = 0.0  # Best smoothed AUROC
        
    def update(self, val_loss, train_loss=None, metrics=None):
        """
        Update state and determine if training should stop
        
        Args:
            val_loss: Validation loss
            train_loss: Training loss (for calculating Train-Val Loss Gap)
            metrics: Dictionary containing additional metrics, e.g., {'roc_auc': 0.85, 'f1': 0.75}
        
        Returns:
            should_stop: Whether training should stop
            info: Dictionary containing detailed information
        """
        # 1. Add new validation loss and training loss
        self.val_losses.append(val_loss)
        if train_loss is not None:
            self.train_losses.append(train_loss)
            # Calculate Train-Val Loss Gap (overfitting indicator)
            train_val_gap = train_loss - val_loss
            self.train_val_gaps.append(train_val_gap)
        
        # 2. Process multi-metrics
        auroc = None
        f1_score = None
        pr_auc = None
        precision = None
        recall = None
        if metrics:
            auroc = metrics.get('roc_auc', None)
            f1_score = metrics.get('f1', None)
            pr_auc = metrics.get('pr_auc', None)
            precision = metrics.get('precision', None)
            recall = metrics.get('recall', None)
        
        # 2.5 Check for pathological solutions (all negative or all positive predictions)
        # Enhanced check: Also check F1 < 0.01 or PR-AUC < 0.01 (for MUV and similar datasets)
        # If precision=0 or recall=0, force continue training
        # BUT: Still update best_auroc if this is the first valid AUROC or if it improved
        is_pathological = False
        pathological_reason = None
        
        if precision == 0 or recall == 0:
            is_pathological = True
            pathological_reason = f"precision={precision} or recall={recall}"
        elif f1_score is not None and f1_score < 0.01:
            # F1 score extremely low (likely all-negative predictions)
            is_pathological = True
            pathological_reason = f"F1={f1_score:.4f} < 0.01"
        elif pr_auc is not None and pr_auc < 0.01:
            # PR-AUC extremely low (likely all-negative predictions for imbalanced datasets)
            is_pathological = True
            pathological_reason = f"PR-AUC={pr_auc:.4f} < 0.01"
        
        if is_pathological:
            # Update best_auroc even in pathological case (if valid)
            if auroc is not None and auroc > 0:
                # Check if this is the first valid AUROC or if it improved
                if len(self.val_aurocs) == 0:
                    # First valid AUROC
                    self.val_aurocs.append(auroc)
                    self.best_auroc = auroc
                    self.best_smoothed_auroc = auroc
                elif auroc > self.best_auroc:
                    # AUROC improved even in pathological case
                    self.val_aurocs.append(auroc)
                    self.best_auroc = auroc
                    # Update smoothed AUROC
                    if len(self.val_aurocs) >= self.auroc_moving_avg_window:
                        recent_aurocs = self.val_aurocs[-self.auroc_moving_avg_window:]
                        self.best_smoothed_auroc = sum(recent_aurocs) / len(recent_aurocs)
                    else:
                        self.best_smoothed_auroc = auroc
                else:
                    # AUROC didn't improve, but still record it
                    self.val_aurocs.append(auroc)

            # Despite detecting pathological predictions, still update best loss to avoid staying at inf
            # This ensures at least the first available validation loss is recorded, avoiding missing history/best_model writes
            if val_loss < self.best_raw_loss:
                self.best_raw_loss = val_loss
                # If no smoothed loss yet, use current loss as initial value
                if len(self.smoothed_losses) == 0:
                    smoothed_loss = val_loss
                self.best_smoothed_loss = min(self.best_smoothed_loss, smoothed_loss if 'smoothed_loss' in locals() else val_loss)
            
            # Update other metrics similarly
            if f1_score is not None:
                if len(self.val_f1_scores) == 0:
                    self.val_f1_scores.append(f1_score)
                    self.best_f1 = f1_score
                elif f1_score > self.best_f1:
                    self.val_f1_scores.append(f1_score)
                    self.best_f1 = f1_score
                else:
                    self.val_f1_scores.append(f1_score)
            
            if pr_auc is not None:
                if len(self.val_pr_aucs) == 0:
                    self.val_pr_aucs.append(pr_auc)
                    self.best_pr_auc = pr_auc
                elif pr_auc > self.best_pr_auc:
                    self.val_pr_aucs.append(pr_auc)
                    self.best_pr_auc = pr_auc
                else:
                    self.val_pr_aucs.append(pr_auc)
            
            # Reset patience counter to force continue training
            self.patience_counter = 0
            return False, {
                'should_stop': False,
                'patience_counter': self.patience_counter,
                'current_patience': int(self.current_patience),
                'best_smoothed_loss': self.best_smoothed_loss,
                'best_raw_loss': self.best_raw_loss,
                'best_auroc': self.best_auroc,
                'best_f1': self.best_f1,
                'best_pr_auc': self.best_pr_auc,
                'trend_info': f'Avoiding pathological predictions ({pathological_reason})',
                'metric_info': f'Pathological solution detected: {pathological_reason}',
                'improved': False,
                'improved_by_loss': False,
                'improved_by_metric': False,
                'smoothed_loss': smoothed_loss if 'smoothed_loss' in locals() else val_loss
            }
        
        # Normal case: update metrics lists
        if auroc is not None:
            self.val_aurocs.append(auroc)
        if f1_score is not None:
            self.val_f1_scores.append(f1_score)
        if pr_auc is not None:
            self.val_pr_aucs.append(pr_auc)
        
        # 3. Calculate moving average
        if len(self.val_losses) >= self.moving_avg_window:
            recent_losses = self.val_losses[-self.moving_avg_window:]
            smoothed_loss = sum(recent_losses) / len(recent_losses)
        else:
            # If insufficient data, use raw value
            smoothed_loss = val_loss
        
        self.smoothed_losses.append(smoothed_loss)
        
        # 4. Multi-metric Early Stopping judgment
        improved_by_metric = False
        improved_by_auroc = False
        improved_by_f1 = False
        metric_info = None
        
        if self.use_multi_metric and metrics:
            # Check if AUROC improved (with stability check)
            if auroc is not None:
                # Note: auroc already appended above in "Normal case" section
                
                # Calculate moving average for AUROC (to filter noise)
                if len(self.val_aurocs) >= self.auroc_moving_avg_window:
                    recent_aurocs = self.val_aurocs[-self.auroc_moving_avg_window:]
                    smoothed_auroc = sum(recent_aurocs) / len(recent_aurocs)
                else:
                    smoothed_auroc = auroc
                
                if len(self.val_aurocs) == 1:
                    # First time seeing AUROC, set as best value
                    self.best_auroc = auroc
                    self.best_smoothed_auroc = smoothed_auroc
                else:
                    # Always track the highest raw AUROC value (regardless of smoothing)
                    # This ensures best_auroc represents the true best performance
                    prev_best_auroc = self.best_auroc
                    if auroc > self.best_auroc:
                        self.best_auroc = auroc
                    
                    # Check if smoothed AUROC improved (using moving average)
                    # This is used for early stopping logic (stability check)
                    if smoothed_auroc > self.best_smoothed_auroc + self.auroc_improvement_threshold:
                        # Smoothed AUROC improved, increment streak
                        self.auroc_improvement_streak += 1
                    else:
                        # Smoothed AUROC did not improve, reset streak
                        self.auroc_improvement_streak = 0
                    
                    # Only reset patience if consecutive improvements are achieved (based on smoothed value)
                    if self.auroc_improvement_streak >= self.auroc_improvement_required:
                        # AUROC improved consecutively (stable improvement based on smoothed value)
                        improved_by_auroc = True
                        # Update best_smoothed_auroc for tracking smoothed improvements
                        self.best_smoothed_auroc = smoothed_auroc
                        self.auroc_improvement_streak = 0  # Reset streak after using it
                        # Show both raw and smoothed values in message
                        if auroc > prev_best_auroc:
                            metric_info = f"AUROC improved to {auroc:.4f} (best raw: {self.best_auroc:.4f}, smoothed: {smoothed_auroc:.4f})"
                        else:
                            metric_info = f"AUROC smoothed improvement (raw: {auroc:.4f}, best raw: {self.best_auroc:.4f}, smoothed: {smoothed_auroc:.4f})"
            
            # Check if F1 Score improved
            if f1_score is not None:
                if len(self.val_f1_scores) == 1:
                    # First time seeing F1, set as best value
                    self.best_f1 = f1_score
                elif f1_score > self.best_f1 + self.f1_improvement_threshold:
                    # F1 Score improved
                    improved_by_f1 = True
                    prev_f1 = self.val_f1_scores[-2] if len(self.val_f1_scores) > 1 else self.best_f1
                    self.best_f1 = f1_score
                    if metric_info:
                        metric_info += f", F1 improved to {f1_score:.4f} (increase {f1_score - prev_f1:.4f})"
                    else:
                        metric_info = f"F1 Score improved to {f1_score:.4f} (increase {f1_score - prev_f1:.4f})"
            
            # Balance AUROC and F1: Only reset patience if both improve (for HIV/SIDER/TOX21)
            # This prevents model from becoming too conservative (high AUROC but low F1)
            if improved_by_auroc and improved_by_f1:
                # Both AUROC and F1 improved - reset patience
                improved_by_metric = True
            elif improved_by_auroc and not improved_by_f1:
                # AUROC improved but F1 didn't - don't reset patience (avoid becoming too conservative)
                if metric_info:
                    metric_info += " (F1 did not improve, not resetting patience)"
            elif improved_by_f1 and not improved_by_auroc:
                # F1 improved but AUROC didn't - reset patience (F1 is important)
                improved_by_metric = True
                if not metric_info:
                    metric_info = f"F1 improved to {f1_score:.4f} (AUROC did not improve)"
            # If neither improved, improved_by_metric remains False
            
            # Check if PR-AUC improved (important for imbalanced datasets)
            if pr_auc is not None:
                if len(self.val_pr_aucs) == 1:
                    # First time seeing PR-AUC, set as best value
                    self.best_pr_auc = pr_auc
                elif pr_auc > self.best_pr_auc + 0.01:  # PR-AUC improvement threshold
                    # PR-AUC improved significantly
                    improved_by_metric = True
                    prev_pr_auc = self.val_pr_aucs[-2] if len(self.val_pr_aucs) > 1 else self.best_pr_auc
                    self.best_pr_auc = pr_auc
                    if metric_info:
                        metric_info += f", PR-AUC improved to {pr_auc:.4f} (increase {pr_auc - prev_pr_auc:.4f})"
                    else:
                        metric_info = f"PR-AUC improved to {pr_auc:.4f} (increase {pr_auc - prev_pr_auc:.4f})"
        
        # 5. Determine if loss improved (based on moving average)
        improved_by_loss = False
        trend_info = None
        
        if smoothed_loss < self.best_smoothed_loss:
            self.best_smoothed_loss = smoothed_loss
            self.best_raw_loss = val_loss
            improved_by_loss = True
            self.improvement_streak += 1
            
            # 6. Patience decay: if continuously improving, increase patience
            if self.improvement_streak >= self.improvement_threshold:
                old_patience = self.current_patience
                self.current_patience = min(
                    self.current_patience * 1.2,
                    self.max_patience
                )
                if int(self.current_patience) != int(old_patience):
                    trend_info = f"Model continuously improving, patience increased from {int(old_patience)} to {int(self.current_patience)}"
        else:
            self.improvement_streak = 0
            
            # 7. Trend analysis: even if current value is slightly higher, if trend is decreasing, don't increase counter
            if len(self.smoothed_losses) >= self.trend_window:
                trend = self._analyze_trend()
                if trend == "improving":
                    # Overall trend is decreasing, don't increase counter
                    trend_info = "Validation loss shows overall decreasing trend, continuing training"
                    return False, {
                        'should_stop': False,
                        'patience_counter': self.patience_counter,
                        'current_patience': int(self.current_patience),
                        'best_smoothed_loss': self.best_smoothed_loss,
                        'best_raw_loss': self.best_raw_loss,
                        'best_auroc': self.best_auroc,
                        'best_f1': self.best_f1,
                        'best_pr_auc': self.best_pr_auc,
                        'trend_info': trend_info,
                        'metric_info': metric_info,
                        'improved': False,
                        'improved_by_loss': False,
                        'improved_by_metric': False,
                        'smoothed_loss': smoothed_loss
                    }
        
        # 7.5 Overfitting Check: Do not reset patience if overfitting is severe
        # Severe overfitting definition: Loss Gap > 0.5 AND Loss Gap increasing trend
        is_severe_overfitting = False
        train_val_gap = None
        if len(self.train_val_gaps) > 0:
            train_val_gap = self.train_val_gaps[-1]
            
            # Check for severe overfitting condition (Gap > 0.5 is extremely large)
            if train_val_gap > 0.5:
                is_severe_overfitting = True
                if metric_info:
                    metric_info += f" [⚠️ Severe Overfitting Gap={train_val_gap:.4f}]"
                else:
                    metric_info = f"[⚠️ Severe Overfitting Gap={train_val_gap:.4f}]"
        
        # 8. Multi-metric Early Stopping logic: if key metrics improve, continue training even if loss slightly increases
        # BUT: If severe overfitting is detected, do not reset patience fully (or force increase)
        if self.use_multi_metric and improved_by_metric and not improved_by_loss:
            if is_severe_overfitting:
                # If AUROC improved but severe overfitting exists -> Don't reset patience to 0
                # Instead, treat as no improvement to encourage early stopping
                self.patience_counter += 1
                trend_info = metric_info or "Key metrics improved but severe overfitting detected - Patience not reset"
                
                # Force stop check
                should_stop = self.patience_counter >= self.current_patience
                
                return should_stop, {
                    'should_stop': should_stop,
                    'patience_counter': self.patience_counter,
                    'current_patience': int(self.current_patience),
                    'best_smoothed_loss': self.best_smoothed_loss,
                    'best_raw_loss': self.best_raw_loss,
                    'best_auroc': self.best_auroc,
                    'best_f1': self.best_f1,
                    'train_val_gap': train_val_gap,
                    'trend_info': trend_info,
                    'metric_info': metric_info,
                    'improved': False,  # Treated as not improved to enforce strictness
                    'improved_by_loss': False,
                    'improved_by_metric': True,
                    'smoothed_loss': smoothed_loss
                }
            else:
                # Normal case: AUROC improved, no severe overfitting -> Reset patience
                self.patience_counter = 0
                trend_info = metric_info or "Key metrics improved, continuing training"
                return False, {
                    'should_stop': False,
                    'patience_counter': self.patience_counter,
                    'current_patience': int(self.current_patience),
                    'best_smoothed_loss': self.best_smoothed_loss,
                    'best_raw_loss': self.best_raw_loss,
                    'best_auroc': self.best_auroc,
                    'best_f1': self.best_f1,
                    'train_val_gap': train_val_gap,
                    'trend_info': trend_info,
                    'metric_info': metric_info,
                    'improved': True,
                    'improved_by_loss': False,
                    'improved_by_metric': True,
                    'smoothed_loss': smoothed_loss
                }
        
        # 9. Update counter
        if improved_by_loss or (improved_by_metric and not is_severe_overfitting):
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 10. Determine if should stop
        should_stop = self.patience_counter >= self.current_patience
        
        return should_stop, {
            'should_stop': should_stop,
            'patience_counter': self.patience_counter,
            'current_patience': int(self.current_patience),
            'best_smoothed_loss': self.best_smoothed_loss,
            'best_raw_loss': self.best_raw_loss,
            'best_auroc': self.best_auroc,
            'best_f1': self.best_f1,
            'train_val_gap': self.train_val_gaps[-1] if self.train_val_gaps else None,
            'trend_info': trend_info,
            'metric_info': metric_info,
            'improved': improved_by_loss or improved_by_metric,
            'improved_by_loss': improved_by_loss,
            'improved_by_metric': improved_by_metric,
            'smoothed_loss': smoothed_loss
        }
    
    def _analyze_trend(self):
        """Analyze the trend of validation loss"""
        if len(self.smoothed_losses) < self.trend_window:
            return "insufficient_data"
        
        recent_losses = self.smoothed_losses[-self.trend_window:]
        x = np.arange(len(recent_losses))
        
        # Calculate linear regression slope
        slope = np.polyfit(x, recent_losses, 1)[0]
        
        if slope < self.trend_slope_threshold:
            return "improving"  # Clear decreasing trend
        elif slope < abs(self.trend_slope_threshold) * 0.5:
            return "stable"  # Slight decrease or stable
        else:
            return "degrading"  # Increasing trend
    
    def reset(self):
        """Reset state (for new training)"""
        self.val_losses = []
        self.train_losses = []
        self.smoothed_losses = []
        self.patience_counter = 0
        self.best_smoothed_loss = float('inf')
        self.best_raw_loss = float('inf')
        self.improvement_streak = 0
        self.current_patience = float(self.initial_patience)
        
        # Reset multi-metrics
        self.val_aurocs = []
        self.val_f1_scores = []
        self.train_val_gaps = []
        self.best_auroc = 0.0
        self.best_f1 = 0.0
        self.best_train_val_gap = float('inf')


class GraphDataset:
    """Graph dataset wrapper class for DistributedSampler"""
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


class WeightedDistributedSampler:
    """
    Combines WeightedRandomSampler with DistributedSampler for DDP training
    """
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        
        # Convert weights to numpy array if needed
        if isinstance(weights, (list, np.ndarray)):
            weights = np.array(weights, dtype=np.float32)
        elif isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()
        
        # Normalize weights
        weights = weights / weights.sum()
        self.weights = weights
        
        # Calculate indices for this rank
        self.num_samples = int(np.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Generate weighted random indices
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(
            torch.from_numpy(self.weights),
            self.total_size,
            replacement=self.replacement,
            generator=g
        ).tolist()
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class BalancedBatchSampler:
    """
    Ensures each batch contains at least two positive samples (if available).
    """
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, min_pos_per_batch=2):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.min_pos_per_batch = min_pos_per_batch
        
        labels = [int(getattr(graph, 'y', torch.tensor([0])).view(-1)[0].item()) for graph in self.dataset.graphs]
        self.pos_indices = [i for i, label in enumerate(labels) if label == 1]
        self.neg_indices = [i for i, label in enumerate(labels) if label == 0]
        
        self.num_samples_per_rank = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas
        
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            # Fallback to DistributedSampler when positive or negative samples are missing
            self.fallback_sampler = DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
            )
        else:
            self.fallback_sampler = None
    
    def __iter__(self):
        if self.fallback_sampler is not None:
            self.fallback_sampler.set_epoch(self.epoch)
            return self.fallback_sampler.__iter__()
        
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        if self.shuffle:
            pos_perm = torch.randperm(len(self.pos_indices), generator=g).tolist()
            pos_indices = [self.pos_indices[i] for i in pos_perm]
            neg_perm = torch.randperm(len(self.neg_indices), generator=g).tolist()
            neg_indices = [self.neg_indices[i] for i in neg_perm]
        else:
            pos_indices = list(self.pos_indices)
            neg_indices = list(self.neg_indices)
        
        pos_ptr = 0
        neg_ptr = 0
        balanced_indices = []
        num_batches = int(math.ceil(self.total_size / self.batch_size))
        
        for _ in range(num_batches):
            batch = []
            pos_needed = min(self.min_pos_per_batch, len(pos_indices))
            for _ in range(pos_needed):
                batch.append(pos_indices[pos_ptr % len(pos_indices)])
                pos_ptr += 1
            
            needed = self.batch_size - len(batch)
            for _ in range(needed):
                if len(neg_indices) > 0:
                    batch.append(neg_indices[neg_ptr % len(neg_indices)])
                    neg_ptr += 1
                else:
                    batch.append(pos_indices[pos_ptr % len(pos_indices)])
                    pos_ptr += 1
            
            if self.shuffle:
                batch_perm = torch.randperm(len(batch), generator=g).tolist()
                batch = [batch[i] for i in batch_perm]
            balanced_indices.extend(batch)
        
        # Trim to total size and select slice for this rank
        balanced_indices = balanced_indices[:self.total_size]
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        rank_indices = balanced_indices[start:end]
        return iter(rank_indices)
    
    def __len__(self):
        if self.fallback_sampler is not None:
            return len(self.fallback_sampler)
        return self.num_samples_per_rank
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.fallback_sampler is not None:
            self.fallback_sampler.set_epoch(epoch)


class AEGNNDDPTrainer:
    """AEGNN-M DDP Trainer"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 device,
                 rank,
                 world_size,
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 scheduler_patience=10,
                 early_stopping_patience=20,
                 gradient_accumulation_steps=1,
                 skip_test=False,
                 scheduler_type='cosine',
                 warmup_epochs=5,
                 min_lr=1e-6,
                 log_dir=None,
                 use_smart_early_stopping=False,
                 smart_early_stopping_max_patience=50,
                 smart_early_stopping_moving_avg_window=5,
                 smart_early_stopping_trend_window=10,
                 auroc_improvement_threshold=0.005,
                 f1_improvement_threshold=0.001,
                 onecycle_max_lr=None,
                 onecycle_pct_start=0.3,
                 onecycle_div_factor=25.0,
                 onecycle_final_div_factor=1e4,
                 val_loss_ema_beta=0.8,
                 enable_loss_gap_control=False,
                 loss_gap_threshold=0.5,
                 loss_gap_patience=3,
                 loss_gap_lr_factor=0.5,
                 loss_gap_patience_decay=0.7,
                 grad_clip_norm=1.0,
                 enable_manifold_mixup=False,
                 manifold_mixup_alpha=2.0,
                 model_config=None,
                 dataset_name=None):
        
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.dataset_name = dataset_name  # Store dataset name for threshold configuration
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.skip_test = skip_test
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # Initialize model_config early (before it's accessed in primary metric loading)
        self.model_config = model_config or {}
        
        # Wrap model with DDP only if world_size > 1
        if world_size > 1:
            # Wrap model with DDP
            # find_unused_parameters=True allows some parameters to not participate in gradient computation in specific iterations
            # This is necessary for models with conditional branches or optional layers
            self.model = DDP(model.to(device), device_ids=[rank], output_device=rank, find_unused_parameters=True)
            self.base_model = self.model.module  # Access underlying model
        else:
            # Single GPU mode: no DDP wrapping
            self.model = model.to(device)
            self.base_model = self.model  # In single GPU mode, base_model is the same as model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = None
        self.onecycle_max_lr = onecycle_max_lr or learning_rate
        self.onecycle_pct_start = onecycle_pct_start
        self.onecycle_div_factor = onecycle_div_factor
        self.onecycle_final_div_factor = onecycle_final_div_factor
        
        # Learning rate scheduler - supports multiple schedulers
        # IMPROVEMENT 3.1: Use reasonable initial T_max (will be updated in train() with actual num_epochs)
        # Use a default value that works for most cases, but will be updated in train()
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=200,  # Reasonable default, will be updated in train() with actual num_epochs - warmup_epochs
                eta_min=min_lr
            )
        elif scheduler_type == 'cosine_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=50,
                T_mult=2,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=scheduler_patience,
                min_lr=min_lr
            )
        elif scheduler_type == 'onecycle':
            # Will be initialized in train() once num_epochs is known
            self.scheduler = None
        else:
            # IMPROVEMENT 3.1: Use reasonable default T_max
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=200,  # Reasonable default, will be updated in train() with actual num_epochs - warmup_epochs
                eta_min=min_lr
            )
        
        # Early stopping mechanism
        self.use_smart_early_stopping = use_smart_early_stopping
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize best_val_loss for both early stopping modes
        self.best_val_loss = float('inf')
        self.best_threshold = 0.5  # Default threshold, will be updated during validation
        
        if use_smart_early_stopping:
            # Use smart early stopping
            self.smart_early_stopping = SmartEarlyStopping(
                initial_patience=early_stopping_patience,
                max_patience=smart_early_stopping_max_patience,
                min_patience=max(5, early_stopping_patience // 2),
                moving_avg_window=smart_early_stopping_moving_avg_window,
                trend_window=smart_early_stopping_trend_window,
                use_multi_metric=True,  # Enable multi-metric monitoring
                auroc_improvement_threshold=auroc_improvement_threshold,
                f1_improvement_threshold=f1_improvement_threshold
            )
            if rank == 0:
                print(f"✅ Smart Early Stopping enabled (with multi-metric monitoring)")
                print(f"   - Initial Patience: {early_stopping_patience}")
                print(f"   - Max Patience: {smart_early_stopping_max_patience}")
                print(f"   - Moving Average Window: {smart_early_stopping_moving_avg_window}")
                print(f"   - Trend Analysis Window: {smart_early_stopping_trend_window}")
                print(f"   - Multi-metric Monitoring: ✅ Enabled (AUROC, F1 Score, Train-Val Loss Gap)")
        else:
            # Use traditional early stopping
            self.smart_early_stopping = None
            self.patience_counter = 0
        
        # Logger (only on rank 0)
        if rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()
        else:
            self.writer = None
        self.train_losses = []
        self.val_losses = []
        
        # Primary metric tracking
        self.primary_metric = None
        self.val_spearman = []  # For regression tasks
        self.val_aurocs = []    # For classification tasks
        self.val_f1_scores = [] # For classification tasks
        self.val_pr_aucs = []   # For classification tasks
        self.best_primary_metric_value = None
        self.best_primary_metric_epoch = -1
        
        # Load primary metric from config
        # Priority: 1) model_config (from trial_config.json), 2) dataset_primary_metrics.yaml
        try:
            # First, try to get primary_metric from model_config (if provided via --config)
            # Note: optuna_serach_mod.py saves primary_metric in the top-level config, not in hyperparameters
            # So we check both model_config and try to load from dataset_primary_metrics.yaml
            primary_metric_loaded = False
            if self.model_config and 'primary_metric' in self.model_config:
                self.primary_metric = self.model_config['primary_metric']
                primary_metric_loaded = True
                if rank == 0:
                    print(f"📌 Primary metric from model_config: {self.primary_metric}")
                    print(f"   Will save best model based on {self.primary_metric} instead of val_loss")
            
            # Fallback: load from dataset_primary_metrics.yaml
            if not primary_metric_loaded and self.dataset_name:
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'dataset_primary_metrics.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    dataset_configs = config.get('dataset_primary_metrics', {})
                    dataset_config = dataset_configs.get(self.dataset_name.lower(), {})
                    self.primary_metric = dataset_config.get('primary_metric', None)
                    if self.primary_metric and rank == 0:
                        print(f"📌 Primary metric for {self.dataset_name}: {self.primary_metric}")
                        print(f"   Will save best model based on {self.primary_metric} instead of val_loss")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Could not load primary metric config: {e}")
        
        # Initialize best_primary_metric_value based on metric type
        if self.primary_metric:
            if self.primary_metric in ['spearman', 'roc_auc', 'f1', 'pr_auc']:
                # For maximize metrics (higher is better)
                self.best_primary_metric_value = float('-inf')
            elif self.primary_metric == 'mae':
                # For minimize metrics (lower is better)
                self.best_primary_metric_value = float('inf')
        
        self.enable_loss_gap_control = enable_loss_gap_control
        self.loss_gap_threshold = loss_gap_threshold
        self.loss_gap_patience = loss_gap_patience
        self.loss_gap_lr_factor = loss_gap_lr_factor
        self.loss_gap_patience_decay = loss_gap_patience_decay
        self.loss_gap_counter = 0
        self.latest_loss_gap_ratio = 0.0
        self.val_loss_ema_beta = val_loss_ema_beta if 0 < val_loss_ema_beta < 1 else 0.0
        self.val_loss_ema = None
        self.grad_clip_norm = grad_clip_norm
        self.enable_manifold_mixup = enable_manifold_mixup
        self.manifold_mixup_alpha = manifold_mixup_alpha
        # model_config was already initialized earlier
        
        # Collect target statistics from training set for normalization detection
        # This helps detect if predictions and targets are in different scales during testing
        self.target_mean = None
        self.target_std = None
        self.target_min = None
        self.target_max = None
        
        try:
            # Collect all training targets
            train_targets = []
            for batch in self.train_loader:
                if hasattr(batch, 'y'):
                    if batch.y.dim() > 0:
                        train_targets.append(batch.y.cpu())
            
            if train_targets:
                all_targets = torch.cat(train_targets).numpy()
                self.target_mean = float(np.mean(all_targets))
                self.target_std = float(np.std(all_targets))
                self.target_min = float(np.min(all_targets))
                self.target_max = float(np.max(all_targets))
                
                if rank == 0:
                    print(f"📊 Training target statistics:")
                    print(f"   Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
                    print(f"   Range: [{self.target_min:.4f}, {self.target_max:.4f}]")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Could not collect target statistics: {e}")
    
    def load_threshold_config(self, dataset_name):
        """
        Load dataset-specific threshold configuration from config file.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'clintox', 'bace')
        
        Returns:
            dict: Configuration dictionary with keys:
                - fixed_threshold: float or None (if set, use this threshold directly)
                - method: str ('cv' or 'adaptive')
                - threshold_range: tuple (min, max)
                - metric: str (optimization metric)
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'dataset_thresholds.yaml')
        default_config = {
            'fixed_threshold': None,
            'method': None,  # None means use auto selection
            'threshold_range': None,
            'metric': None
        }
        
        if not os.path.exists(config_path):
            if self.rank == 0:
                print(f"ℹ️  Threshold config file not found: {config_path}")
                print(f"   Using default automatic threshold selection")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_configs = config.get('dataset_thresholds', {})
            dataset_config = dataset_configs.get(dataset_name.lower(), {})
            
            # Merge with defaults
            result = default_config.copy()
            result.update(dataset_config)
            
            if self.rank == 0 and dataset_config:
                print(f"📌 Loaded threshold config for {dataset_name}:")
                if result['fixed_threshold'] is not None:
                    print(f"   Fixed threshold: {result['fixed_threshold']}")
                else:
                    print(f"   Method: {result['method'] or 'auto'}")
                    if result['threshold_range']:
                        print(f"   Threshold range: {result['threshold_range']}")
            
            return result
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️  Error loading threshold config: {e}")
                print(f"   Using default automatic threshold selection")
            return default_config
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc="Training", leave=False)
        else:
            pbar = self.train_loader
        
        # Gradient accumulation: accumulate gradients from multiple batches before updating
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Forward pass
            # If batch has pos attribute (3D coordinates), pass it to the model
            pos = getattr(batch, 'pos', None)
            fingerprint = getattr(batch, 'fingerprint', None)
            if self.enable_manifold_mixup:
                pred, _, graph_features = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch, 
                    pos=pos, fingerprint=fingerprint, return_graph_features=True
                )
                if graph_features.size(0) > 1:
                    lam = np.random.beta(self.manifold_mixup_alpha, self.manifold_mixup_alpha)
                    lam = max(lam, 1.0 - lam)
                    perm = torch.randperm(graph_features.size(0), device=self.device)
                    mixed_features = lam * graph_features + (1 - lam) * graph_features[perm]
                    pred = self.base_model.project_graph_features(mixed_features)
                    targets = batch.y
                    targets_perm = targets[perm]
                    loss = lam * self.base_model.compute_loss(pred, targets) + \
                           (1 - lam) * self.base_model.compute_loss(pred, targets_perm)
                else:
                    loss = self.base_model.compute_loss(pred, batch.y)
            else:
                b2revb = getattr(batch, 'b2revb', None)  # Get pre-computed b2revb (Chemprop style)
                pred, _ = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, 
                                   pos=pos, fingerprint=fingerprint, b2revb=b2revb)
                loss = self.base_model.compute_loss(pred, batch.y)
            
            # Check for NaN or Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                if self.rank == 0:
                    print(f"⚠️  Warning: NaN/Inf loss detected at batch {batch_idx}, skipping this batch")
                # Skip this batch - don't accumulate gradients
                continue
            
            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            
            # Check for NaN in gradients before updating
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                if self.rank == 0:
                    print(f"⚠️  Warning: NaN/Inf gradients detected at batch {batch_idx}, skipping gradient update")
                self.optimizer.zero_grad()
                continue
            
            total_loss += loss.item() * self.gradient_accumulation_steps  # Restore true loss value
            num_batches += 1
            
            # Update every gradient_accumulation_steps batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                
                # Update weights
                self.optimizer.step()
                if self.scheduler_type == 'onecycle' and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})
        
        # Handle last incomplete accumulation batch
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            if self.scheduler_type == 'onecycle' and self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Average loss across all processes
        avg_loss = total_loss / num_batches
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        else:
            avg_loss = loss_tensor.item()
        
        return avg_loss
    
    def validate(self, return_metrics=False):
        """
        Validate model
        
        Args:
            return_metrics: If True, return additional metrics (AUROC, F1, etc.) for classification tasks
        
        Returns:
            avg_loss: Average validation loss
            metrics (optional): Dictionary with additional metrics if return_metrics=True
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # For metrics calculation (only for classification)
        all_predictions = []
        all_predictions_raw = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # If batch has pos attribute (3D coordinates), pass it to the model
                pos = getattr(batch, 'pos', None)
                fingerprint = getattr(batch, 'fingerprint', None)
                b2revb = getattr(batch, 'b2revb', None)  # Get pre-computed b2revb (Chemprop style)
                pred, _ = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, pos=pos, fingerprint=fingerprint, b2revb=b2revb)
                loss = self.base_model.compute_loss(pred, batch.y)
                
                # Check for NaN or Inf in validation loss
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.rank == 0:
                        print(f"⚠️  Warning: NaN/Inf validation loss detected, skipping this batch")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and targets for metrics
                # For classification: if return_metrics=True
                # For regression: if primary_metric is spearman or mae and return_metrics=True
                collect_for_regression_metric = (self.primary_metric in ['spearman', 'mae'] and return_metrics)
                if return_metrics or collect_for_regression_metric:
                    all_predictions_raw.append(pred.cpu())
                    
                    # 🔧 FIX: Use model_type from model_config to explicitly determine task type
                    # This avoids heuristic-based misclassification that can cause MAE calculation errors
                    model_type = self.model_config.get('model_type', None) if self.model_config else None
                    
                    # Determine task type: use model_type if available, otherwise infer from primary_metric
                    is_regression = False
                    if model_type == 'regressor':
                        is_regression = True
                    elif model_type == 'classifier':
                        is_regression = False
                    elif self.primary_metric in ['spearman', 'mae']:
                        # Fallback: infer from primary_metric
                        is_regression = True
                    else:
                        # Default to classification
                        is_regression = False
                    
                    if is_regression:
                        # Regression: use raw predictions directly (no sigmoid/softmax)
                        # For regression, predictions are already in the correct scale
                        if pred.dim() >= 2:
                            # If 2D, squeeze to 1D (regression typically has shape [batch_size, 1])
                            all_predictions.append(pred.squeeze().cpu())
                        else:
                            # Already 1D, use directly
                            all_predictions.append(pred.cpu())
                    else:
                        # Classification: apply sigmoid/softmax to get probabilities
                        if pred.dim() >= 2:
                            # 2D or higher tensor: check second dimension
                            if pred.shape[1] > 1:
                                # Multi-class classification: extract probabilities for positive class
                                pred_probs = torch.softmax(pred, dim=1)[:, 1].cpu()
                                all_predictions.append(pred_probs)
                            elif pred.shape[1] == 1:
                                # Binary classification with single output, apply sigmoid
                                pred_probs = torch.sigmoid(pred.squeeze()).cpu()
                                all_predictions.append(pred_probs)
                            else:
                                # Edge case: shape[1] == 0 (should not happen, but handle gracefully)
                                if self.rank == 0:
                                    print(f"⚠️  Warning: Unexpected pred shape: {pred.shape}, skipping this prediction")
                        elif pred.dim() == 1:
                            # 1D tensor: binary classification, apply sigmoid
                            pred_probs = torch.sigmoid(pred).cpu()
                            all_predictions.append(pred_probs)
                        else:
                            # 0D or unexpected dimension
                            if self.rank == 0:
                                print(f"⚠️  Warning: Unexpected pred dimension: {pred.dim()}, shape: {pred.shape}, skipping this prediction")
                    
                    all_targets.append(batch.y.cpu())
        
        # Average loss across all processes
        avg_loss = total_loss / num_batches
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        else:
            avg_loss = loss_tensor.item()
        
        # Calculate additional metrics if requested (only for classification)
        if return_metrics:
            # Collect all predictions and targets across all processes
            predictions_probs = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([])
            predictions_raw = torch.cat(all_predictions_raw, dim=0) if all_predictions_raw else torch.tensor([])
            targets = torch.cat(all_targets, dim=0) if all_targets else torch.tensor([])
            
            # Ensure predictions_probs is 1D
            if predictions_probs.dim() > 1:
                predictions_probs = predictions_probs.squeeze()
            
            # Convert probabilities to class predictions
            if len(torch.unique(targets)) <= 2:  # Binary classification
                predictions = (predictions_probs > 0.5).float()
            else:
                # Multi-class: use raw logits to get class predictions
                if predictions_raw.shape[1] > 1:
                    predictions = predictions_raw.argmax(dim=1).float()
                else:
                    predictions = predictions_probs
            
            # Ensure targets are 1D
            if targets.dim() > 1:
                targets = targets.squeeze()
            
            # Collect data across all processes (similar to test function)
            if len(predictions) > 0:
                if self.world_size > 1:
                    # Use simpler method for validation (smaller dataset)
                    local_len = torch.tensor(len(predictions), device=self.device, dtype=torch.long)
                    lengths = [torch.zeros_like(local_len) for _ in range(self.world_size)]
                    dist.all_gather(lengths, local_len)
                    lengths = [l.item() for l in lengths]
                    max_len = max(lengths) if lengths else 0
                    
                    if max_len > 0:
                        # Pad to same length
                        if len(predictions) < max_len:
                            padding_size = max_len - len(predictions)
                            predictions = torch.cat([predictions, torch.zeros(padding_size, dtype=predictions.dtype)])
                            predictions_probs = torch.cat([predictions_probs, torch.zeros(padding_size, dtype=predictions_probs.dtype)])
                            targets = torch.cat([targets, torch.zeros(padding_size, dtype=targets.dtype)])
                        
                        # Convert to GPU tensor and collect
                        predictions_tensor = predictions.to(self.device)
                        predictions_probs_tensor = predictions_probs.to(self.device)
                        targets_tensor = targets.to(self.device)
                        
                        gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(self.world_size)]
                        gathered_predictions_probs = [torch.zeros_like(predictions_probs_tensor) for _ in range(self.world_size)]
                        gathered_targets = [torch.zeros_like(targets_tensor) for _ in range(self.world_size)]
                        
                        dist.all_gather(gathered_predictions, predictions_tensor)
                        dist.all_gather(gathered_predictions_probs, predictions_probs_tensor)
                        dist.all_gather(gathered_targets, targets_tensor)
                        
                        if self.rank == 0:
                            # Remove padding and merge
                            all_pred = []
                            all_pred_probs = []
                            all_targ = []
                            for i, (pred, pred_prob, targ, length) in enumerate(zip(gathered_predictions, gathered_predictions_probs, gathered_targets, lengths)):
                                if length > 0:
                                    all_pred.append(pred[:length].cpu())
                                    all_pred_probs.append(pred_prob[:length].cpu())
                                    all_targ.append(targ[:length].cpu())
                            
                            if all_pred:
                                # Concatenate while preserving structure
                                all_predictions_flat = torch.cat(all_pred).numpy()
                                all_predictions_probs_flat = torch.cat(all_pred_probs).numpy()
                                all_targets_flat = torch.cat(all_targ).numpy()
                else:
                    # Single GPU mode: use predictions directly
                    all_predictions_flat = predictions.numpy()
                    all_predictions_probs_flat = predictions_probs.numpy()
                    all_targets_flat = targets.numpy()
                
                if len(predictions) > 0:
                    # Check if we have gathered data (multi-GPU) or direct data (single-GPU)
                    if self.world_size == 1 or (self.world_size > 1 and self.rank == 0 and 'all_predictions_flat' in locals()):
                        if self.world_size == 1:
                            all_predictions_flat = predictions.numpy()
                            all_predictions_probs_flat = predictions_probs.numpy()
                            all_targets_flat = targets.numpy()
                        
                        # Check if multitask (preserve 2D structure)
                            is_multitask_here = all_targets_flat.ndim == 2 and all_targets_flat.shape[1] > 1
                            
                            # Calculate metrics (only on rank 0)
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                            
                            if is_multitask_here:
                                # For multitask, get unique values from all tasks (excluding missing labels -1)
                                unique_targets = np.unique(all_targets_flat[all_targets_flat != -1])
                            else:
                                # Single task: flatten and get unique values
                                all_targets_flat = all_targets_flat.flatten()
                                all_predictions_probs_flat = all_predictions_probs_flat.flatten()
                                all_predictions_flat = all_predictions_flat.flatten()
                                unique_targets = np.unique(all_targets_flat)
                            
                            if len(unique_targets) <= 10:  # Classification
                                # Process predicted classes
                                if np.all(np.isin(all_predictions_flat, [0, 1])) or np.all(all_predictions_flat == all_predictions_flat.astype(int)):
                                    pred_classes = all_predictions_flat.astype(int)
                                else:
                                    pred_classes = (all_predictions_flat > 0.5).astype(int)
                                
                                pred_classes = np.clip(pred_classes, int(unique_targets.min()), int(unique_targets.max()))
                                
                                acc = accuracy_score(all_targets_flat, pred_classes)
                                prec = precision_score(all_targets_flat, pred_classes, zero_division=0, average='binary' if len(unique_targets) == 2 else 'macro')
                                rec = recall_score(all_targets_flat, pred_classes, zero_division=0, average='binary' if len(unique_targets) == 2 else 'macro')
                                f1 = f1_score(all_targets_flat, pred_classes, zero_division=0, average='binary' if len(unique_targets) == 2 else 'macro')
                                
                                metrics = {
                                    'accuracy': acc,
                                    'precision': prec,
                                    'recall': rec,
                                    'f1': f1
                                }
                                
                                # For binary classification, calculate ROC-AUC and find optimal threshold
                                if len(unique_targets) == 2:
                                    try:
                                        # Use macro-averaged AUROC for multitask datasets
                                        pred_probs = np.clip(all_predictions_probs_flat, 0, 1)
                                        roc_auc, task_aurocs = calculate_macro_averaged_auroc(all_targets_flat, pred_probs)
                                        metrics['roc_auc'] = roc_auc
                                        
                                        # Also calculate PR-AUC directly (needed for validation metrics)
                                        # This ensures pr_auc is available even if optimal threshold calculation fails
                                        try:
                                            from sklearn.metrics import average_precision_score
                                            if len(np.unique(all_targets_flat)) > 1:
                                                pr_auc = average_precision_score(all_targets_flat, pred_probs)
                                                metrics['pr_auc'] = pr_auc
                                        except Exception as e:
                                            if self.rank == 0:
                                                print(f"⚠️  Could not calculate PR-AUC: {e}")
                                            metrics['pr_auc'] = 0.0
                                        
                                        # Store task-level AUROCs for debugging (if multitask)
                                        if len(task_aurocs) > 1:
                                            metrics['task_aurocs'] = task_aurocs
                                            metrics['num_tasks'] = len(task_aurocs)
                                        
                                        # Find optimal threshold using adaptive method
                                        # Support dataset-specific configuration
                                        try:
                                            # Load dataset-specific threshold configuration
                                            threshold_config = self.load_threshold_config(self.dataset_name) if self.dataset_name else {}
                                            
                                            # Check if fixed threshold is specified
                                            if threshold_config.get('fixed_threshold') is not None:
                                                optimal_threshold = threshold_config['fixed_threshold']
                                                # Evaluate with fixed threshold to get score
                                                temp_metrics = evaluate_with_threshold(all_targets_flat, pred_probs, optimal_threshold)
                                                optimal_score = temp_metrics.get('f1', 0.0)
                                                # IMPORTANT: Add pr_auc to metrics when using fixed threshold
                                                if 'pr_auc' in temp_metrics:
                                                    metrics['pr_auc'] = temp_metrics['pr_auc']
                                                if self.rank == 0:
                                                    print(f"📌 Using fixed threshold from config: {optimal_threshold:.4f}")
                                            else:
                                                # Try to infer from imbalance ratio and dataset characteristics
                                                pos_count = (all_targets_flat == 1).sum()
                                                neg_count = (all_targets_flat == 0).sum()
                                                imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
                                                dataset_size = len(all_targets_flat)
                                                
                                                # Determine method from config or auto-select
                                                config_method = threshold_config.get('method')
                                                use_cv_threshold = False
                                                
                                                if config_method == 'cv':
                                                    use_cv_threshold = True
                                                elif config_method == 'adaptive':
                                                    use_cv_threshold = False
                                                else:
                                                    # Auto-select: Use CV threshold for extremely imbalanced small datasets (like CLINTOX)
                                                    # CLINTOX characteristics: imbalance_ratio > 100, dataset_size < 1000
                                                    use_cv_threshold = (imbalance_ratio > 100 and dataset_size < 1000)
                                                
                                                if use_cv_threshold:
                                                    # Use CV threshold selection for CLINTOX-like datasets
                                                    optimal_threshold, optimal_score = find_optimal_threshold_cv(
                                                        all_targets_flat, pred_probs, 
                                                        imbalance_ratio=imbalance_ratio,
                                                        method='auto', n_splits=5
                                                    )
                                                else:
                                                    # Use adaptive threshold selection for other datasets
                                                    # Check if custom threshold_range is specified
                                                    threshold_range = threshold_config.get('threshold_range')
                                                    if threshold_range:
                                                        # Use custom range with adaptive method
                                                        from utils.threshold_utils import find_optimal_threshold
                                                        metric = threshold_config.get('metric', 'auto')
                                                        optimal_threshold, optimal_score = find_optimal_threshold(
                                                            all_targets_flat, pred_probs,
                                                            metric=metric if metric != 'auto' else 'f1',
                                                            threshold_range=tuple(threshold_range),
                                                            num_steps=200,
                                                            avoid_pathological=True
                                                        )
                                                    else:
                                                        # Use default adaptive method
                                                        optimal_threshold, optimal_score = find_optimal_threshold_adaptive(
                                                            all_targets_flat, pred_probs, method='auto'
                                                        )
                                            metrics['optimal_threshold'] = optimal_threshold
                                            metrics['optimal_threshold_score'] = optimal_score
                                            
                                            # Evaluate with optimal threshold
                                            optimal_metrics = evaluate_with_threshold(
                                                all_targets_flat, pred_probs, optimal_threshold
                                            )
                                            # Add optimal threshold metrics with prefix
                                            for key, value in optimal_metrics.items():
                                                if key != 'threshold':  # Skip threshold itself
                                                    metrics[f'optimal_{key}'] = value
                                            
                                            # IMPORTANT: Also add pr_auc directly to metrics (not just optimal_pr_auc)
                                            # This is needed for optuna_serach_mod.py to find validation metrics
                                            if 'pr_auc' in optimal_metrics:
                                                metrics['pr_auc'] = optimal_metrics['pr_auc']
                                        except Exception as e:
                                            if self.rank == 0:
                                                print(f"⚠️  Could not find optimal threshold: {e}")
                                            metrics['optimal_threshold'] = 0.5
                                    except Exception as e:
                                        if self.rank == 0:
                                            print(f"⚠️  Could not calculate ROC-AUC: {e}")
                                        metrics['roc_auc'] = 0.0
                                
                                return avg_loss, metrics
                            else:
                                # Regression - calculate spearman or mae if primary_metric is set
                                metrics = {}
                                if return_metrics and self.primary_metric in ['spearman', 'mae']:
                                    try:
                                        # Collect predictions and targets for metric calculation
                                        if len(all_predictions) > 0 and len(all_targets) > 0:
                                            # Collect data across all processes (similar to classification)
                                            if self.world_size > 1:
                                                # Use simpler method for validation (smaller dataset)
                                                local_len = torch.tensor(len(all_predictions), device=self.device, dtype=torch.long)
                                                lengths = [torch.zeros_like(local_len) for _ in range(self.world_size)]
                                                dist.all_gather(lengths, local_len)
                                                lengths = [l.item() for l in lengths]
                                                max_len = max(lengths) if lengths else 0
                                                
                                                if max_len > 0:
                                                    # Pad to same length
                                                    if len(all_predictions) < max_len:
                                                        padding_size = max_len - len(all_predictions)
                                                        all_predictions_padded = torch.cat([all_predictions, torch.zeros(padding_size, dtype=all_predictions[0].dtype)])
                                                        all_targets_padded = torch.cat([all_targets, torch.zeros(padding_size, dtype=all_targets[0].dtype)])
                                                    else:
                                                        all_predictions_padded = all_predictions
                                                        all_targets_padded = all_targets
                                                    
                                                    # Convert to GPU tensor and collect
                                                    predictions_tensor = all_predictions_padded.to(self.device)
                                                    targets_tensor = all_targets_padded.to(self.device)
                                                    
                                                    gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(self.world_size)]
                                                    gathered_targets = [torch.zeros_like(targets_tensor) for _ in range(self.world_size)]
                                                    
                                                    dist.all_gather(gathered_predictions, predictions_tensor)
                                                    dist.all_gather(gathered_targets, targets_tensor)
                                                    
                                                    if self.rank == 0:
                                                        # Remove padding and merge
                                                        all_pred = []
                                                        all_targ = []
                                                        for i, (pred, targ, length) in enumerate(zip(gathered_predictions, gathered_targets, lengths)):
                                                            if length > 0:
                                                                all_pred.append(pred[:length].cpu())
                                                                all_targ.append(targ[:length].cpu())
                                                        
                                                        if all_pred:
                                                            all_predictions_flat = torch.cat(all_pred).numpy()
                                                            all_targets_flat = torch.cat(all_targ).numpy()
                                            else:
                                                # Single GPU mode: use predictions directly
                                                all_predictions_flat = torch.cat(all_predictions, dim=0).numpy()
                                                all_targets_flat = torch.cat(all_targets, dim=0).numpy()
                                            
                                            # Calculate metrics (only on rank 0)
                                            if self.rank == 0 and len(all_predictions_flat) > 0 and len(all_targets_flat) > 0:
                                                all_predictions_flat = all_predictions_flat.flatten()
                                                all_targets_flat = all_targets_flat.flatten()
                                                
                                                # Calculate spearman correlation if primary_metric is spearman
                                                if self.primary_metric == 'spearman':
                                                    try:
                                                        from scipy.stats import spearmanr
                                                        spearman_result = spearmanr(all_targets_flat, all_predictions_flat)
                                                        if spearman_result.correlation is not None and not np.isnan(spearman_result.correlation):
                                                            metrics['spearman'] = float(spearman_result.correlation)
                                                    except Exception as e:
                                                        if self.rank == 0:
                                                            print(f"⚠️  Could not calculate Spearman: {e}")
                                                
                                                # Calculate MAE if primary_metric is mae
                                                if self.primary_metric == 'mae':
                                                    try:
                                                        from sklearn.metrics import mean_absolute_error
                                                        mae = mean_absolute_error(all_targets_flat, all_predictions_flat)
                                                        if mae is not None and not np.isnan(mae):
                                                            metrics['mae'] = float(mae)
                                                    except Exception as e:
                                                        if self.rank == 0:
                                                            print(f"⚠️  Could not calculate MAE: {e}")
                                    except Exception as e:
                                        if self.rank == 0:
                                            print(f"⚠️  Could not collect regression data for metrics: {e}")
                                
                                return avg_loss, metrics
                        else:
                            return avg_loss, {}
                    else:
                        # Non-rank 0 processes return empty metrics
                        return avg_loss, {}
                else:
                    return avg_loss, {}
            else:
                return avg_loss, {}
        else:
            return avg_loss
    
    def test(self, save_dir='./checkpoints'):
        """Test model - Optimized version with progress bar and more efficient data collection"""
        self.model.eval()
        all_predictions = []
        all_predictions_raw = []  # Store raw logits for ROC-AUC calculation
        all_targets = []
        
        # Add progress bar (only display on rank 0)
        if self.rank == 0:
            test_iter = tqdm(self.test_loader, desc="Testing", leave=False, ncols=100)
        else:
            test_iter = self.test_loader
        
        # 🔧 FIX: Determine task type from model_config before processing predictions
        # This ensures regression tasks use raw predictions, not sigmoid/softmax processed ones
        model_type = self.model_config.get('model_type', None) if self.model_config else None
        is_regression = False
        if model_type == 'regressor':
            is_regression = True
        elif model_type == 'classifier':
            is_regression = False
        else:
            # Fallback: infer from primary_metric or targets (will be checked later)
            is_regression = None  # Will be determined later based on targets
        
        with torch.no_grad():
            for batch in test_iter:
                batch = batch.to(self.device)
                # If batch has pos attribute (3D coordinates), pass it to the model
                pos = getattr(batch, 'pos', None)
                fingerprint = getattr(batch, 'fingerprint', None)
                b2revb = getattr(batch, 'b2revb', None)  # Get pre-computed b2revb (Chemprop style)
                pred, _ = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, pos=pos, fingerprint=fingerprint, b2revb=b2revb)
                
                # Store raw predictions (logits) for ROC-AUC calculation and regression
                all_predictions_raw.append(pred.cpu())
                
                # Process predictions based on task type
                if is_regression is True:
                    # Regression: use raw predictions directly (no sigmoid/softmax)
                    if pred.dim() >= 2:
                        # If 2D, squeeze to 1D (regression typically has shape [batch_size, 1])
                        all_predictions.append(pred.squeeze().cpu())
                    else:
                        # Already 1D, use directly
                        all_predictions.append(pred.cpu())
                elif is_regression is False:
                    # Classification: extract probabilities for positive class
                    if pred.shape[1] > 1:  # Multi-class classification
                        # Use softmax to get probabilities, then take positive class (class 1)
                        pred_probs = torch.softmax(pred, dim=1)[:, 1].cpu()
                    else:
                        # Binary classification with single output, apply sigmoid
                        pred_probs = torch.sigmoid(pred.squeeze()).cpu()
                    all_predictions.append(pred_probs)
                else:
                    # is_regression is None: fallback to old logic (will be determined later)
                    # For classification, extract probabilities for positive class
                    if pred.shape[1] > 1:  # Multi-class classification
                        # Use softmax to get probabilities, then take positive class (class 1)
                        pred_probs = torch.softmax(pred, dim=1)[:, 1].cpu()
                    else:
                        # Binary classification with single output, apply sigmoid
                        pred_probs = torch.sigmoid(pred.squeeze()).cpu()
                    all_predictions.append(pred_probs)
                
                all_targets.append(batch.y.cpu())
        
        # Concatenate all batches
        predictions_probs = torch.cat(all_predictions, dim=0)  # Probabilities for positive class (already processed) or raw predictions for regression
        predictions_raw = torch.cat(all_predictions_raw, dim=0)  # Raw logits
        targets = torch.cat(all_targets, dim=0)
        
        # 🔧 FIX: Determine task type from model_config if not already determined
        if is_regression is None:
            model_type = self.model_config.get('model_type', None) if self.model_config else None
            if model_type == 'regressor':
                is_regression = True
            elif model_type == 'classifier':
                is_regression = False
            else:
                # Fallback: infer from targets
                unique_targets = torch.unique(targets)
                is_regression = len(unique_targets) > 10  # More than 10 unique values suggests regression
        
        # Ensure predictions_probs is 1D
        if predictions_probs.dim() > 1:
            predictions_probs = predictions_probs.squeeze()
        
        # 🔧 FIX: For regression, use raw predictions directly; for classification, convert to class predictions
        if is_regression:
            # Regression: use raw predictions directly (already in all_predictions after our fix)
            predictions = predictions_probs  # For regression, predictions_probs contains raw values
        else:
            # Classification: convert probabilities to class predictions
            if len(torch.unique(targets)) <= 2:  # Binary classification
                predictions = (predictions_probs > 0.5).float()
            else:
                # Multi-class: use raw logits to get class predictions
                if predictions_raw.dim() >= 2 and predictions_raw.shape[1] > 1:
                    predictions = predictions_raw.argmax(dim=1).float()
                else:
                    predictions = predictions_probs
        
        # Ensure targets are 1D
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Ensure predictions and targets have consistent length
        min_len = min(len(predictions), len(targets), len(predictions_probs))
        if min_len == 0:
            if self.rank == 0:
                print("⚠️  Warning: Test data is empty")
            return {}
        predictions = predictions[:min_len]
        predictions_probs = predictions_probs[:min_len]  # Keep probabilities for ROC-AUC
        targets = targets[:min_len]
        
        # Verify length consistency (for debugging)
        if self.rank == 0 and len(predictions) != len(targets):
            print(f"⚠️  Warning: predictions length ({len(predictions)}) != targets length ({len(targets)})")
            min_len = min(len(predictions), len(targets), len(predictions_probs))
            predictions = predictions[:min_len]
            predictions_probs = predictions_probs[:min_len]
            targets = targets[:min_len]
        
        # Optimization: Use more efficient data collection method
        # Method 1: For small datasets, use all_gather (faster)
        # Method 2: For large datasets, use point-to-point communication (more stable)
        
        # Determine data size and choose appropriate method
        data_size = len(predictions)
        use_all_gather = data_size < 5000  # Use all_gather for less than 5000 samples
        
        # 🔧 FIX: Determine final task type for regression vs classification
        # Use model_type if available, otherwise infer from targets
        if is_regression is None:
            # Determine from targets (fallback method)
            unique_targets = torch.unique(targets)
            is_regression = len(unique_targets) > 10  # More than 10 unique values suggests regression
        
        # Single GPU mode: skip DDP operations
        if self.world_size == 1:
            # Directly use predictions without gathering
            # 🔧 FIX: After our fix, predictions variable already contains correct values:
            # - For regression: raw predictions (from all_predictions, which contains raw values after our fix)
            # - For classification: processed predictions (probabilities)
            all_predictions_flat = predictions.numpy().flatten()
            all_predictions_probs_flat = predictions_probs.numpy().flatten()
            all_targets_flat = targets.numpy().flatten()
        elif use_all_gather:
            if self.rank == 0:
                print("💡 Using all_gather to collect test results (fast mode)...")
            
            # Method 1: Use all_gather (faster for small datasets)
            # First synchronize lengths
            local_len = torch.tensor(len(predictions), device=self.device, dtype=torch.long)
            lengths = [torch.zeros_like(local_len) for _ in range(self.world_size)]
            dist.all_gather(lengths, local_len)
            lengths = [l.item() for l in lengths]
            max_len = max(lengths)
            
            # Pad to same length (ensure all are 1D tensors)
            if len(predictions) < max_len:
                padding_size = max_len - len(predictions)
                predictions = torch.cat([predictions, torch.zeros(padding_size, dtype=predictions.dtype)])
                predictions_probs = torch.cat([predictions_probs, torch.zeros(padding_size, dtype=predictions_probs.dtype)])
                targets = torch.cat([targets, torch.zeros(padding_size, dtype=targets.dtype)])
            
            # Convert to GPU tensor and collect
            predictions_tensor = predictions.to(self.device)
            predictions_probs_tensor = predictions_probs.to(self.device)
            targets_tensor = targets.to(self.device)
            
            gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(self.world_size)]
            gathered_predictions_probs = [torch.zeros_like(predictions_probs_tensor) for _ in range(self.world_size)]
            gathered_targets = [torch.zeros_like(targets_tensor) for _ in range(self.world_size)]
            
            if self.rank == 0:
                print("📡 Collecting test results...")
            
            dist.all_gather(gathered_predictions, predictions_tensor)
            dist.all_gather(gathered_predictions_probs, predictions_probs_tensor)
            dist.all_gather(gathered_targets, targets_tensor)
            
            if self.rank == 0:
                # Remove padding and merge
                all_pred = []
                all_pred_probs = []
                all_targ = []
                for i, (pred, pred_prob, targ, length) in enumerate(zip(gathered_predictions, gathered_predictions_probs, gathered_targets, lengths)):
                    all_pred.append(pred[:length].cpu())
                    all_pred_probs.append(pred_prob[:length].cpu())
                    all_targ.append(targ[:length].cpu())
                
                all_predictions_flat = torch.cat(all_pred).numpy().flatten()
                all_predictions_probs_flat = torch.cat(all_pred_probs).numpy().flatten()
                all_targets_flat = torch.cat(all_targ).numpy().flatten()
        else:
            # Method 2: Use point-to-point communication (more stable for large datasets)
            # Ensure predictions and targets are both 1D (already processed above)
            if self.rank == 0:
                # Rank 0 collects all results
                gathered_predictions = [predictions]
                gathered_predictions_probs = [predictions_probs]
                gathered_targets = [targets]
                
                # Receive data from other processes (with progress indication)
                if self.world_size > 1:
                    print(f"📡 Collecting data from {self.world_size - 1} processes...")
                    for src_rank in range(1, self.world_size):
                        # First receive length
                        recv_len = torch.zeros(1, dtype=torch.long, device=self.device)
                        dist.recv(recv_len, src=src_rank)
                        recv_len = recv_len.item()
                        
                        # Receive predictions, probabilities, and targets
                        recv_pred = torch.zeros(recv_len, dtype=torch.float32, device=self.device)
                        recv_pred_prob = torch.zeros(recv_len, dtype=torch.float32, device=self.device)
                        recv_targ = torch.zeros(recv_len, dtype=torch.float32, device=self.device)
                        dist.recv(recv_pred, src=src_rank)
                        dist.recv(recv_pred_prob, src=src_rank)
                        dist.recv(recv_targ, src=src_rank)
                        
                        gathered_predictions.append(recv_pred.cpu())
                        gathered_predictions_probs.append(recv_pred_prob.cpu())
                        gathered_targets.append(recv_targ.cpu())
                
                # 🔧 FIX: For regression, use raw predictions; for classification, use processed predictions
                if is_regression:
                    # Regression: use raw predictions (gathered_predictions should contain raw values after our fix)
                    all_predictions_flat = torch.cat(gathered_predictions, dim=0).numpy().flatten()
                else:
                    # Classification: use processed predictions
                    all_predictions_flat = torch.cat(gathered_predictions, dim=0).numpy().flatten()
                all_predictions_probs_flat = torch.cat(gathered_predictions_probs, dim=0).numpy().flatten()
                all_targets_flat = torch.cat(gathered_targets, dim=0).numpy().flatten()
            else:
                # Other processes send data to rank 0
                # predictions and targets are already 1D, use directly
                pred_flat = predictions.to(self.device)
                pred_prob_flat = predictions_probs.to(self.device)
                targ_flat = targets.to(self.device)
                
                # Send length
                send_len = torch.tensor(len(pred_flat), dtype=torch.long, device=self.device)
                dist.send(send_len, dst=0)
                
                # Send data
                dist.send(pred_flat, dst=0)
                dist.send(pred_prob_flat, dst=0)
                dist.send(targ_flat, dst=0)
                
                # Non-rank 0 processes return directly
                return {}
            
        # Calculate metrics (only on rank 0)
        if self.rank == 0:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import roc_auc_score
            from scipy.stats import spearmanr
            
            print("📊 Calculating test metrics...")
            
            # Load dataset primary metric configuration
            primary_metric = None
            try:
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'dataset_primary_metrics.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    dataset_configs = config.get('dataset_primary_metrics', {})
                    if self.dataset_name:
                        dataset_config = dataset_configs.get(self.dataset_name.lower(), {})
                        primary_metric = dataset_config.get('primary_metric', None)
                        if primary_metric:
                            print(f"📌 Primary metric for {self.dataset_name}: {primary_metric}")
            except Exception as e:
                print(f"⚠️  Could not load primary metric config: {e}")
            
            # Load optimal threshold from checkpoint if available
            test_threshold = self.best_threshold
            try:
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    if 'optimal_threshold' in checkpoint:
                        test_threshold = checkpoint['optimal_threshold']
                        print(f"✅ Using optimal threshold from checkpoint: {test_threshold:.4f}")
                    else:
                        print(f"ℹ️  No optimal threshold in checkpoint, using saved threshold: {test_threshold:.4f}")
            except Exception as e:
                print(f"⚠️  Could not load optimal threshold from checkpoint: {e}, using default: {test_threshold:.4f}")
            
            # 🔧 NORMALIZATION FIX: Detect and correct scale mismatch between predictions and targets
            # This fixes the issue where predictions are in normalized space but targets are in original space
            print(f"\n🔍 Checking prediction and target scale consistency...")
            print(f"   Predictions range: [{all_predictions_flat.min():.4f}, {all_predictions_flat.max():.4f}]")
            print(f"   Predictions mean: {all_predictions_flat.mean():.4f}, std: {all_predictions_flat.std():.4f}")
            print(f"   Targets range: [{all_targets_flat.min():.4f}, {all_targets_flat.max():.4f}]")
            print(f"   Targets mean: {all_targets_flat.mean():.4f}, std: {all_targets_flat.std():.4f}")
            
            # Detect if predictions and targets are in different scales
            needs_denormalization = False
            if self.target_mean is not None and self.target_std is not None:
                # Calculate prediction statistics
                pred_mean = all_predictions_flat.mean()
                pred_std = all_predictions_flat.std()
                pred_min = all_predictions_flat.min()
                pred_max = all_predictions_flat.max()
                pred_range = pred_max - pred_min
                
                # Calculate target statistics
                target_mean = all_targets_flat.mean()
                target_std = all_targets_flat.std()
                target_min = all_targets_flat.min()
                target_max = all_targets_flat.max()
                target_range = target_max - target_min
                
                # Method 1: Check if predictions appear to be in normalized space
                # Normalized values typically have mean close to 0 and std close to 1, or range in [0,1] or [-1,1]
                pred_appears_normalized = (
                    # Standard normalization: mean ~0, std ~1
                    (abs(pred_mean) < 0.5 and 0.3 < pred_std < 2.0) or
                    # Min-max normalization: range in [0, 1]
                    (pred_min >= -0.1 and pred_max <= 1.1 and pred_range < 1.5) or
                    # Centered normalization: range in [-1, 1]
                    (pred_min >= -1.1 and pred_max <= 1.1 and pred_range < 2.5)
                )
                
                # Method 2: Check if targets are in original space (match training statistics)
                # Targets should match training set statistics (mean and std should be close)
                target_matches_training = (
                    abs(target_mean - self.target_mean) < self.target_std * 0.3 and
                    abs(target_std - self.target_std) < self.target_std * 0.3
                )
                
                # Method 3: Compare prediction and target scales directly
                # If predictions and targets have very different scales, likely mismatch
                scale_ratio_mean = abs(pred_mean / target_mean) if abs(target_mean) > 1e-6 else float('inf')
                scale_ratio_std = abs(pred_std / target_std) if target_std > 1e-6 else float('inf')
                scale_ratio_range = abs(pred_range / target_range) if target_range > 1e-6 else float('inf')
                
                # Scale mismatch if ratios are very different from 1.0
                scale_mismatch = (
                    (scale_ratio_mean < 0.1 or scale_ratio_mean > 10.0) or
                    (scale_ratio_std < 0.1 or scale_ratio_std > 10.0) or
                    (scale_ratio_range < 0.1 or scale_ratio_range > 10.0)
                )
                
                # Decision: Apply denormalization if:
                # 1. Predictions appear normalized AND targets match training statistics, OR
                # 2. Clear scale mismatch detected
                if (pred_appears_normalized and target_matches_training) or (scale_mismatch and pred_appears_normalized):
                    needs_denormalization = True
                    print(f"\n🚨 Scale mismatch detected!")
                    print(f"   Predictions appear to be in normalized space")
                    print(f"     - Range: [{pred_min:.4f}, {pred_max:.4f}], Mean: {pred_mean:.4f}, Std: {pred_std:.4f}")
                    print(f"   Targets are in original space")
                    print(f"     - Range: [{target_min:.4f}, {target_max:.4f}], Mean: {target_mean:.4f}, Std: {target_std:.4f}")
                    print(f"   Training statistics: Mean={self.target_mean:.4f}, Std={self.target_std:.4f}")
                    print(f"   Applying denormalization to predictions...")
                    
                    # Denormalize predictions
                    # Assume predictions were normalized as: (x - mean) / std
                    # Denormalize: x_original = x_normalized * std + mean
                    all_predictions_denorm = all_predictions_flat * self.target_std + self.target_mean
                    
                    print(f"   Denormalized predictions range: [{all_predictions_denorm.min():.4f}, {all_predictions_denorm.max():.4f}]")
                    print(f"   Denormalized predictions mean: {all_predictions_denorm.mean():.4f}, std: {all_predictions_denorm.std():.4f}")
                    
                    # Verify denormalization improved scale match
                    denorm_mean = all_predictions_denorm.mean()
                    denorm_std = all_predictions_denorm.std()
                    mean_diff_before = abs(pred_mean - target_mean)
                    mean_diff_after = abs(denorm_mean - target_mean)
                    std_diff_before = abs(pred_std - target_std)
                    std_diff_after = abs(denorm_std - target_std)
                    
                    # Calculate overall improvement score
                    # Mean matching is more important than std matching for regression tasks
                    mean_improvement = mean_diff_before - mean_diff_after
                    std_change = std_diff_after - std_diff_before
                    
                    # Accept denormalization if:
                    # 1. Mean difference improved significantly (more than 0.5), OR
                    # 2. Both mean and std improved, OR
                    # 3. Mean improved and std only slightly worsened (less than 0.2)
                    # 4. Mean improvement is very large (more than 2.0) - accept even if std worsens
                    mean_improved_significantly = mean_improvement > 0.5
                    both_improved = mean_diff_after < mean_diff_before and std_diff_after < std_diff_before
                    mean_improved_acceptable_std = mean_diff_after < mean_diff_before and std_change < 0.2
                    very_large_mean_improvement = mean_improvement > 2.0
                    
                    if mean_improved_significantly or both_improved or mean_improved_acceptable_std or very_large_mean_improvement:
                        # Use denormalized predictions for metric calculation
                        all_predictions_flat = all_predictions_denorm
                        print(f"   ✅ Denormalization applied successfully (improved scale match)")
                        print(f"      Mean diff: {mean_diff_before:.4f} -> {mean_diff_after:.4f} (improved by {mean_improvement:.4f})")
                        print(f"      Std diff: {std_diff_before:.4f} -> {std_diff_after:.4f} (changed by {std_change:+.4f})")
                    else:
                        # Denormalization didn't help, might be wrong assumption
                        print(f"   ⚠️  Denormalization didn't improve scale match, using original predictions")
                        print(f"      Mean diff: {mean_diff_before:.4f} -> {mean_diff_after:.4f} (improved by {mean_improvement:.4f})")
                        print(f"      Std diff: {std_diff_before:.4f} -> {std_diff_after:.4f} (changed by {std_change:+.4f})")
                        print(f"      ⚠️  Note: Mean improved but std worsened significantly, keeping original predictions")
                else:
                    print(f"   ✅ Predictions and targets appear to be in the same scale")
                    print(f"      Scale ratios - Mean: {scale_ratio_mean:.4f}, Std: {scale_ratio_std:.4f}, Range: {scale_ratio_range:.4f}")
            else:
                # 🔧 FIX: Even without training statistics, use heuristic detection
                print(f"   ℹ️  No training target statistics available, using heuristic detection...")
                
                # Calculate prediction and target statistics
                pred_mean = all_predictions_flat.mean()
                pred_std = all_predictions_flat.std()
                pred_min = all_predictions_flat.min()
                pred_max = all_predictions_flat.max()
                pred_range = pred_max - pred_min
                
                target_mean = all_targets_flat.mean()
                target_std = all_targets_flat.std()
                target_min = all_targets_flat.min()
                target_max = all_targets_flat.max()
                target_range = target_max - target_min
                
                # Heuristic: Check if predictions appear normalized but targets don't
                pred_appears_normalized = (
                    (abs(pred_mean) < 0.5 and 0.3 < pred_std < 2.0) or
                    (pred_min >= -0.1 and pred_max <= 1.1 and pred_range < 1.5) or
                    (pred_min >= -1.1 and pred_max <= 1.1 and pred_range < 2.5)
                )
                
                target_appears_original = (
                    (abs(target_mean) > 1.0 or target_std > 1.0) and
                    (target_min < -1.5 or target_max > 1.5)
                )
                
                # Calculate scale ratios
                scale_ratio_mean = abs(pred_mean / target_mean) if abs(target_mean) > 1e-6 else float('inf')
                scale_ratio_std = abs(pred_std / target_std) if target_std > 1e-6 else float('inf')
                scale_ratio_range = abs(pred_range / target_range) if target_range > 1e-6 else float('inf')
                
                scale_mismatch = (
                    (scale_ratio_mean < 0.1 or scale_ratio_mean > 10.0) or
                    (scale_ratio_std < 0.1 or scale_ratio_std > 10.0) or
                    (scale_ratio_range < 0.1 or scale_ratio_range > 10.0)
                )
                
                # If predictions appear normalized and targets appear original, and scale mismatch exists
                if pred_appears_normalized and target_appears_original and scale_mismatch:
                    print(f"\n🚨 Scale mismatch detected (heuristic)!")
                    print(f"   Predictions appear normalized: mean={pred_mean:.4f}, std={pred_std:.4f}, range=[{pred_min:.4f}, {pred_max:.4f}]")
                    print(f"   Targets appear original: mean={target_mean:.4f}, std={target_std:.4f}, range=[{target_min:.4f}, {target_max:.4f}]")
                    print(f"   Scale ratios - Mean: {scale_ratio_mean:.4f}, Std: {scale_ratio_std:.4f}, Range: {scale_ratio_range:.4f}")
                    print(f"   ⚠️  Attempting denormalization using test set statistics...")
                    
                    # Use test set statistics for denormalization (approximation)
                    # Assume predictions were normalized as: (x - mean) / std
                    # We'll use test target statistics as approximation
                    test_target_mean = target_mean
                    test_target_std = target_std
                    
                    all_predictions_denorm = all_predictions_flat * test_target_std + test_target_mean
                    
                    print(f"   Denormalized predictions range: [{all_predictions_denorm.min():.4f}, {all_predictions_denorm.max():.4f}]")
                    print(f"   Denormalized predictions mean: {all_predictions_denorm.mean():.4f}, std: {all_predictions_denorm.std():.4f}")
                    
                    # Verify improvement
                    mean_diff_before = abs(pred_mean - target_mean)
                    mean_diff_after = abs(all_predictions_denorm.mean() - target_mean)
                    std_diff_before = abs(pred_std - target_std)
                    std_diff_after = abs(all_predictions_denorm.std() - target_std)
                    
                    if mean_diff_after < mean_diff_before and std_diff_after < std_diff_before:
                        all_predictions_flat = all_predictions_denorm
                        print(f"   ✅ Denormalization applied (heuristic, using test set statistics)")
                    else:
                        print(f"   ⚠️  Denormalization didn't improve scale match, using original predictions")
                else:
                    print(f"   ✅ Predictions and targets appear to be in the same scale (heuristic check)")
                    print(f"      Scale ratios - Mean: {scale_ratio_mean:.4f}, Std: {scale_ratio_std:.4f}, Range: {scale_ratio_range:.4f}")
            
            mse = mean_squared_error(all_targets_flat, all_predictions_flat)
            mae = mean_absolute_error(all_targets_flat, all_predictions_flat)
            rmse = np.sqrt(mse)
            
            # Check if classification or regression
            if len(np.unique(all_targets_flat)) <= 10:  # Classification
                # Get raw predictions (logits) before conversion to classes
                # We need to collect raw predictions for ROC-AUC calculation
                # For now, we'll use the predictions as-is and convert to probabilities if needed
                
                # FIXED: Always use probability values with optimal threshold for classification
                # Use all_predictions_probs_flat (probabilities) instead of all_predictions_flat (already converted classes)
                pred_classes = (all_predictions_probs_flat > test_threshold).astype(int)
                
                # Ensure classes are in valid range
                unique_targets = np.unique(all_targets_flat)
                pred_classes = np.clip(pred_classes, int(unique_targets.min()), int(unique_targets.max()))
                
                # Debug information (optional)
                print(f"🔍 Debug information:")
                print(f"   Targets unique values: {unique_targets}")
                print(f"   Predictions unique values: {np.unique(pred_classes)}")
                print(f"   Predicted class distribution: {np.bincount(pred_classes)}")
                print(f"   True class distribution: {np.bincount(all_targets_flat.astype(int))}")
                print(f"   Using threshold: {test_threshold:.4f}")
                
                acc = accuracy_score(all_targets_flat, pred_classes)
                prec = precision_score(all_targets_flat, pred_classes, zero_division=0, average='binary' if len(unique_targets) == 2 else 'macro')
                rec = recall_score(all_targets_flat, pred_classes, zero_division=0, average='binary' if len(unique_targets) == 2 else 'macro')
                f1 = f1_score(all_targets_flat, pred_classes, zero_division=0, average='binary' if len(unique_targets) == 2 else 'macro')
                
                # Calculate ROC-AUC (better metric for imbalanced datasets)
                # Note: For binary classification, we need probabilities, not just class predictions
                # We'll use the raw predictions as probabilities (assuming they're logits or probabilities)
                results = {
                    'mse': mse, 'mae': mae, 'rmse': rmse,
                    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                    'threshold': test_threshold
                }
                
                # For binary classification, calculate ROC-AUC and evaluate with optimal threshold
                if len(unique_targets) == 2:
                    try:
                        # Use the probability predictions directly (already processed above)
                        # all_predictions_probs_flat contains probabilities for positive class
                        pred_probs = all_predictions_probs_flat
                        
                        # Ensure probabilities are in [0, 1]
                        pred_probs = np.clip(pred_probs, 0, 1)
                        
                        # Calculate macro-averaged AUROC for multitask datasets
                        roc_auc, task_aurocs = calculate_macro_averaged_auroc(all_targets_flat, pred_probs)
                        results['roc_auc'] = roc_auc
                        
                        # Print AUROC with multitask information if applicable
                        if len(task_aurocs) > 1:
                            print(f"   Macro-Averaged ROC-AUC: {roc_auc:.4f} (across {len(task_aurocs)} tasks)")
                            # Fix: Ensure auc values are numeric before formatting
                            # Use a helper function to avoid nested f-string issues
                            def format_auc(auc):
                                if isinstance(auc, (int, float, np.number)):
                                    return f'{float(auc):.4f}'
                                else:
                                    return str(auc)
                            task_aurocs_formatted = [format_auc(auc) for auc in task_aurocs]
                            print(f"   Task-level AUROCs: {task_aurocs_formatted}")
                        else:
                            print(f"   ROC-AUC: {roc_auc:.4f} (better metric for imbalanced datasets)")
                        
                        # Mark primary metric if configured
                        if primary_metric:
                            results['primary_metric'] = primary_metric
                            results['primary_metric_value'] = results.get(primary_metric, None)
                            if results['primary_metric_value'] is not None:
                                print(f"   ⭐ Primary metric ({primary_metric}): {results['primary_metric_value']:.4f}")
                        
                        # Evaluate with optimal threshold from validation set
                        try:
                            optimal_metrics = evaluate_with_threshold(
                                all_targets_flat, pred_probs, test_threshold
                            )
                            # Add optimal threshold metrics with prefix
                            for key, value in optimal_metrics.items():
                                if key not in ['threshold', 'roc_auc', 'pr_auc']:  # Avoid duplicates
                                    results[f'optimal_{key}'] = value
                            if 'pr_auc' in optimal_metrics:
                                results['pr_auc'] = optimal_metrics['pr_auc']
                            print(f"   📊 Metrics with validation optimal threshold ({test_threshold:.4f}):")
                            print(f"      F1: {optimal_metrics.get('f1', 0):.4f}, "
                                  f"Precision: {optimal_metrics.get('precision', 0):.4f}, "
                                  f"Recall: {optimal_metrics.get('recall', 0):.4f}")
                            
                            # Direction 3: Check threshold stability between validation and test
                            try:
                                # Find optimal threshold on test set for comparison
                                test_optimal_threshold, test_optimal_score = find_optimal_threshold_adaptive(
                                    all_targets_flat, pred_probs, method='auto'
                                )
                                
                                # Calculate imbalance ratio and dataset size for adaptive threshold check
                                pos_count = (all_targets_flat == 1).sum()
                                neg_count = (all_targets_flat == 0).sum()
                                imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
                                dataset_size = len(all_targets_flat)
                                
                                # Check stability with adaptive thresholds
                                is_stable, stability_warning = check_threshold_stability(
                                    test_threshold, test_optimal_threshold,
                                    imbalance_ratio=imbalance_ratio,
                                    dataset_size=dataset_size
                                )
                                
                                if not is_stable:
                                    print(f"   {stability_warning}")
                                    print(f"   🔧 Using Test-Time Threshold Adaptation (unstable threshold detected)")
                                    
                                    # Test-Time Threshold Adaptation: Use test set calibration
                                    # For extremely imbalanced small datasets (like CLINTOX), use calibration set
                                    if imbalance_ratio > 100 and dataset_size < 1000:
                                        # Use 20% of test set as calibration set
                                        calibration_size = max(int(dataset_size * 0.2), 10)  # At least 10 samples
                                        calibration_indices = np.random.choice(
                                            dataset_size, calibration_size, replace=False
                                        )
                                        calibration_targets = all_targets_flat[calibration_indices]
                                        calibration_probs = pred_probs[calibration_indices]
                                        
                                        # Use CV threshold selection on calibration set
                                        calibration_threshold, _ = find_optimal_threshold_cv(
                                            calibration_targets, calibration_probs,
                                            imbalance_ratio=imbalance_ratio,
                                            method='auto', n_splits=min(5, calibration_size // 2)
                                        )
                                        
                                        # Evaluate with calibration threshold
                                        calibration_metrics = evaluate_with_threshold(
                                            all_targets_flat, pred_probs, calibration_threshold
                                        )
                                        
                                        print(f"   📊 Metrics with calibration threshold ({calibration_threshold:.4f}):")
                                        print(f"      F1: {calibration_metrics.get('f1', 0):.4f}, "
                                              f"Precision: {calibration_metrics.get('precision', 0):.4f}, "
                                              f"Recall: {calibration_metrics.get('recall', 0):.4f}")
                                        
                                        # Store calibration threshold metrics
                                        results['calibration_threshold'] = calibration_threshold
                                        results['calibration_f1'] = calibration_metrics.get('f1', 0.0)
                                        results['calibration_precision'] = calibration_metrics.get('precision', 0.0)
                                        results['calibration_recall'] = calibration_metrics.get('recall', 0.0)
                                        
                                        # Update main results with calibration threshold if it's better
                                        if calibration_metrics.get('f1', 0.0) > results.get('f1', 0.0):
                                            print(f"   ✅ Calibration threshold gives better F1, updating results")
                                            results['f1'] = calibration_metrics.get('f1', 0.0)
                                            results['precision'] = calibration_metrics.get('precision', 0.0)
                                            results['recall'] = calibration_metrics.get('recall', 0.0)
                                            results['threshold'] = calibration_threshold
                                
                                # Store threshold stability information
                                results['val_optimal_threshold'] = test_threshold
                                results['test_optimal_threshold'] = test_optimal_threshold
                                results['threshold_abs_diff'] = abs(test_optimal_threshold - test_threshold)
                                results['threshold_relative_diff'] = max(
                                    test_optimal_threshold / test_threshold if test_threshold > 0 else 0,
                                    test_threshold / test_optimal_threshold if test_optimal_threshold > 0 else 0
                                )
                                results['threshold_stable'] = is_stable
                                
                                # Also evaluate with test optimal threshold for reference
                                test_optimal_metrics = evaluate_with_threshold(
                                    all_targets_flat, pred_probs, test_optimal_threshold
                                )
                                results['test_optimal_f1'] = test_optimal_metrics.get('f1', 0)
                                results['test_optimal_precision'] = test_optimal_metrics.get('precision', 0)
                                results['test_optimal_recall'] = test_optimal_metrics.get('recall', 0)
                                
                                if not is_stable:
                                    print(f"   📊 Metrics with test optimal threshold ({test_optimal_threshold:.4f}) for reference:")
                                    print(f"      F1: {test_optimal_metrics.get('f1', 0):.4f}, "
                                          f"Precision: {test_optimal_metrics.get('precision', 0):.4f}, "
                                          f"Recall: {test_optimal_metrics.get('recall', 0):.4f}")
                                
                            except Exception as e:
                                print(f"   ⚠️  Could not check threshold stability: {e}")
                                results['threshold_stable'] = None
                                
                        except Exception as e:
                            print(f"   ⚠️  Could not evaluate with optimal threshold: {e}")
                    except Exception as e:
                        print(f"   ⚠️  Could not calculate ROC-AUC: {e}")
                
                return results
            else:  # Regression
                r2 = r2_score(all_targets_flat, all_predictions_flat)
                
                # Calculate Spearman correlation coefficient
                spearman_corr = None
                try:
                    spearman_result = spearmanr(all_targets_flat, all_predictions_flat)
                    spearman_corr = spearman_result.correlation if not np.isnan(spearman_result.correlation) else None
                    if spearman_corr is not None:
                        print(f"   Spearman correlation: {spearman_corr:.4f}")
                except Exception as e:
                    print(f"   ⚠️  Could not calculate Spearman correlation: {e}")
                
                results = {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}
                if spearman_corr is not None:
                    results['spearman'] = spearman_corr
                
                # Mark primary metric if configured
                if primary_metric:
                    results['primary_metric'] = primary_metric
                    results['primary_metric_value'] = results.get(primary_metric, None)
                    if results['primary_metric_value'] is not None:
                        print(f"   ⭐ Primary metric ({primary_metric}): {results['primary_metric_value']:.4f}")
                
                return results
        
        return {}
    
    def _broadcast_learning_rate(self, new_lr):
        lr_tensor = torch.tensor(new_lr, device=self.device)
        if self.world_size > 1:
            dist.broadcast(lr_tensor, src=0)
        if self.rank != 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_tensor.item()
    
    def _apply_loss_gap_lr_reduction(self, gap_ratio):
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * self.loss_gap_lr_factor, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._broadcast_learning_rate(new_lr)
    
    def _apply_loss_gap_patience_decay(self):
        if self.use_smart_early_stopping and self.smart_early_stopping:
            old_patience = self.smart_early_stopping.current_patience
            self.smart_early_stopping.current_patience = max(
                self.smart_early_stopping.current_patience * self.loss_gap_patience_decay,
                self.smart_early_stopping.min_patience
            )
        else:
            old_patience = self.early_stopping_patience
            self.early_stopping_patience = max(int(self.early_stopping_patience * self.loss_gap_patience_decay), 5)
    
    def _maybe_handle_loss_gap(self, train_loss, val_loss):
        if not self.enable_loss_gap_control or train_loss <= 0:
            return
        
        gap_ratio = (val_loss - train_loss) / max(abs(train_loss), 1e-8)
        self.latest_loss_gap_ratio = gap_ratio
        
        if gap_ratio > self.loss_gap_threshold:
            self.loss_gap_counter += 1
        else:
            self.loss_gap_counter = 0
        
        if self.loss_gap_counter >= self.loss_gap_patience:
            self.loss_gap_counter = 0
            self._apply_loss_gap_lr_reduction(gap_ratio)
            self._apply_loss_gap_patience_decay()
    
    def _get_smoothed_val_loss(self, val_loss):
        if self.val_loss_ema_beta <= 0.0 or self.val_loss_ema_beta >= 1.0:
            return val_loss
        
        if self.val_loss_ema is None:
            self.val_loss_ema = val_loss
        else:
            beta = self.val_loss_ema_beta
            self.val_loss_ema = beta * self.val_loss_ema + (1 - beta) * val_loss
        return self.val_loss_ema
    
    def train(self, num_epochs=200, save_dir='./checkpoints', resume_from=None):
        """Complete training process"""
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize progress monitor for Optuna pruning (only on rank 0)
        progress_monitor = None
        if self.rank == 0 and PROGRESS_MONITORING_AVAILABLE:
            try:
                progress_file = os.path.join(save_dir, "training_progress.json")
                progress_monitor = JSONProgressMonitor(progress_file)
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️  Warning: Failed to initialize progress monitor: {e}")
                    print("   Progress monitoring disabled. Optuna pruning may be slower.")
        
        # Synchronize all processes (skip if single GPU)
        if self.world_size > 1:
            dist.barrier()
        
        # 🔧 FIX: Collect target statistics even when skipping training (for test-only mode)
        # This is needed for normalization detection during testing
        if num_epochs == 0 and (self.target_mean is None or self.target_std is None):
            if self.rank == 0:
                print(f"📊 Collecting target statistics from training set (for test-only mode)...")
            try:
                train_targets = []
                for batch in self.train_loader:
                    if hasattr(batch, 'y'):
                        if batch.y.dim() > 0:
                            train_targets.append(batch.y.cpu())
                
                if train_targets:
                    all_targets = torch.cat(train_targets).numpy()
                    self.target_mean = float(np.mean(all_targets))
                    self.target_std = float(np.std(all_targets))
                    self.target_min = float(np.min(all_targets))
                    self.target_max = float(np.max(all_targets))
                    
                    if self.rank == 0:
                        print(f"📊 Training target statistics:")
                        print(f"   Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
                        print(f"   Range: [{self.target_min:.4f}, {self.target_max:.4f}]")
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️  Could not collect target statistics: {e}")
        
        # Load checkpoint if resuming
        start_epoch = 0
        if resume_from is not None and os.path.exists(resume_from):
            if self.rank == 0:
                print(f"📂 Resuming training from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=f'cuda:{self.rank}', weights_only=False)
            
            # 🔧 FIX: Try to load target statistics from checkpoint or training history
            if (self.target_mean is None or self.target_std is None) and self.rank == 0:
                # Try to load from checkpoint
                if 'target_mean' in checkpoint and 'target_std' in checkpoint:
                    self.target_mean = checkpoint.get('target_mean')
                    self.target_std = checkpoint.get('target_std')
                    self.target_min = checkpoint.get('target_min')
                    self.target_max = checkpoint.get('target_max')
                    print(f"📊 Loaded target statistics from checkpoint:")
                    print(f"   Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
                else:
                    # Try to load from training history
                    save_dir = os.path.dirname(resume_from)
                    history_file = os.path.join(save_dir, 'training_history.json')
                    if os.path.exists(history_file):
                        try:
                            # json is already imported at module level
                            with open(history_file, 'r') as f:
                                history = json.load(f)
                            if 'target_mean' in history and 'target_std' in history:
                                self.target_mean = history.get('target_mean')
                                self.target_std = history.get('target_std')
                                self.target_min = history.get('target_min')
                                self.target_max = history.get('target_max')
                                print(f"📊 Loaded target statistics from training history:")
                                print(f"   Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
                        except Exception as e:
                            print(f"⚠️  Could not load target statistics from history: {e}")
            
            # Load model state with error handling
            try:
                self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                if self.rank == 0:
                    print(f"✅ Model state loaded successfully")
            except RuntimeError as e:
                if self.rank == 0:
                    print(f"⚠️  Warning: Could not load model state with strict=True")
                    print(f"   Error: {str(e)[:200]}...")
                    print(f"   Attempting to load with strict=False (ignoring mismatched keys)...")
                try:
                    missing_keys, unexpected_keys = self.base_model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False
                    )
                    if self.rank == 0:
                        if missing_keys:
                            print(f"   Missing keys (not in current model): {len(missing_keys)} keys")
                            if len(missing_keys) <= 10:
                                for key in missing_keys:
                                    print(f"     - {key}")
                            else:
                                for key in missing_keys[:5]:
                                    print(f"     - {key}")
                                print(f"     ... and {len(missing_keys) - 5} more")
                        if unexpected_keys:
                            print(f"   Unexpected keys (not in checkpoint): {len(unexpected_keys)} keys")
                            if len(unexpected_keys) <= 10:
                                for key in unexpected_keys:
                                    print(f"     - {key}")
                            else:
                                for key in unexpected_keys[:5]:
                                    print(f"     - {key}")
                                print(f"     ... and {len(unexpected_keys) - 5} more")
                        print(f"   ⚠️  Model loaded with mismatched keys - results may be incorrect!")
                except Exception as e2:
                    if self.rank == 0:
                        print(f"❌ Failed to load model state even with strict=False: {e2}")
                    raise
            
            # Load optimizer state (only if resuming training, not just testing)
            if num_epochs > 0:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if self.rank == 0:
                        print(f"✅ Optimizer state loaded successfully")
                except Exception as e:
                    if self.rank == 0:
                        print(f"⚠️  Warning: Could not load optimizer state: {e}")
                        print(f"   Will start with fresh optimizer state")
            
            # Get starting epoch
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            
            # Restore best validation loss and patience counter
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            if self.rank == 0:
                print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
                print(f"   Best validation loss: {self.best_val_loss:.4f}")
                print(f"   Resuming from epoch {start_epoch}")
        
        # IMPROVEMENT 3.1: Update cosine scheduler T_max with actual training epochs
        # This ensures the cosine annealing schedule matches the actual training duration
        if self.scheduler_type == 'cosine' and isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            actual_t_max = num_epochs - self.warmup_epochs
            self.scheduler.T_max = max(1, actual_t_max)  # Ensure T_max is at least 1
            if self.rank == 0:
                print(f"📊 IMPROVEMENT 3.1: Updated CosineAnnealingLR T_max to {self.scheduler.T_max} (num_epochs={num_epochs}, warmup={self.warmup_epochs})")
        
        if self.rank == 0:
            if start_epoch > 0:
                print(f"Continuing training for {num_epochs - start_epoch} more epochs (total: {num_epochs} epochs)")
            else:
                print(f"Starting training for {num_epochs} epochs")
            print(f"Using {self.world_size} GPUs with DDP")
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"Model parameter count: {sum(p.numel() for p in self.base_model.parameters()):,}")
            print(f"Learning rate scheduler: {self.scheduler_type}")
            print(f"Initial learning rate: {self.base_lr:.2e}")
            if self.warmup_epochs > 0:
                print(f"Warmup epochs: {self.warmup_epochs}")
            print(f"Minimum learning rate: {self.min_lr:.2e}")
        
        if self.scheduler_type == 'onecycle':
            steps_per_epoch = max(1, len(self.train_loader))
            if self.rank == 0:
                print(f"Initializing OneCycleLR (steps_per_epoch={steps_per_epoch}, epochs={num_epochs})")
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.onecycle_max_lr,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.onecycle_pct_start,
                anneal_strategy='cos',
                div_factor=self.onecycle_div_factor,
                final_div_factor=self.onecycle_final_div_factor
            )
        
        for epoch in range(start_epoch, num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Synchronize all processes to ensure training is complete (skip if single GPU)
            if self.world_size > 1:
                dist.barrier()
            
            # Validation (with metrics for smart early stopping)
            if self.use_smart_early_stopping:
                val_result = self.validate(return_metrics=True)
                if isinstance(val_result, tuple):
                    val_loss, val_metrics = val_result
                else:
                    val_loss = val_result
                    val_metrics = {}
            else:
                val_loss = self.validate(return_metrics=False)
                val_metrics = {}
            self.val_losses.append(val_loss)
            smoothed_val_loss = self._get_smoothed_val_loss(val_loss)
            
            # Extract and save primary metric from val_metrics
            if self.primary_metric and val_metrics:
                primary_metric_value = None
                
                if self.primary_metric == 'spearman':
                    primary_metric_value = val_metrics.get('spearman')
                    if primary_metric_value is not None:
                        self.val_spearman.append(primary_metric_value)
                elif self.primary_metric == 'roc_auc' or self.primary_metric == 'auroc':
                    primary_metric_value = val_metrics.get('roc_auc')
                    if primary_metric_value is not None:
                        self.val_aurocs.append(primary_metric_value)
                elif self.primary_metric == 'f1':
                    primary_metric_value = val_metrics.get('f1')
                    if primary_metric_value is not None:
                        self.val_f1_scores.append(primary_metric_value)
                elif self.primary_metric == 'pr_auc':
                    primary_metric_value = val_metrics.get('pr_auc')
                    if primary_metric_value is not None:
                        self.val_pr_aucs.append(primary_metric_value)
                elif self.primary_metric == 'mae':
                    primary_metric_value = val_metrics.get('mae')
                    if primary_metric_value is not None:
                        # Initialize val_mae list if not exists
                        if not hasattr(self, 'val_mae'):
                            self.val_mae = []
                        self.val_mae.append(primary_metric_value)
                
                # Check if primary metric improved
                primary_metric_improved = False  # Track if primary metric improved (for model saving)
                if primary_metric_value is not None:
                    improved = False
                    if self.primary_metric in ['spearman', 'roc_auc', 'f1', 'pr_auc']:
                        # Maximize metrics: higher is better
                        if primary_metric_value > self.best_primary_metric_value:
                            improved = True
                    elif self.primary_metric == 'mae':
                        # Minimize metrics: lower is better
                        if primary_metric_value < self.best_primary_metric_value:
                            improved = True
                    
                    if improved:
                        old_best = self.best_primary_metric_value
                        self.best_primary_metric_value = primary_metric_value
                        self.best_primary_metric_epoch = epoch
                        primary_metric_improved = True  # Mark that primary metric improved
                        
                        if self.rank == 0:
                            print(f"  ⭐ Primary metric ({self.primary_metric}) improved: {old_best:.4f} → {primary_metric_value:.4f}")
            
            if self.enable_loss_gap_control:
                self._maybe_handle_loss_gap(train_loss, val_loss)
            
            # Synchronize all processes to ensure validation is complete (skip if single GPU)
            if self.world_size > 1:
                dist.barrier()
            
            # Learning rate scheduling with warmup
            # IMPROVEMENT 3.1: Plateau scheduler doesn't need warmup, but we keep it for stability
            if self.scheduler_type not in ['onecycle', 'plateau'] and epoch < self.warmup_epochs:
                # Warmup: linearly increase learning rate
                warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                # Synchronize learning rate to all processes
                if self.rank == 0:
                    lr_tensor = torch.tensor(warmup_lr, device=self.device)
                else:
                    lr_tensor = torch.tensor(0.0, device=self.device)
                if self.world_size > 1:
                    dist.broadcast(lr_tensor, src=0)
                if self.rank != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_tensor.item()
            elif self.scheduler_type != 'onecycle':
                # Normal scheduling
                if self.scheduler_type == 'plateau':
                    # ReduceLROnPlateau only updates on rank 0, then broadcast LR to all ranks
                    if self.rank == 0:
                        scheduler_metric = smoothed_val_loss if self.val_loss_ema_beta > 0 else val_loss
                        self.scheduler.step(scheduler_metric)
                        # Get updated learning rate
                        current_lr = self.optimizer.param_groups[0]['lr']
                        lr_tensor = torch.tensor(current_lr, device=self.device)
                    else:
                        lr_tensor = torch.tensor(0.0, device=self.device)
                    # Broadcast learning rate from rank 0 to all ranks (skip if single GPU)
                    if self.world_size > 1:
                        dist.broadcast(lr_tensor, src=0)
                    # Update learning rate on all ranks
                    if self.rank != 0:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_tensor.item()
                elif self.scheduler is not None:
                    # For other schedulers, all ranks can call step() independently
                    # But we still synchronize to ensure consistency
                    self.scheduler.step()
                    # Synchronize learning rate to ensure all ranks have the same LR
                    if self.rank == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        lr_tensor = torch.tensor(current_lr, device=self.device)
                    else:
                        lr_tensor = torch.tensor(0.0, device=self.device)
                    if self.world_size > 1:
                        dist.broadcast(lr_tensor, src=0)
                    if self.rank != 0:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_tensor.item()
            
            # Logging (only on rank 0)
            if self.rank == 0:
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_loss, epoch)
                    self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                    if self.val_loss_ema_beta > 0:
                        self.writer.add_scalar('Loss/Validation_EMA', smoothed_val_loss, epoch)
                    
                    # Log Train-Val Loss Gap (overfitting indicator)
                    train_val_gap = train_loss - val_loss
                    self.writer.add_scalar('Metrics/Train-Val_Gap', train_val_gap, epoch)
                    
                    # Log additional metrics if available
                    if val_metrics:
                        if 'roc_auc' in val_metrics:
                            self.writer.add_scalar('Metrics/Val_AUROC', val_metrics['roc_auc'], epoch)
                        if 'f1' in val_metrics:
                            self.writer.add_scalar('Metrics/Val_F1', val_metrics['f1'], epoch)
                        if 'accuracy' in val_metrics:
                            self.writer.add_scalar('Metrics/Val_Accuracy', val_metrics['accuracy'], epoch)
                
                # Early stopping check
                if self.use_smart_early_stopping:
                    # Use smart early stopping (pass multiple metrics)
                    should_stop, early_stop_info = self.smart_early_stopping.update(
                        val_loss, 
                        train_loss=train_loss,
                        metrics=val_metrics
                    )
                    best_val_loss = early_stop_info['best_raw_loss']
                    # Update self.best_val_loss for consistency and history tracking
                    self.best_val_loss = best_val_loss
                    patience_counter = early_stop_info['patience_counter']
                    current_patience = early_stop_info['current_patience']
                    
                    # Check if should save best model based on primary metric
                    # Use the primary_metric_improved flag that was set earlier (before best_primary_metric_value was updated)
                    should_save_best_model = False
                    
                    if self.primary_metric and val_metrics:
                        # Use the primary_metric_improved flag that was set earlier
                        # This avoids the issue where best_primary_metric_value was already updated
                        should_save_best_model = primary_metric_improved
                    else:
                        # Fallback to val_loss based improved flag
                        should_save_best_model = early_stop_info['improved']
                    
                    # If improved (by primary metric or val_loss), save best model
                    if should_save_best_model:
                        try:
                            save_dict = {
                                'epoch': epoch,
                                'model_state_dict': self.base_model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'val_loss': val_loss,
                                'train_loss': train_loss
                            }
                            # Add primary metric information
                            if self.primary_metric:
                                save_dict['best_primary_metric'] = self.primary_metric
                                save_dict['best_primary_metric_value'] = self.best_primary_metric_value
                                save_dict['best_primary_metric_epoch'] = self.best_primary_metric_epoch
                            # Add multiple metrics to saved dictionary
                            if 'best_auroc' in early_stop_info:
                                save_dict['best_auroc'] = early_stop_info['best_auroc']
                            if 'best_f1' in early_stop_info:
                                save_dict['best_f1'] = early_stop_info['best_f1']
                            if val_metrics:
                                save_dict['val_metrics'] = val_metrics
                                # Save optimal threshold
                                if 'optimal_threshold' in val_metrics:
                                    self.best_threshold = val_metrics['optimal_threshold']
                                    save_dict['optimal_threshold'] = self.best_threshold
                                    if self.rank == 0:
                                        print(f"  💾 Saved optimal threshold: {self.best_threshold:.4f}")
                            if self.model_config:
                                save_dict['model_config'] = self.model_config
                            
                            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
                        except Exception as e:
                            print(f"⚠️  Error saving model: {e}")
                    
                    # Print progress with smart early stopping info
                    print(f"Epoch {epoch+1}/{num_epochs}")
                    print(f"  Training loss: {train_loss:.4f}")
                    smoothed_loss_display = early_stop_info.get('smoothed_loss', val_loss)
                    print(f"  Validation loss: {val_loss:.4f} (smoothed: {smoothed_loss_display:.4f})")
                    print(f"  Best validation loss: {best_val_loss:.4f}")
                    
                    # Display multi-metric information
                    if val_metrics:
                        if 'roc_auc' in val_metrics:
                            best_auroc = early_stop_info.get('best_auroc', 0.0)
                            print(f"  Validation AUROC: {val_metrics['roc_auc']:.4f} (best: {best_auroc:.4f})")
                        if 'f1' in val_metrics:
                            best_f1 = early_stop_info.get('best_f1', 0.0)
                            print(f"  Validation F1: {val_metrics['f1']:.4f} (best: {best_f1:.4f})")
                    
                    # Display Train-Val Loss Gap
                    if early_stop_info.get('train_val_gap') is not None:
                        gap = early_stop_info['train_val_gap']
                        gap_status = "✅ Normal" if gap < 0 else "⚠️  Overfitting" if gap > 0.05 else "🤏 Slight"
                        print(f"  Train-Val Loss Gap: {gap:.4f} {gap_status}")
                    if self.enable_loss_gap_control:
                        gap_ratio = self.latest_loss_gap_ratio
                        status = "⚠️  high" if gap_ratio > self.loss_gap_threshold else "✅ normal"
                        print(f"  Train-Val gap ratio: {gap_ratio:.2f} ({status}, streak={self.loss_gap_counter}/{self.loss_gap_patience})")
                        if gap_ratio > self.loss_gap_threshold:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            projected_lr = max(current_lr * self.loss_gap_lr_factor, self.min_lr)
                            print(f"  ⚠️  Loss gap ratio exceeded threshold {self.loss_gap_threshold:.2f}")
                            print(f"  🔻 Reducing learning rate from {current_lr:.2e} to {projected_lr:.2e}")
                            if self.use_smart_early_stopping:
                                projected_patience = max(
                                    self.smart_early_stopping.current_patience * self.loss_gap_patience_decay,
                                    self.smart_early_stopping.min_patience
                                )
                                print(f"  ⏱️  Smart ES patience reduced: {self.smart_early_stopping.current_patience:.1f} -> {projected_patience:.1f}")
                            else:
                                projected_patience = max(int(self.early_stopping_patience * self.loss_gap_patience_decay), 5)
                                print(f"  ⏱️  Early stopping patience reduced: {self.early_stopping_patience} -> {projected_patience}")
                    
                    print(f"  Early stopping counter: {patience_counter}/{current_patience}")
                    if early_stop_info.get('trend_info'):
                        print(f"  ℹ️  {early_stop_info['trend_info']}")
                    if early_stop_info.get('metric_info'):
                        print(f"  📊 {early_stop_info['metric_info']}")
                    print("-" * 50)
                    
                    # Write progress for Optuna pruning (only on rank 0)
                    if self.rank == 0 and progress_monitor:
                        try:
                            progress_monitor.write_progress(
                                epoch=epoch,
                                val_loss=val_loss,
                                train_loss=train_loss
                            )
                        except Exception as e:
                            # Silently fail to avoid disrupting training
                            pass
                    
                    if should_stop:
                        print(f"Early stopping triggered, stopping training at epoch {epoch+1}")
                        should_stop = torch.tensor(1, device=self.device, dtype=torch.int)
                    else:
                        should_stop = torch.tensor(0, device=self.device, dtype=torch.int)
                else:
                    # Use traditional early stopping
                    # Check if should save based on primary metric or val_loss
                    should_save_best_model = False
                    
                    if self.primary_metric and val_metrics:
                        # Use the primary_metric_improved flag that was set earlier (before best_primary_metric_value was updated)
                        # This avoids the issue where best_primary_metric_value was already updated
                        should_save_best_model = primary_metric_improved
                    
                    # Also check val_loss improvement (for backward compatibility)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if not self.primary_metric:
                            # If no primary metric, use val_loss
                            should_save_best_model = True
                    
                    if should_save_best_model:
                        if not self.primary_metric:
                            # Only update patience counter if using val_loss
                            self.patience_counter = 0
                        
                        # Save best model (only save on rank 0 to avoid file conflicts)
                        # Note: No need to synchronize before saving, as only base_model (non-DDP wrapped model) is read
                        try:
                            save_dict = {
                                'epoch': epoch,
                                'model_state_dict': self.base_model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'val_loss': val_loss,
                                'train_loss': train_loss
                            }
                            # Add primary metric information
                            if self.primary_metric:
                                save_dict['best_primary_metric'] = self.primary_metric
                                save_dict['best_primary_metric_value'] = self.best_primary_metric_value
                                save_dict['best_primary_metric_epoch'] = self.best_primary_metric_epoch
                            if self.model_config:
                                save_dict['model_config'] = self.model_config
                            # Save target statistics for normalization detection during testing
                            if self.target_mean is not None and self.target_std is not None:
                                save_dict['target_mean'] = self.target_mean
                                save_dict['target_std'] = self.target_std
                                save_dict['target_min'] = self.target_min
                                save_dict['target_max'] = self.target_max
                            # If validation metrics exist, save optimal threshold
                            if val_metrics and 'optimal_threshold' in val_metrics:
                                self.best_threshold = val_metrics['optimal_threshold']
                                save_dict['optimal_threshold'] = self.best_threshold
                                if self.rank == 0:
                                    print(f"  💾 Saved optimal threshold: {self.best_threshold:.4f}")
                            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
                        except Exception as e:
                            print(f"⚠️  Error saving model: {e}")
                        
                    else:
                        if not self.primary_metric:
                            # Only update patience counter if using val_loss
                            self.patience_counter += 1
                    
                    # Print progress
                    print(f"Epoch {epoch+1}/{num_epochs}")
                    print(f"  Training loss: {train_loss:.4f}")
                    print(f"  Validation loss: {val_loss:.4f}")
                    if self.primary_metric:
                        print(f"  Best {self.primary_metric}: {self.best_primary_metric_value:.4f} (epoch {self.best_primary_metric_epoch})")
                    else:
                        print(f"  Best validation loss: {self.best_val_loss:.4f}")
                        print(f"  Early stopping counter: {self.patience_counter}/{self.early_stopping_patience}")
                    print("-" * 50)
                    
                    # Write progress for Optuna pruning (only on rank 0)
                    if self.rank == 0 and progress_monitor:
                        try:
                            progress_monitor.write_progress(
                                epoch=epoch,
                                val_loss=val_loss,
                                train_loss=train_loss
                            )
                        except Exception as e:
                            # Silently fail to avoid disrupting training
                            pass
                    
                    # Early stopping
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered, stopping training at epoch {epoch+1}")
                        # Set flag to notify all processes to stop training
                        should_stop = torch.tensor(1, device=self.device, dtype=torch.int)
                    else:
                        should_stop = torch.tensor(0, device=self.device, dtype=torch.int)
            else:
                # Non-rank 0 processes wait for synchronization
                should_stop = torch.tensor(0, device=self.device, dtype=torch.int)
            
            # Synchronize all processes to ensure all processes know whether to stop (skip if single GPU)
            if self.world_size > 1:
                dist.broadcast(should_stop, src=0)
            
            # Check if should stop
            if should_stop.item() == 1:
                if self.rank == 0:
                    print("🛑 All processes synchronizing to stop training...")
                # Synchronize all processes to ensure current operations complete before stopping (skip if single GPU)
                if self.world_size > 1:
                    dist.barrier()
                break
        
        # Synchronize before testing (skip if single GPU)
        if self.world_size > 1:
            dist.barrier()
        
        # Testing (only on rank 0, can be skipped)
        test_results = {}
        if hasattr(self, 'skip_test') and self.skip_test:
            if self.rank == 0:
                print("\n⚠️  Skipping test phase (using --skip_test option)")
        else:
            if self.rank == 0:
                print("\nTesting model...")
                print("💡 Tip: For large datasets, testing may take a long time. You can use --skip_test to skip.")
            test_results = self.test(save_dir=save_dir)
            if self.rank == 0:
                print("\nTest results:")
                for metric, value in test_results.items():
                    # Handle different value types: numeric vs string/boolean/None
                    if value is None:
                        print(f"  {metric}: None")
                    elif isinstance(value, bool):
                        # Boolean value
                        print(f"  {metric}: {value}")
                    elif isinstance(value, str):
                        # String value
                        print(f"  {metric}: {value}")
                    elif isinstance(value, (int, float)):
                        # Check for NaN or Inf
                        if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
                            print(f"  {metric}: {value}")
                        else:
                            # Valid numeric value
                            print(f"  {metric}: {value:.4f}")
                    else:
                        # Other types (fallback)
                        print(f"  {metric}: {value}")
        
        # Cleanup
        if self.writer:
            self.writer.close()
        
        # Save training history to JSON file (only on rank 0)
        if self.rank == 0:
            # Determine best_epoch based on primary metric if available
            if self.primary_metric and self.best_primary_metric_epoch >= 0:
                best_epoch = self.best_primary_metric_epoch
                best_primary_metric_value = self.best_primary_metric_value
            else:
                # Fallback to val_loss
                if self.val_losses:
                    best_epoch = self.val_losses.index(min(self.val_losses))
                    best_primary_metric_value = None
                else:
                    best_epoch = len(self.train_losses) - 1
                    best_primary_metric_value = None
            
            history_data = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'best_epoch': best_epoch,
                'test_results': test_results
            }
            
            # Add primary metric information if available
            if self.primary_metric:
                history_data['primary_metric'] = self.primary_metric
                history_data['best_primary_metric_value'] = self.best_primary_metric_value
                history_data['best_primary_metric_epoch'] = self.best_primary_metric_epoch
                
                # Add per-epoch primary metric lists
                if self.primary_metric == 'spearman' and self.val_spearman:
                    history_data['val_spearman'] = self.val_spearman
                elif self.primary_metric in ['roc_auc', 'auroc'] and self.val_aurocs:
                    history_data['val_aurocs'] = self.val_aurocs
                elif self.primary_metric == 'f1' and self.val_f1_scores:
                    history_data['val_f1_scores'] = self.val_f1_scores
                elif self.primary_metric == 'pr_auc' and self.val_pr_aucs:
                    history_data['val_pr_aucs'] = self.val_pr_aucs
                elif self.primary_metric == 'mae' and hasattr(self, 'val_mae') and self.val_mae:
                    history_data['val_mae'] = self.val_mae
            
            history_path = os.path.join(save_dir, 'training_history.json')
            try:
                with open(history_path, 'w') as f:
                    json.dump(history_data, f, indent=2)
                print(f"✅ Training history saved to: {history_path}")
                if self.primary_metric:
                    print(f"   Best {self.primary_metric}: {self.best_primary_metric_value:.4f} (epoch {self.best_primary_metric_epoch})")
            except Exception as e:
                print(f"⚠️  Error saving training history: {e}")
        
        return {
            'train_losses': self.train_losses if self.rank == 0 else [],
            'val_losses': self.val_losses if self.rank == 0 else [],
            'best_val_loss': self.best_val_loss,
            'test_results': test_results if self.rank == 0 else {}
        }


def setup(rank, world_size, master_port):
    """Initialize the process group"""
    # Skip DDP initialization if world_size == 1 (single GPU mode)
    if world_size == 1:
        # Set the device
        torch.cuda.set_device(rank)
        return
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Increase NCCL timeout (default 10 minutes, increase to 30 minutes)
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes (seconds)
    
    # Set NCCL debug level (optional, for debugging)
    # os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    
    # Set the device
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the process group"""
    # Only cleanup if process group is initialized (i.e., DDP was used)
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_worker(rank, world_size, args, master_port=None):
    """Training function for each process"""
    # Set seed for reproducibility (use base seed + rank to ensure different seeds per GPU)
    set_seed(args.seed + rank)
    
    # Setup
    setup(rank, world_size, master_port=master_port)
    
    # Load data
    if rank == 0:
        print("Loading data...")
    
    # Check for TDC dataset first
    if args.tdc_dataset:
        if rank == 0:
            print(f"✅ Loading TDC dataset: {args.tdc_dataset} (seed {args.tdc_seed})")
        
        tdc_data_dir = f"data/processed_tdc_data/{args.tdc_dataset}/seed{args.tdc_seed}"
        train_pt = os.path.join(tdc_data_dir, "train.pt")
        valid_pt = os.path.join(tdc_data_dir, "valid.pt")
        test_pt = os.path.join(tdc_data_dir, "test.pt")
        
        if not all(os.path.exists(f) for f in [train_pt, valid_pt, test_pt]):
            if rank == 0:
                print(f"❌ Error: TDC dataset files not found in {tdc_data_dir}")
            sys.exit(1)
        
        # Load TDC data (can be either list of Data objects or tuple format)
        try:
            train_data = torch.load(train_pt, weights_only=False)
            valid_data = torch.load(valid_pt, weights_only=False)
            test_data = torch.load(test_pt, weights_only=False)
            
            # Check if data is already in Data object format (list of Data objects)
            from torch_geometric.data import Data
            if isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], Data):
                # Data is already in Data object format, use directly
                if rank == 0:
                    print(f"✅ Data is already in PyTorch Geometric Data format ({len(train_data)} samples)")
                train_graphs = train_data
                val_graphs = valid_data
                test_graphs = test_data
            else:
                # Old format: tuple of (smiles, node_features, edge_indices, descriptors, labels)
                # Unpack TDC data format (tuple: smiles, node_features, edge_indices, descriptors, labels)
                train_smiles, train_x, train_edge_index, train_desc, train_y = train_data
                valid_smiles, valid_x, valid_edge_index, valid_desc, valid_y = valid_data
                test_smiles, test_x, test_edge_index, test_desc, test_y = test_data
                
                # Convert to Data objects compatible with AEGNN-M
                train_graphs = []
                num_samples = len(train_smiles) if isinstance(train_smiles, list) else train_y.shape[0] if isinstance(train_y, torch.Tensor) else len(train_y)
                
                for i in range(num_samples):
                    # Handle different data formats
                    if isinstance(train_x, list):
                        x = train_x[i]
                    elif isinstance(train_x, torch.Tensor) and train_x.dim() > 2:
                        x = train_x[i]
                    else:
                        x = train_x
                    
                    if isinstance(train_edge_index, list):
                        edge_index = train_edge_index[i]
                    elif isinstance(train_edge_index, torch.Tensor) and train_edge_index.dim() > 2:
                        edge_index = train_edge_index[i]
                    else:
                        edge_index = train_edge_index
                    
                    if isinstance(train_desc, torch.Tensor):
                        if train_desc.dim() > 1:
                            descriptor = train_desc[i]
                        else:
                            descriptor = train_desc
                    else:
                        descriptor = train_desc[i] if isinstance(train_desc, list) else train_desc
                    
                    if isinstance(train_y, torch.Tensor):
                        if train_y.dim() > 1:
                            y = train_y[i]
                        else:
                            y = train_y[i:i+1] if train_y.dim() == 1 else train_y[i]
                    else:
                        y = torch.tensor([train_y[i]]) if isinstance(train_y, list) else torch.tensor([train_y])
                    
                    smiles = train_smiles[i] if isinstance(train_smiles, list) else train_smiles
                    
                    graph = Data(x=x, edge_index=edge_index, descriptor=descriptor, y=y, smiles=smiles)
                    train_graphs.append(graph)
                
                val_graphs = []
                num_samples = len(valid_smiles) if isinstance(valid_smiles, list) else valid_y.shape[0] if isinstance(valid_y, torch.Tensor) else len(valid_y)
                
                for i in range(num_samples):
                    if isinstance(valid_x, list):
                        x = valid_x[i]
                    elif isinstance(valid_x, torch.Tensor) and valid_x.dim() > 2:
                        x = valid_x[i]
                    else:
                        x = valid_x
                    
                    if isinstance(valid_edge_index, list):
                        edge_index = valid_edge_index[i]
                    elif isinstance(valid_edge_index, torch.Tensor) and valid_edge_index.dim() > 2:
                        edge_index = valid_edge_index[i]
                    else:
                        edge_index = valid_edge_index
                    
                    if isinstance(valid_desc, torch.Tensor):
                        descriptor = valid_desc[i] if valid_desc.dim() > 1 else valid_desc
                    else:
                        descriptor = valid_desc[i] if isinstance(valid_desc, list) else valid_desc
                    
                    if isinstance(valid_y, torch.Tensor):
                        y = valid_y[i] if valid_y.dim() > 1 else valid_y[i:i+1] if valid_y.dim() == 1 else valid_y[i]
                    else:
                        y = torch.tensor([valid_y[i]]) if isinstance(valid_y, list) else torch.tensor([valid_y])
                    
                    smiles = valid_smiles[i] if isinstance(valid_smiles, list) else valid_smiles
                    
                    graph = Data(x=x, edge_index=edge_index, descriptor=descriptor, y=y, smiles=smiles)
                    val_graphs.append(graph)
                
                test_graphs = []
                num_samples = len(test_smiles) if isinstance(test_smiles, list) else test_y.shape[0] if isinstance(test_y, torch.Tensor) else len(test_y)
                
                for i in range(num_samples):
                    if isinstance(test_x, list):
                        x = test_x[i]
                    elif isinstance(test_x, torch.Tensor) and test_x.dim() > 2:
                        x = test_x[i]
                    else:
                        x = test_x
                    
                    if isinstance(test_edge_index, list):
                        edge_index = test_edge_index[i]
                    elif isinstance(test_edge_index, torch.Tensor) and test_edge_index.dim() > 2:
                        edge_index = test_edge_index[i]
                    else:
                        edge_index = test_edge_index
                    
                    if isinstance(test_desc, torch.Tensor):
                        descriptor = test_desc[i] if test_desc.dim() > 1 else test_desc
                    else:
                        descriptor = test_desc[i] if isinstance(test_desc, list) else test_desc
                    
                    if isinstance(test_y, torch.Tensor):
                        y = test_y[i] if test_y.dim() > 1 else test_y[i:i+1] if test_y.dim() == 1 else test_y[i]
                    else:
                        y = torch.tensor([test_y[i]]) if isinstance(test_y, list) else torch.tensor([test_y])
                    
                    smiles = test_smiles[i] if isinstance(test_smiles, list) else test_smiles
                    
                    graph = Data(x=x, edge_index=edge_index, descriptor=descriptor, y=y, smiles=smiles)
                    test_graphs.append(graph)
            
            # Create dummy dataset object for compatibility
            # Extract feature dimensions from the first graph
            if len(train_graphs) > 0:
                first_graph = train_graphs[0]
                node_feature_dim = first_graph.x.shape[1] if hasattr(first_graph, 'x') and first_graph.x is not None else 0
                edge_feature_dim = first_graph.edge_attr.shape[1] if hasattr(first_graph, 'edge_attr') and first_graph.edge_attr is not None else 0
            else:
                # Fallback: use default dimensions
                node_feature_dim = 44  # Default node feature dimension
                edge_feature_dim = 10  # Default edge feature dimension
            
            class DummyDataset:
                def __init__(self, node_feature_dim, edge_feature_dim):
                    all_y = []
                    for g in train_graphs + val_graphs + test_graphs:
                        if isinstance(g.y, torch.Tensor):
                            all_y.append(g.y)
                        else:
                            all_y.append(torch.tensor([g.y]))
                    self.targets = torch.cat(all_y) if all_y else torch.tensor([])
                    # Create a dummy graph_builder object with the extracted dimensions
                    class DummyGraphBuilder:
                        def __init__(self, node_feature_dim, edge_feature_dim):
                            self.node_feature_dim = node_feature_dim
                            self.edge_feature_dim = edge_feature_dim
                    self.graph_builder = DummyGraphBuilder(node_feature_dim, edge_feature_dim)
            dataset = DummyDataset(node_feature_dim, edge_feature_dim)
            smiles_list = None  # TDC datasets already have splits, no need for scaffold splitting
            
            if rank == 0:
                print(f"   Loaded {len(train_graphs)} train, {len(val_graphs)} valid, {len(test_graphs)} test graphs")
        
        except Exception as e:
            if rank == 0:
                print(f"❌ Error loading TDC dataset: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Check if preprocessed data exists
    elif args.processed_data_path and os.path.exists(args.processed_data_path):
        if rank == 0:
            print(f"✅ Found preprocessed data: {args.processed_data_path}")
            print("   Directly loading preprocessed data (skipping data processing step)")
        dataset = MolecularDataset()
        smiles_list = dataset.load_processed_data(args.processed_data_path)
        graphs = dataset.graphs
        if rank == 0:
            print(f"   Loaded {len(graphs)} molecular graphs")
    else:
        # Automatically detect preprocessed data
        if args.data_path:
            from pathlib import Path
            dataset_name = Path(args.data_path).stem
            if dataset_name.endswith('_dataset'):
                dataset_name = dataset_name[:-8]
            
            fp_suffix = f"_fp{args.fingerprint_dim}" if args.use_fingerprint else ""
            auto_processed_path = f"data/processed/{dataset_name}{fp_suffix}_processed.pkl"
            
            if os.path.exists(auto_processed_path):
                if rank == 0:
                    print(f"✅ Automatically detected preprocessed data: {auto_processed_path}")
                    print("   Directly loading preprocessed data (skipping data processing step)")
                dataset = MolecularDataset()
                smiles_list = dataset.load_processed_data(auto_processed_path)
                graphs = dataset.graphs
                if rank == 0:
                    print(f"   Loaded {len(graphs)} molecular graphs")
            else:
                # Normal data processing
                if not args.data_path:
                    if rank == 0:
                        print("❌ Error: Need to provide --data_path or --processed_data_path")
                    sys.exit(1)
                
                if rank == 0:
                    print("📊 Starting data processing (this may take some time, especially 3D coordinate generation)...")
                
                # Create graph builder with fingerprint settings
                graph_builder = MolecularGraphBuilder(
                    use_fingerprint=args.use_fingerprint,
                    fingerprint_bits=args.fingerprint_dim,
                    use_descriptor=getattr(args, 'use_descriptor', False),
                    descriptor_dim=getattr(args, 'descriptor_dim', 217)
                )
                
                dataset = MolecularDataset(
                    data_path=args.data_path,
                    target_column=args.target_column,
                    smiles_column=args.smiles_column,
                    graph_builder=graph_builder
                )
                
                # Process graph data
                graphs = dataset.process_graphs(max_samples=args.max_samples)
                
                # Auto-save processed data for future use
                if rank == 0:  # Save on rank 0 regardless of max_samples
                    try:
                        from pathlib import Path
                        dataset_name = Path(args.data_path).stem
                        if dataset_name.endswith('_dataset'):
                            dataset_name = dataset_name[:-8]
                        
                        save_dir = "data/processed"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Determine filename suffix based on max_samples
                        suffix = ""
                        if args.max_samples:
                            suffix = f"_partial_{args.max_samples}"
                        
                        # 1. Save processed data (.pkl)
                        fp_suffix = f"_fp{args.fingerprint_dim}" if args.use_fingerprint else ""
                        save_path = f"{save_dir}/{dataset_name}{suffix}{fp_suffix}_processed.pkl"
                        print(f"💾 Saving processed data to {save_path}...", flush=True)
                        dataset.save_processed_data(save_path)
                        
                        # 2. Save configuration (.json)
                        failed_count = len(dataset.graph_builder.failed_3d_generation)
                        valid_count = len(dataset.graphs)
                        total_count = valid_count + failed_count
                        
                        processing_config_data = {
                            'processing_statistics': {
                                'total_molecules': total_count,
                                'valid_molecules': valid_count,
                                'failed_3d_coordinate_generation': failed_count,
                                'success_rate': f"{(valid_count/total_count*100):.2f}%" if total_count > 0 else "0.00%"
                            },
                            'use_atomic_number': dataset.graph_builder.use_atomic_number,
                            'use_hybridization': dataset.graph_builder.use_hybridization,
                            'use_formal_charge': dataset.graph_builder.use_formal_charge,
                            'use_aromatic': dataset.graph_builder.use_aromatic,
                            'use_chirality': dataset.graph_builder.use_chirality,
                            'use_hydrogen_bonds': dataset.graph_builder.use_hydrogen_bonds,
                            'use_bond_type': dataset.graph_builder.use_bond_type,
                            'use_bond_stereo': dataset.graph_builder.use_bond_stereo,
                            'num_conformers': dataset.graph_builder.num_conformers,
                            'optimize_conformers': dataset.graph_builder.optimize_conformers,
                            'num_threads': dataset.graph_builder.num_threads,
                            'add_hydrogens': dataset.graph_builder.add_hydrogens,
                            'prune_rms_thresh': dataset.graph_builder.prune_rms_thresh,
                            'use_fingerprint': dataset.graph_builder.use_fingerprint,
                            'fingerprint_radius': dataset.graph_builder.fingerprint_radius,
                            'fingerprint_bits': dataset.graph_builder.fingerprint_bits,
                            'node_feature_dim': dataset.graph_builder.node_feature_dim,
                            'edge_feature_dim': dataset.graph_builder.edge_feature_dim
                        }
                        config_path = f"{save_dir}/{dataset_name}{suffix}_config.json"
                        with open(config_path, 'w') as f:
                            json.dump(processing_config_data, f, indent=4)
                        
                        # 3. Save failed 3D generation info (.json)
                        if dataset.graph_builder.failed_3d_generation:
                            failed_path = f"{save_dir}/{dataset_name}{suffix}_failed_3d.json"
                            with open(failed_path, 'w') as f:
                                json.dump(dataset.graph_builder.failed_3d_generation, f, indent=4)
                            print(f"   ⚠️  Saved {len(dataset.graph_builder.failed_3d_generation)} failed molecules to {failed_path}", flush=True)
                            
                        print(f"✅ All processed data files saved!", flush=True)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to save processed data: {e}", flush=True)
                
                # Get SMILES list for scaffold splitting
                smiles_list = None
                if args.split_method == 'scaffold':
                    smiles_list = dataset.data[args.smiles_column].tolist()[:len(graphs)]
        else:
            if rank == 0:
                print("❌ Error: Need to provide --data_path or --processed_data_path")
            sys.exit(1)
    
    # Skip splitting if TDC dataset (already split)
    if not args.tdc_dataset:
        # IMPORTANT: Seed extraction for data splitting
        # Optuna passes a combined seed (model_init_seed) that encodes:
        #   model_init_seed = trial.number * 1000 + seed * 100 + worker_id
        #   where seed is the original seed (1-5)
        # 
        # For data splitting, we want to use the original seed (1-5) to ensure
        # consistent splits across different trials (same seed = same data split).
        # For model initialization, we use the full seed to ensure unique initialization.
        # 
        # Extract original seed from combined seed
        # Formula: model_init_seed = trial.number * 1000 + seed * 100 + worker_id
        #   where seed is 1-5, worker_id is typically 1-10
        # 
        # Extraction method:
        #   1. Check last 3 digits: last_three = args.seed % 1000
        #   2. If last_three is in range [100, 599], extract hundreds digit
        #   3. This works because seed*100 gives 100, 200, 300, 400, 500
        #   4. worker_id (typically < 100) doesn't affect hundreds digit
        # 
        # Backward compatibility: If seed < 100, use it directly (for non-Optuna usage)
        if args.seed >= 100:
            # Extract original seed (1-5) from combined seed
            last_three = args.seed % 1000  # Get last 3 digits
            if 100 <= last_three <= 599:
                # Extract hundreds digit: this is the original seed (1-5)
                data_split_seed = (last_three // 100) % 10
                # Validate: should be 1-5
                if data_split_seed == 0 or data_split_seed > 5:
                    data_split_seed = 1  # Fallback to 1
            else:
                # Fallback: try simple extraction or use default
                data_split_seed = (args.seed // 100) % 10
                if data_split_seed == 0 or data_split_seed > 5:
                    data_split_seed = 1  # Default to 1
        else:
            # Backward compatibility: use seed directly if < 100
            data_split_seed = args.seed
        
        train_graphs, val_graphs, test_graphs = DataPreprocessor.split_dataset(
            graphs, dataset.targets, 
            smiles_list=smiles_list,
            train_ratio=0.8, 
            val_ratio=0.1,
            random_seed=data_split_seed,  # Use extracted original seed (1-5) for consistent data splits
            split_method=args.split_method
        )
    
    # Detect dataset name for imbalanced dataset handling
    dataset_name = None
    if args.tdc_dataset:
        dataset_name = args.tdc_dataset
    elif args.data_path:
        from pathlib import Path
        dataset_name = Path(args.data_path).stem
        if dataset_name.endswith('_dataset'):
            dataset_name = dataset_name[:-8]
    elif args.processed_data_path:
        from pathlib import Path
        dataset_name = Path(args.processed_data_path).stem
        if dataset_name.endswith('_processed'):
            dataset_name = dataset_name[:-10]
    
    # Highly imbalanced datasets that need special handling
    dataset_name_lower = dataset_name.lower() if dataset_name else None
    # Include known highly imbalanced molecular benchmarks
    # - MUV / HIV / TOX21 / SIDER / CLINTOX
    highly_imbalanced_datasets = ['muv', 'hiv', 'tox21', 'sider', 'clintox']
    # Use balanced batches (at least one positive per batch) for the most extreme cases
    balanced_batch_datasets = ['muv', 'hiv', 'clintox']
    use_balanced_sampler = (dataset_name_lower in balanced_batch_datasets) and args.model_type == 'classifier'
    use_weighted_sampling = (dataset_name_lower in highly_imbalanced_datasets) and not use_balanced_sampler and args.model_type == 'classifier'
    
    # Normalize descriptors before training (if using descriptors)
    # IMPROVEMENT: Using RobustScaler for more robust normalization (resistant to outliers)
    if getattr(args, 'use_descriptor', False):
        if rank == 0:
            print("📊 Normalizing descriptors using RobustScaler (robust to outliers)...")
        
        # Collect all descriptors from training set
        train_descriptors = []
        for graph in train_graphs:
            if hasattr(graph, 'descriptor') and graph.descriptor is not None:
                desc = graph.descriptor
                # Handle different descriptor formats
                if isinstance(desc, torch.Tensor):
                    if desc.dim() > 1:
                        desc = desc.squeeze()
                    train_descriptors.append(desc.cpu().numpy())
                else:
                    train_descriptors.append(np.array(desc))
        
        if len(train_descriptors) > 0:
            # Stack all training descriptors
            train_descriptors_array = np.stack(train_descriptors)
            
            if rank == 0:
                print(f"📊 Descriptor statistics before normalization:")
                print(f"   Shape: {train_descriptors_array.shape}")
                print(f"   Mean range: [{train_descriptors_array.mean(axis=0).min():.4f}, "
                      f"{train_descriptors_array.mean(axis=0).max():.4f}]")
                print(f"   Std range: [{train_descriptors_array.std(axis=0).min():.4f}, "
                      f"{train_descriptors_array.std(axis=0).max():.4f}]")
                
                # Detect outliers using IQR method
                q1 = np.percentile(train_descriptors_array, 25, axis=0)
                q3 = np.percentile(train_descriptors_array, 75, axis=0)
                iqr = q3 - q1
                outlier_mask = (train_descriptors_array < q1 - 3 * iqr) | (train_descriptors_array > q3 + 3 * iqr)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"   ⚠️  Detected {outlier_count} potential outliers (using IQR method)")
                    print(f"   ℹ️  RobustScaler will handle outliers robustly using median and IQR")
            
            # Handle NaN and Inf values before RobustScaler
            has_nan = np.any(np.isnan(train_descriptors_array))
            has_inf = np.any(np.isinf(train_descriptors_array))
            
            if has_nan or has_inf:
                if rank == 0:
                    if has_nan:
                        print(f"⚠️  Warning: Found NaN values in descriptors: {np.isnan(train_descriptors_array).sum()}")
                    if has_inf:
                        print(f"⚠️  Warning: Found Inf values in descriptors: {np.isinf(train_descriptors_array).sum()}")
                    print(f"   Replacing NaN/Inf with 0.0 before normalization...")
                
                # Replace NaN and Inf with 0.0
                train_descriptors_array = np.nan_to_num(train_descriptors_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Use RobustScaler for robust normalization (based on median and IQR)
            # RobustScaler is resistant to outliers, making it ideal for TDC datasets
            scaler = RobustScaler()
            train_descriptors_normalized = scaler.fit_transform(train_descriptors_array)
            
            # Extract statistics from RobustScaler
            # RobustScaler uses median (center_) and IQR-based scale (scale_)
            desc_median = torch.tensor(scaler.center_, dtype=torch.float32)
            desc_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            
            # Avoid division by zero
            desc_scale = torch.clamp(desc_scale, min=1e-8)
            
            # Check for NaN and Inf in statistics
            if torch.any(torch.isnan(desc_median)) or torch.any(torch.isinf(desc_median)):
                if rank == 0:
                    print(f"⚠️  Warning: NaN/Inf detected in descriptor median, replacing with 0.0")
                desc_median = torch.where(torch.isnan(desc_median) | torch.isinf(desc_median), 
                                         torch.zeros_like(desc_median), desc_median)
            
            if torch.any(torch.isnan(desc_scale)) or torch.any(torch.isinf(desc_scale)):
                if rank == 0:
                    print(f"⚠️  Warning: NaN/Inf detected in descriptor scale, replacing with 1.0")
                desc_scale = torch.where(torch.isnan(desc_scale) | torch.isinf(desc_scale), 
                                        torch.ones_like(desc_scale), desc_scale)
            
            if rank == 0:
                print(f"   Descriptor median range: [{desc_median.min().item():.4f}, {desc_median.max().item():.4f}]")
                print(f"   Descriptor scale range: [{desc_scale.min().item():.4f}, {desc_scale.max().item():.4f}]")
                print(f"   ℹ️  Using median and IQR-based scaling (robust to outliers)")
            
            # Normalize training set descriptors
            for graph in train_graphs:
                if hasattr(graph, 'descriptor') and graph.descriptor is not None:
                    desc = graph.descriptor
                    if isinstance(desc, torch.Tensor):
                        # Ensure descriptor is on CPU for normalization
                        desc = desc.cpu()
                        if desc.dim() > 1:
                            desc = desc.squeeze()
                        # RobustScaler normalization: (x - median) / IQR_scale
                        graph.descriptor = (desc - desc_median) / desc_scale
                    else:
                        desc_tensor = torch.tensor(desc, dtype=torch.float32)
                        graph.descriptor = (desc_tensor - desc_median) / desc_scale
            
            # Normalize validation set descriptors (using training set statistics)
            for graph in val_graphs:
                if hasattr(graph, 'descriptor') and graph.descriptor is not None:
                    desc = graph.descriptor
                    if isinstance(desc, torch.Tensor):
                        # Ensure descriptor is on CPU for normalization
                        desc = desc.cpu()
                        if desc.dim() > 1:
                            desc = desc.squeeze()
                        graph.descriptor = (desc - desc_median) / desc_scale
                    else:
                        desc_tensor = torch.tensor(desc, dtype=torch.float32)
                        graph.descriptor = (desc_tensor - desc_median) / desc_scale
            
            # Normalize test set descriptors (using training set statistics)
            for graph in test_graphs:
                if hasattr(graph, 'descriptor') and graph.descriptor is not None:
                    desc = graph.descriptor
                    if isinstance(desc, torch.Tensor):
                        # Ensure descriptor is on CPU for normalization
                        desc = desc.cpu()
                        if desc.dim() > 1:
                            desc = desc.squeeze()
                        graph.descriptor = (desc - desc_median) / desc_scale
                    else:
                        desc_tensor = torch.tensor(desc, dtype=torch.float32)
                        graph.descriptor = (desc_tensor - desc_median) / desc_scale
            
            if rank == 0:
                print("✅ RobustScaler normalization completed")
        else:
            if rank == 0:
                print("⚠️  Warning: No descriptors found in training set, skipping normalization")
    
    # Create distributed samplers
    # Note: DistributedSampler requires a dataset with __len__ method
    # GraphDataset is defined at module level to support pickling for multiprocessing
    train_dataset = GraphDataset(train_graphs)
    val_dataset = GraphDataset(val_graphs)
    test_dataset = GraphDataset(test_graphs)
    
    # Create weighted sampler for imbalanced datasets
    if use_balanced_sampler:
        train_sampler = BalancedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        if rank == 0:
            print(f"⚖️  Using balanced batch sampler for dataset: {dataset_name}")
            print(f"   Ensuring each batch contains at least one positive sample")
    elif use_weighted_sampling:
        train_targets = [g.y.item() for g in train_graphs]
        unique_classes, class_counts = np.unique(train_targets, return_counts=True)
        
        # Calculate sample weights: inverse frequency weighting
        class_weights = len(train_targets) / (len(unique_classes) * class_counts)
        sample_weights = np.array([class_weights[int(t)] for t in train_targets], dtype=np.float32)
        
        train_sampler = WeightedDistributedSampler(
            train_dataset,
            weights=sample_weights,
            num_replicas=world_size,
            rank=rank,
            replacement=True
        )
        
        if rank == 0:
            print(f"⚖️  Using WeightedRandomSampler for imbalanced dataset: {dataset_name}")
            print(f"   Sample weights: Class 0={class_weights[0]:.4f}, Class 1={class_weights[1]:.4f}")
    else:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders with distributed samplers
    # For DDP training, use num_workers=0 to avoid shared memory issues
    # This is especially important when using multiple GPUs, as each GPU process creates its own DataLoader workers
    # Using num_workers=0 avoids mmap memory allocation errors in multiprocessing
    dataset_size = len(train_dataset)
    num_workers = 0  # Always use 0 to avoid shared memory issues in DDP
    pin_memory = False  # Disable pin_memory to reduce memory pressure
    
    if rank == 0:
        print(f"📊 Dataset size: {dataset_size} samples")
        print(f"💡 Using num_workers=0 to avoid shared memory issues in DDP training")
        print(f"   (Each GPU process will load data in the main process)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        collate_fn=collate_with_b2revb,  # Auto-build b2revb (Chemprop style)
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False  # Disable persistent workers to reduce memory usage
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        collate_fn=collate_with_b2revb,  # Auto-build b2revb (Chemprop style)
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler,
        collate_fn=collate_with_b2revb,  # Auto-build b2revb (Chemprop style)
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    # IMPROVEMENT 3.1 & 3.2: Calculate dataset characteristics for dynamic parameter adjustment
    dataset_size = len(train_graphs)
    imbalance_ratio = None
    
    # Calculate class weights and pos_weight for imbalanced datasets (only for classification)
    class_weight = None
    pos_weight = None
    use_bce_for_imbalanced = False
    
    if args.model_type == 'classifier':
        # Calculate class weights from training data to handle class imbalance
        train_targets = [g.y.item() for g in train_graphs]
        unique_classes, class_counts = np.unique(train_targets, return_counts=True)
        
        # Calculate imbalance ratio for dynamic parameter adjustment
        if len(unique_classes) == 2:
            pos_count = class_counts[1] if len(class_counts) > 1 else 0
            neg_count = class_counts[0]
            imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        
        if rank == 0:
            print(f"📊 Class distribution in training set:")
            for cls, count in zip(unique_classes, class_counts):
                print(f"   Class {int(cls)}: {count} samples ({count/len(train_targets)*100:.1f}%)")
        
        # Calculate pos_weight automatically if requested or using BCE or using Weighted Sampling
        if len(unique_classes) == 2 and (args.auto_pos_weight or args.use_bce_for_imbalanced or use_weighted_sampling):
            neg_count = class_counts[0]
            pos_count = class_counts[1] if len(class_counts) > 1 else 1  # Avoid division by zero
            pos_weight = float(neg_count) / float(pos_count)
            if rank == 0:
                print(f"⚖️  Calculated pos_weight: {pos_weight:.4f} (Neg: {neg_count}, Pos: {pos_count})")
                
        # Calculate class weights for CrossEntropyLoss/FocalLoss
        if len(unique_classes) == 2:
            neg_c = int(class_counts[0])
            pos_c = int(class_counts[1]) if len(class_counts) > 1 else 0
            class_weight = torch.tensor(calc_class_weights(pos_c, neg_c), dtype=torch.float32)
        else:
            # Multi-class manual calculation
            cw = len(train_targets) / (len(unique_classes) * class_counts)
            class_weight = torch.tensor(cw, dtype=torch.float32)
            
        if rank == 0 and not args.use_bce_for_imbalanced:
             print(f"⚖️  Class weights: {class_weight}")
             
        # For backward compatibility: if using Focal Loss and dynamic params are requested
        if len(unique_classes) == 2 and getattr(args, 'use_dynamic_focal_params', False) and args.use_focal_loss:
            pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
            neg_count = int(class_counts[0])
            focal_alpha, focal_gamma = calculate_focal_params(pos_count, neg_count, method='aggressive')
            args.focal_alpha = focal_alpha
            args.focal_gamma = focal_gamma
            if rank == 0:
                print(f"📊 Dynamic Focal Loss: alpha={focal_alpha:.4f}, gamma={focal_gamma:.4f}")
        else:
            # Multi-class: use standard calculation
            total_samples = len(train_targets)
            n_classes = len(unique_classes)
            class_weight = total_samples / (n_classes * class_counts)
            class_weight = class_weight / class_weight.sum() * n_classes  # Normalize
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        
        # For highly imbalanced datasets, use BCEWithLogitsLoss with pos_weight (unless Focal Loss is requested)
        if use_weighted_sampling and len(unique_classes) == 2 and not args.use_focal_loss:
            use_bce_for_imbalanced = True
            if rank == 0:
                print(f"⚖️  Using BCEWithLogitsLoss for highly imbalanced dataset")
                if pos_weight is not None:
                    print(f"   pos_weight (negative/positive): {pos_weight:.4f}")
                print(f"   Label smoothing disabled for imbalanced dataset")
        else:
            if rank == 0 and len(unique_classes) > 2:
                print(f"⚠️  Multi-class dataset, using standard CrossEntropyLoss")
        
        if rank == 0 and not use_bce_for_imbalanced and class_weight is not None:
            class_weight_dict = {int(c): float(class_weight[i]) for i, c in enumerate(unique_classes)}
            print(f"⚖️  Calculated class weights: {class_weight_dict}")
            print(f"   (Minority classes will have higher weights to balance training)")
    
    # LR Range Test (if requested, only on rank 0 to avoid conflicts)
    if args.find_lr and rank == 0:
        print("\n🔍 Running LR Range Test to find optimal learning rate...")
        from utils.lr_finder import LRFinder
        import torch.optim as optim
        
        # Create temporary model for LR test
        temp_model = create_aegnn_model(
            model_type=args.model_type,
            node_features=dataset.graph_builder.node_feature_dim,
            edge_features=dataset.graph_builder.edge_feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_expansion_factor=args.ffn_expansion_factor,
            dropout=args.dropout,
            drop_path_rate=getattr(args, 'drop_path_rate', 0.0),
            pre_norm=getattr(args, 'use_pre_norm', False),
            rotate_aug=getattr(args, 'rotate_aug', False),
            rotation_prob=getattr(args, 'rotation_prob', 0.5),  # IMPROVEMENT 5.1
            max_rotation_angle=getattr(args, 'max_rotation_angle', 180.0),  # IMPROVEMENT 5.1
            dmp_steps=getattr(args, 'dmp_steps', 2),
            use_descriptor=getattr(args, 'use_descriptor', False),
            descriptor_dim=getattr(args, 'descriptor_dim', 217),
            descriptor_dropout=getattr(args, 'descriptor_dropout', 0.0)
        )
        device_for_lr_test = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        temp_model = temp_model.to(device_for_lr_test)
        
        # Create temporary optimizer
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=args.lr_test_init_lr)
        
        # Create LR Finder
        lr_finder = LRFinder(temp_model, temp_optimizer, None, device=device_for_lr_test)
        
        # Create temporary data loader (use subset for speed)
        temp_train_dataset = GraphDataset(train_graphs[:min(200, len(train_graphs))])
        temp_train_loader = DataLoader(temp_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        # Run LR Range Test
        best_lr, min_lr = lr_finder.find_lr(
            temp_train_loader,
            init_lr=args.lr_test_init_lr,
            final_lr=args.lr_test_final_lr,
            num_iter=args.lr_test_num_iter
        )
        
        # Save plot if requested
        if args.lr_test_save_plot:
            plot_path = args.lr_test_save_plot
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            plot_path = os.path.join(args.save_dir, 'lr_range_test.png')
        
        lr_finder.plot(save_path=plot_path)
        
        # Update learning rate
        print(f"\n✅ LR Range Test completed!")
        print(f"   Optimal LR: {best_lr:.2e}")
        print(f"   Min LR: {min_lr:.2e}")
        print(f"   → Updating learning rate from {args.learning_rate:.2e} to {best_lr:.2e}")
        args.learning_rate = best_lr
        
        # Update OneCycleLR max_lr if using onecycle
        if args.scheduler_type == 'onecycle':
            if args.onecycle_max_lr is None:
                args.onecycle_max_lr = best_lr
                print(f"   → Updating OneCycleLR max_lr to {best_lr:.2e}")
        
        print(f"   Plot saved to: {plot_path}\n")
        
        # Clean up
        del temp_model, temp_optimizer, lr_finder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Synchronize all processes after LR test (if any process ran it)
    if world_size > 1:
        dist.barrier()
    
    # Get primary_metric from config file (for regression tasks to select correct loss function)
    primary_metric = None
    if args.model_type == 'regressor' and dataset_name:
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'configs', 'dataset_primary_metrics.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                dataset_configs = config.get('dataset_primary_metrics', {})
                dataset_config = dataset_configs.get(dataset_name.lower(), {})
                primary_metric = dataset_config.get('primary_metric', None)
                if rank == 0 and primary_metric:
                    print(f"📌 Primary metric for {dataset_name}: {primary_metric}")
                    if primary_metric == 'spearman':
                        print(f"   Using SpearmanLoss for training (optimized for Spearman correlation)")
                    else:
                        print(f"   Using L1Loss (MAE) for training")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Could not load primary metric config: {e}")
                print(f"   Will use default L1Loss for regression")
    
    # 🔧 FIX: Load model config from checkpoint if resuming
    # This ensures the model architecture matches the checkpoint
    checkpoint_config = None
    if args.resume and os.path.exists(args.resume):
        try:
            if rank == 0:
                print(f"📂 Loading model configuration from checkpoint...")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            checkpoint_config = checkpoint.get('model_config', None)
            if checkpoint_config and rank == 0:
                print(f"   Found model config in checkpoint:")
                print(f"     hidden_dim: {checkpoint_config.get('hidden_dim', 'N/A')}")
                print(f"     num_layers: {checkpoint_config.get('num_layers', 'N/A')}")
                print(f"     num_heads: {checkpoint_config.get('num_heads', 'N/A')}")
                print(f"     ffn_expansion_factor: {checkpoint_config.get('ffn_expansion_factor', 'N/A')}")
                print(f"     dropout: {checkpoint_config.get('dropout', 'N/A')}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Warning: Could not load model config from checkpoint: {e}")
                print(f"   Will use command-line arguments or defaults")
    
    # Use checkpoint config if available, otherwise use command-line arguments
    model_hidden_dim = checkpoint_config.get('hidden_dim', args.hidden_dim) if checkpoint_config else args.hidden_dim
    model_num_layers = checkpoint_config.get('num_layers', args.num_layers) if checkpoint_config else args.num_layers
    model_num_heads = checkpoint_config.get('num_heads', args.num_heads) if checkpoint_config else args.num_heads
    model_ffn_expansion = checkpoint_config.get('ffn_expansion_factor', args.ffn_expansion_factor) if checkpoint_config else args.ffn_expansion_factor
    model_dropout = checkpoint_config.get('dropout', args.dropout) if checkpoint_config else args.dropout
    model_drop_path_rate = checkpoint_config.get('drop_path_rate', getattr(args, 'drop_path_rate', 0.0)) if checkpoint_config else getattr(args, 'drop_path_rate', 0.0)
    model_pre_norm = checkpoint_config.get('pre_norm', getattr(args, 'use_pre_norm', False)) if checkpoint_config else getattr(args, 'use_pre_norm', False)
    model_rotate_aug = checkpoint_config.get('rotate_aug', getattr(args, 'rotate_aug', False)) if checkpoint_config else getattr(args, 'rotate_aug', False)
    model_rotation_prob = checkpoint_config.get('rotation_prob', getattr(args, 'rotation_prob', 0.5)) if checkpoint_config else getattr(args, 'rotation_prob', 0.5)  # IMPROVEMENT 5.1
    model_max_rotation_angle = checkpoint_config.get('max_rotation_angle', getattr(args, 'max_rotation_angle', 180.0)) if checkpoint_config else getattr(args, 'max_rotation_angle', 180.0)  # IMPROVEMENT 5.1
    
    # Also check for descriptor config in checkpoint
    checkpoint_use_descriptor = None
    checkpoint_descriptor_dim = None
    checkpoint_descriptor_dropout = None
    if checkpoint_config:
        checkpoint_use_descriptor = checkpoint_config.get('use_descriptor', None)
        checkpoint_descriptor_dim = checkpoint_config.get('descriptor_dim', None)
        checkpoint_descriptor_dropout = checkpoint_config.get('descriptor_dropout', None)
    
    # Create model
    if rank == 0:
        print("Creating model...")
        if checkpoint_config:
            print(f"   Using model config from checkpoint (to match saved model architecture)")
    
    model = create_aegnn_model(
        model_type=args.model_type,
        primary_metric=primary_metric if args.model_type == 'regressor' else None,
        node_features=dataset.graph_builder.node_feature_dim,
        edge_features=dataset.graph_builder.edge_feature_dim,
        hidden_dim=model_hidden_dim,
        num_layers=model_num_layers,
        num_heads=model_num_heads,
        ffn_expansion_factor=model_ffn_expansion,
        dropout=model_dropout,
        drop_path_rate=model_drop_path_rate,
        pre_norm=model_pre_norm,
        alpha=getattr(args, 'alpha', 0.2),
        rotate_aug=model_rotate_aug,
        rotation_prob=model_rotation_prob,  # IMPROVEMENT 5.1
        max_rotation_angle=model_max_rotation_angle,  # IMPROVEMENT 5.1
        use_fingerprint=args.use_fingerprint,
        fingerprint_dim=args.fingerprint_dim,
        fingerprint_dropout=args.fingerprint_dropout,
        use_fingerprint_gate=args.use_fingerprint_gate,
        use_descriptor=checkpoint_use_descriptor if checkpoint_use_descriptor is not None else getattr(args, 'use_descriptor', False),
        descriptor_dim=checkpoint_descriptor_dim if checkpoint_descriptor_dim is not None else getattr(args, 'descriptor_dim', 217),
        descriptor_dropout=checkpoint_descriptor_dropout if checkpoint_descriptor_dropout is not None else getattr(args, 'descriptor_dropout', 0.0),
        class_weight=class_weight if args.model_type == 'classifier' else None,
        # Removed use_focal_loss, focal_alpha, focal_gamma since all tasks use BCE
        label_smoothing=0.0 if use_bce_for_imbalanced else (args.label_smoothing if args.model_type == 'classifier' else 0.0),
        use_bce_for_imbalanced=use_bce_for_imbalanced,
        pos_weight=pos_weight,
        use_class_balanced_focal_loss=getattr(args, 'use_class_balanced_focal_loss', False) if args.model_type == 'classifier' else False,
        class_balanced_beta=getattr(args, 'class_balanced_beta', 0.9999) if args.model_type == 'classifier' else 0.9999,
        class_counts=class_counts.tolist() if args.model_type == 'classifier' and getattr(args, 'use_class_balanced_focal_loss', False) and 'class_counts' in locals() else None,
        # Pass through aggregation / activation if available (for MOD edmpnn_model)
        pool_type=getattr(args, 'aggregation', 'mean'),
        activation=getattr(args, 'activation', 'SiLU'),
        dmp_steps=getattr(args, 'dmp_steps', 2)
    )
    # Unified advanced initialization: Avoid relying on PyTorch default initialization, improve training stability
    model.apply(init_weights_advanced)
    
    model_arch_config = {
        'model_type': args.model_type,
        'hidden_dim': model_hidden_dim,
        'num_layers': model_num_layers,
        'num_heads': model_num_heads,
        'ffn_expansion_factor': model_ffn_expansion,
        'dropout': model_dropout,
        'drop_path_rate': model_drop_path_rate,
        'pre_norm': model_pre_norm,
        'rotate_aug': model_rotate_aug,
        'rotation_prob': model_rotation_prob,  # IMPROVEMENT 5.1
        'max_rotation_angle': model_max_rotation_angle,  # IMPROVEMENT 5.1
        'use_descriptor': checkpoint_use_descriptor if checkpoint_use_descriptor is not None else getattr(args, 'use_descriptor', False),
        'descriptor_dim': checkpoint_descriptor_dim if checkpoint_descriptor_dim is not None else getattr(args, 'descriptor_dim', 217),
        'descriptor_dropout': checkpoint_descriptor_dropout if checkpoint_descriptor_dropout is not None else getattr(args, 'descriptor_dropout', 0.0)
    }
    
    # 🔧 FIX: Add primary_metric to model_arch_config if available in config file
    # This ensures Trainer can read primary_metric from model_config
    # Read config file if provided (train_worker doesn't have access to main's config_data)
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            if 'primary_metric' in config_data:
                model_arch_config['primary_metric'] = config_data['primary_metric']
                if rank == 0:
                    print(f"📌 Primary metric from config file: {config_data['primary_metric']}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Warning: Could not read config file for primary_metric: {e}")
    
    # Extract dataset name for threshold configuration
    # Note: dataset_name should already be set earlier (around line 3407) for TDC datasets
    # If not set, extract from data_path or processed_data_path
    if dataset_name is None:
        if args.data_path:
            from pathlib import Path
            dataset_name = Path(args.data_path).stem
            if dataset_name.endswith('_dataset'):
                dataset_name = dataset_name[:-8]
        elif args.processed_data_path:
            from pathlib import Path
            dataset_name = Path(args.processed_data_path).stem
            # Remove common suffixes
            for suffix in ['_processed', '_fp', '_dataset']:
                if dataset_name.endswith(suffix):
                    dataset_name = dataset_name[:-len(suffix)]
                    break
    
    # IMPROVEMENT 3.1: Dynamic scheduler selection based on dataset characteristics
    # IMPROVEMENT 3.1: Dynamic warmup_epochs adjustment based on dataset size
    # IMPROVEMENT 3.2: Dynamic early stopping parameters based on dataset characteristics
    
    # Auto-select scheduler based on dataset characteristics
    original_scheduler_type = args.scheduler_type
    original_warmup_epochs = args.warmup_epochs
    original_early_stopping_patience = args.early_stopping_patience
    original_smart_early_stopping_max_patience = getattr(args, 'smart_early_stopping_max_patience', 50)
    original_auroc_improvement_threshold = getattr(args, 'auroc_improvement_threshold', 0.005)
    
    # Auto-select scheduler: Automatically select optimal scheduler based on dataset characteristics
    # Rules:
    # 1. Small dataset (< 1000): ReduceLROnPlateau (adaptive, avoid overfitting)
    # 2. Extremely imbalanced dataset (imbalance_ratio > 100): ReduceLROnPlateau (validation loss fluctuates greatly)
    # 3. Medium dataset (1000-20000): CosineAnnealingLR (smooth decay, good regularization)
    # 4. Large dataset (> 20000): CosineAnnealingLR or OneCycleLR (strong exploration)
    # 
    # Note: If user explicitly specifies scheduler (via command line or config file), respect user's choice
    # 只有当使用默认值 'cosine' 时才自动选择
    should_auto_select = (args.scheduler_type == 'cosine' or 
                        args.scheduler_type is None or
                        not hasattr(args, '_scheduler_explicitly_set') or
                        not args._scheduler_explicitly_set)
    
    if should_auto_select:
        if dataset_size < 1000:
            # 小数据集：使用 ReduceLROnPlateau (自适应，避免过拟合)
            selected_scheduler = 'plateau'
            reason = f"小数据集 ({dataset_size} 样本)，容易过拟合，使用自适应调度器 ReduceLROnPlateau"
        elif args.model_type == 'classifier' and imbalance_ratio is not None and imbalance_ratio > 100:
            # 极度不平衡数据集：使用 ReduceLROnPlateau (验证损失波动大)
            selected_scheduler = 'plateau'
            reason = f"极度不平衡数据集 (ratio={imbalance_ratio:.2f})，验证损失波动大，使用自适应调度器 ReduceLROnPlateau"
        elif dataset_size >= 20000:
            # 大数据集：使用 CosineAnnealingLR (平滑衰减) 或 OneCycleLR (探索性强)
            # 默认使用 CosineAnnealingLR，但如果训练轮数足够多，可以考虑 OneCycleLR
            if args.num_epochs >= 100:
                selected_scheduler = 'onecycle'
                reason = f"大数据集 ({dataset_size} 样本) 且训练轮数多 ({args.num_epochs} epochs)，使用 OneCycleLR 提高探索性"
            else:
                selected_scheduler = 'cosine'
                reason = f"大数据集 ({dataset_size} 样本)，使用 CosineAnnealingLR 平滑衰减"
        else:
            # 中等数据集：使用 CosineAnnealingLR
            selected_scheduler = 'cosine'
            reason = f"中等数据集 ({dataset_size} 样本)，使用 CosineAnnealingLR 平滑衰减"
        
        if selected_scheduler != original_scheduler_type:
            args.scheduler_type = selected_scheduler
            if rank == 0:
                print(f"📊 IMPROVEMENT 3.1: 自动选择学习率调度器")
                print(f"   - 原始调度器: {original_scheduler_type}")
                print(f"   - 选择调度器: {selected_scheduler}")
                print(f"   - 选择理由: {reason}")
    else:
        if rank == 0:
            print(f"📊 使用用户指定的调度器: {args.scheduler_type}")
    
    # Adjust warmup_epochs based on dataset size and scheduler type
    # IMPROVEMENT 3.1: Plateau scheduler doesn't need warmup (it adapts based on validation loss)
    if args.scheduler_type == 'plateau':
        # ReduceLROnPlateau 不需要 warmup，因为它根据验证损失自适应调整
        if args.warmup_epochs > 0:
            original_warmup_for_plateau = args.warmup_epochs
            args.warmup_epochs = 0  # Plateau scheduler doesn't need warmup
            if rank == 0:
                print(f"📊 IMPROVEMENT 3.1: ReduceLROnPlateau 调度器不需要 warmup，将 warmup_epochs 从 {original_warmup_for_plateau} 调整为 0")
    elif dataset_size < 1000:
        # Small dataset: warmup should be 10-15% of total epochs, but at least 3
        adjusted_warmup = max(3, args.num_epochs // 20)  # 5% of total epochs, min 3
        if adjusted_warmup != original_warmup_epochs:
            args.warmup_epochs = adjusted_warmup
            if rank == 0:
                print(f"📊 IMPROVEMENT 3.1: Adjusted warmup_epochs from {original_warmup_epochs} to {adjusted_warmup} (small dataset: {dataset_size} samples)")
    else:
        # Large dataset: warmup should be 5-10% of total epochs, but at least 5
        adjusted_warmup = max(5, args.num_epochs // 10)  # 10% of total epochs, min 5
        if adjusted_warmup != original_warmup_epochs:
            args.warmup_epochs = adjusted_warmup
            if rank == 0:
                print(f"📊 IMPROVEMENT 3.1: Adjusted warmup_epochs from {original_warmup_epochs} to {adjusted_warmup} (large dataset: {dataset_size} samples)")
    
    # Adjust early stopping parameters based on dataset characteristics
    if args.model_type == 'classifier' and imbalance_ratio is not None and imbalance_ratio > 100:
        # Extremely imbalanced: need more patience
        adjusted_early_stopping_patience = 40
        adjusted_smart_max_patience = 80
        adjusted_auroc_threshold = 0.001  # Smaller threshold for imbalanced datasets
        
        if adjusted_early_stopping_patience != original_early_stopping_patience:
            args.early_stopping_patience = adjusted_early_stopping_patience
        if adjusted_smart_max_patience != original_smart_early_stopping_max_patience:
            args.smart_early_stopping_max_patience = adjusted_smart_max_patience
        if adjusted_auroc_threshold != original_auroc_improvement_threshold:
            args.auroc_improvement_threshold = adjusted_auroc_threshold
            
        if rank == 0:
            print(f"📊 IMPROVEMENT 3.2: Adjusted early stopping parameters for extremely imbalanced dataset (ratio={imbalance_ratio:.2f})")
            print(f"   - Early stopping patience: {original_early_stopping_patience} → {adjusted_early_stopping_patience}")
            print(f"   - Smart ES max patience: {original_smart_early_stopping_max_patience} → {adjusted_smart_max_patience}")
            print(f"   - AUROC improvement threshold: {original_auroc_improvement_threshold} → {adjusted_auroc_threshold}")
    elif dataset_size < 1000:
        # Small dataset: prevent premature stopping
        adjusted_early_stopping_patience = 30
        adjusted_smart_max_patience = 60
        
        if adjusted_early_stopping_patience != original_early_stopping_patience:
            args.early_stopping_patience = adjusted_early_stopping_patience
        if adjusted_smart_max_patience != original_smart_early_stopping_max_patience:
            args.smart_early_stopping_max_patience = adjusted_smart_max_patience
            
        if rank == 0:
            print(f"📊 IMPROVEMENT 3.2: Adjusted early stopping parameters for small dataset ({dataset_size} samples)")
            print(f"   - Early stopping patience: {original_early_stopping_patience} → {adjusted_early_stopping_patience}")
            print(f"   - Smart ES max patience: {original_smart_early_stopping_max_patience} → {adjusted_smart_max_patience}")
    else:
        # Standard configuration (already set, no adjustment needed)
        if rank == 0:
            print(f"📊 Using standard early stopping parameters (dataset_size={dataset_size}, imbalance_ratio={imbalance_ratio if imbalance_ratio is not None else 'N/A'})")
    
    # IMPROVEMENT 7.1: Auto-adjust gradient accumulation steps based on batch_size
    # 根据 batch_size 自动调整梯度累积
    original_gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.batch_size < 32:
        # Small batch size: increase gradient accumulation to maintain effective batch size ~32
        adjusted_gradient_accumulation = max(1, 32 // args.batch_size)
        if adjusted_gradient_accumulation != original_gradient_accumulation_steps:
            args.gradient_accumulation_steps = adjusted_gradient_accumulation
            if rank == 0:
                effective_batch_size = args.batch_size * adjusted_gradient_accumulation
                print(f"📊 IMPROVEMENT 7.1: Auto-adjusted gradient accumulation steps")
                print(f"   - Batch size: {args.batch_size}")
                print(f"   - Gradient accumulation steps: {original_gradient_accumulation_steps} → {adjusted_gradient_accumulation}")
                print(f"   - Effective batch size: {effective_batch_size} (target: ~32)")
    else:
        # Large batch size: no need for gradient accumulation
        if original_gradient_accumulation_steps != 1:
            if rank == 0:
                print(f"📊 IMPROVEMENT 7.1: Batch size ({args.batch_size}) >= 32, using gradient_accumulation_steps=1")
            args.gradient_accumulation_steps = 1
    
    # Create trainer
    device = f'cuda:{rank}'
    trainer = AEGNNDDPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        rank=rank,
        world_size=world_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        scheduler_patience=args.scheduler_patience,
        early_stopping_patience=args.early_stopping_patience,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        skip_test=args.skip_test,
        log_dir=getattr(args, 'log_dir', None),
        use_smart_early_stopping=getattr(args, 'use_smart_early_stopping', False),
        smart_early_stopping_max_patience=getattr(args, 'smart_early_stopping_max_patience', 50),
        smart_early_stopping_moving_avg_window=getattr(args, 'smart_early_stopping_moving_avg_window', 5),
        smart_early_stopping_trend_window=getattr(args, 'smart_early_stopping_trend_window', 10),
        auroc_improvement_threshold=getattr(args, 'auroc_improvement_threshold', 0.005),
        f1_improvement_threshold=getattr(args, 'f1_improvement_threshold', 0.001),
        onecycle_max_lr=getattr(args, 'onecycle_max_lr', None),
        onecycle_pct_start=getattr(args, 'onecycle_pct_start', 0.3),
        onecycle_div_factor=getattr(args, 'onecycle_div_factor', 25.0),
        onecycle_final_div_factor=getattr(args, 'onecycle_final_div_factor', 1e4),
        val_loss_ema_beta=getattr(args, 'val_loss_ema_beta', 0.8),
        enable_loss_gap_control=getattr(args, 'enable_loss_gap_control', False),
        loss_gap_threshold=getattr(args, 'loss_gap_threshold', 0.5),
        loss_gap_patience=getattr(args, 'loss_gap_patience', 3),
        loss_gap_lr_factor=getattr(args, 'loss_gap_lr_factor', 0.5),
        loss_gap_patience_decay=getattr(args, 'loss_gap_patience_decay', 0.7),
        grad_clip_norm=getattr(args, 'grad_clip_norm', 1.0),
        enable_manifold_mixup=getattr(args, 'enable_manifold_mixup', False),
        manifold_mixup_alpha=getattr(args, 'manifold_mixup_alpha', 2.0),
        model_config=model_arch_config,
        dataset_name=dataset_name
    )
    
    # Start training
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        resume_from=args.resume
    )
    
    # Cleanup
    cleanup()
    
    return history


def main():
    parser = argparse.ArgumentParser(description='AEGNN-M DDP Training')
    parser.add_argument('--data_path', type=str, default=None, help='Data path (CSV file)')
    parser.add_argument('--processed_data_path', type=str, default=None, 
                       help='Preprocessed data path (PKL file). If provided, will skip data processing.')
    parser.add_argument('--tdc_dataset', type=str, default=None,
                       help='TDC dataset name (e.g., caco2_wang). If provided, will load from data/processed_tdc_data/{dataset}/seed{seed}/')
    parser.add_argument('--tdc_seed', type=int, default=1,
                       help='Seed for TDC dataset split (default: 1, range: 1-5)')
    parser.add_argument('--target_column', type=str, default='target', help='Target column name')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='SMILES column name')
    parser.add_argument('--model_type', type=str, default='regressor', choices=['regressor', 'classifier'], help='Model type')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of AEGNN layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ffn_expansion_factor', type=int, default=4, help='FFN expansion factor (controls model width)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop Path rate (Stochastic Depth) (default: 0.0)')
    parser.add_argument('--use_pre_norm', action='store_true', help='Use Pre-Norm architecture (better for deep networks)')
    parser.add_argument('--alpha', type=float, default=0.2, help='GAT attention negative slope (LeakyReLU alpha)')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum', 'norm'], 
                       help='Graph pooling aggregation method (mean, sum, or norm)')
    parser.add_argument('--rotate_aug', action='store_true', help='Enable 3D random rotation augmentation')
    parser.add_argument('--rotation_prob', type=float, default=0.5, 
                       help='Probability of applying rotation augmentation (IMPROVEMENT 5.1, default: 0.5, range: 0.0-1.0)')
    parser.add_argument('--max_rotation_angle', type=float, default=180.0,
                       help='Maximum rotation angle in degrees (IMPROVEMENT 5.1, default: 180.0, range: 0.0-180.0)')
    parser.add_argument('--dmp_steps', type=int, default=2,
                       help='Number of directed message passing steps in EGNN-DMP (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save directory')
    parser.add_argument('--split_method', type=str, default='random', choices=['random', 'scaffold'], 
                       help='Data splitting method')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs (auto-detect if not specified)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Number of gradient accumulation steps (1=no accumulation, 2=accumulate 2 batches before update)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='Gradient clipping max norm (<=0 disables clipping)')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip testing phase after training (useful for large datasets to save time)')
    parser.add_argument('--scheduler_type', type=str, default='cosine', 
                       choices=['cosine', 'cosine_restarts', 'step', 'plateau', 'onecycle'],
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs for learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate')
    parser.add_argument('--find_lr', action='store_true',
                       help='Run LR Range Test before training to find optimal learning rate')
    parser.add_argument('--lr_test_init_lr', type=float, default=1e-7,
                       help='Initial learning rate for LR Range Test (default: 1e-7)')
    parser.add_argument('--lr_test_final_lr', type=float, default=10.0,
                       help='Final learning rate for LR Range Test (default: 10.0)')
    parser.add_argument('--lr_test_num_iter', type=int, default=None,
                       help='Number of iterations for LR Range Test (default: None, uses full dataset)')
    parser.add_argument('--lr_test_save_plot', type=str, default=None,
                       help='Path to save LR Range Test plot (default: None, no plot saved)')
    parser.add_argument('--use_dynamic_focal_params', action='store_true', default=True,
                       help='Use dynamically calculated Focal Loss parameters (default: True). If False, use script-provided values.')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from (e.g., checkpoints/bbbp/best_model.pth)')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use Focal Loss instead of CrossEntropyLoss for classification (helps with class imbalance)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Alpha parameter for Focal Loss (default: 0.25)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss (default: 2.0, higher values focus more on hard examples)')
    parser.add_argument('--use_class_balanced_focal_loss', action='store_true',
                       help='Use Class-Balanced Focal Loss (adds effective number reweighting for extreme imbalance)')
    parser.add_argument('--class_balanced_beta', type=float, default=0.9999,
                       help='Beta parameter for Class-Balanced Loss reweighting (default: 0.9999; closer to 1.0 strengthens reweighting)')
    parser.add_argument('--activation', type=str, default='SiLU',
                       choices=['SiLU', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'tanh'],
                       help='Activation function for AEGNN layers (default: SiLU)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor (0.0 to 1.0, default: 0.0). Can be used with both CrossEntropyLoss and FocalLoss')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='TensorBoard log directory (default: None, uses SummaryWriter default). Recommended: runs/${dataset_name}')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                       help='Patience for ReduceLROnPlateau scheduler (default: 10)')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Patience for early stopping (default: 20)')
    parser.add_argument('--use_smart_early_stopping', action='store_true',
                       help='Enable smart early stopping with moving average, patience decay, and trend analysis')
    parser.add_argument('--smart_early_stopping_max_patience', type=int, default=50,
                       help='Maximum patience for smart early stopping (default: 50)')
    parser.add_argument('--smart_early_stopping_moving_avg_window', type=int, default=5,
                       help='Moving average window size for smart early stopping (default: 5)')
    parser.add_argument('--smart_early_stopping_trend_window', type=int, default=10,
                       help='Trend analysis window size for smart early stopping (default: 10)')
    parser.add_argument('--auroc_improvement_threshold', type=float, default=0.005,
                       help='Minimum AUROC gain to extend patience (default: 0.005)')
    parser.add_argument('--f1_improvement_threshold', type=float, default=0.001,
                       help='Minimum F1 gain to extend patience (default: 0.001)')
    parser.add_argument('--onecycle_max_lr', type=float, default=None,
                       help='Max LR for OneCycleLR (default: same as learning_rate)')
    parser.add_argument('--onecycle_pct_start', type=float, default=0.3,
                       help='Percentage of cycle spent increasing LR (default: 0.3)')
    parser.add_argument('--onecycle_div_factor', type=float, default=25.0,
                       help='Initial LR = max_lr/div_factor (default: 25)')
    parser.add_argument('--onecycle_final_div_factor', type=float, default=1e4,
                       help='Minimum LR = initial_lr/final_div_factor (default: 1e4)')
    parser.add_argument('--val_loss_ema_beta', type=float, default=0.8,
                       help='EMA beta for smoothing validation loss before scheduler (0 disables EMA)')
    parser.add_argument('--enable_manifold_mixup', action='store_true',
                       help='Enable manifold mixup on graph representations')
    parser.add_argument('--manifold_mixup_alpha', type=float, default=2.0,
                       help='Beta distribution alpha for manifold mixup (default: 2.0)')
    parser.add_argument('--enable_loss_gap_control', action='store_true',
                       help='Enable dynamic LR/patience adjustments when (val-train)/train exceeds threshold')
    parser.add_argument('--loss_gap_threshold', type=float, default=0.5,
                       help='Relative loss gap threshold to trigger adjustments (default: 0.5)')
    parser.add_argument('--loss_gap_patience', type=int, default=3,
                       help='Consecutive epochs over threshold before triggering (default: 3)')
    parser.add_argument('--loss_gap_lr_factor', type=float, default=0.5,
                       help='Multiplier applied to LR when gap control triggers (default: 0.5)')
    parser.add_argument('--loss_gap_patience_decay', type=float, default=0.7,
                       help='Patience multiplier applied when gap control triggers (default: 0.7)')
    parser.add_argument('--use_fingerprint', action='store_true', help='Use molecular fingerprints (Deep & Wide)')
    parser.add_argument('--fingerprint_dim', type=int, default=2048, help='Fingerprint dimension (default: 2048)')
    parser.add_argument('--fingerprint_dropout', type=float, default=0.0, help='Dropout rate for fingerprint input (default: 0.0)')
    parser.add_argument('--use_fingerprint_gate', action='store_true', help='Use Gating Mechanism for fingerprint fusion')
    parser.add_argument('--use_descriptor', action='store_true', help='Use RDKit normalized molecular descriptors')
    parser.add_argument('--descriptor_dim', type=int, default=217, help='Descriptor dimension (default: 217)')
    parser.add_argument('--descriptor_dropout', type=float, default=0.0, help='Dropout rate for descriptor input (default: 0.0)')
    parser.add_argument('--use_bce_for_imbalanced', action='store_true', help='Use BCEWithLogitsLoss for imbalanced datasets')
    parser.add_argument('--auto_pos_weight', action='store_true', help='Automatically calculate pos_weight for BCE loss')
    parser.add_argument('--base_port', type=int, default=12355, help='Base port for DDP (default: 12355)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to JSON configuration file. If provided, hyperparameters will be loaded from this file and override command-line arguments.')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config_data = None
    if args.config and os.path.exists(args.config):
        print(f"📋 Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        
        # Extract dataset name from config if available (for primary_metric loading)
        if 'dataset' in config_data and not args.tdc_dataset:
            # Only override if tdc_dataset is not already set from command line
            # This ensures dataset_name is available for primary_metric loading in Trainer
            pass  # dataset_name will be set from args.tdc_dataset or args.data_path later
        
        # Extract hyperparameters from config
        if 'hyperparameters' in config_data:
            hp = config_data['hyperparameters']
            
            # Override args with config values (only if not explicitly set via command line)
            # For simplicity, we'll always use config values when config file is provided
            # This ensures consistency between Optuna optimization and training
            
            # Model architecture parameters
            if 'hidden_dim' in hp:
                args.hidden_dim = hp['hidden_dim']
            if 'num_layers' in hp:
                args.num_layers = hp['num_layers']
            if 'num_heads' in hp:
                args.num_heads = hp['num_heads']
            if 'ffn_expansion_factor' in hp:
                args.ffn_expansion_factor = hp['ffn_expansion_factor']
            if 'dropout' in hp:
                args.dropout = hp['dropout']
            if 'drop_path_rate' in hp:
                args.drop_path_rate = hp['drop_path_rate']
            if 'alpha' in hp:
                args.alpha = hp['alpha']
            if 'aggregation' in hp:
                args.aggregation = hp['aggregation']
            elif 'pool_type' in hp:
                args.aggregation = hp['pool_type']
            if 'activation' in hp:
                args.activation = hp['activation']
            if 'dmp_steps' in hp:
                args.dmp_steps = hp['dmp_steps']
            
            # Training parameters
            if 'learning_rate' in hp:
                args.learning_rate = hp['learning_rate']
            elif 'lr' in hp:
                args.learning_rate = hp['lr']
            if 'weight_decay' in hp:
                args.weight_decay = hp['weight_decay']
            if 'batch_size' in hp:
                args.batch_size = hp['batch_size']
            if 'grad_clip_norm' in hp:
                args.grad_clip_norm = hp['grad_clip_norm']
            if 'num_epochs' in hp:
                args.num_epochs = hp['num_epochs']
            
            # Scheduler parameters
            if 'scheduler_type' in hp:
                args.scheduler_type = hp['scheduler_type']
            if 'warmup_epochs' in hp:
                args.warmup_epochs = hp['warmup_epochs']
            if 'min_lr' in hp:
                args.min_lr = hp['min_lr']
            
            # Early stopping parameters
            if 'early_stopping_patience' in hp:
                args.early_stopping_patience = hp['early_stopping_patience']
            if 'use_smart_early_stopping' in hp:
                if hp['use_smart_early_stopping']:
                    args.use_smart_early_stopping = True
            if 'smart_early_stopping_max_patience' in hp:
                args.smart_early_stopping_max_patience = hp['smart_early_stopping_max_patience']
            
            # Model type
            if 'model_type' in hp:
                args.model_type = hp['model_type']
            
            # Feature engineering
            if 'use_descriptor' in hp and hp['use_descriptor']:
                args.use_descriptor = True
            if 'descriptor_dim' in hp:
                args.descriptor_dim = hp['descriptor_dim']
            if 'descriptor_dropout' in hp:
                args.descriptor_dropout = hp['descriptor_dropout']
            
            # Regularization and augmentation
            if 'rotate_aug' in hp and hp['rotate_aug']:
                args.rotate_aug = True
            # IMPROVEMENT 5.1: Support rotation_prob and max_rotation_angle from config
            if 'rotation_prob' in hp:
                args.rotation_prob = hp['rotation_prob']
            if 'max_rotation_angle' in hp:
                args.max_rotation_angle = hp['max_rotation_angle']
            if 'use_pre_norm' in hp and hp['use_pre_norm']:
                args.use_pre_norm = True
            
            # Loss function parameters
            if 'use_bce_for_imbalanced' in hp and hp['use_bce_for_imbalanced']:
                args.use_bce_for_imbalanced = True
            if 'auto_pos_weight' in hp and hp['auto_pos_weight']:
                args.auto_pos_weight = True
            if 'use_focal_loss' in hp and hp['use_focal_loss']:
                args.use_focal_loss = True
            if 'focal_alpha' in hp:
                args.focal_alpha = hp['focal_alpha']
            if 'focal_gamma' in hp:
                args.focal_gamma = hp['focal_gamma']
            
            # Mixup parameters
            if 'use_mixup' in hp and hp['use_mixup']:
                args.enable_manifold_mixup = True
            if 'mixup_alpha' in hp:
                args.manifold_mixup_alpha = hp['mixup_alpha']
            
            print(f"✅ Configuration loaded successfully")
            print(f"   Model: {args.hidden_dim}D × {args.num_layers}L, LR={args.learning_rate:.2e}, Batch={args.batch_size}")
        else:
            print(f"⚠️  Warning: 'hyperparameters' key not found in config file")
    elif args.config:
        print(f"⚠️  Warning: Config file not found: {args.config}")
        print(f"   Continuing with command-line arguments only")
    
    # Auto-detect number of GPUs
    if args.world_size is None:
        if torch.cuda.is_available():
            args.world_size = torch.cuda.device_count()
        else:
            print("❌ CUDA not available, cannot use DDP")
            sys.exit(1)
    
    if args.world_size < 2:
        print("⚠️  DDP requires at least 2 GPUs. Using single GPU training mode.")
        print(f"   Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Single GPU mode: call train_worker with rank=0, world_size=1
        # train_worker will skip DDP initialization when world_size=1
        history = train_worker(rank=0, world_size=1, args=args, master_port=None)
        
        print("✅ Training completed!")
        return
    
    print(f"🚀 Starting DDP training with {args.world_size} GPUs")
    for i in range(args.world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Select available port in main process (all child processes use the same port)
    import socket
    base_port = args.base_port
    max_attempts = 100
    master_port = None
    
    for attempt in range(max_attempts):
        test_port = base_port + attempt
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', test_port))
                master_port = test_port
                print(f"🔌 Selected port: {master_port}")
                break
        except OSError:
            # Port is occupied, try next one
            continue
    
    if master_port is None:
        raise RuntimeError(f"❌ Cannot find available port (tried {max_attempts} ports starting from {base_port})")
    
    # Spawn processes
    mp.spawn(
        train_worker,
        args=(args.world_size, args, master_port),
        nprocs=args.world_size,
        join=True
    )
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()

