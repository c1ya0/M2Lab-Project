"""
Loss Function Utilities for Imbalanced Datasets and Regression Tasks
Provides loss function tools for imbalanced datasets and regression tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def calculate_imbalance_ratio(pos_count: int, neg_count: int) -> float:
    """
    Calculate class imbalance ratio
    
    Args:
        pos_count: Number of positive samples
        neg_count: Number of negative samples
    
    Returns:
        Imbalance ratio (neg_count / pos_count)
    """
    if pos_count == 0:
        return float('inf')
    return neg_count / pos_count


def calculate_focal_params(pos_count: int, neg_count: int, 
                           method: str = 'adaptive') -> Tuple[float, float]:
    """
    Dynamically calculate Focal Loss parameters based on dataset imbalance ratio
    
    Args:
        pos_count: Number of positive samples
        neg_count: Number of negative samples
        method: Calculation method ('adaptive', 'aggressive', 'conservative')
    
    Returns:
        (alpha, gamma) tuple
    """
    imbalance_ratio = calculate_imbalance_ratio(pos_count, neg_count)
    
    if method == 'aggressive':
        # Aggressive strategy: Use larger alpha for extremely imbalanced datasets
        if imbalance_ratio > 100:  # Extremely imbalanced (e.g., HIV, MUV)
            alpha = 0.9
            gamma = 1.0  # Reduce gamma to avoid over-focusing on hard samples
        elif imbalance_ratio > 50:
            alpha = 0.75
            gamma = 1.3
        elif imbalance_ratio > 20:
            alpha = 0.6
            gamma = 1.5
        else:
            alpha = 0.5
            gamma = 2.0
    
    elif method == 'conservative':
        # Conservative strategy: Smaller adjustments
        if imbalance_ratio > 100:
            alpha = 0.75
            gamma = 1.3
        elif imbalance_ratio > 50:
            alpha = 0.6
            gamma = 1.5
        else:
            alpha = 0.5
            gamma = 2.0
    
    else:  # 'adaptive' - default strategy
        # Adaptive strategy: Linear adjustment based on imbalance ratio
        if imbalance_ratio > 100:
            alpha = 0.85
            gamma = 1.1
        elif imbalance_ratio > 50:
            alpha = 0.7
            gamma = 1.3
        elif imbalance_ratio > 20:
            alpha = 0.6
            gamma = 1.5
        elif imbalance_ratio > 10:
            alpha = 0.5
            gamma = 1.8
        else:
            alpha = 0.25
            gamma = 2.0
    
    return alpha, gamma


def calculate_class_weights(pos_count: int, neg_count: int, 
                            method: str = 'balanced') -> np.ndarray:
    """
    Calculate class weights
    
    Args:
        pos_count: Number of positive samples
        neg_count: Number of negative samples
        method: Calculation method ('balanced', 'inverse', 'sqrt')
    
    Returns:
        Class weight array [neg_weight, pos_weight]
    """
    total = pos_count + neg_count
    
    if method == 'balanced':
        # sklearn's balanced method
        pos_weight = total / (2.0 * pos_count) if pos_count > 0 else 1.0
        neg_weight = total / (2.0 * neg_count) if neg_count > 0 else 1.0
    elif method == 'inverse':
        # Inverse proportional weights
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        neg_weight = 1.0
    elif method == 'sqrt':
        # Square root weights (gentler)
        pos_weight = np.sqrt(neg_count / pos_count) if pos_count > 0 else 1.0
        neg_weight = 1.0
    else:
        pos_weight = 1.0
        neg_weight = 1.0
    
    # Normalize
    weights = np.array([neg_weight, pos_weight])
    weights = weights / weights.sum() * 2.0  # Keep total weight as 2
    
    return weights


def calculate_pos_weight(pos_count: int, neg_count: int) -> float:
    """
    Calculate pos_weight for BCEWithLogitsLoss
    
    Args:
        pos_count: Number of positive samples
        neg_count: Number of negative samples
    
    Returns:
        pos_weight = neg_count / pos_count
    """
    if pos_count == 0:
        return 1.0
    return neg_count / pos_count


class DiceLoss(nn.Module):
    """
    Dice Loss for imbalanced binary classification
    Suitable for extremely imbalanced datasets
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing coefficient to avoid division by zero
            reduction: 'mean' or 'sum'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predicted logits [batch_size] or [batch_size, 1]
            target: True labels [batch_size], values are 0 or 1
        """
        # Convert to probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1).float()
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss = 1 - Dice
        loss = 1 - dice
        
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss
    Combines Class-Balanced Loss and Focal Loss
    Paper: Class-Balanced Loss Based on Effective Number of Samples
    """
    
    def __init__(self, beta: float = 0.9999, gamma: float = 2.0, 
                 alpha: float = 0.25, reduction: str = 'mean'):
        """
        Args:
            beta: Beta parameter for Class-Balanced Loss (0.9-0.9999)
                  Larger beta is more effective for extremely imbalanced datasets
            gamma: Gamma parameter for Focal Loss
            alpha: Alpha parameter for Focal Loss
            reduction: 'mean' or 'sum'
        """
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                class_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Model predicted logits [batch_size, num_classes]
            target: True labels [batch_size], values are class indices
            class_counts: Number of samples per class [num_classes], if None then Class-Balanced is not used
        """
        target = target.long()
        
        # Calculate Class-Balanced weights
        if class_counts is not None:
            # Effective number of samples
            effective_num = 1.0 - torch.pow(self.beta, class_counts.float())
            # Weight = (1 - beta) / effective_num
            weights = (1.0 - self.beta) / effective_num
            # Normalize
            weights = weights / weights.sum() * len(weights)
        else:
            weights = torch.ones(pred.size(1), device=pred.device)
        
        # Calculate standard Cross Entropy Loss
        # Note: label_smoothing can be added here if needed, but for Class-Balanced Focal Loss,
        # we rely on the focal weight and class weights for regularization
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Calculate predicted probabilities
        probs = torch.exp(-ce_loss)  # p_t = exp(-CE_loss)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = torch.pow(1 - probs, self.gamma)
        
        # Class-Balanced weight
        class_weights = weights[target]
        
        # Alpha weight
        alpha_t = self.alpha * (target == 1).float() + (1 - self.alpha) * (target == 0).float()
        
        # Combine loss
        loss = alpha_t * class_weights * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions
    Example: Focal Loss + Dice Loss
    """
    
    def __init__(self, focal_weight: float = 0.7, dice_weight: float = 0.3,
                 focal_alpha: float = 0.9, focal_gamma: float = 1.0):
        """
        Args:
            focal_weight: Weight for Focal Loss
            dice_weight: Weight for Dice Loss
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
        """
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        # Import FocalLoss from models.aegnn_model
        from models.aegnn_model import FocalLoss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        self.dice_loss = DiceLoss(reduction='mean')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predicted logits [batch_size, num_classes] or [batch_size]
            target: True labels [batch_size]
        """
        # If multi-class, convert to binary classification format
        if pred.dim() > 1 and pred.size(1) > 1:
            # Binary classification: use positive class logits
            if pred.size(1) == 2:
                pred_binary = pred[:, 1]
            else:
                # Multi-class: use maximum logit (simplified handling)
                pred_binary = pred.max(dim=1)[0]
        else:
            pred_binary = pred.squeeze()
        
        # Calculate both losses
        focal = self.focal_loss(pred, target.long())
        dice = self.dice_loss(pred_binary, target.float())
        
        # Combine
        loss = self.focal_weight * focal + self.dice_weight * dice
        
        return loss


class SpearmanLoss(nn.Module):
    """
    Differentiable Spearman Loss for Regression Tasks
    
    This loss function approximates Spearman correlation using soft ranking.
    Spearman correlation measures the rank correlation between two variables,
    which is useful when the relationship is monotonic but not necessarily linear.
    
    The loss is defined as: loss = 1 - spearman_correlation
    So minimizing the loss maximizes the Spearman correlation.
    """
    
    def __init__(self, temperature=1.0, reduction='mean'):
        """
        Args:
            temperature: Temperature parameter for soft ranking (lower = sharper ranking)
            reduction: 'mean' or 'sum' (currently only 'mean' is used)
        """
        super(SpearmanLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Spearman Loss using Pearson correlation on ranks
        
        Args:
            pred: Predicted values [batch_size] or [batch_size, 1]
            target: True values [batch_size]
        
        Returns:
            Loss value (1 - spearman_correlation)
        """
        # Flatten to 1D
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Ensure same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Handle scalar case
        if pred.dim() == 0:
            # Scalar tensor - return a loss that maintains gradient flow
            return pred * 0.0 + 1.0
        
        n = pred.shape[0] if len(pred.shape) > 0 else 0
        if n < 2:
            # Return a tensor that requires grad (use pred to maintain gradient flow)
            # Use pred[0] * 0.0 to create a tensor connected to computation graph
            if n == 1:
                return pred[0] * 0.0 + 1.0
            else:
                return pred.mean() * 0.0 + 1.0 if pred.numel() > 0 else torch.tensor(1.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        # Check for NaN or Inf in inputs
        if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
            # If predictions contain NaN/Inf, return a large loss value that maintains gradient flow
            # Use pred.mean() * 0.0 + 2.0 to ensure gradient flow
            return pred.mean() * 0.0 + 2.0
        
        if torch.any(torch.isnan(target)) or torch.any(torch.isinf(target)):
            # If targets contain NaN/Inf, return a large loss value that maintains gradient flow
            return pred.mean() * 0.0 + 2.0
        
        # Clamp inputs to reasonable range to avoid numerical instability
        pred = torch.clamp(pred, min=-1e6, max=1e6)
        target = torch.clamp(target, min=-1e6, max=1e6)
        
        # Compute soft ranks
        pred_ranks = self._soft_rank(pred, temperature=self.temperature)
        target_ranks = self._soft_rank(target, temperature=self.temperature)
        
        # Check for NaN in ranks
        if torch.any(torch.isnan(pred_ranks)) or torch.any(torch.isnan(target_ranks)):
            # Return a loss that maintains gradient flow
            return pred.mean() * 0.0 + 2.0
        
        # Center the ranks (subtract mean)
        pred_ranks_centered = pred_ranks - pred_ranks.mean()
        target_ranks_centered = target_ranks - target_ranks.mean()
        
        # Compute Pearson correlation on ranks (equivalent to Spearman)
        numerator = torch.sum(pred_ranks_centered * target_ranks_centered)
        pred_var = torch.sum(pred_ranks_centered ** 2)
        target_var = torch.sum(target_ranks_centered ** 2)
        
        # Enhanced protection against division by zero
        # Check if variance is too small (all values are nearly identical)
        min_var_threshold = 1e-8
        if pred_var < min_var_threshold or target_var < min_var_threshold:
            # If variance is too small, return a moderate loss (not perfect correlation)
            # Use pred to maintain gradient flow
            return pred.mean() * 0.0 + 1.0
        
        # Avoid division by zero with larger epsilon
        denominator = torch.sqrt(pred_var * target_var) + 1e-6
        
        # Pearson correlation (on ranks = Spearman correlation)
        spearman = numerator / denominator
        
        # Clamp spearman to valid range [-1, 1] to avoid numerical issues
        spearman = torch.clamp(spearman, min=-1.0, max=1.0)
        
        # Loss = 1 - spearman (to minimize, we want to maximize spearman)
        loss = 1.0 - spearman
        
        # Final clamp to ensure valid loss range
        loss = torch.clamp(loss, min=0.0, max=2.0)
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            # Return a loss that maintains gradient flow
            return pred.mean() * 0.0 + 2.0
        
        return loss
    
    def _soft_rank(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute soft ranks using differentiable sorting
        
        This uses a soft version of ranking that is differentiable.
        The idea is to use pairwise comparisons to determine ranks.
        
        Args:
            x: Input tensor [n]
            temperature: Temperature for softmax (lower = sharper)
        
        Returns:
            Soft ranks [n]
        """
        n = x.shape[0]
        if n == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        
        # Clamp temperature to avoid division by zero
        temperature = max(temperature, 1e-6)
        
        # Clamp input to avoid numerical instability in sigmoid
        x = torch.clamp(x, min=-1e4, max=1e4)
        
        # Expand for pairwise comparison
        x_expanded = x.unsqueeze(1)  # [n, 1]
        x_expanded_t = x.unsqueeze(0)  # [1, n]
        
        # Pairwise differences
        diff = x_expanded - x_expanded_t  # [n, n]
        
        # Clamp diff to avoid extreme values in sigmoid
        diff = torch.clamp(diff, min=-100.0, max=100.0)
        
        # Soft indicator using sigmoid with numerical stability
        indicator = torch.sigmoid(-diff / temperature)  # [n, n]
        
        # Check for NaN in indicator
        if torch.any(torch.isnan(indicator)):
            # Fallback to simple ranking if sigmoid produces NaN
            _, indices = torch.sort(x)
            ranks = torch.zeros_like(x)
            ranks[indices] = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
            return ranks
        
        # Sum to get soft rank
        soft_rank = torch.sum(indicator, dim=1) + 1.0  # [n]
        
        # Final NaN check
        if torch.any(torch.isnan(soft_rank)):
            # Fallback to simple ranking
            _, indices = torch.sort(x)
            ranks = torch.zeros_like(x)
            ranks[indices] = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
            return ranks
        
        return soft_rank


def select_loss_function(dataset_name: str, pos_count: int, neg_count: int,
                        imbalance_ratio: Optional[float] = None) -> str:
    """
    Select the most suitable loss function based on dataset characteristics
    
    Args:
        dataset_name: Dataset name
        pos_count: Number of positive samples
        neg_count: Number of negative samples
        imbalance_ratio: Imbalance ratio (if None, automatically calculated)
    
    Returns:
        Loss function type string
    """
    if imbalance_ratio is None:
        imbalance_ratio = calculate_imbalance_ratio(pos_count, neg_count)
    
    # Extremely imbalanced datasets (e.g., HIV, MUV)
    highly_imbalanced = ['hiv', 'muv']
    
    if dataset_name.lower() in highly_imbalanced or imbalance_ratio > 100:
        # Use combined loss: Class-Balanced Focal + Dice
        return 'combined_cbfocal_dice'
    elif imbalance_ratio > 50:
        # Use Class-Balanced Focal Loss
        return 'cbfocal'
    elif imbalance_ratio > 20:
        # Use Focal Loss (dynamic parameters)
        return 'focal_adaptive'
    else:
        # Use standard Focal Loss or CrossEntropy
        return 'focal'


if __name__ == "__main__":
    # Test
    pos_count = 50
    neg_count = 5000
    imbalance_ratio = calculate_imbalance_ratio(pos_count, neg_count)
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    alpha, gamma = calculate_focal_params(pos_count, neg_count, method='aggressive')
    print(f"Focal Loss parameters: alpha={alpha:.2f}, gamma={gamma:.2f}")
    
    class_weights = calculate_class_weights(pos_count, neg_count)
    print(f"Class weights: {class_weights}")
    
    pos_weight = calculate_pos_weight(pos_count, neg_count)
    print(f"BCE pos_weight: {pos_weight:.2f}")
    
    loss_type = select_loss_function('hiv', pos_count, neg_count)
    print(f"Recommended loss function: {loss_type}")


