"""
Threshold Selection Utilities for Imbalanced Classification
Provides dynamic threshold selection tools for imbalanced datasets
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    balanced_accuracy_score, roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import KFold
from typing import Optional, Tuple, Dict, Callable


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           metric: str = 'f1',
                           threshold_range: Tuple[float, float] = (0.1, 0.9),
                           num_steps: int = 81,
                           avoid_pathological: bool = True,
                           pr_auc_weight: float = 0.0) -> Tuple[float, float]:
    """
    Find optimal threshold on validation set
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Optimization metric ('f1', 'balanced_accuracy', 'f1_precision_recall', 'youden', 'pr_auc_weighted')
        threshold_range: Threshold search range (min, max)
        num_steps: Number of search steps
        avoid_pathological: If True, filter out thresholds that lead to all-negative or all-positive predictions
        pr_auc_weight: Weight for PR-AUC in composite score (0.0 = ignore, 1.0 = PR-AUC only)
                       Used when metric='pr_auc_weighted'
    
    Returns:
        (optimal_threshold, optimal_score) tuple
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    best_threshold = 0.5
    best_score = -np.inf
    
    # Calculate overall PR-AUC once (for pr_auc_weighted metric)
    overall_pr_auc = 0.0
    if pr_auc_weight > 0.0 or metric == 'pr_auc_weighted':
        try:
            overall_pr_auc = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
        except:
            overall_pr_auc = 0.0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate basic metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Avoid pathological solutions (all negative or all positive)
        if avoid_pathological:
            unique_preds = np.unique(y_pred)
            if len(unique_preds) == 1:
                # All predictions are the same class - skip this threshold
                continue
            # Also check if precision or recall is extremely low (likely pathological)
            if precision < 0.01 and recall < 0.01:
                continue
        
        # Calculate score based on metric
        if metric == 'f1':
            score = f1
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'f1_precision_recall':
            # Weighted average of F1, Precision, Recall
            score = (f1 * 0.5 + precision * 0.25 + recall * 0.25)
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            # Equivalent to TPR - FPR
            from sklearn.metrics import confusion_matrix
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                score = tpr - fpr
            except:
                score = 0.0
        elif metric == 'pr_auc_weighted':
            # Composite score: weighted combination of PR-AUC (overall) and F1 (at this threshold)
            # This balances overall PR curve quality with point-wise F1 performance
            score = pr_auc_weight * overall_pr_auc + (1 - pr_auc_weight) * f1
        elif metric == 'pr_auc_f1_balanced':
            # For imbalanced datasets: balance PR-AUC contribution with F1
            # This is used internally by find_optimal_threshold_adaptive
            score = 0.6 * overall_pr_auc + 0.4 * f1
        else:
            score = f1
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Fallback: if no valid threshold found, use 0.5
    if best_score == -np.inf:
        best_threshold = 0.5
        best_score = f1_score(y_true, (y_pred_proba >= 0.5).astype(int), zero_division=0)
    
    return best_threshold, best_score


def find_optimal_threshold_multi_metric(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       metrics: list = ['f1', 'balanced_accuracy'],
                                       threshold_range: Tuple[float, float] = (0.1, 0.9),
                                       num_steps: int = 81,
                                       avoid_pathological: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Find optimal threshold using multiple metrics
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metrics: List of metrics
        threshold_range: Threshold search range
        num_steps: Number of search steps
        avoid_pathological: If True, filter out pathological thresholds
    
    Returns:
        Dictionary with metric names as keys and (optimal_threshold, optimal_score) tuples as values
    """
    results = {}
    
    for metric in metrics:
        threshold, score = find_optimal_threshold(
            y_true, y_pred_proba, metric=metric,
            threshold_range=threshold_range, num_steps=num_steps,
            avoid_pathological=avoid_pathological
        )
        results[metric] = (threshold, score)
    
    return results


def evaluate_with_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model with specified threshold
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Evaluation metrics dictionary
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': (y_pred == y_true).mean(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'threshold': threshold
    }
    
    return metrics


def check_threshold_stability(val_threshold: float, test_threshold: float,
                              threshold_diff_threshold: float = 0.2,
                              relative_diff_threshold: float = 2.0,
                              imbalance_ratio: Optional[float] = None,
                              dataset_size: Optional[int] = None) -> Tuple[bool, str]:
    """
    Check if threshold is stable between validation and test sets (Direction 3)
    Now with adaptive thresholds based on dataset characteristics
    
    Args:
        val_threshold: Optimal threshold found on validation set
        test_threshold: Optimal threshold found on test set
        threshold_diff_threshold: Absolute difference threshold (default: 0.2, will be adjusted if imbalance_ratio/dataset_size provided)
        relative_diff_threshold: Relative difference threshold (default: 2.0, will be adjusted if imbalance_ratio/dataset_size provided)
        imbalance_ratio: Imbalance ratio (neg_count / pos_count) for adaptive threshold adjustment
        dataset_size: Dataset size for adaptive threshold adjustment
    
    Returns:
        (is_stable, warning_message) tuple
        - is_stable: True if threshold is stable, False otherwise
        - warning_message: Warning message if unstable, empty string if stable
    """
    if val_threshold <= 0 or test_threshold <= 0:
        return False, "Invalid threshold values (must be > 0)"
    
    # Adaptive threshold adjustment based on dataset characteristics
    if imbalance_ratio is not None:
        # For extremely imbalanced datasets, use stricter standards
        if imbalance_ratio > 100:
            # Extremely imbalanced (e.g., MUV, CLINTOX)
            threshold_diff_threshold = 0.05  # Stricter
            relative_diff_threshold = 1.5  # Stricter
        elif imbalance_ratio > 50:
            # Moderately imbalanced (e.g., HIV)
            threshold_diff_threshold = 0.08
            relative_diff_threshold = 1.8
        elif imbalance_ratio > 20:
            # Mildly imbalanced (e.g., TOX21, SIDER)
            threshold_diff_threshold = 0.12
            relative_diff_threshold = 1.9
        # else: balanced datasets use default values
    
    # For small datasets, use stricter standards (more unstable)
    if dataset_size is not None and dataset_size < 1000:
        threshold_diff_threshold *= 0.7  # Stricter
        relative_diff_threshold *= 0.8  # Stricter
    
    # Calculate absolute and relative differences
    abs_diff = abs(test_threshold - val_threshold)
    relative_diff = max(test_threshold / val_threshold, val_threshold / test_threshold)
    
    # Check stability
    is_stable = abs_diff <= threshold_diff_threshold and relative_diff <= relative_diff_threshold
    
    if not is_stable:
        warning = (
            f"⚠️  Threshold instability detected: "
            f"Validation threshold={val_threshold:.4f}, "
            f"Test threshold={test_threshold:.4f}, "
            f"Absolute diff={abs_diff:.4f} (threshold={threshold_diff_threshold:.4f}), "
            f"Relative diff={relative_diff:.2f}x (threshold={relative_diff_threshold:.2f}x). "
            f"This may indicate overfitting to validation set or dataset distribution shift."
        )
        return False, warning
    
    return True, ""


def find_optimal_threshold_cv(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             imbalance_ratio: Optional[float] = None,
                             n_splits: int = 5,
                             random_state: int = 42,
                             method: str = 'auto') -> Tuple[float, float]:
    """
    Find optimal threshold using Cross-Validation for more robust threshold selection
    Particularly useful for datasets with threshold instability (e.g., CLINTOX)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        imbalance_ratio: Imbalance ratio (neg_count / pos_count, if None, automatically calculated)
        n_splits: Number of CV folds
        random_state: Random seed for CV splits
        method: Selection method ('auto', 'f1', 'balanced_accuracy', 'youden', 'pr_auc_weighted')
    
    Returns:
        (optimal_threshold, optimal_score) tuple
        - optimal_threshold: Median threshold from CV folds (more robust than mean)
        - optimal_score: Average score across CV folds
    """
    if imbalance_ratio is None:
        pos_count = (y_true == 1).sum()
        neg_count = (y_true == 0).sum()
        imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Use KFold for cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    thresholds = []
    scores = []
    
    for train_idx, val_idx in kf.split(y_true):
        y_val_fold = y_true[val_idx]
        y_pred_fold = y_pred_proba[val_idx]
        
        # Find optimal threshold for this fold
        threshold, score = find_optimal_threshold_adaptive(
            y_val_fold, y_pred_fold, 
            imbalance_ratio=imbalance_ratio,
            method=method
        )
        thresholds.append(threshold)
        scores.append(score)
    
    # Use median threshold (more robust to outliers than mean)
    optimal_threshold = np.median(thresholds)
    optimal_score = np.mean(scores)
    
    return optimal_threshold, optimal_score


def find_optimal_threshold_adaptive(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   imbalance_ratio: Optional[float] = None,
                                   method: str = 'auto') -> Tuple[float, float]:
    """
    Adaptive threshold selection: Select optimal metric based on dataset imbalance ratio
    
    Strategy:
    - Balanced (ratio < 20): F1 maximization (most intuitive for balanced datasets)
    - Imbalanced (20 <= ratio <= 100): PR-AUC + F1 weighted combination
    - Extremely imbalanced (ratio > 100): PR-AUC dominated + pathological solution check
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        imbalance_ratio: Imbalance ratio (neg_count / pos_count, if None, automatically calculated)
        method: Selection method ('auto', 'f1', 'balanced_accuracy', 'youden', 'pr_auc_weighted')
    
    Returns:
        (optimal_threshold, optimal_score) tuple
    """
    if imbalance_ratio is None:
        pos_count = (y_true == 1).sum()
        neg_count = (y_true == 0).sum()
        imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Select metric & threshold search range based on imbalance ratio (Direction 1 & 2)
    if method == 'auto':
        if imbalance_ratio > 100:  # Extremely imbalanced (e.g., MUV, CLINTOX)
            # Strategy: PR-AUC dominated + strict pathological check
            # Use finer search in lower threshold range (Direction 2)
            # More conservative range to avoid extreme thresholds
            metric = 'pr_auc_f1_balanced'  # 60% PR-AUC + 40% F1
            threshold_range = (0.05, 0.25)  # More conservative: from (0.01, 0.3) to (0.05, 0.25)
            num_steps = 200  # Finer search for extreme imbalance (Direction 2)
            avoid_pathological = True
        elif imbalance_ratio > 50:
            # Moderately imbalanced (e.g., HIV)
            metric = 'pr_auc_f1_balanced'  # 60% PR-AUC + 40% F1
            threshold_range = (0.05, 0.5)
            num_steps = 150
            avoid_pathological = True
        elif imbalance_ratio > 20:
            # Mildly imbalanced (e.g., TOX21, SIDER)
            metric = 'pr_auc_f1_balanced'  # 60% PR-AUC + 40% F1
            threshold_range = (0.05, 0.7)
            num_steps = 120
            avoid_pathological = True
        else:
            # Balanced (e.g., BACE, BBBP)
            # Keep F1 maximization for balanced datasets (no change from original)
            metric = 'f1'
            threshold_range = (0.1, 0.9)
            num_steps = 81
            avoid_pathological = False  # Less strict for balanced datasets
    else:
        # Manual method override
        metric = method
        threshold_range = (0.1, 0.9)
        num_steps = 81
        avoid_pathological = True
    
    return find_optimal_threshold(
        y_true, y_pred_proba, 
        metric=metric, 
        threshold_range=threshold_range,
        num_steps=num_steps,
        avoid_pathological=avoid_pathological
    )


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    
    # Generate imbalanced data
    n_pos = 50
    n_neg = 5000
    y_true = np.concatenate([
        np.ones(n_pos),
        np.zeros(n_neg)
    ])
    
    # Generate predicted probabilities (simulate model output)
    y_pred_proba = np.concatenate([
        np.random.beta(2, 1, n_pos),  # Positive class: biased towards high probability
        np.random.beta(1, 2, n_neg)   # Negative class: biased towards low probability
    ])
    
    # Find optimal threshold
    optimal_threshold, optimal_score = find_optimal_threshold_adaptive(
        y_true, y_pred_proba, method='auto'
    )
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Optimal score: {optimal_score:.4f}")
    
    # Evaluate with optimal threshold
    metrics = evaluate_with_threshold(y_true, y_pred_proba, optimal_threshold)
    print(f"\nEvaluation results with optimal threshold:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compare with default threshold 0.5
    metrics_default = evaluate_with_threshold(y_true, y_pred_proba, 0.5)
    print(f"\nEvaluation results with default threshold 0.5:")
    for key, value in metrics_default.items():
        print(f"  {key}: {value:.4f}")


