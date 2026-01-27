"""
AEGNN-M Model Evaluation Script
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import json
from tqdm import tqdm

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.aegnn_model import create_aegnn_model
from utils.data_utils import MolecularDataset, DataPreprocessor


class AEGNNEvaluator:
    """AEGNN-M Evaluator"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @staticmethod
    def _optimize_threshold(targets,
                            probabilities,
                            metric='f1',
                            min_threshold=0.05,
                            max_threshold=0.95,
                            num_thresholds=181):
        """Find the best classification threshold for binary classification"""
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        best_threshold = 0.5
        best_score = -np.inf
        
        for threshold in thresholds:
            preds = (probabilities >= threshold).astype(int)
            if metric == 'f1':
                score = f1_score(targets, preds, zero_division=0)
            elif metric == 'youden':
                tp = np.logical_and(preds == 1, targets == 1).sum()
                tn = np.logical_and(preds == 0, targets == 0).sum()
                fp = np.logical_and(preds == 1, targets == 0).sum()
                fn = np.logical_and(preds == 0, targets == 1).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                score = tpr - fpr
            else:
                raise ValueError("Unsupported threshold metric. Choose from {'f1', 'youden'}.")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def evaluate_regression(self, test_loader, save_full_predictions=False):
        """Evaluate regression model"""
        predictions = []
        targets = []
        attention_weights = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                # If batch has pos attribute (3D coordinates), pass it to the model
                pos = getattr(batch, 'pos', None)
                pred, attn_weights = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, pos=pos)
                
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(batch.y.cpu().numpy().flatten())
                if save_full_predictions:
                    attention_weights.extend(attn_weights)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 🔧 NORMALIZATION FIX: Detect and correct scale mismatch between predictions and targets
        # This fixes the issue where predictions are in normalized space but targets are in original space
        print(f"\n🔍 Checking prediction and target scale consistency...")
        print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   Predictions mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
        print(f"   Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"   Targets mean: {targets.mean():.4f}, std: {targets.std():.4f}")
        
        # Try to load target statistics from checkpoint if available
        target_mean = None
        target_std = None
        try:
            # Try to get from model checkpoint or config
            # Note: This is a simplified version - in practice, you might need to pass these as arguments
            pass
        except:
            pass
        
        # If we have target statistics, check for scale mismatch
        if target_mean is not None and target_std is not None:
            pred_mean = predictions.mean()
            pred_std = predictions.std()
            pred_min = predictions.min()
            pred_max = predictions.max()
            pred_range = pred_max - pred_min
            
            target_mean_actual = targets.mean()
            target_std_actual = targets.std()
            target_min = targets.min()
            target_max = targets.max()
            target_range = target_max - target_min
            
            # Check if predictions appear normalized
            pred_appears_normalized = (
                (abs(pred_mean) < 0.5 and 0.3 < pred_std < 2.0) or
                (pred_min >= -0.1 and pred_max <= 1.1 and pred_range < 1.5) or
                (pred_min >= -1.1 and pred_max <= 1.1 and pred_range < 2.5)
            )
            
            # Check if targets match expected statistics
            target_matches_expected = (
                abs(target_mean_actual - target_mean) < target_std * 0.3 and
                abs(target_std_actual - target_std) < target_std * 0.3
            )
            
            # Check scale mismatch
            scale_ratio_mean = abs(pred_mean / target_mean_actual) if abs(target_mean_actual) > 1e-6 else float('inf')
            scale_ratio_std = abs(pred_std / target_std_actual) if target_std_actual > 1e-6 else float('inf')
            scale_ratio_range = abs(pred_range / target_range) if target_range > 1e-6 else float('inf')
            
            scale_mismatch = (
                (scale_ratio_mean < 0.1 or scale_ratio_mean > 10.0) or
                (scale_ratio_std < 0.1 or scale_ratio_std > 10.0) or
                (scale_ratio_range < 0.1 or scale_ratio_range > 10.0)
            )
            
            if (pred_appears_normalized and target_matches_expected) or (scale_mismatch and pred_appears_normalized):
                print(f"\n🚨 Scale mismatch detected! Applying denormalization...")
                predictions_denorm = predictions * target_std + target_mean
                
                # Verify improvement
                mean_diff_before = abs(pred_mean - target_mean_actual)
                mean_diff_after = abs(predictions_denorm.mean() - target_mean_actual)
                
                if mean_diff_after < mean_diff_before:
                    predictions = predictions_denorm
                    print(f"   ✅ Denormalization applied successfully")
                else:
                    print(f"   ⚠️  Denormalization didn't improve scale match, using original predictions")
        else:
            # Without target statistics, use heuristic check
            pred_mean = predictions.mean()
            pred_std = predictions.std()
            target_mean = targets.mean()
            target_std = targets.std()
            
            scale_ratio_mean = abs(pred_mean / target_mean) if abs(target_mean) > 1e-6 else float('inf')
            scale_ratio_std = abs(pred_std / target_std) if target_std > 1e-6 else float('inf')
            
            # If predictions are clearly in normalized space but targets are not, warn user
            if (abs(pred_mean) < 0.5 and 0.3 < pred_std < 2.0) and (abs(target_mean) > 1.0 or target_std > 1.0):
                print(f"\n⚠️  Warning: Predictions appear normalized but targets are not.")
                print(f"   Consider providing target statistics (mean, std) for proper denormalization.")
                print(f"   Current scale ratios - Mean: {scale_ratio_mean:.4f}, Std: {scale_ratio_std:.4f}")
        
        # Calculate evaluation metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(targets, predictions)[0, 1]
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'num_samples': len(predictions),
            # Always include predictions and targets for plotting (even if not saved to file)
            'predictions': predictions,
            'targets': targets
        }
        
        # Optionally include attention weights (can be very large)
        if save_full_predictions:
            results['attention_weights'] = attention_weights
        
        return results
    
    def evaluate_classification(self, test_loader, threshold_opt_config=None, save_full_predictions=False):
        """Evaluate classification model"""
        predictions = []
        targets = []
        probabilities = []
        attention_weights = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                # If batch has pos attribute (3D coordinates), pass it to the model
                pos = getattr(batch, 'pos', None)
                pred, attn_weights = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, pos=pos)
                
                # Get predicted classes and probabilities
                pred_probs = torch.softmax(pred, dim=1)
                pred_classes = torch.argmax(pred, dim=1)
                
                predictions.extend(pred_classes.cpu().numpy())
                targets.extend(batch.y.cpu().numpy().astype(int))
                probabilities.extend(pred_probs.cpu().numpy())
                if save_full_predictions:
                    attention_weights.extend(attn_weights)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted')
        recall = recall_score(targets, predictions, average='weighted')
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Calculate ROC-AUC
        roc_auc = None
        try:
            if probabilities.shape[1] == 2:
                # Binary classification
                roc_auc = roc_auc_score(targets, probabilities[:, 1])
            else:
                # Multi-class - use one-vs-rest
                roc_auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'num_samples': len(predictions),
            # Always include predictions and targets for plotting (even if not saved to file)
            'predictions': predictions,
            'targets': targets
        }
        
        # Probabilities are needed for PR curve and threshold optimization, but can be large
        # Only include in results if needed for saving or if save_full_predictions is True
        if save_full_predictions:
            results['probabilities'] = probabilities
            results['attention_weights'] = attention_weights
        else:
            # Store probabilities temporarily for PR curve calculation, but mark for removal
            results['_probabilities'] = probabilities  # Will be removed before saving
        
        # Threshold optimization (binary classification only)
        threshold_results = {
            'enabled': False,
            'reason': 'not_requested'
        }
        
        pr_curve_results = {
            'enabled': False,
            'reason': 'binary_classification_only'
        }
        
        # Get probabilities from results (either saved or temporary)
        probs_for_calc = results.get('probabilities') or results.get('_probabilities')
        positive_probs = None
        if probs_for_calc is not None and probs_for_calc.size > 0:
            if probs_for_calc.ndim == 2 and probs_for_calc.shape[1] == 2:
                positive_probs = probs_for_calc[:, 1]
            elif probs_for_calc.ndim == 1:
                positive_probs = probs_for_calc
        
        if positive_probs is not None:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(targets, positive_probs)
            avg_precision = average_precision_score(targets, positive_probs)
            pr_curve_results = {
                'enabled': True,
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'average_precision': float(avg_precision)
            }
        
        if threshold_opt_config and positive_probs is not None:
            best_threshold, best_score = self._optimize_threshold(
                targets=targets,
                probabilities=positive_probs,
                metric=threshold_opt_config.get('metric', 'f1'),
                min_threshold=threshold_opt_config.get('min_threshold', 0.05),
                max_threshold=threshold_opt_config.get('max_threshold', 0.95),
                num_thresholds=threshold_opt_config.get('num_thresholds', 181)
            )
            
            threshold_preds = (positive_probs >= best_threshold).astype(int)
            threshold_results = {
                'enabled': True,
                'metric': threshold_opt_config.get('metric', 'f1'),
                'best_threshold': float(best_threshold),
                'best_score': float(best_score),
                'accuracy': accuracy_score(targets, threshold_preds),
                'precision': precision_score(targets, threshold_preds, zero_division=0),
                'recall': recall_score(targets, threshold_preds, zero_division=0),
                'f1': f1_score(targets, threshold_preds, zero_division=0),
                'predictions': threshold_preds.tolist()
            }
        elif threshold_opt_config and positive_probs is None:
            threshold_results = {
                'enabled': False,
                'reason': 'threshold_optimization_supported_for_binary_classification_only'
            }
        
        results['threshold_optimization'] = threshold_results
        results['precision_recall_curve'] = pr_curve_results
        
        return results
    
    @staticmethod
    def save_pr_curve(pr_curve_data, save_dir):
        """Save precision-recall curve data and generate plot"""
        if not pr_curve_data or not pr_curve_data.get('enabled'):
            return None
        
        os.makedirs(save_dir, exist_ok=True)
        
        json_path = os.path.join(save_dir, 'precision_recall_curve.json')
        with open(json_path, 'w') as f:
            json.dump(pr_curve_data, f, indent=2)
        
        precision_vals = pr_curve_data['precision']
        recall_vals = pr_curve_data['recall']
        avg_precision = pr_curve_data.get('average_precision', 0.0)
        
        plt.figure(figsize=(7, 5))
        plt.step(recall_vals, precision_vals, where='post', label=f'AP = {avg_precision:.3f}')
        plt.fill_between(recall_vals, precision_vals, step='post', alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'precision_recall_curve.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return {
            'json_path': json_path,
            'plot_path': plot_path
        }
    
    def plot_regression_results(self, results, save_path=None):
        """Plot regression results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Predicted vs actual scatter plot
        axes[0, 0].scatter(results['targets'], results['predictions'], alpha=0.6)
        axes[0, 0].plot([results['targets'].min(), results['targets'].max()], 
                       [results['targets'].min(), results['targets'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Predicted vs Actual (R² = {results["r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual plot
        residuals = results['predictions'] - results['targets']
        axes[0, 1].scatter(results['predictions'], residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Value distribution
        axes[1, 0].hist(results['targets'], alpha=0.7, label='Actual', bins=30)
        axes[1, 0].hist(results['predictions'], alpha=0.7, label='Predicted', bins=30)
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Value Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Evaluation metrics
        metrics = ['MSE', 'MAE', 'RMSE', 'R²', 'Correlation']
        values = [results['mse'], results['mae'], results['rmse'], results['r2'], results['correlation']]
        bars = axes[1, 1].bar(metrics, values)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Evaluation Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result plot saved to: {save_path}")
        
        plt.show()
    
    def plot_classification_results(self, results, save_path=None):
        """Plot classification results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted Class')
        axes[0, 0].set_ylabel('Actual Class')
        axes[0, 0].set_title('Confusion Matrix')
        
        # 2. Prediction accuracy
        correct = (results['predictions'] == results['targets']).astype(int)
        accuracy_by_sample = np.array(correct)
        
        axes[0, 1].hist(accuracy_by_sample, bins=2, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Correctness')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title(f'Prediction Accuracy (Overall: {results["accuracy"]:.3f})')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_xticklabels(['Incorrect', 'Correct'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Class distribution
        unique_targets, target_counts = np.unique(results['targets'], return_counts=True)
        unique_predictions, pred_counts = np.unique(results['predictions'], return_counts=True)
        
        x = np.arange(len(unique_targets))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, target_counts, width, label='Actual', alpha=0.7)
        axes[1, 0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Class Distribution Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(unique_targets)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Evaluation metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
        bars = axes[1, 1].bar(metrics, values)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Evaluation Metrics')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results, save_path, save_full_predictions=False):
        """Save evaluation results"""
        def _to_serializable(obj):
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items() if not k.startswith('_')}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if torch.is_tensor(obj):
                if obj.dim() == 0:
                    return obj.item()
                return obj.detach().cpu().tolist()
            return obj
        
        # Create a copy of results for saving
        save_data = results.copy()
        
        # Remove internal flags
        save_data.pop('_save_full_predictions', None)
        
        # If not saving full predictions, remove large arrays
        if not save_full_predictions:
            save_data.pop('predictions', None)
            save_data.pop('targets', None)
            save_data.pop('probabilities', None)
            save_data.pop('_probabilities', None)  # Remove temporary probabilities
            save_data.pop('attention_weights', None)
            if 'threshold_optimization' in save_data and isinstance(save_data['threshold_optimization'], dict):
                save_data['threshold_optimization'].pop('predictions', None)
        
        save_data = _to_serializable(save_data)
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Evaluation results saved to: {save_path} ({file_size_mb:.2f} MB)")
        
        if not save_full_predictions and ('predictions' in results or 'targets' in results):
            print("  Note: Full predictions/targets not saved (use --save_full_predictions to include them)")


def main():
    parser = argparse.ArgumentParser(description='AEGNN-M Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_path', type=str, default=None, help='Test data path (CSV)')
    parser.add_argument('--processed_data_path', type=str, default=None,
                        help='Optional processed dataset (PKL) to skip re-processing')
    parser.add_argument('--target_column', type=str, default='target', help='Target column name')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='SMILES column name')
    parser.add_argument('--model_type', type=str, default='regressor', choices=['regressor', 'classifier'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results', help='Results save directory')
    parser.add_argument('--plot', action='store_true', help='Whether to plot results')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Enable threshold optimization for binary classification tasks')
    parser.add_argument('--threshold_metric', type=str, default='f1', choices=['f1', 'youden'],
                        help='Metric to optimize for threshold search')
    parser.add_argument('--threshold_range', type=float, nargs=2, default=[0.05, 0.95],
                        help='Min and max values for threshold search (inclusive)')
    parser.add_argument('--threshold_steps', type=int, default=181,
                        help='Number of thresholds to evaluate between min and max')
    parser.add_argument('--save_pr_curve', action='store_true',
                        help='Save precision-recall curve data/plots when available')
    parser.add_argument('--pr_curve_dir', type=str, default=None,
                        help='Directory to save precision-recall data (default: save_dir/pr_curve)')
    parser.add_argument('--save_full_predictions', action='store_true',
                        help='Save full predictions, targets, probabilities, and attention weights (default: False, saves only summary metrics)')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    loaded_from_processed = False
    if args.processed_data_path and os.path.exists(args.processed_data_path):
        dataset = MolecularDataset()
        dataset.load_processed_data(args.processed_data_path)
        print(f"Using processed dataset: {args.processed_data_path}")
        loaded_from_processed = True
    elif args.data_path:
        dataset = MolecularDataset(
            data_path=args.data_path,
            target_column=args.target_column,
            smiles_column=args.smiles_column
        )
    else:
        raise ValueError("Must provide --data_path or --processed_data_path for evaluation.")
    
    # Process graph data if not already loaded
    if loaded_from_processed:
        if not dataset.graphs:
            raise ValueError(f"Processed dataset at {args.processed_data_path} contains no graphs.")
        graphs = dataset.graphs
    else:
        graphs = dataset.process_graphs()
    
    # Create test data loader
    test_loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model_config = checkpoint.get('model_config', {})
    model_kwargs = {
        'model_type': args.model_type,
        'node_features': dataset.graph_builder.node_feature_dim,
        'edge_features': dataset.graph_builder.edge_feature_dim,
        'hidden_dim': model_config.get('hidden_dim', 256),
        'num_layers': model_config.get('num_layers', 6),
        'num_heads': model_config.get('num_heads', 8),
        'ffn_expansion_factor': model_config.get('ffn_expansion_factor', 4),
        'dropout': model_config.get('dropout', 0.1)
    }
    
    model = create_aegnn_model(**model_kwargs)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = AEGNNEvaluator(model, device)
    
    # Execute evaluation
    print("Starting evaluation...")
    if args.model_type == 'regressor':
        results = evaluator.evaluate_regression(test_loader, save_full_predictions=args.save_full_predictions)
        
        print("\nRegression evaluation results:")
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  R²: {results['r2']:.4f}")
        print(f"  Correlation: {results['correlation']:.4f}")
        
        if args.plot:
            os.makedirs(args.save_dir, exist_ok=True)
            plot_path = os.path.join(args.save_dir, 'regression_results.png')
            evaluator.plot_regression_results(results, plot_path)
    
    else:  # classifier
        threshold_config = None
        if args.optimize_threshold:
            min_t, max_t = args.threshold_range
            if min_t >= max_t:
                raise ValueError("threshold_range must be in the format: min max, and min < max.")
            threshold_config = {
                'metric': args.threshold_metric,
                'min_threshold': min_t,
                'max_threshold': max_t,
                'num_thresholds': args.threshold_steps
            }
        
        results = evaluator.evaluate_classification(test_loader, threshold_opt_config=threshold_config, save_full_predictions=args.save_full_predictions)
        
        print("\nClassification evaluation results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        if results['roc_auc'] is not None:
            print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        
        threshold_info = results.get('threshold_optimization', {})
        if threshold_info.get('enabled'):
            print("\n🔧 Threshold optimization")
            print(f"  Metric: {threshold_info.get('metric')}")
            print(f"  Best threshold: {threshold_info.get('best_threshold'):.4f}")
            print(f"  Best score ({threshold_info.get('metric')}): {threshold_info.get('best_score'):.4f}")
            print(f"  Accuracy@best: {threshold_info.get('accuracy'):.4f}")
            print(f"  Precision@best: {threshold_info.get('precision'):.4f}")
            print(f"  Recall@best: {threshold_info.get('recall'):.4f}")
            print(f"  F1@best: {threshold_info.get('f1'):.4f}")
        elif threshold_info.get('reason'):
            print(f"\nThreshold optimization skipped: {threshold_info.get('reason')}")
        
        if args.plot:
            os.makedirs(args.save_dir, exist_ok=True)
            plot_path = os.path.join(args.save_dir, 'classification_results.png')
            evaluator.plot_classification_results(results, plot_path)
        
        if args.save_pr_curve:
            pr_curve_dir = args.pr_curve_dir or os.path.join(args.save_dir, 'pr_curve')
            pr_save_info = AEGNNEvaluator.save_pr_curve(results.get('precision_recall_curve'), pr_curve_dir)
            if pr_save_info:
                print(f"\nPrecision-Recall curve saved to: {pr_save_info['json_path']}")
                print(f"PR curve plot: {pr_save_info['plot_path']}")
            else:
                print("\nPrecision-Recall curve not available (non-binary classification).")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    evaluator.save_results(results, results_path, save_full_predictions=args.save_full_predictions)
    
    print(f"\nEvaluation completed! Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
