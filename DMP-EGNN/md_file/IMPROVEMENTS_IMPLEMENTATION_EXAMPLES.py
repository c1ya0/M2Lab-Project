"""
TDC Datasets 改进实施示例代码

这些代码示例展示了如何实施分析报告中提到的主要改进。
可以直接集成到现有的 train_edmpnn.py 和 optuna_serach_mod.py 中。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from scipy import stats


# ============================================================================
# 改进 1: 稳健的描述符归一化
# ============================================================================

def robust_normalize_descriptors(train_graphs, val_graphs, test_graphs, rank=0):
    """
    使用 RobustScaler 进行更稳健的描述符归一化
    
    优势：
    - 对异常值不敏感（使用中位数和 IQR）
    - 数值稳定性更好
    - 适合包含极端值的 TDC 数据集
    """
    # 收集训练集描述符
    train_descriptors = []
    for graph in train_graphs:
        if hasattr(graph, 'descriptor') and graph.descriptor is not None:
            desc = graph.descriptor
            if isinstance(desc, torch.Tensor):
                desc = desc.cpu().numpy()
            if desc.ndim > 1:
                desc = desc.squeeze()
            train_descriptors.append(desc)
    
    if len(train_descriptors) == 0:
        if rank == 0:
            print("⚠️  No descriptors found, skipping normalization")
        return None, None
    
    # 转换为 numpy 数组
    train_descriptors_array = np.stack(train_descriptors)
    
    # 检查异常值
    if rank == 0:
        print(f"📊 Descriptor statistics before normalization:")
        print(f"   Shape: {train_descriptors_array.shape}")
        print(f"   Mean range: [{train_descriptors_array.mean(axis=0).min():.4f}, "
              f"{train_descriptors_array.mean(axis=0).max():.4f}]")
        print(f"   Std range: [{train_descriptors_array.std(axis=0).min():.4f}, "
              f"{train_descriptors_array.std(axis=0).max():.4f}]")
        
        # 检测异常值（使用 IQR 方法）
        q1 = np.percentile(train_descriptors_array, 25, axis=0)
        q3 = np.percentile(train_descriptors_array, 75, axis=0)
        iqr = q3 - q1
        outlier_mask = (train_descriptors_array < q1 - 3 * iqr) | (train_descriptors_array > q3 + 3 * iqr)
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            print(f"   ⚠️  Detected {outlier_count} potential outliers (using IQR method)")
    
    # 使用 RobustScaler（基于中位数和 IQR）
    scaler = RobustScaler()
    train_descriptors_normalized = scaler.fit_transform(train_descriptors_array)
    
    # 转换为 torch tensor
    desc_median = torch.tensor(scaler.center_, dtype=torch.float32)
    desc_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    
    # 避免除零
    desc_scale = torch.clamp(desc_scale, min=1e-8)
    
    # 归一化所有数据集（使用训练集统计信息）
    for graph in train_graphs + val_graphs + test_graphs:
        if hasattr(graph, 'descriptor') and graph.descriptor is not None:
            desc = graph.descriptor
            if isinstance(desc, torch.Tensor):
                desc = desc.cpu()
                if desc.dim() > 1:
                    desc = desc.squeeze()
            else:
                desc = torch.tensor(desc, dtype=torch.float32)
            
            # 手动归一化（与 RobustScaler 一致）
            graph.descriptor = (desc - desc_median) / desc_scale
    
    if rank == 0:
        print("✅ Robust normalization completed")
        print(f"   Median range: [{desc_median.min().item():.4f}, {desc_median.max().item():.4f}]")
        print(f"   Scale range: [{desc_scale.min().item():.4f}, {desc_scale.max().item():.4f}]")
    
    return desc_median, desc_scale


# ============================================================================
# 改进 2: 根据数据集特性动态调整超参数搜索空间
# ============================================================================

def get_dataset_characteristics(dataset_name, train_graphs, task_type='classification'):
    """
    分析数据集特性，用于动态调整超参数搜索空间
    
    返回:
        dict: 包含数据集大小、不平衡程度等特性
    """
    dataset_size = len(train_graphs)
    
    characteristics = {
        'dataset_size': dataset_size,
        'is_small': dataset_size < 1000,
        'is_medium': 1000 <= dataset_size < 5000,
        'is_large': dataset_size >= 5000,
    }
    
    if task_type == 'classification':
        # 计算类别分布
        train_targets = [g.y.item() if hasattr(g.y, 'item') else float(g.y) for g in train_graphs]
        unique_classes, class_counts = np.unique(train_targets, return_counts=True)
        
        if len(unique_classes) == 2:
            pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
            neg_count = int(class_counts[0])
            imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
            
            characteristics.update({
                'is_binary': True,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'imbalance_ratio': imbalance_ratio,
                'is_extremely_imbalanced': imbalance_ratio > 100,
                'is_highly_imbalanced': imbalance_ratio > 50,
                'is_moderately_imbalanced': 10 < imbalance_ratio <= 50,
                'is_balanced': imbalance_ratio <= 10,
            })
        else:
            characteristics.update({
                'is_binary': False,
                'num_classes': len(unique_classes),
            })
    else:
        # 回归任务：分析目标值分布
        train_targets = [g.y.item() if hasattr(g.y, 'item') else float(g.y) for g in train_graphs]
        targets_array = np.array(train_targets)
        
        characteristics.update({
            'target_mean': float(np.mean(targets_array)),
            'target_std': float(np.std(targets_array)),
            'target_min': float(np.min(targets_array)),
            'target_max': float(np.max(targets_array)),
            'target_range': float(np.max(targets_array) - np.min(targets_array)),
        })
    
    return characteristics


def suggest_hyperparameters_dynamic(trial, characteristics, primary_metric=None):
    """
    根据数据集特性动态建议超参数
    
    这个函数应该在 optuna_serach_mod.py 的 objective 函数中使用
    """
    dataset_size = characteristics['dataset_size']
    is_small = characteristics.get('is_small', False)
    is_large = characteristics.get('is_large', False)
    
    # 1. 模型大小：根据数据集大小调整
    if is_small:
        # 小数据集：使用更小的模型防止过拟合
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        ffn_expansion_factor = trial.suggest_int("ffn_expansion_factor", 2, 4)
    elif is_large:
        # 大数据集：可以使用更大的模型
        hidden_dim = trial.suggest_int("hidden_dim", 256, 512)
        num_layers = trial.suggest_int("num_layers", 5, 10)
        ffn_expansion_factor = trial.suggest_int("ffn_expansion_factor", 4, 8)
    else:
        # 中等数据集：标准配置
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
        num_layers = trial.suggest_int("num_layers", 3, 8)
        ffn_expansion_factor = trial.suggest_int("ffn_expansion_factor", 2, 6)
    
    # 2. DMP Steps：根据数据集大小调整
    if is_small:
        dmp_steps = trial.suggest_int("dmp_steps", 1, 3)
    elif is_large:
        dmp_steps = trial.suggest_int("dmp_steps", 3, 6)
    else:
        dmp_steps = trial.suggest_int("dmp_steps", 2, 5)
    
    # 3. 注意力头数：确保能整除 hidden_dim
    valid_heads = [h for h in [2, 4, 8, 16, 32] if hidden_dim % h == 0]
    if not valid_heads:
        valid_heads = [1, 2, 4]  # Fallback
    num_heads = trial.suggest_categorical("num_heads", valid_heads)
    
    # 4. Batch Size：根据数据集大小调整
    if is_small:
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    elif is_large:
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    else:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # 5. 学习率：根据模型大小调整
    model_size = hidden_dim * num_layers
    if model_size < 1000:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    elif model_size < 3000:
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    
    # 6. 对于分类任务，根据不平衡程度调整
    if characteristics.get('is_binary', False):
        imbalance_ratio = characteristics.get('imbalance_ratio', 1.0)
        is_extremely_imbalanced = characteristics.get('is_extremely_imbalanced', False)
        
        # Label Smoothing：对不平衡数据集可能有用
        if is_extremely_imbalanced:
            label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1, step=0.02)
        else:
            label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
        
        # Drop Path Rate：对不平衡数据集可能需要更多正则化
        if is_extremely_imbalanced:
            drop_path_rate = trial.suggest_float("drop_path_rate", 0.1, 0.3, step=0.05)
        else:
            drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.2, step=0.05)
    else:
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
        drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.2, step=0.05)
    
    # 7. 添加缺失的超参数
    descriptor_dropout = trial.suggest_float("descriptor_dropout", 0.0, 0.3, step=0.05)
    fingerprint_dropout = trial.suggest_float("fingerprint_dropout", 0.0, 0.3, step=0.05)
    
    # 8. Mixup：对小数据集或不平衡数据集可能有用
    if is_small or characteristics.get('is_highly_imbalanced', False):
        use_mixup = trial.suggest_categorical("use_mixup", [True, False])
        if use_mixup:
            mixup_alpha = trial.suggest_float("mixup_alpha", 0.5, 4.0)
        else:
            mixup_alpha = 2.0
    else:
        use_mixup = False
        mixup_alpha = 2.0
    
    return {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'ffn_expansion_factor': ffn_expansion_factor,
        'dmp_steps': dmp_steps,
        'batch_size': batch_size,
        'lr': lr,
        'label_smoothing': label_smoothing,
        'drop_path_rate': drop_path_rate,
        'descriptor_dropout': descriptor_dropout,
        'fingerprint_dropout': fingerprint_dropout,
        'use_mixup': use_mixup,
        'mixup_alpha': mixup_alpha,
    }


# ============================================================================
# 改进 3: 根据数据集特性选择损失函数
# ============================================================================

def select_loss_function_dynamic(characteristics, primary_metric=None):
    """
    根据数据集特性和主要指标动态选择损失函数
    
    返回:
        dict: 损失函数配置
    """
    loss_config = {
        'use_bce_for_imbalanced': False,
        'use_focal_loss': False,
        'use_class_balanced_focal_loss': False,
        'auto_pos_weight': False,
        'label_smoothing': 0.0,
    }
    
    if not characteristics.get('is_binary', False):
        # 多分类：使用标准 CrossEntropyLoss
        return loss_config
    
    imbalance_ratio = characteristics.get('imbalance_ratio', 1.0)
    is_extremely_imbalanced = characteristics.get('is_extremely_imbalanced', False)
    is_highly_imbalanced = characteristics.get('is_highly_imbalanced', False)
    
    # 根据主要指标选择
    if primary_metric == 'pr_auc':
        # PR-AUC 数据集：使用 Focal Loss 或 Class-Balanced Focal Loss
        if is_extremely_imbalanced:
            loss_config['use_class_balanced_focal_loss'] = True
        else:
            loss_config['use_focal_loss'] = True
    elif primary_metric == 'roc_auc':
        # ROC-AUC 数据集：根据不平衡程度选择
        if is_extremely_imbalanced:
            loss_config['use_class_balanced_focal_loss'] = True
        elif is_highly_imbalanced:
            loss_config['use_focal_loss'] = True
        else:
            loss_config['use_bce_for_imbalanced'] = True
            loss_config['auto_pos_weight'] = True
    else:
        # 默认：根据不平衡程度选择
        if is_extremely_imbalanced:
            loss_config['use_class_balanced_focal_loss'] = True
        elif is_highly_imbalanced:
            loss_config['use_focal_loss'] = True
        else:
            loss_config['use_bce_for_imbalanced'] = True
            loss_config['auto_pos_weight'] = True
    
    return loss_config


# ============================================================================
# 改进 4: 根据数据集特性调整早停参数
# ============================================================================

def get_early_stopping_config(characteristics, primary_metric=None):
    """
    根据数据集特性调整早停参数
    
    返回:
        dict: 早停配置
    """
    dataset_size = characteristics['dataset_size']
    is_small = characteristics.get('is_small', False)
    imbalance_ratio = characteristics.get('imbalance_ratio', 1.0)
    is_extremely_imbalanced = characteristics.get('is_extremely_imbalanced', False)
    
    # 基础配置
    config = {
        'use_smart_early_stopping': True,
        'initial_patience': 20,
        'max_patience': 50,
        'auroc_improvement_threshold': 0.005,
        'f1_improvement_threshold': 0.001,
    }
    
    # 根据数据集特性调整
    if is_extremely_imbalanced:
        # 极度不平衡：需要更多耐心
        config['initial_patience'] = 40
        config['max_patience'] = 80
        config['auroc_improvement_threshold'] = 0.001  # 更小的阈值
        config['f1_improvement_threshold'] = 0.0005
    elif is_small:
        # 小数据集：防止过早停止
        config['initial_patience'] = 30
        config['max_patience'] = 60
        config['auroc_improvement_threshold'] = 0.003
    elif imbalance_ratio > 50:
        # 高度不平衡
        config['initial_patience'] = 30
        config['max_patience'] = 60
        config['auroc_improvement_threshold'] = 0.002
    
    # 根据主要指标调整
    if primary_metric == 'pr_auc':
        # PR-AUC 可能需要更多耐心
        config['initial_patience'] = max(config['initial_patience'], 30)
        config['max_patience'] = max(config['max_patience'], 60)
    
    return config


# ============================================================================
# 改进 5: 根据数据集大小调整 Warmup Epochs
# ============================================================================

def get_warmup_epochs(dataset_size, num_epochs):
    """
    根据数据集大小和总训练轮数调整 warmup epochs
    """
    if dataset_size < 1000:
        # 小数据集：更少 warmup
        warmup_epochs = max(3, num_epochs // 20)
    elif dataset_size < 5000:
        # 中等数据集：标准 warmup
        warmup_epochs = max(5, num_epochs // 15)
    else:
        # 大数据集：更多 warmup
        warmup_epochs = max(5, num_epochs // 10)
    
    # 确保不超过总轮数的 20%
    warmup_epochs = min(warmup_epochs, max(1, int(num_epochs * 0.2)))
    
    return warmup_epochs


# ============================================================================
# 改进 6: 权重初始化
# ============================================================================

def init_model_weights(model, init_method='xavier_uniform'):
    """
    初始化模型权重
    
    Args:
        model: PyTorch 模型
        init_method: 初始化方法 ('xavier_uniform', 'kaiming_uniform', 'orthogonal')
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model


# ============================================================================
# 改进 7: 数据质量检查
# ============================================================================

def check_data_quality(train_graphs, val_graphs, test_graphs, task_type='classification', rank=0):
    """
    检查数据质量并生成报告
    
    返回:
        dict: 数据质量报告
    """
    report = {
        'train_size': len(train_graphs),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
        'total_size': len(train_graphs) + len(val_graphs) + len(test_graphs),
    }
    
    if task_type == 'classification':
        # 检查类别分布
        train_targets = [g.y.item() if hasattr(g.y, 'item') else float(g.y) for g in train_graphs]
        val_targets = [g.y.item() if hasattr(g.y, 'item') else float(g.y) for g in val_graphs]
        test_targets = [g.y.item() if hasattr(g.y, 'item') else float(g.y) for g in test_graphs]
        
        train_unique, train_counts = np.unique(train_targets, return_counts=True)
        val_unique, val_counts = np.unique(val_targets, return_counts=True)
        test_unique, test_counts = np.unique(test_targets, return_counts=True)
        
        report['train_class_distribution'] = dict(zip(train_unique.tolist(), train_counts.tolist()))
        report['val_class_distribution'] = dict(zip(val_unique.tolist(), val_counts.tolist()))
        report['test_class_distribution'] = dict(zip(test_unique.tolist(), test_counts.tolist()))
        
        if len(train_unique) == 2:
            pos_count = int(train_counts[1]) if len(train_counts) > 1 else 0
            neg_count = int(train_counts[0])
            imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
            report['imbalance_ratio'] = imbalance_ratio
            report['is_extremely_imbalanced'] = imbalance_ratio > 100
    else:
        # 回归任务：检查目标值分布
        train_targets = [g.y.item() if hasattr(g.y, 'item') else float(g.y) for g in train_graphs]
        targets_array = np.array(train_targets)
        
        report['target_statistics'] = {
            'mean': float(np.mean(targets_array)),
            'std': float(np.std(targets_array)),
            'min': float(np.min(targets_array)),
            'max': float(np.max(targets_array)),
            'median': float(np.median(targets_array)),
        }
        
        # 检查异常值
        q1 = np.percentile(targets_array, 25)
        q3 = np.percentile(targets_array, 75)
        iqr = q3 - q1
        outliers = (targets_array < q1 - 3 * iqr) | (targets_array > q3 + 3 * iqr)
        report['outlier_count'] = int(outliers.sum())
        report['outlier_ratio'] = float(outliers.sum() / len(targets_array))
    
    # 检查图特征
    if len(train_graphs) > 0:
        sample_graph = train_graphs[0]
        report['has_pos'] = hasattr(sample_graph, 'pos') and sample_graph.pos is not None
        report['has_descriptor'] = hasattr(sample_graph, 'descriptor') and sample_graph.descriptor is not None
        report['has_fingerprint'] = hasattr(sample_graph, 'fingerprint') and sample_graph.fingerprint is not None
        
        if hasattr(sample_graph, 'x'):
            report['node_feature_dim'] = sample_graph.x.shape[1] if sample_graph.x is not None else 0
        if hasattr(sample_graph, 'edge_attr'):
            report['edge_feature_dim'] = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0
    
    if rank == 0:
        print("\n📊 Data Quality Report:")
        print(f"   Dataset sizes: Train={report['train_size']}, Val={report['val_size']}, Test={report['test_size']}")
        if task_type == 'classification' and 'imbalance_ratio' in report:
            print(f"   Imbalance ratio: {report['imbalance_ratio']:.2f}")
            if report.get('is_extremely_imbalanced', False):
                print(f"   ⚠️  Extremely imbalanced dataset!")
        elif task_type == 'regression' and 'target_statistics' in report:
            stats = report['target_statistics']
            print(f"   Target range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"   Target mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
            if report.get('outlier_count', 0) > 0:
                print(f"   ⚠️  Detected {report['outlier_count']} outliers ({report['outlier_ratio']*100:.1f}%)")
    
    return report


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：如何在 train_edmpnn.py 中使用这些改进
    
    # 1. 数据质量检查
    # characteristics = get_dataset_characteristics(dataset_name, train_graphs, task_type)
    # quality_report = check_data_quality(train_graphs, val_graphs, test_graphs, task_type, rank)
    
    # 2. 稳健的描述符归一化
    # desc_median, desc_scale = robust_normalize_descriptors(train_graphs, val_graphs, test_graphs, rank)
    
    # 3. 动态早停配置
    # early_stopping_config = get_early_stopping_config(characteristics, primary_metric)
    
    # 4. 动态损失函数选择
    # loss_config = select_loss_function_dynamic(characteristics, primary_metric)
    
    # 5. 动态 warmup epochs
    # warmup_epochs = get_warmup_epochs(len(train_graphs), num_epochs)
    
    # 6. 权重初始化
    # model = init_model_weights(model, init_method='xavier_uniform')
    
    print("✅ Improvement examples loaded. See comments for usage.")

