# 修改 train_edmpnn_new.py 以基於 Primary Metric 保存最佳模型

## 需要修改的地方

### 1. 在 `__init__` 方法中（約第 830-1000 行）

添加：
- 讀取 primary_metric 配置
- 初始化追蹤 primary metric 的列表
- 初始化 best_primary_metric 相關變量

### 2. 在訓練循環中（約第 2365-2620 行）

修改：
- 在每個 epoch 的 validation 後，提取並保存 primary metric
- 根據 primary metric 來保存 best_model 和更新 best_epoch
- 而不是只基於 val_loss

### 3. 在保存 training_history.json 時（約第 2674-2689 行）

修改：
- 保存每個 epoch 的 primary metric 列表
- 使用 primary metric 的最佳 epoch 作為 best_epoch

## 具體修改步驟

### 步驟 1：在 `__init__` 中添加 primary metric 追蹤

位置：約第 987-988 行（在 `self.train_losses = []` 和 `self.val_losses = []` 之後）

添加：
```python
# Primary metric tracking
self.primary_metric = None
self.val_spearman = []  # For regression tasks
self.val_aurocs = []    # For classification tasks
self.val_f1_scores = [] # For classification tasks
self.val_pr_aucs = []   # For classification tasks
self.best_primary_metric_value = None
self.best_primary_metric_epoch = -1

# Load primary metric from config
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'dataset_primary_metrics.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        dataset_configs = config.get('dataset_primary_metrics', {})
        if self.dataset_name:
            dataset_config = dataset_configs.get(self.dataset_name.lower(), {})
            self.primary_metric = dataset_config.get('primary_metric', None)
            if self.primary_metric and rank == 0:
                print(f"📌 Primary metric for {self.dataset_name}: {self.primary_metric}")
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
```

### 步驟 2：在訓練循環中提取並保存 primary metric

位置：約第 2379 行（在 `self.val_losses.append(val_loss)` 之後）

添加：
```python
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
    
    # Check if primary metric improved
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
            self.best_primary_metric_value = primary_metric_value
            self.best_primary_metric_epoch = epoch
```

### 步驟 3：根據 primary metric 保存 best_model

位置：約第 2476-2504 行（Smart Early Stopping 的 improved 檢查）和約第 2572-2602 行（Traditional Early Stopping）

修改邏輯：
- 不僅檢查 `early_stop_info['improved']`（基於 val_loss）
- 還要檢查 primary metric 是否改善
- 如果 primary metric 改善，保存 best_model

### 步驟 4：在保存 training_history.json 時包含 primary metric

位置：約第 2674-2689 行

修改：
```python
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

# Add primary metric lists if available
if self.primary_metric:
    history_data['primary_metric'] = self.primary_metric
    history_data['best_primary_metric_value'] = self.best_primary_metric_value
    history_data['best_primary_metric_epoch'] = self.best_primary_metric_epoch
    
    if self.primary_metric == 'spearman' and self.val_spearman:
        history_data['val_spearman'] = self.val_spearman
    elif self.primary_metric in ['roc_auc', 'auroc'] and self.val_aurocs:
        history_data['val_aurocs'] = self.val_aurocs
    elif self.primary_metric == 'f1' and self.val_f1_scores:
        history_data['val_f1_scores'] = self.val_f1_scores
    elif self.primary_metric == 'pr_auc' and self.val_pr_aucs:
        history_data['val_pr_aucs'] = self.val_pr_aucs
```

## 注意事項

1. **回歸任務的 spearman**：需要在 validate 方法中計算 spearman（目前可能沒有）
2. **分類任務的 metrics**：已經在 validate 方法中計算了（roc_auc, f1, pr_auc）
3. **向後兼容**：如果沒有 primary_metric 配置，回退到使用 val_loss

## 驗證

修改後需要驗證：
1. 每個 epoch 的 primary metric 是否正確保存
2. best_model 是否在 primary metric 最佳時保存
3. best_epoch 是否記錄為 primary metric 最佳的 epoch


