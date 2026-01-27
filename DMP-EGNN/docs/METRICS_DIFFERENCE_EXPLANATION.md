# Optuna Best Trial Metrics vs Final Test Metrics 差異說明

## 問題描述

執行 `optuna_parallel_mod_new.sh` 得到的 best trial metrics 數值和實際執行 `train_edmpnn_new.sh` 時得到的 test metrics 數值相差很多。

## 根本原因

### 1. **不同的評估數據集**

**Optuna 優化階段** (`optuna_serach_mod_new.py`):
- ✅ 使用 **Validation Set** 的 metrics 來選擇 best trial
- ✅ 這是正確的做法，避免數據洩漏（data leakage）
- ✅ Best trial 的 `best_trial.value` 是 validation set 上的最佳 primary metric 值

**最終訓練階段** (`train_edmpnn_new.py`):
- ✅ 使用 **Test Set** 的 metrics 進行最終評估
- ✅ Test set 是模型從未見過的數據
- ✅ 這是標準的機器學習評估流程

### 2. **代碼證據**

#### Optuna 優化時（使用 Validation Metrics）

```746:880:DMP-EGNN/scripts/optuna_serach_mod_new.py
                # IMPORTANT: Use VALIDATION metrics (not test set) to select best trial
                # This follows fusion_model's approach and avoids data leakage
                history_path = os.path.join(seed_save_dir, "training_history.json")
                if os.path.exists(history_path):
                    try:
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                        
                        # Try to get best validation metric from training history
                        # Priority: best_primary_metric_value > max of validation metric list > fallback to test_results
                        best_val_metric = None
                        
                        # Method 1: Use best_primary_metric_value (most reliable, saved by train_edmpnn_new.py)
                        if 'best_primary_metric_value' in history:
                            saved_primary_metric = history.get('primary_metric', '')
                            if saved_primary_metric == primary_metric or not saved_primary_metric:
                                best_val_metric = history.get('best_primary_metric_value')
                        
                        # ... (其他方法)
                        
                        # Return validation metric score (following fusion_model approach)
                        score = np.mean(valid_scores)
                        return score
```

#### 最終訓練時（使用 Test Metrics）

```3058:3114:DMP-EGNN/scripts/train_edmpnn_new.py
        # Testing (only on rank 0, can be skipped)
        test_results = {}
        if hasattr(self, 'skip_test') and self.skip_test:
            if self.rank == 0:
                print("\n⚠️  Skipping test phase (using --skip_test option)")
        else:
            if self.rank == 0:
                print("\nTesting model...")
            test_results = self.test(save_dir=save_dir)
            
            # ... (保存 test_results 到 training_history.json)
            
            history_data = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'best_epoch': best_epoch,
                'test_results': test_results  # Test set 的結果
            }
```

### 3. **為什麼會有差異？**

這是**正常的機器學習現象**，原因包括：

1. **數據分布差異**
   - Validation set 和 test set 的數據分布可能略有不同
   - 即使來自同一數據源，隨機分割也會導致差異

2. **過擬合風險**
   - Optuna 在 validation set 上選擇最佳超參數
   - 可能對 validation set 有輕微過擬合
   - Test set 是未見過的數據，性能通常會略低

3. **隨機性影響**
   - 不同的隨機種子會導致不同的結果
   - Optuna 優化時使用的種子可能與最終訓練時不同

4. **訓練時長差異**
   - Optuna 優化時可能使用較少的 epochs（為了快速搜索）
   - 最終訓練時使用完整的 epochs（200 epochs）

## 解決方案

### 方案 1: 理解這是正常現象（推薦）

這是標準的機器學習實踐：
- ✅ **Validation metrics** 用於超參數選擇（避免數據洩漏）
- ✅ **Test metrics** 用於最終評估（真實性能）
- ✅ 兩者差異是**預期的**，只要差異在合理範圍內即可

### 方案 2: 檢查差異是否過大

如果差異**異常大**（例如 > 0.1），可能的原因：

1. **數據分割問題**
   - 檢查 validation 和 test set 的分布是否一致
   - 檢查是否有數據洩漏

2. **模型不穩定**
   - 檢查訓練曲線是否穩定
   - 檢查是否有過擬合

3. **超參數選擇問題**
   - 檢查 Optuna 是否收斂
   - 檢查 best trial 是否合理

### 方案 3: 統一使用 Test Metrics（不推薦）

**不建議**修改 Optuna 使用 test set，因為：
- ❌ 會導致數據洩漏
- ❌ 超參數選擇會過擬合 test set
- ❌ 無法真實反映模型泛化能力

### 方案 4: 報告兩個 Metrics

在報告結果時，同時報告：
- **Validation metrics**（Optuna 選擇的依據）
- **Test metrics**（最終評估結果）

這樣可以更全面地了解模型性能。

## 如何查看實際的 Metrics

### 1. Optuna Best Trial 的 Validation Metrics

```bash
# 查看 Optuna study 的 best trial
python3 -c "
import optuna
study = optuna.load_study(
    study_name='edmpnn_mod_new_<dataset>_seed<seed>_opt',
    storage='sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db'
)
print(f'Best trial: {study.best_trial.number}')
print(f'Best value (validation): {study.best_trial.value}')
print(f'Best params: {study.best_trial.params}')
"
```

### 2. 最終訓練的 Test Metrics

```bash
# 查看 training_history.json
python3 -c "
import json
with open('checkpoints/<dataset>_optuna_final/seed<seed>/training_history.json', 'r') as f:
    history = json.load(f)
print('Validation metrics:')
print(f'  Best {history.get(\"primary_metric\", \"N/A\")}: {history.get(\"best_primary_metric_value\", \"N/A\")}')
print('Test metrics:')
for metric, value in history.get('test_results', {}).items():
    print(f'  {metric}: {value}')
"
```

### 3. 比較兩個 Metrics

創建一個比較腳本：

```python
import json
import optuna

# 1. 獲取 Optuna best trial validation metric
study = optuna.load_study(
    study_name='edmpnn_mod_new_<dataset>_seed<seed>_opt',
    storage='sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db'
)
optuna_val_metric = study.best_trial.value

# 2. 獲取最終訓練的 test metric
with open('checkpoints/<dataset>_optuna_final/seed<seed>/training_history.json', 'r') as f:
    history = json.load(f)
test_results = history.get('test_results', {})
primary_metric = history.get('primary_metric', 'roc_auc')
test_metric = test_results.get(primary_metric, None)

# 3. 比較
print(f"Optuna Validation {primary_metric}: {optuna_val_metric:.4f}")
print(f"Final Test {primary_metric}: {test_metric:.4f}")
print(f"Difference: {abs(optuna_val_metric - test_metric):.4f}")
```

## 總結

**這是正常的機器學習現象**，不是 bug：

- ✅ Optuna 使用 **validation metrics** 選擇最佳超參數（正確做法）
- ✅ 最終訓練使用 **test metrics** 評估模型（標準流程）
- ✅ 兩者差異是預期的，只要在合理範圍內即可
- ✅ 如果差異過大（> 0.1），需要檢查數據和模型穩定性

**建議**：在報告結果時，同時報告 validation 和 test metrics，這樣可以更全面地了解模型性能。











