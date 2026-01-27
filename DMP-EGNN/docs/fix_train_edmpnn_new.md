# 修復 train_edmpnn_new 的具體步驟

## 問題 1: RobustScaler 導致分佈不匹配

### 修復方法：改回標準標準化

**文件**: `scripts/train_edmpnn_new.py`

**位置**: 約第 3662-3670 行

**原代碼**:
```python
# Use RobustScaler for robust normalization (based on median and IQR)
# RobustScaler is resistant to outliers, making it ideal for TDC datasets
scaler = RobustScaler()
train_descriptors_normalized = scaler.fit_transform(train_descriptors_array)

# Extract statistics from RobustScaler
# RobustScaler uses median (center_) and IQR-based scale (scale_)
desc_median = torch.tensor(scaler.center_, dtype=torch.float32)
desc_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
```

**修改為**:
```python
# Use standard normalization (mean/std) to match Optuna optimization
# Calculate statistics from training set only (after clipping)
train_descriptors_array_f64 = train_descriptors_array.astype(np.float64)

# Calculate mean using float64 to avoid overflow
desc_mean_f64 = train_descriptors_array_f64.mean(axis=0)

# Calculate std using a more stable method to avoid overflow
centered = train_descriptors_array_f64 - desc_mean_f64
max_abs_centered = 1e6
centered = np.clip(centered, -max_abs_centered, max_abs_centered)
desc_var_f64 = np.mean(centered ** 2, axis=0)
desc_std_f64 = np.sqrt(desc_var_f64)
desc_std_f64 = np.clip(desc_std_f64, 0.0, 1e6)

# Convert to float32 and create tensors
desc_mean = torch.tensor(desc_mean_f64.astype(np.float32), dtype=torch.float32)
desc_std = torch.tensor(desc_std_f64.astype(np.float32), dtype=torch.float32)
```

**同時修改標準化邏輯** (約第 3702 行):
```python
# 原代碼:
# RobustScaler normalization: (x - median) / IQR_scale
graph.descriptor = (desc - desc_median) / desc_scale

# 修改為:
# Standard normalization: (x - mean) / std
graph.descriptor = (desc - desc_mean) / desc_std
```

**並更新統計輸出** (約第 3688-3691 行):
```python
# 原代碼:
print(f"   Descriptor median range: [{desc_median.min().item():.4f}, {desc_median.max().item():.4f}]")
print(f"   Descriptor scale range: [{desc_scale.min().item():.4f}, {desc_scale.max().item():.4f}]")
print(f"   ℹ️  Using median and IQR-based scaling (robust to outliers)")

# 修改為:
print(f"   Descriptor mean range: [{desc_mean.min().item():.4f}, {desc_mean.max().item():.4f}]")
print(f"   Descriptor std range: [{desc_std.min().item():.4f}, {desc_std.max().item():.4f}]")
print(f"   ℹ️  Using standard normalization (mean/std)")
```

**並更新完成消息** (約第 3737 行):
```python
# 原代碼:
print("✅ RobustScaler normalization completed")

# 修改為:
print("✅ Standard normalization completed")
```

**移除 RobustScaler import** (約第 29 行):
```python
# 刪除或註釋掉:
# from sklearn.preprocessing import RobustScaler
```

---

## 問題 2: 權重初始化過於保守

### 修復方法：調整初始化參數

**文件**: `scripts/train_edmpnn_new.py`

**位置**: 約第 59-85 行

**原代碼**:
```python
def init_weights_advanced(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            lname = name.lower()
            if ("output_proj" in lname) or ("output" in lname):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # 過小
            elif "embedding" in lname:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif ("gate" in lname) or ("attention" in lname):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 過小
            else:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
```

**修改為**:
```python
def init_weights_advanced(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            lname = name.lower()
            if ("output_proj" in lname) or ("output" in lname):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 從 0.1 改為 0.5
            elif "embedding" in lname:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif ("gate" in lname) or ("attention" in lname):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)  # 從 0.01 改為 0.1
            else:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
```

---

## 問題 3: 缺少模型初始化 Seed

### 修復方法：添加 model_init_seed 參數

**文件**: `train_edmpnn_new.sh`

**位置**: 約第 362-383 行，在 `train_single_seed` 函數中

**在 `load_optuna_mod_params` 之後添加**:
```bash
# Model initialization seed: ensure each seed has unique model initialization
# Formula: seed * 1000 + seed (e.g., seed 1 → 1001, seed 2 → 2002, etc.)
# This ensures different random initialization for each seed while maintaining reproducibility
local model_init_seed=$((seed * 1000 + seed))
```

**在 train_cmd 中添加 seed 參數** (約第 397-434 行):
```bash
local train_cmd=(
    python3 scripts/train_edmpnn_new.py
    --tdc_dataset "${dataset_name}"
    --tdc_seed "${seed}"
    --seed "${model_init_seed}"  # 添加這一行
    --model_type "${task_type}"
    # ... 其他參數 ...
)
```

**並更新輸出信息** (約第 385-389 行):
```bash
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}🌱 [GPU ${gpu_id}] Training Seed ${GREEN}${seed}${NC} / 5"
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "   Data Split Seed: ${seed} (TDC dataset)"
echo -e "   Model Init Seed: ${model_init_seed} (for reproducibility)"  # 添加這一行
echo -e "   Model: Dim ${hidden_dim}, Layers ${num_layers}, Heads ${num_heads}, DMP Steps ${dmp_steps}"
echo -e "   Config: LR ${lr}, Batch ${batch_size}, WD ${weight_decay}, Dropout ${dropout}"
```

---

## 完整修復檢查清單

- [ ] 修改 `train_edmpnn_new.py` 中的 RobustScaler 為標準標準化
- [ ] 更新所有相關的標準化邏輯（train/val/test）
- [ ] 調整 `init_weights_advanced()` 中的 gain 和 std 值
- [ ] 在 `train_edmpnn_new.sh` 中添加 `model_init_seed` 參數
- [ ] 在 train_cmd 中添加 `--seed "${model_init_seed}"`
- [ ] 測試修復後的版本（建議先用一個小數據集測試）

---

## 測試建議

1. **小規模測試**: 選擇 `ames` 數據集，只訓練 seed 1
   ```bash
   ./train_edmpnn_new.sh ames
   ```

2. **對比實驗**: 比較修復前後的訓練曲線和最終性能

3. **檢查日誌**: 確認：
   - 描述符標準化使用 mean/std 而非 median/IQR
   - 模型初始化 seed 正確設置
   - 訓練 loss 正常下降

---

## 如果問題仍然存在

如果修復後問題仍然存在，請檢查：

1. **Optuna 優化時使用的標準化方法**: 確認 Optuna 優化時是否也使用了 RobustScaler
2. **超參數文件**: 檢查 `optuna_edmpnn_results_new/all_best_hyperparameters_mod.json` 中的參數是否合理
3. **數據預處理**: 確認訓練數據與 Optuna 優化時使用的數據一致
4. **模型架構**: 確認模型架構與 Optuna 優化時一致



