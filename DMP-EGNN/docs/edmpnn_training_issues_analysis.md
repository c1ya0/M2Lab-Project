# E-DMPNN 模型訓練問題分析與優化建議

## 問題概述

`train_edmpnn_new.sh` 的訓練結果比 `train_edmpnn.sh` 還差，可能存在以下結構性或配置性問題。

## 關鍵差異分析

### 1. 描述符標準化方法差異 ⚠️ **可能的主要問題**

**train_edmpnn.py (舊版本)**:
- 使用標準的 **mean/std 標準化**: `(x - mean) / std`
- 使用百分位數裁剪處理異常值
- 標準化後的分佈接近標準正態分佈

**train_edmpnn_new.py (新版本)**:
- 使用 **RobustScaler**: `(x - median) / IQR_scale`
- 基於中位數和四分位距 (IQR) 的標準化
- 對異常值更魯棒，但可能改變數據分佈

**潛在問題**:
- RobustScaler 使用中位數而非均值，可能導致標準化後的數據分佈與 Optuna 優化時不一致
- IQR-based scaling 可能使某些特徵的尺度變化過大或過小
- 如果 Optuna 優化時使用的是標準標準化，而訓練時使用 RobustScaler，會導致分佈不匹配

### 2. 權重初始化策略差異 ⚠️ **可能影響訓練穩定性**

**train_edmpnn.py**:
- 使用 PyTorch 默認初始化（通常是 Xavier/Kaiming）

**train_edmpnn_new.py**:
- 使用 `init_weights_advanced()` 自定義初始化：
  - Output layer: `gain=0.1` (非常小的初始化)
  - Gate/attention layers: `std=0.01` (非常小的初始化)
  - Embedding: `gain=1.0` (標準)

**潛在問題**:
- Output layer 的 `gain=0.1` 可能過於保守，導致初始輸出接近零，影響梯度流動
- Gate/attention 的 `std=0.01` 可能過小，導致注意力機制初始化不良

### 3. 模型初始化 Seed 差異

**train_edmpnn.sh**:
- 有 `model_init_seed` 參數：`seed * 1000 + seed`
- 確保每個 seed 有獨特的模型初始化

**train_edmpnn_new.sh**:
- **缺少 `model_init_seed` 參數**
- 可能導致不同 seed 的模型初始化相同或相似

**潛在問題**:
- 如果所有 seed 使用相同的初始化，會降低模型多樣性
- 可能導致所有 seed 都收斂到相似的次優解

### 4. Rotation Augmentation 參數

**train_edmpnn_new.sh**:
- 支持 `rotation_prob` 和 `max_rotation_angle` 參數
- 這些參數在 Optuna 優化結果中存在

**train_edmpnn.sh**:
- 只支持基本的 `rotate_aug` 開關

**潛在問題**:
- 如果 rotation augmentation 配置不當，可能破壞分子的幾何結構信息

## 優化建議

### 建議 1: 統一描述符標準化方法 🔴 **高優先級**

**選項 A: 回退到標準標準化（推薦）**
```python
# 在 train_edmpnn_new.py 中，將 RobustScaler 改回標準標準化
# 確保與 Optuna 優化時一致
desc_mean = torch.tensor(train_descriptors_array.mean(axis=0), dtype=torch.float32)
desc_std = torch.tensor(train_descriptors_array.std(axis=0), dtype=torch.float32)
desc_std = torch.clamp(desc_std, min=1e-8)
```

**選項 B: 在 Optuna 優化時也使用 RobustScaler**
- 需要重新運行 Optuna 優化，確保訓練和優化使用相同的標準化方法

### 建議 2: 調整權重初始化策略 🟡 **中優先級**

**修改 `init_weights_advanced()` 函數**:
```python
def init_weights_advanced(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            lname = name.lower()
            if ("output_proj" in lname) or ("output" in lname):
                # 改為 gain=0.5 或 1.0，避免過於保守
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 從 0.1 改為 0.5
            elif "embedding" in lname:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif ("gate" in lname) or ("attention" in lname):
                # 改為 std=0.1，避免過小
                nn.init.normal_(m.weight, mean=0.0, std=0.1)  # 從 0.01 改為 0.1
            else:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
```

### 建議 3: 添加模型初始化 Seed 🟡 **中優先級**

**修改 `train_edmpnn_new.sh`**:
```bash
# 在 train_single_seed 函數中添加
local model_init_seed=$((seed * 1000 + seed))

# 在 train_cmd 中添加
--seed "${model_init_seed}"
```

**修改 `train_edmpnn_new.py`**:
- 確保 `args.seed` 被正確使用於模型初始化
- 檢查 `set_seed()` 函數是否在模型創建前被調用

### 建議 4: 檢查 Rotation Augmentation 配置 🟢 **低優先級**

**驗證 Optuna 優化結果中的 rotation 參數是否合理**:
- `rotation_prob` 應該在 0.0-1.0 之間
- `max_rotation_angle` 應該在 0.0-180.0 之間
- 對於分子數據，建議使用較小的角度（< 90度）

### 建議 5: 診斷步驟

1. **比較標準化後的描述符分佈**:
   ```python
   # 在兩個腳本中都添加統計輸出
   print(f"Normalized descriptor mean: {desc_mean.mean():.4f}")
   print(f"Normalized descriptor std: {desc_std.mean():.4f}")
   print(f"Normalized descriptor range: [{desc_normalized.min():.4f}, {desc_normalized.max():.4f}]")
   ```

2. **檢查初始權重分佈**:
   ```python
   # 在模型創建後添加
   for name, param in model.named_parameters():
       if 'weight' in name:
           print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}, min={param.min():.4f}, max={param.max():.4f}")
   ```

3. **比較訓練曲線**:
   - 檢查 loss 是否正常下降
   - 檢查梯度是否正常（無梯度爆炸或消失）
   - 檢查驗證指標是否正常提升

## 立即行動建議

### 優先級 1: 修復描述符標準化
1. 將 `train_edmpnn_new.py` 中的 RobustScaler 改回標準標準化
2. 或確認 Optuna 優化時也使用了 RobustScaler

### 優先級 2: 修復權重初始化
1. 調整 `init_weights_advanced()` 中的 gain 和 std 值
2. 或暫時禁用自定義初始化，使用 PyTorch 默認初始化

### 優先級 3: 添加模型初始化 Seed
1. 在 `train_edmpnn_new.sh` 中添加 `model_init_seed` 參數
2. 確保每個 seed 有獨特的模型初始化

## 測試建議

1. **小規模測試**: 選擇一個小數據集（如 `ames`），分別測試：
   - 標準標準化 vs RobustScaler
   - 自定義初始化 vs 默認初始化
   - 有 model_init_seed vs 無 model_init_seed

2. **對比實驗**: 使用相同的超參數，只改變一個變量，觀察性能差異

3. **檢查訓練日誌**: 比較兩個版本的訓練曲線，找出性能下降的具體階段

## 結論

最可能的問題是：
1. **RobustScaler 導致描述符分佈與 Optuna 優化時不一致**（最可能）
2. **權重初始化過於保守，影響訓練**（次可能）
3. **缺少 model_init_seed，降低模型多樣性**（可能）

建議先修復描述符標準化問題，這是最可能導致性能下降的原因。



