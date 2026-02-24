# 常數特徵特殊處理詳細說明

## 一、什麼是"特殊處理"？

**特殊處理**是指在標準化過程中，對**常數特徵**（在訓練集中所有樣本值都相同的特徵）採用不同於正常特徵的處理方式，避免它們在標準化後變為 0。

## 二、為什麼需要特殊處理？

### 當前問題

```python
# 標準化公式
normalized = (x - median) / scale

# 對於常數特徵（所有值都是 c）
normalized = (c - c) / scale = 0 / scale = 0
```

**結果**：常數特徵標準化後全部變為 0，在訓練時無法發揮作用。

### 特殊處理的目的

1. **避免特徵變為 0**：保持特徵有非零值
2. **保留資訊**：盡可能保留特徵的相對位置資訊
3. **模型可學習**：讓模型能夠從這些特徵中學習（雖然資訊量有限）

## 三、特殊處理的完整流程

### 步驟 1：檢測常數特徵

```python
# 方法 1：使用 IQR 檢測（推薦，與 RobustScaler 一致）
q1 = np.percentile(descriptors_array, 25, axis=0)
q3 = np.percentile(descriptors_array, 75, axis=0)
iqr = q3 - q1
constant_mask = iqr < 1e-6  # 閾值可調整

# 方法 2：使用標準差檢測
std = np.std(descriptors_array, axis=0)
constant_mask = std < 1e-6

# 方法 3：檢查所有值是否完全相同
constant_mask = np.array([
    np.allclose(descriptors_array[:, i], descriptors_array[0, i], atol=1e-6) 
    for i in range(descriptors_array.shape[1])
])
```

### 步驟 2：標準化變異特徵

```python
# 使用 RobustScaler 標準化所有特徵
scaler = RobustScaler()
descriptors_normalized = scaler.fit_transform(descriptors_array)
```

### 步驟 3：對常數特徵進行特殊處理

根據不同的策略，有多種處理方法：

## 四、特殊處理方法詳解

### 方法 A：設置為固定值（最簡單）

```python
# 將常數特徵設置為固定非零值（如 1.0）
descriptors_normalized[:, constant_mask] = 1.0
```

**優點**：
- 實現簡單
- 所有常數特徵統一處理
- 不會變為 0

**缺點**：
- 所有常數特徵值相同，無法區分
- 可能引入虛假資訊

**適用場景**：快速修復，不需要額外資訊

---

### 方法 B：保持原始值（不標準化）

```python
# 常數特徵保持原始值，不進行標準化
descriptors_normalized[:, constant_mask] = descriptors_array[:, constant_mask]
```

**優點**：
- 保留原始資訊
- 實現簡單

**缺點**：
- 與標準化特徵尺度不一致
- 可能影響模型訓練（數值範圍差異大）

**適用場景**：當原始值範圍合理時

---

### 方法 C：設置為 0.0（當前做法，不推薦）

```python
# 當前做法：標準化後自然變為 0
# 不需要額外處理，但特徵失去作用
```

**優點**：簡單

**缺點**：特徵完全失去作用

---

### 方法 D：設置為小的隨機值（不推薦）

```python
# 為常數特徵添加小的隨機雜訊
np.random.seed(42)
for i in range(descriptors_array.shape[1]):
    if constant_mask[i]:
        descriptors_normalized[:, i] = np.random.normal(0, 0.01, descriptors_array.shape[0])
```

**優點**：
- 增加變異，模型可以學習

**缺點**：
- 引入虛假資訊
- 可能誤導模型
- 不穩定（每次運行結果不同）

**適用場景**：不推薦使用

---

### 方法 E：使用全局統計量標準化（最佳，類似 fusion_model）

```python
# 假設我們有全局統計量（從更大的數據集計算）
global_median = ...  # 全局中位數
global_iqr = ...     # 全局 IQR

# 對常數特徵使用全局統計量標準化
for i in range(descriptors_array.shape[1]):
    if constant_mask[i]:
        constant_value = descriptors_array[0, i]  # 所有值都相同
        normalized_value = (constant_value - global_median[i]) / global_iqr[i]
        descriptors_normalized[:, i] = normalized_value
```

**優點**：
- 保留相對位置資訊（相對於全局分佈）
- 類似 fusion_model 的做法
- 特徵值有意義

**缺點**：
- 需要全局統計量（需要額外計算或使用預定義值）

**適用場景**：最佳方案，但需要全局統計量

---

### 方法 F：基於全局範圍的相對位置

```python
# 假設我們知道特徵的全局範圍
global_min = ...  # 全局最小值
global_max = ...  # 全局最大值

# 計算常數值在全局範圍中的相對位置
for i in range(descriptors_array.shape[1]):
    if constant_mask[i]:
        constant_value = descriptors_array[0, i]
        # 計算相對位置 [0, 1]
        relative_position = (constant_value - global_min[i]) / (global_max[i] - global_min[i])
        # 標準化到 [-1, 1] 範圍
        normalized_value = 2 * relative_position - 1
        descriptors_normalized[:, i] = normalized_value
```

**優點**：
- 保留相對位置資訊
- 不需要完整的統計量

**缺點**：
- 需要知道全局範圍

**適用場景**：次佳方案，需要全局範圍資訊

---

## 五、實際代碼實現示例

### 在 train_edmpnn.py 中的實現（方法 A：最簡單）

```python
# 在現有代碼的 3654 行之後添加

# 檢測常數特徵（IQR < threshold）
constant_feature_mask = desc_scale < 1e-6  # 或使用原始 IQR 檢測

if rank == 0 and constant_feature_mask.any():
    num_constant = constant_feature_mask.sum().item()
    print(f"   ⚠️  Detected {num_constant} constant features (IQR < 1e-6)")
    print(f"   ℹ️  Applying special handling: setting to 1.0")

# 在標準化循環中（約 3684 行）
for graph in train_graphs:
    if hasattr(graph, 'descriptor') and graph.descriptor is not None:
        desc = graph.descriptor
        if isinstance(desc, torch.Tensor):
            desc = desc.cpu()
            if desc.dim() > 1:
                desc = desc.squeeze()
            
            # 標準化所有特徵
            normalized_desc = (desc - desc_median) / desc_scale
            
            # 特殊處理：對常數特徵設置為 1.0
            normalized_desc[constant_feature_mask] = 1.0
            
            graph.descriptor = normalized_desc
        else:
            desc_tensor = torch.tensor(desc, dtype=torch.float32)
            normalized_desc = (desc_tensor - desc_median) / desc_scale
            normalized_desc[constant_feature_mask] = 1.0
            graph.descriptor = normalized_desc
```

### 在 train_edmpnn.py 中的實現（方法 E：使用全局統計量）

```python
# 需要預先計算或載入全局統計量
# 這裡假設我們有全局統計量（可以從所有數據集計算或使用預定義值）

# 檢測常數特徵
constant_feature_mask = desc_scale < 1e-6

if rank == 0 and constant_feature_mask.any():
    num_constant = constant_feature_mask.sum().item()
    print(f"   ⚠️  Detected {num_constant} constant features")
    print(f"   ℹ️  Applying special handling: using global statistics")

# 假設我們有全局統計量（需要預先計算）
# global_desc_median = ...  # 從所有數據集計算
# global_desc_scale = ...    # 從所有數據集計算

# 在標準化循環中
for graph in train_graphs:
    if hasattr(graph, 'descriptor') and graph.descriptor is not None:
        desc = graph.descriptor
        if isinstance(desc, torch.Tensor):
            desc = desc.cpu()
            if desc.dim() > 1:
                desc = desc.squeeze()
            
            # 標準化變異特徵
            normalized_desc = (desc - desc_median) / desc_scale
            
            # 特殊處理：對常數特徵使用全局統計量
            for i in range(len(desc)):
                if constant_feature_mask[i]:
                    constant_value = desc[i].item()
                    # 使用全局統計量標準化
                    normalized_value = (constant_value - global_desc_median[i]) / global_desc_scale[i]
                    normalized_desc[i] = normalized_value
            
            graph.descriptor = normalized_desc
```

## 六、方法選擇建議

| 方法 | 複雜度 | 效果 | 推薦度 | 適用場景 |
|------|--------|------|--------|---------|
| 方法 A（固定值 1.0） | ⭐ | ⭐⭐ | ⭐⭐⭐ | 快速修復，不需要額外資訊 |
| 方法 B（保持原值） | ⭐ | ⭐⭐ | ⭐⭐ | 原始值範圍合理時 |
| 方法 E（全局統計量） | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 最佳方案，有全局統計量 |
| 方法 F（相對位置） | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 有全局範圍資訊時 |

## 七、推薦實現方案

### 方案 1：快速修復（方法 A）

**適合**：需要快速解決問題，不需要額外資訊

**實現**：約 10-15 行代碼

### 方案 2：最佳方案（方法 E）

**適合**：追求最佳效果，願意計算全局統計量

**實現**：約 20-30 行代碼 + 全局統計量計算

## 八、注意事項

1. **閾值選擇**：`1e-6` 是常用的閾值，可根據實際情況調整
2. **一致性**：確保 train/val/test 使用相同的處理方式
3. **記錄**：建議記錄哪些特徵被特殊處理，便於分析
4. **測試**：修改後需要測試模型性能是否改善




