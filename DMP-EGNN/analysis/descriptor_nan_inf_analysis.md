# Descriptor NaN/Inf 異常數值分析報告

## 問題概述

根據 `preprocess_quality_all_v2.csv` 和 `preprocess_quality_all_v2.json` 的分析結果，所有檢測到的 NaN/Inf 異常數值都來自 `descriptor` 欄位，而其他欄位（`x`, `pos`, `edge_attr`, `y`）都沒有發現異常。

## Descriptor 產生 NaN/Inf 的主要原因

### 1. RDKit 描述符計算的固有問題

**重要概念**：Descriptor 不是簡單的「記錄分子特性的數值」，而是**通過化學理論和經驗公式計算得出的衍生指標**。這些計算自然涉及各種數學運算。

RDKit 的 `CalcMolDescriptors` 函數會計算約 217 個不同的分子描述符，這些描述符包括：

- **拓撲描述符**：基於分子圖結構的計算
- **幾何描述符**：基於分子幾何形狀的計算
- **電子描述符**：基於電子結構的計算
- **物理化學描述符**：如 LogP、分子量等

**為什麼需要複雜運算？**

Descriptor 是為了**預測或描述分子的化學性質**而設計的，這些性質通常無法直接測量，需要通過計算得出。例如：

- **LogP（親脂性）**：`LogP = log10(分子在辛醇中的濃度 / 分子在水中的濃度)`
  - 需要對數運算來壓縮數值範圍
  - 當濃度為 0 時會產生 Inf 或 NaN
  
- **分子極性表面積（PSA）**：需要計算 3D 分子結構中每個原子的暴露表面積
  - 涉及 3D 幾何計算、表面積積分等複雜運算
  
- **複雜度指標**：可能涉及比率計算，如「平均分支長度 = 總分支長度 / 分支數」
  - 當分支數為 0 時，分母為 0 → Inf

**問題根源**：
- 某些描述符涉及**除法運算**（例如比率描述符），當分母為 0 時會產生 Inf
- 某些描述符涉及**對數運算**（例如 LogP），當輸入 ≤ 0 時會產生 NaN
- 某些描述符涉及**開方運算**，當輸入為負數時會產生 NaN
- 對於**特殊分子結構**（如極大分子、複雜環狀結構、異常鍵合等），某些描述符可能無法正確計算

### 2. 數值溢出問題

- 某些描述符的計算結果可能**超出浮點數表示範圍**，導致 Inf
- 在計算過程中，**中間結果的累積**可能導致數值溢出

### 3. 標準化過程中的問題

在 `train_edmpnn.py` 的標準化過程中（使用 RobustScaler）：

```python
# RobustScaler normalization: (x - median) / IQR_scale
graph.descriptor = (desc - desc_median) / desc_scale
```

**可能產生的問題**：
- 如果某個描述符的 **IQR（四分位距）為 0**，則 `desc_scale` 為 0，除以 0 會產生 Inf
- 如果 `desc_median` 或 `desc_scale` 本身是 NaN/Inf，標準化後也會產生 NaN/Inf
- 即使原始 descriptor 中沒有 NaN/Inf，標準化過程也可能引入異常值

### 4. 數據處理流程中的缺陷

雖然在 `data_utils.py` 的 `_generate_descriptor` 函數中有處理 NaN/Inf 的邏輯：

```python
# Replace NaN and Inf with 0
if np.isnan(val) or np.isinf(val):
    val = 0.0
```

**但這個處理可能不夠完善**：
- 只處理了單個值，但沒有處理整個數組的邊緣情況
- 在後續的數據轉換（numpy → torch）過程中，可能又產生了 NaN/Inf
- 標準化過程中的計算可能重新引入異常值

## 為什麼其他欄位沒有問題？

- **`x` (節點特徵)**：是基於原子類型的 one-hot 編碼，數值穩定
- **`pos` (3D 座標)**：是通過力場優化得到的座標，數值範圍有限
- **`edge_attr` (邊特徵)**：是基於鍵類型的編碼，數值穩定
- **`y` (標籤)**：是預先定義的標籤值，不涉及複雜計算

只有 **`descriptor`** 涉及複雜的分子描述符計算，因此容易產生 NaN/Inf。

## 影響範圍

根據分析數據：
- **ames** 數據集：每個 split 約有 1-3 個異常樣本
- **bbb_martins** 數據集：每個 split 約有 3-29 個異常樣本
- **bioavailability_ma** 數據集：每個 split 約有 1-11 個異常樣本
- **caco2_wang** 數據集：每個 split 約有 1-7 個異常樣本

異常樣本通常具有以下特徵：
- 分子結構較複雜（節點數較多）
- 可能包含特殊的功能基團或環狀結構

## 解決方案建議

### 1. 改進 descriptor 生成邏輯

在 `data_utils.py` 的 `_generate_descriptor` 函數中：
- 加強 NaN/Inf 檢測和處理
- 對整個 descriptor 數組進行檢查，而不只是單個值
- 使用 `np.nan_to_num` 進行批量處理

### 2. 改進標準化過程

在 `train_edmpnn.py` 的標準化過程中：
- 在計算 median 和 IQR 之前，先過濾掉 NaN/Inf 值
- 對於 IQR 為 0 的描述符，使用替代的標準化方法（如使用 std 代替 IQR）
- 添加 epsilon 值防止除以 0：`desc_scale = desc_scale + epsilon`

### 3. 預處理階段過濾

在 `preprocess_tdc_data_new.py` 中：
- 已經實現了 `_filter_nonfinite_descriptor` 函數來過濾異常樣本
- 建議在預處理階段就過濾掉這些樣本，而不是在訓練時處理

### 4. 使用更穩定的描述符子集

- 可以考慮只使用數值穩定的描述符子集
- 或者使用自定義的描述符計算方法，避免使用容易產生 NaN/Inf 的描述符

## 結論

Descriptor 產生 NaN/Inf 的主要原因是：
1. **RDKit 描述符計算的固有問題**：某些描述符對於特殊分子結構會產生異常值
2. **標準化過程中的數值不穩定**：除以 0 或使用 NaN/Inf 進行計算
3. **數據處理流程中的缺陷**：雖然有處理邏輯，但可能不夠完善

建議在預處理階段就過濾掉這些異常樣本，並改進 descriptor 生成和標準化的邏輯，以提高數據質量和訓練穩定性。

