# train_edmpnn_new.sh 改動方案

## 📋 改動目標

創建一個新腳本 `train_edmpnn_new.sh`，用於訓練**移除 output_norm 並使用簡單 Linear 層**的模型版本。

## 🔍 分析現有架構

### 當前 train_edmpnn.sh 的結構

1. **調用 train_edmpnn.py**：腳本本身只是調用 Python 訓練腳本
2. **模型定義在 models/edmpnn_model.py**：實際的模型架構在這裡
3. **需要修改的地方**：
   - 模型文件（models/edmpnn_model.py）中的 `output_norm` 和 `output_proj`
   - 或者創建新的模型文件

## 🎯 改動方案

### 方案 1：創建新的模型文件（推薦）

**優點**：
- 不影響原有代碼
- 可以同時保留兩個版本
- 易於對比和切換

**步驟**：
1. 創建 `models/edmpnn_model_new.py`（複製自 `edmpnn_model.py`）
2. 修改 `project_graph_features` 方法，移除 `output_norm`
3. 修改 `output_proj` 為簡單的 `nn.Linear`（或保持多層但移除 output_norm）
4. 創建 `train_edmpnn_new.sh`，調用新的模型文件

### 方案 2：通過參數控制（不推薦）

**缺點**：
- 需要修改現有的 `train_edmpnn.py`
- 增加代碼複雜度
- 不符合"不更動原本檔案"的要求

## 📝 具體改動內容

### 1. 創建新模型文件：`models/edmpnn_model_new.py`

**改動點 1：移除 output_norm 定義**
```python
# 原代碼（第 924 行）：
self.output_norm = nn.LayerNorm(self.graph_repr_dim)

# 新代碼：刪除這一行
```

**改動點 2：簡化 output_proj（選項 A：簡單 Linear）**
```python
# 原代碼（第 925-930 行）：
self.output_proj = nn.Sequential(
    nn.Linear(self.graph_repr_dim, hidden_dim // 2),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, output_dim)
)

# 新代碼（選項 A：簡單 Linear，類似 fusion_model）：
self.output_proj = nn.Linear(self.graph_repr_dim, output_dim)
```

**或保持多層但移除 output_norm（選項 B）**
```python
# 新代碼（選項 B：保持多層但移除 output_norm）：
self.output_proj = nn.Sequential(
    nn.Linear(self.graph_repr_dim, hidden_dim // 2),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, output_dim)
)
# 注意：沒有 output_norm
```

**改動點 3：修改 project_graph_features 方法**
```python
# 原代碼（第 1098-1103 行）：
def project_graph_features(self, graph_features):
    graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
    logits_input = self.output_norm(graph_features)  # 移除這行
    logits = self.output_proj(logits_input)
    return logits

# 新代碼：
def project_graph_features(self, graph_features):
    graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
    logits = self.output_proj(graph_features)  # 直接使用 graph_features
    return logits
```

### 2. 創建新訓練腳本：`train_edmpnn_new.sh`

**改動點 1：更新標題和說明**
```bash
# 原代碼（第 4 行）：
# AEGNN-M Training Script (Optimized Version 10 - TDC Multi-Seed Training)

# 新代碼：
# AEGNN-M Training Script (NEW VERSION - Without output_norm, Using Simple Linear)
```

**改動點 2：更新調用的 Python 腳本**
```bash
# 原代碼（第 358 行）：
python3 scripts/train_edmpnn.py

# 新代碼：
python3 scripts/train_edmpnn_new.py
```

**改動點 3：更新輸出目錄**
```bash
# 原代碼（第 351 行）：
local save_dir="checkpoints/${dataset_name}_optuna_final/seed${seed}"

# 新代碼：
local save_dir="checkpoints/${dataset_name}_optuna_final_new/seed${seed}"
```

**改動點 4：更新日誌目錄**
```bash
# 原代碼（第 354 行）：
local log_dir="runs/${dataset_name}_optuna_final/seed${seed}"

# 新代碼：
local log_dir="runs/${dataset_name}_optuna_final_new/seed${seed}"
```

### 3. 創建新訓練 Python 腳本：`scripts/train_edmpnn_new.py`

**改動點：導入新模型**
```python
# 原代碼（第 33 行）：
from models.edmpnn_model import create_aegnn_model

# 新代碼：
from models.edmpnn_model_new import create_aegnn_model
```

**其他部分保持不變**（因為只是模型定義改變）

## 🎯 推薦方案

### 選項 A：簡單 Linear（最接近 fusion_model）

**優點**：
- 最簡單，最接近 fusion_model 的做法
- 參數最少
- 計算最快

**缺點**：
- 可能表達能力較弱
- 需要重新訓練和 Optuna

**代碼**：
```python
self.output_proj = nn.Linear(self.graph_repr_dim, output_dim)
```

### 選項 B：保持多層但移除 output_norm（平衡）

**優點**：
- 保持一定的表達能力
- 只移除 output_norm，其他結構不變
- 可能性能更好

**缺點**：
- 仍然需要重新訓練和 Optuna

**代碼**：
```python
self.output_proj = nn.Sequential(
    nn.Linear(self.graph_repr_dim, hidden_dim // 2),
    nn.SiLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, output_dim)
)
# 注意：沒有 output_norm
```

## 📊 文件結構

```
AEGNN-M_TDC/
├── models/
│   ├── edmpnn_model.py          # 原有模型（不變）
│   └── edmpnn_model_new.py      # 新模型（移除 output_norm）
├── scripts/
│   ├── train_edmpnn.py          # 原有訓練腳本（不變）
│   └── train_edmpnn_new.py      # 新訓練腳本（導入新模型）
├── train_edmpnn.sh              # 原有腳本（不變）
└── train_edmpnn_new.sh          # 新腳本（調用新訓練腳本）
```

## 🔧 實施步驟

1. **創建新模型文件**：
   - 複製 `models/edmpnn_model.py` → `models/edmpnn_model_new.py`
   - 修改 `output_norm` 和 `project_graph_features`

2. **創建新訓練 Python 腳本**：
   - 複製 `scripts/train_edmpnn.py` → `scripts/train_edmpnn_new.py`
   - 修改導入語句

3. **創建新 Shell 腳本**：
   - 複製 `train_edmpnn.sh` → `train_edmpnn_new.sh`
   - 修改標題、調用腳本、輸出目錄

## ❓ 需要確認的問題

1. **output_proj 的結構**：
   - 選項 A：簡單 Linear（`nn.Linear(graph_repr_dim, output_dim)`）
   - 選項 B：保持多層（`nn.Sequential(Linear → SiLU → Dropout → Linear)`）
   - **您希望使用哪個？**

2. **輸出目錄命名**：
   - 建議：`checkpoints/{dataset_name}_optuna_final_new/seed{seed}`
   - **您希望使用什麼命名？**

3. **是否保留反標準化邏輯**：
   - 移除 output_norm 後，理論上不需要反標準化
   - 但可以保留作為安全措施
   - **您希望保留還是移除？**

## 📝 總結

**改動範圍**：
- ✅ 創建新文件，不修改原有文件
- ✅ 新模型文件：`models/edmpnn_model_new.py`
- ✅ 新訓練腳本：`scripts/train_edmpnn_new.py`
- ✅ 新 Shell 腳本：`train_edmpnn_new.sh`

**主要改動**：
1. 移除 `output_norm` 定義
2. 修改 `project_graph_features` 方法（移除 output_norm 調用）
3. 可選：簡化 `output_proj` 為簡單 Linear

**等待確認**：
1. output_proj 的結構選擇（選項 A 或 B）
2. 輸出目錄命名
3. 是否保留反標準化邏輯

