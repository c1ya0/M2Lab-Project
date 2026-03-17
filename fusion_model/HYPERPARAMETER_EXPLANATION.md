# 超參數詳細說明與項目對比

## 一、超參數詳細說明

### 1. 優化器相關超參數

#### 1.1 學習率 (Learning Rate, `lr`)
- **作用**：控制模型參數更新的步長大小
- **影響**：
  - 過大：訓練不穩定，可能無法收斂或發散
  - 過小：訓練速度慢，可能陷入局部最優
- **fusion_model 範圍**：`1e-5` 到 `1e-2`（對數尺度）
- **DMP-EGNN 範圍**：`1e-5` 到 `3e-3`（對數尺度）
- **對比**：DMP-EGNN 的上限更保守（3e-3 vs 1e-2）

#### 1.2 批次大小 (Batch Size, `batch_size`)
- **作用**：每次訓練迭代使用的樣本數量
- **影響**：
  - 較大：訓練更穩定，但需要更多記憶體，梯度估計更準確
  - 較小：記憶體需求低，但梯度估計噪音較大，訓練可能不穩定
- **fusion_model 選項**：`[16, 32, 64]`
- **DMP-EGNN 選項**：`[32, 64]`
- **對比**：DMP-EGNN 不搜尋較小的批次大小（16），可能因為模型較大

#### 1.3 權重衰減 (Weight Decay, `weight_decay`)
- **作用**：L2 正則化係數，防止過擬合
- **影響**：
  - 較大：更強的正則化，可能欠擬合
  - 較小：較弱的正則化，可能過擬合
- **fusion_model 範圍**：`1e-5` 到 `1e-2`（對數尺度）
- **DMP-EGNN 範圍**：`1e-8` 到 `1e-3`（對數尺度）
- **對比**：DMP-EGNN 的下限更小（1e-8 vs 1e-5），允許更弱的正則化

#### 1.4 梯度裁剪 (Gradient Clipping Norm, `grad_clip_norm`) - **僅 DMP-EGNN**
- **作用**：限制梯度的大小，防止梯度爆炸
- **影響**：
  - 較大：允許更大的梯度更新
  - 較小：更保守的更新，訓練更穩定但可能較慢
- **DMP-EGNN 範圍**：`0.1` 到 `1.0`
- **fusion_model**：未使用此參數

---

### 2. MLP（多層感知機）相關超參數

#### 2.1 MLP 隱藏層維度 (`mlp_hidden_dim` / `hidden_dim`)
- **作用**：MLP 隱藏層的神經元數量，決定模型容量
- **影響**：
  - 較大：模型容量大，表達能力強，但可能過擬合，計算成本高
  - 較小：模型容量小，計算快，但可能欠擬合
- **fusion_model 選項**：`[16, 32, 64, 128, 256]`
- **DMP-EGNN 選項**：`[64, 128, 256]`（作為 `hidden_dim`）
- **對比**：DMP-EGNN 不搜尋較小的維度（16, 32），可能因為模型架構需要較大容量

#### 2.2 MLP 層數 (`mlp_num_layers` / `num_layers`)
- **作用**：MLP 的深度，決定模型的非線性轉換能力
- **影響**：
  - 較深：可以學習更複雜的特徵，但可能出現梯度消失/爆炸
  - 較淺：訓練容易，但表達能力有限
- **fusion_model 範圍**：`1` 到 `5`
- **DMP-EGNN 範圍**：`2` 到 `8`（作為 `num_layers`）
- **對比**：DMP-EGNN 允許更深的網絡（最多 8 層 vs 5 層）

#### 2.3 MLP 激活函數 (`mlp_activation` / `activation`)
- **作用**：引入非線性，使模型能學習複雜模式
- **常見選擇**：
  - `ReLU`：最常用，計算快，但可能出現死亡 ReLU
  - `GELU`：平滑的 ReLU，在 Transformer 中常用
  - `SiLU`：Sigmoid Linear Unit，類似 GELU
  - `ELU`：指數線性單元，輸出可為負值
  - `LeakyReLU`：解決死亡 ReLU 問題
- **fusion_model 選項**：`['relu', 'gelu']`
- **DMP-EGNN 選項**：`['SiLU', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'tanh']`
- **對比**：DMP-EGNN 提供更多激活函數選擇

#### 2.4 MLP Dropout (`mlp_dropout` / `dropout`)
- **作用**：隨機將部分神經元輸出設為 0，防止過擬合
- **影響**：
  - 較大：更強的正則化，但可能欠擬合
  - 較小：較弱的正則化，可能過擬合
- **fusion_model 範圍**：`0.0` 到 `0.5`，步長 `0.05`
- **DMP-EGNN 範圍**：`0.0` 到 `0.5`，步長 `0.05`
- **對比**：兩者相同

#### 2.5 MLP 正規化類型 (`mlp_norm_type`) - **僅 fusion_model**
- **作用**：標準化層的輸出，穩定訓練
- **選項**：
  - `LayerNorm`：對每個樣本的所有特徵進行標準化（適合序列數據）
  - `BatchNorm`：對批次內的所有樣本進行標準化（需要較大批次）
- **fusion_model 選項**：`['LayerNorm', 'BatchNorm']`
- **DMP-EGNN**：未明確搜尋此參數（可能固定使用某種正規化）

---

### 3. GCN（圖卷積網絡）相關超參數 - **僅 fusion_model**

#### 3.1 GCN 隱藏層維度 (`gcn_hidden_dim`)
- **作用**：GCN 層的隱藏特徵維度
- **選項**：`[16, 32, 64, 128, 256]`

#### 3.2 GCN 輸出維度 (`gcn_output_dim`)
- **作用**：GCN 最終輸出的特徵維度（用於與其他模態融合）
- **選項**：`[16, 32, 64, 128, 256]`

#### 3.3 GCN 層數 (`gcn_num_layers`)
- **作用**：GCN 的深度，決定消息傳遞的跳數
- **範圍**：`1` 到 `3`
- **注意**：層數過多可能導致過平滑（over-smoothing）

#### 3.4 GCN 激活函數 (`gcn_activation`)
- **選項**：`['relu', 'gelu']`

#### 3.5 GCN Dropout (`gcn_dropout`)
- **範圍**：`0.0` 到 `0.5`，步長 `0.05`

#### 3.6 GCN 正規化類型 (`gcn_norm_type`)
- **選項**：`['LayerNorm', 'BatchNorm']`

#### 3.7 GCN 池化方式 (`gcn_pooling`)
- **作用**：將圖節點特徵聚合為圖級特徵
- **選項**：
  - `mean`：平均池化，對所有節點特徵取平均
  - `max`：最大池化，取每個維度的最大值
  - `add`：求和池化，對所有節點特徵求和
- **fusion_model 選項**：`['mean', 'max', 'add']`
- **DMP-EGNN**：使用 `pool_type`，選項相同

---

### 4. MPN（消息傳遞網絡）相關超參數

#### 4.1 MPN 隱藏層大小 (`mpn_hidden_size`)
- **作用**：MPN 的隱藏特徵維度
- **fusion_model 選項**：`[64, 128, 256, 300]`
- **DMP-EGNN**：使用 `hidden_dim`（類似概念）

#### 4.2 MPN 深度 (`mpn_depth` / `dmp_steps`)
- **作用**：消息傳遞的步數/深度
- **fusion_model 範圍**：`2` 到 `6`（作為 `mpn_depth`）
- **DMP-EGNN 範圍**：`1` 到 `6`（作為 `dmp_steps`）
- **對比**：DMP-EGNN 允許更淺的網絡（1 步 vs 2 步）

#### 4.3 MPN Dropout (`mpn_dropout`)
- **fusion_model 範圍**：`0.0` 到 `0.3`，步長 `0.05`
- **DMP-EGNN**：使用統一的 `dropout` 參數

#### 4.4 MPN 激活函數 (`mpn_activation`)
- **fusion_model 選項**：`['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']`
- **DMP-EGNN**：使用統一的 `activation` 參數

#### 4.5 MPN 聚合方式 (`mpn_aggregation` / `pool_type`)
- **作用**：如何聚合分子圖的節點特徵為圖級特徵
- **fusion_model 選項**：`['mean', 'sum', 'norm']`（作為 `mpn_aggregation`）
- **DMP-EGNN 選項**：`['mean', 'sum', 'norm']`（作為 `pool_type`）
- **對比**：兩者相同

---

### 5. DMP-EGNN 專有超參數

#### 5.1 注意力頭數 (`num_heads`)
- **作用**：多頭注意力機制的頭數，允許模型關注不同類型的特徵
- **影響**：
  - 較多：模型可以學習更豐富的表示，但計算成本高
  - 較少：計算快，但表達能力可能受限
- **DMP-EGNN 選項**：`[4, 8]`
- **fusion_model**：未使用（可能不使用注意力機制）

#### 5.2 學習率調度器類型 (`scheduler_type`)
- **作用**：控制學習率隨訓練進程的變化
- **選項**：
  - `cosine`：餘弦退火，學習率平滑下降
  - `step`：階梯式下降，在特定 epoch 降低學習率
  - `plateau`：當驗證指標不再改善時降低學習率
- **DMP-EGNN 選項**：`['cosine', 'step', 'plateau']`
- **fusion_model**：未明確搜尋（可能使用固定調度器）

#### 5.3 最小學習率 (`min_lr`)
- **作用**：學習率調度器的最低學習率下限
- **DMP-EGNN 範圍**：`1e-7` 到 `1e-5`（對數尺度）
- **fusion_model**：未使用

#### 5.4 Warmup Epochs (`warmup_epochs`)
- **作用**：訓練初期逐漸增加學習率的 epoch 數
- **影響**：幫助模型在訓練初期更穩定地收斂
- **DMP-EGNN 範圍**：`5` 到 `15`，步長 `5`
- **fusion_model**：未使用

#### 5.5 Drop Path Rate (`drop_path_rate`)
- **作用**：隨機深度（Stochastic Depth）的正則化技術，隨機跳過某些層
- **影響**：類似 Dropout，但作用於整個層而非神經元
- **DMP-EGNN 範圍**：`0.0` 到 `0.2`，步長 `0.05`
- **fusion_model**：未使用

#### 5.6 Alpha (`alpha`)
- **作用**：可能用於控制某些組件的權重或混合比例
- **DMP-EGNN 範圍**：`0.1` 到 `0.3`，步長 `0.05`
- **fusion_model**：未使用

#### 5.7 FFN 擴展因子 (`ffn_expansion_factor`)
- **作用**：前饋網絡（Feed-Forward Network）的擴展倍數
- **影響**：決定 FFN 中間層相對於輸入層的維度擴展
- **DMP-EGNN 選項**：`[2, 4, 6, 8]`
- **fusion_model**：未使用

#### 5.8 旋轉增強 (`rotate_aug`)
- **作用**：是否使用旋轉數據增強（可能用於 3D 分子結構）
- **DMP-EGNN 選項**：`[True, False]`
- **fusion_model**：未使用

#### 5.9 描述符 Dropout (`descriptor_dropout`)
- **作用**：對輸入描述符特徵的 Dropout
- **影響**：防止模型過度依賴描述符特徵
- **DMP-EGNN 範圍**：`0.0` 到 `0.3`，步長 `0.05`
- **fusion_model**：未使用

#### 5.10 Mixup (`use_mixup`, `mixup_alpha`)
- **作用**：數據增強技術，混合兩個樣本及其標籤
- **影響**：提高模型泛化能力，減少過擬合
- **DMP-EGNN**：
  - `use_mixup`：`[True, False]`
  - `mixup_alpha`：`0.1` 到 `0.5`（當 `use_mixup=True` 時）
- **fusion_model**：未使用

---

## 二、項目對比總結

### 共同使用的超參數

| 超參數 | fusion_model | DMP-EGNN | 備註 |
|--------|-------------|----------|------|
| `lr` | ✓ | ✓ | 學習率 |
| `batch_size` | ✓ | ✓ | 批次大小 |
| `weight_decay` | ✓ | ✓ | 權重衰減 |
| `hidden_dim` | ✓ (mlp_hidden_dim) | ✓ | 隱藏層維度 |
| `num_layers` | ✓ (mlp_num_layers) | ✓ | 層數 |
| `dropout` | ✓ (mlp_dropout) | ✓ | Dropout |
| `activation` | ✓ (mlp_activation) | ✓ | 激活函數 |
| `pool_type` | ✓ (gcn_pooling/mpn_aggregation) | ✓ | 池化/聚合方式 |

### fusion_model 獨有超參數

1. **MLP 正規化類型** (`mlp_norm_type`)：`['LayerNorm', 'BatchNorm']`
2. **GCN 相關參數**（當模型包含 GCN 時）：
   - `gcn_hidden_dim`, `gcn_output_dim`, `gcn_num_layers`
   - `gcn_activation`, `gcn_dropout`, `gcn_norm_type`, `gcn_pooling`
3. **MPN 相關參數**（當模型包含 MPN 時）：
   - `mpn_hidden_size`, `mpn_depth`, `mpn_dropout`
   - `mpn_activation`, `mpn_aggregation`

### DMP-EGNN 獨有超參數

1. **訓練穩定性**：
   - `grad_clip_norm`：梯度裁剪
   - `warmup_epochs`：學習率預熱
   - `drop_path_rate`：隨機深度

2. **學習率調度**：
   - `scheduler_type`：調度器類型
   - `min_lr`：最小學習率

3. **模型架構**：
   - `num_heads`：注意力頭數
   - `ffn_expansion_factor`：FFN 擴展因子
   - `alpha`：混合權重
   - `dmp_steps`：消息傳遞步數

4. **數據增強**：
   - `rotate_aug`：旋轉增強
   - `use_mixup`, `mixup_alpha`：Mixup 增強

5. **特徵處理**：
   - `descriptor_dropout`：描述符 Dropout

---

## 三、設計差異分析

### 1. 模型架構差異

- **fusion_model**：
  - 多模態融合架構（GCN + MegaMolBART + Descriptors）
  - 針對不同模態分別搜尋超參數
  - 更靈活的模態組合

- **DMP-EGNN**：
  - 統一的圖神經網絡架構（基於 D-MPNN 和 EGNN）
  - 使用統一的超參數（如 `hidden_dim`, `dropout`）
  - 更專注於單一架構的優化

### 2. 訓練策略差異

- **fusion_model**：
  - 較簡單的訓練設置
  - 固定學習率或簡單調度器
  - 較少使用數據增強

- **DMP-EGNN**：
  - 更複雜的訓練策略
  - 多種學習率調度器選擇
  - 使用 Mixup 和旋轉增強
  - 梯度裁剪和隨機深度等正則化技術

### 3. 超參數搜尋範圍差異

- **fusion_model**：
  - 允許較小的模型（如 `mlp_hidden_dim: [16, 32, ...]`）
  - 較淺的網絡（`mlp_num_layers: 1-5`）
  - 較大的學習率範圍（`lr: 1e-5 到 1e-2`）

- **DMP-EGNN**：
  - 傾向於較大的模型（`hidden_dim: [64, 128, 256]`）
  - 允許更深的網絡（`num_layers: 2-8`）
  - 更保守的學習率範圍（`lr: 1e-5 到 3e-3`）

---

## 四、建議

1. **融合兩者優點**：
   - 從 DMP-EGNN 學習：梯度裁剪、學習率調度、數據增強
   - 從 fusion_model 學習：多模態超參數獨立搜尋

2. **根據任務選擇**：
   - 簡單任務：使用 fusion_model 的較小模型範圍
   - 複雜任務：使用 DMP-EGNN 的較大模型和更複雜訓練策略

3. **超參數搜尋策略**：
   - 先搜尋核心超參數（lr, batch_size, hidden_dim）
   - 再搜尋正則化參數（dropout, weight_decay）
   - 最後搜尋高級技術（mixup, drop_path_rate）








