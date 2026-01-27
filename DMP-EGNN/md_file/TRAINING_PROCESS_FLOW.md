# 模型訓練過程在流程圖中的位置

## 📊 訓練過程完整流程圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         【訓練循環開始】                                  │
└─────────────────────────────────────────────────────────────────────────┘

for epoch in range(num_epochs):
    for batch in train_loader:
        
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【1. 數據載入】                                                    │
        │                                                                   │
        │  batch = {                                                        │
        │      'x': node_features,          # [N, node_features]          │
        │      'edge_index': edge_index,    # [2, E]                      │
        │      'edge_attr': edge_attr,      # [E, edge_features]         │
        │      'batch': batch_info,         # [N]                         │
        │      'y': targets,                # [batch_size]                │
        │      'pos': positions,             # [N, 3] (optional)           │
        │      'fingerprint': fingerprint,  # [batch_size, fp_dim] (opt) │
        │      'descriptor': descriptor     # [batch_size, desc_dim] (opt)│
        │  }                                                                │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【2. 前向傳播 (Forward Pass)】                                    │
        │                                                                   │
        │  model.train()  # 設置為訓練模式                                │
        │  optimizer.zero_grad()  # 清零梯度                               │
        │                                                                   │
        │  ┌───────────────────────────────────────────────────────────┐ │
        │  │ 2.1 模型前向傳播                                           │ │
        │  │                                                             │ │
        │  │  pred, attn_weights = model(                               │ │
        │  │      x=batch.x,                                            │ │
        │  │      edge_index=batch.edge_index,                           │ │
        │  │      edge_attr=batch.edge_attr,                             │ │
        │  │      batch=batch.batch,                                    │ │
        │  │      pos=batch.pos,                                        │ │
        │  │      fingerprint=batch.fingerprint,                        │ │
        │  │      descriptor=batch.descriptor                           │ │
        │  │  )                                                          │ │
        │  │                                                             │ │
        │  │  數據流過整個模型架構：                                      │ │
        │  │                                                             │ │
        │  │  ┌─────────────────────────────────────────────────────┐ │ │
        │  │  │ 【Embedding 層】                                     │ │ │
        │  │  │   x = node_embedding(batch.x)                       │ │ │
        │  │  │   edge_attr = edge_embedding(batch.edge_attr)       │ │ │
        │  │  └─────────────────────────────────────────────────────┘ │ │
        │  │                    │                                       │ │
        │  │                    ▼                                       │ │
        │  │  ┌─────────────────────────────────────────────────────┐ │ │
        │  │  │ 【GAT-EGNN 層 × num_layers】                         │ │ │
        │  │  │   for layer in aegnn_layers:                         │ │ │
        │  │  │       x, attn, pos = layer(x, edge_index, ...)      │ │ │
        │  │  └─────────────────────────────────────────────────────┘ │ │
        │  │                    │                                       │ │
        │  │                    ▼                                       │ │
        │  │  ┌─────────────────────────────────────────────────────┐ │ │
        │  │  │ 【Graph Pooling】                                    │ │ │
        │  │  │   node_mean = global_mean_pool(x, batch)            │ │ │
        │  │  │   coord_mean = global_mean_pool(pos, batch)        │ │ │
        │  │  └─────────────────────────────────────────────────────┘ │ │
        │  │                    │                                       │ │
        │  │                    ▼                                       │ │
        │  │  ┌─────────────────────────────────────────────────────┐ │ │
        │  │  │ 【Modality Processing】                               │ │ │
        │  │  │   h_processed = h_mlp(node_mean)                     │ │ │
        │  │  │   x_processed = x_mlp(coord_mean)                   │ │ │
        │  │  │   desc_processed = descriptor_mlp(descriptor)        │ │ │
        │  │  └─────────────────────────────────────────────────────┘ │ │
        │  │                    │                                       │ │
        │  │                    ▼                                       │ │
        │  │  ┌─────────────────────────────────────────────────────┐ │ │
        │  │  │ 【Feature Concatenation】                            │ │ │
        │  │  │   graph_features = cat([h, x, desc, ...])           │ │ │
        │  │  └─────────────────────────────────────────────────────┘ │ │
        │  │                    │                                       │ │
        │  │                    ▼                                       │ │
        │  │  ┌─────────────────────────────────────────────────────┐ │ │
        │  │  │ 【project_graph_features】                          │ │ │
        │  │  │   graph_features = nan_to_num(graph_features)        │ │ │
        │  │  │   logits_input = output_norm(graph_features)        │ │ │
        │  │  │   logits = output_proj(logits_input)                │ │ │
        │  │  └─────────────────────────────────────────────────────┘ │ │
        │  │                    │                                       │ │
        │  │                    ▼                                       │ │
        │  │  pred = logits  # [batch_size, output_dim]              │ │
        │  └───────────────────────────────────────────────────────────┘ │
        │                                                                   │
        │  輸出: pred [batch_size, output_dim]                            │
        │       (回歸: [batch_size, 1], 分類: [batch_size, num_classes]) │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【3. 損失計算 (Loss Computation)】                               │
        │                                                                   │
        │  ⭐ 訓練發生在這裡！                                             │
        │                                                                   │
        │  loss = model.compute_loss(pred, batch.y)                       │
        │                                                                   │
        │  根據任務類型：                                                  │
        │                                                                   │
        │  ┌───────────────────────────────────────────────────────────┐ │
        │  │ 回歸任務:                                                   │ │
        │  │   loss = L1Loss(pred.squeeze(), batch.y)                   │ │
        │  │   或                                                       │ │
        │  │   loss = MSELoss(pred.squeeze(), batch.y)                  │ │
        │  │   或                                                       │ │
        │  │   loss = SpearmanLoss(pred.squeeze(), batch.y)             │ │
        │  └───────────────────────────────────────────────────────────┘ │
        │                                                                   │
        │  ┌───────────────────────────────────────────────────────────┐ │
        │  │ 分類任務:                                                   │ │
        │  │   loss = CrossEntropyLoss(pred, batch.y.long())            │ │
        │  │   或                                                       │ │
        │  │   loss = BCEWithLogitsLoss(pred, batch.y.float())          │ │
        │  │   或                                                       │ │
        │  │   loss = FocalLoss(pred, batch.y.long())                   │ │
        │  └───────────────────────────────────────────────────────────┘ │
        │                                                                   │
        │  輸出: loss (標量)                                               │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【4. 反向傳播 (Backward Pass)】                                   │
        │                                                                   │
        │  ⭐ 梯度計算發生在這裡！                                         │
        │                                                                   │
        │  loss.backward()  # 計算梯度                                     │
        │                                                                   │
        │  梯度反向傳播路徑：                                              │
        │                                                                   │
        │  loss                                                           │
        │    ↓                                                            │
        │  compute_loss                                                    │
        │    ↓                                                            │
        │  pred (logits)                                                  │
        │    ↓                                                            │
        │  output_proj (Sequential)                                       │
        │    ↓                                                            │
        │  output_norm (LayerNorm)  ← 梯度會流過這裡！                    │
        │    ↓                                                            │
        │  graph_features                                                 │
        │    ↓                                                            │
        │  Feature Concatenation                                          │
        │    ↓                                                            │
        │  Modality Processing (h_mlp, x_mlp, descriptor_mlp)            │
        │    ↓                                                            │
        │  Graph Pooling                                                  │
        │    ↓                                                            │
        │  GAT-EGNN 層 (反向傳播到每一層)                                 │
        │    ↓                                                            │
        │  Embedding 層 (node_embedding, edge_embedding)                  │
        │                                                                   │
        │  所有層的參數都會收到梯度！                                      │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【5. 參數更新 (Parameter Update)】                                │
        │                                                                   │
        │  ⭐ 模型參數更新發生在這裡！                                     │
        │                                                                   │
        │  optimizer.step()  # 更新參數                                    │
        │                                                                   │
        │  更新的參數包括：                                                 │
        │  - node_embedding.weight, node_embedding.bias                   │
        │  - edge_embedding.weight, edge_embedding.bias                   │
        │  - aegnn_layers[0..num_layers-1] 的所有參數                     │
        │  - h_mlp, x_mlp, descriptor_mlp 的所有參數                      │
        │  - output_norm.weight, output_norm.bias  ← 這裡！                │
        │  - output_proj 的所有參數                                       │
        │                                                                   │
        │  使用優化器（如 Adam）更新：                                     │
        │  param = param - lr * gradient                                   │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【6. 梯度裁剪 (Gradient Clipping) - 可選】                        │
        │                                                                   │
        │  if grad_clip_norm > 0:                                         │
        │      torch.nn.utils.clip_grad_norm_(                             │
        │          model.parameters(),                                     │
        │          max_norm=grad_clip_norm                                 │
        │      )                                                           │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【7. 記錄訓練指標】                                               │
        │                                                                   │
        │  train_losses.append(loss.item())                               │
        │  # 記錄到 TensorBoard 或其他日誌系統                            │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【8. 驗證 (Validation) - 每個 epoch 結束時】                      │
        │                                                                   │
        │  model.eval()  # 設置為評估模式                                 │
        │  with torch.no_grad():  # 不計算梯度                            │
        │      for batch in val_loader:                                    │
        │          pred = model(...)  # 前向傳播                          │
        │          val_loss = compute_loss(pred, batch.y)                 │
        │                                                                   │
        │  # 不進行反向傳播和參數更新                                      │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【9. 學習率調度 (Learning Rate Scheduling)】                      │
        │                                                                   │
        │  scheduler.step()  # 更新學習率                                 │
        │  # 根據驗證損失或 epoch 數調整學習率                             │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【10. 保存 Checkpoint - 可選】                                   │
        │                                                                   │
        │  if val_loss < best_val_loss:                                   │
        │      torch.save({                                                │
        │          'model_state_dict': model.state_dict(),                 │
        │          'optimizer_state_dict': optimizer.state_dict(),        │
        │          'val_loss': val_loss,                                   │
        │          'epoch': epoch,                                         │
        │          'model_config': {...}                                   │
        │      }, 'best_model.pth')                                       │
        └─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 【11. 繼續下一個 batch 或 epoch】                                 │
        │                                                                   │
        │  回到步驟 1，處理下一個 batch                                    │
        └─────────────────────────────────────────────────────────────────┘
```

## 🎯 訓練發生的關鍵位置

### 1. **損失計算**（步驟 3）

**位置**：在前向傳播之後，反向傳播之前

```python
# 在 train_edmpnn.py 的 train_epoch 方法中
loss = self.base_model.compute_loss(pred, batch.y)
```

**作用**：
- 計算預測值和目標值之間的差異
- 這是訓練的"目標"，模型要最小化這個損失

### 2. **反向傳播**（步驟 4）

**位置**：在損失計算之後，參數更新之前

```python
loss.backward()  # 計算所有參數的梯度
```

**作用**：
- 計算損失對所有參數的梯度
- 梯度會從損失反向傳播到所有層，包括 `output_norm`

### 3. **參數更新**（步驟 5）

**位置**：在反向傳播之後

```python
optimizer.step()  # 根據梯度更新參數
```

**作用**：
- 根據梯度更新所有參數
- 包括 `output_norm.weight` 和 `output_norm.bias`

## 📊 訓練過程中的數據流

### 前向傳播（Forward Pass）

```
輸入數據
    ↓
[模型架構的所有層]
    ↓
output_norm  ← 數據流過這裡
    ↓
output_proj
    ↓
pred (預測值)
    ↓
loss = compute_loss(pred, target)  ← 計算損失
```

### 反向傳播（Backward Pass）

```
loss
    ↓
compute_loss
    ↓
pred
    ↓
output_proj  ← 梯度流過這裡
    ↓
output_norm  ← 梯度流過這裡，更新參數！
    ↓
graph_features
    ↓
[所有前面的層]
    ↓
所有參數都收到梯度並更新
```

## 🔍 關鍵要點

### 1. **訓練發生在整個流程中**

- **前向傳播**：數據流過所有層（包括 `output_norm`）
- **損失計算**：在模型輸出後計算
- **反向傳播**：梯度流過所有層（包括 `output_norm`）
- **參數更新**：所有參數（包括 `output_norm`）都會更新

### 2. **output_norm 在訓練中的作用**

- **前向傳播時**：標準化 `graph_features`，影響最終輸出
- **反向傳播時**：接收梯度，更新 `output_norm.weight` 和 `output_norm.bias`
- **參數更新時**：`output_norm` 的參數會根據梯度更新

### 3. **訓練模式 vs 評估模式**

- **訓練模式** (`model.train()`):
  - `output_norm` 使用訓練時的統計信息（batch 統計）
  - Dropout 層會隨機丟棄神經元
  - 計算梯度並更新參數

- **評估模式** (`model.eval()`):
  - `output_norm` 使用訓練時保存的統計信息（running 統計）
  - Dropout 層不丟棄神經元
  - 不計算梯度，不更新參數

## 📝 代碼位置總結

| 步驟 | 位置 | 說明 |
|------|------|------|
| **前向傳播** | `models/edmpnn_model.py:forward()` | 數據流過所有層 |
| **損失計算** | `scripts/train_edmpnn.py:train_epoch()` | `loss = compute_loss(pred, target)` |
| **反向傳播** | `scripts/train_edmpnn.py:train_epoch()` | `loss.backward()` |
| **參數更新** | `scripts/train_edmpnn.py:train_epoch()` | `optimizer.step()` |

## 🎯 總結

1. **訓練發生在整個流程中**，不是單一位置
2. **關鍵步驟**：
   - 前向傳播：數據流過所有層（包括 `output_norm`）
   - 損失計算：計算預測值和目標值的差異
   - 反向傳播：計算梯度（包括 `output_norm` 的梯度）
   - 參數更新：更新所有參數（包括 `output_norm` 的參數）

3. **output_norm 在訓練中**：
   - 前向傳播時標準化特徵
   - 反向傳播時接收梯度
   - 參數更新時更新權重和偏置

4. **訓練循環**：
   - 每個 batch 都執行：前向傳播 → 損失計算 → 反向傳播 → 參數更新
   - 每個 epoch 結束時進行驗證（不更新參數）

