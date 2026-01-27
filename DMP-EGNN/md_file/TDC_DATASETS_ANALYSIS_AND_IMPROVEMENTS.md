# TDC 22 Datasets 模型训练问题分析与改进建议

## 📊 当前问题诊断

### 1. **数据预处理问题**

#### 1.1 描述符归一化数值稳定性
**问题位置**: `train_edmpnn.py` 第 3569-3765 行

**问题**:
- 描述符归一化时可能存在数值溢出风险
- 极端值处理不够稳健
- 某些 TDC 数据集的描述符可能包含异常值

**影响**:
- 训练不稳定
- 梯度爆炸/消失
- 模型性能下降

**改进建议**:
```python
# 建议使用更稳健的归一化方法
# 1. 使用 RobustScaler (基于中位数和 IQR)
# 2. 添加更严格的异常值检测
# 3. 对每个描述符维度单独处理
```

#### 1.2 TDC 数据加载验证不足
**问题位置**: `train_edmpnn.py` 第 3176-3357 行

**问题**:
- 缺少数据完整性检查
- 没有验证标签分布
- 缺少数据质量报告

**改进建议**:
- 添加数据统计报告（类别分布、缺失值、异常值）
- 验证数据格式一致性
- 检查标签分布是否合理

---

### 2. **模型架构问题**

#### 2.1 DMP Steps 参数可能不合适
**问题位置**: `optuna_serach_mod.py` 第 195 行

**当前搜索空间**:
```python
dmp_steps = trial.suggest_int("dmp_steps", 1, 4)
```

**问题**:
- 搜索范围可能不够大（某些数据集可能需要更多步骤）
- 没有考虑数据集大小对 DMP steps 的影响
- 固定搜索范围可能不适合所有 TDC 数据集

**改进建议**:
```python
# 根据数据集大小动态调整搜索范围
if dataset_size < 1000:
    dmp_steps = trial.suggest_int("dmp_steps", 1, 3)
elif dataset_size < 5000:
    dmp_steps = trial.suggest_int("dmp_steps", 2, 5)
else:
    dmp_steps = trial.suggest_int("dmp_steps", 3, 6)
```

#### 2.2 模型深度和宽度可能不匹配
**问题位置**: `optuna_serach_mod.py` 第 150-180 行

**当前配置**:
- `hidden_dim`: 128-512
- `num_layers`: 3-8
- `ffn_expansion_factor`: 2-6

**问题**:
- 某些小数据集可能被过深/过宽的模型过拟合
- 某些大数据集可能模型容量不足
- 没有考虑数据集特性（类别不平衡程度、任务复杂度）

**改进建议**:
```python
# 根据数据集大小和任务复杂度调整搜索空间
dataset_size = len(train_data)
imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0

if dataset_size < 1000:
    # 小数据集：使用更小的模型防止过拟合
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    num_layers = trial.suggest_int("num_layers", 2, 5)
elif imbalance_ratio > 100:
    # 极度不平衡：使用更深的模型学习复杂模式
    hidden_dim = trial.suggest_int("hidden_dim", 256, 512)
    num_layers = trial.suggest_int("num_layers", 5, 10)
else:
    # 标准配置
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    num_layers = trial.suggest_int("num_layers", 3, 8)
```

#### 2.3 注意力头数可能不够优化
**问题位置**: `optuna_serach_mod.py` 第 185 行

**当前配置**:
```python
num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
```

**问题**:
- 搜索空间较小
- 没有考虑 hidden_dim 和 num_heads 的匹配关系（num_heads 必须能整除 hidden_dim）

**改进建议**:
```python
# 确保 num_heads 能整除 hidden_dim
# 在 hidden_dim 确定后，动态调整 num_heads 搜索空间
valid_heads = [h for h in [2, 4, 8, 16, 32] if hidden_dim % h == 0]
if not valid_heads:
    valid_heads = [1, 2, 4]  # Fallback
num_heads = trial.suggest_categorical("num_heads", valid_heads)
```

---

### 3. **训练策略问题**

#### 3.1 学习率调度可能不够灵活
**问题位置**: `train_edmpnn.py` 第 915-951 行

**当前问题**:
- CosineAnnealingLR 的 T_max 固定为 1000（第 919 行），但实际训练轮数可能更少
- 没有根据数据集特性选择最优调度器
- Warmup 轮数固定为 5，可能不适合所有数据集

**改进建议**:
```python
# 1. 根据实际训练轮数设置 T_max
if scheduler_type == 'cosine':
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=num_epochs - self.warmup_epochs,  # 使用实际训练轮数
        eta_min=min_lr
    )

# 2. 根据数据集大小调整 warmup_epochs
if dataset_size < 1000:
    warmup_epochs = max(3, num_epochs // 20)  # 小数据集：更少 warmup
else:
    warmup_epochs = max(5, num_epochs // 10)  # 大数据集：更多 warmup
```

#### 3.2 早停策略可能过于激进
**问题位置**: `train_edmpnn.py` 第 176-660 行 (SmartEarlyStopping)

**当前问题**:
- 初始 patience=20 可能对某些数据集太短
- 多指标早停逻辑复杂，可能过早停止
- 对极度不平衡数据集，AUROC 改善阈值可能不合适

**改进建议**:
```python
# 根据数据集特性调整早停参数
if imbalance_ratio > 100:
    # 极度不平衡：需要更多耐心
    initial_patience = 40
    max_patience = 80
    auroc_improvement_threshold = 0.001  # 更小的阈值
elif dataset_size < 1000:
    # 小数据集：防止过早停止
    initial_patience = 30
    max_patience = 60
else:
    # 标准配置
    initial_patience = 20
    max_patience = 50
```

#### 3.3 损失函数选择可能不合适
**问题位置**: `train_edmpnn.py` 第 3866-3934 行

**当前问题**:
- 所有分类任务默认使用 BCEWithLogitsLoss，但某些数据集可能需要 Focal Loss
- 没有根据类别不平衡程度动态选择损失函数
- PR-AUC 作为主要指标的数据集应该使用不同的损失函数

**改进建议**:
```python
# 根据数据集特性和主要指标选择损失函数
if primary_metric == 'pr_auc':
    # PR-AUC 数据集：使用 Focal Loss 或 Class-Balanced Focal Loss
    if imbalance_ratio > 100:
        use_class_balanced_focal_loss = True
    else:
        use_focal_loss = True
elif imbalance_ratio > 50:
    # 高度不平衡：使用 Focal Loss
    use_focal_loss = True
else:
    # 标准：使用 BCE
    use_bce_for_imbalanced = True
```

---

### 4. **超参数优化问题**

#### 4.1 Optuna 搜索空间可能不够全面
**问题位置**: `optuna_serach_mod.py` 第 150-260 行

**缺失的关键超参数**:
1. **Label Smoothing**: 没有优化（固定为 0.0）
2. **Drop Path Rate**: 搜索范围可能不够（0.0-0.2）
3. **Fingerprint Dropout**: 没有优化（固定为 0.0）
4. **Batch Size**: 搜索范围可能不够（16-128）

**改进建议**:
```python
# 添加缺失的超参数搜索
label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
fingerprint_dropout = trial.suggest_float("fingerprint_dropout", 0.0, 0.3, step=0.05)

# 根据数据集大小调整 batch_size 搜索范围
if dataset_size < 1000:
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
elif dataset_size < 5000:
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
else:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
```

#### 4.2 学习率搜索范围可能不合适
**问题位置**: `optuna_serach_mod.py` 第 160 行

**当前配置**:
```python
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
```

**问题**:
- 范围可能太宽，导致某些试验使用不合适的学习率
- 没有考虑模型大小对学习率的影响

**改进建议**:
```python
# 根据模型大小调整学习率范围
model_size = hidden_dim * num_layers
if model_size < 1000:
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
elif model_size < 3000:
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
else:
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
```

---

### 5. **数据增强问题**

#### 5.1 3D 旋转增强可能不够
**问题位置**: `optuna_serach_mod.py` 第 201 行

**当前配置**:
```python
rotate_aug = trial.suggest_categorical("rotate_aug", [True, False])
```

**问题**:
- 只有开关，没有强度控制
- 某些数据集可能需要更强的增强

**改进建议**:
```python
# 添加增强强度控制
if rotate_aug:
    # 可以添加旋转角度范围、旋转概率等参数
    rotation_prob = trial.suggest_float("rotation_prob", 0.3, 1.0)
    max_rotation_angle = trial.suggest_float("max_rotation_angle", 15.0, 180.0)
```

#### 5.2 Manifold Mixup 使用不足
**问题位置**: `optuna_serach_mod.py` 第 253-254 行

**当前配置**:
- Mixup 是可选超参数，但可能对某些数据集很有用

**改进建议**:
```python
# 对小数据集或极度不平衡数据集，强制考虑 mixup
if dataset_size < 2000 or imbalance_ratio > 50:
    use_mixup = trial.suggest_categorical("use_mixup", [True, False])
    if use_mixup:
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.5, 4.0)
```

---

### 6. **评估指标问题**

#### 6.1 阈值选择可能不合适
**问题位置**: `train_edmpnn.py` 第 1509-1592 行

**当前问题**:
- 某些极度不平衡数据集（如 CLINTOX）的阈值选择可能不稳定
- CV 阈值选择可能计算成本高

**改进建议**:
```python
# 根据数据集特性选择阈值方法
if imbalance_ratio > 100 and dataset_size < 1000:
    # 极度不平衡小数据集：使用固定阈值或简单方法
    optimal_threshold = 0.3  # 或从配置读取
elif imbalance_ratio > 50:
    # 高度不平衡：使用 CV 方法
    optimal_threshold, _ = find_optimal_threshold_cv(...)
else:
    # 标准：使用自适应方法
    optimal_threshold, _ = find_optimal_threshold_adaptive(...)
```

#### 6.2 回归任务的目标归一化
**问题位置**: `train_edmpnn.py` 第 1990-2177 行

**当前问题**:
- 检测和修正归一化问题的逻辑复杂
- 可能在某些情况下误判

**改进建议**:
```python
# 在训练开始时就记录目标统计信息
# 在测试时直接使用，避免复杂的检测逻辑
# 在模型保存时保存目标统计信息
save_dict['target_mean'] = self.target_mean
save_dict['target_std'] = self.target_std
```

---

### 7. **其他潜在问题**

#### 7.1 梯度累积可能不够
**问题位置**: `train_edmpnn.py` 第 1154-1239 行

**问题**:
- 默认 gradient_accumulation_steps=1
- 对于小 batch size，可能需要更多累积

**改进建议**:
```python
# 根据 batch_size 自动调整梯度累积
if batch_size < 32:
    gradient_accumulation_steps = max(1, 32 // batch_size)
else:
    gradient_accumulation_steps = 1
```

#### 7.2 权重初始化可能不合适
**问题位置**: 模型初始化（未在训练脚本中显式设置）

**当前状况分析**:
通过代码审查发现：
1. **部分层有初始化**：
   - `GATLayer` 中的 `reset_parameters()` 方法对注意力权重使用 `xavier_uniform_` 初始化（第 243-250 行）
   - 某些门控机制使用 `normal_(mean=0, std=0.01)` 进行保守初始化（第 692-693 行）

2. **大量层缺少显式初始化**：
   - `node_embedding` 和 `edge_embedding`（第 833-834 行）：使用 PyTorch 默认初始化
   - `fingerprint_mlp` 中的所有 `nn.Linear` 层（第 843-846 行）
   - `descriptor_mlp` 中的所有 `nn.Linear` 层（第 866-869 行）
   - `output_proj` 中的所有 `nn.Linear` 层（第 931-934 行）
   - `h_mlp` 和 `x_mlp` 中的所有层（第 908-918 行）
   - 所有 `nn.LayerNorm` 层（如 `output_norm`，第 929 行）

3. **PyTorch 默认初始化的问题**：
   - `nn.Linear` 默认使用 Kaiming Uniform 初始化（针对 ReLU），但模型使用 SiLU 激活函数
   - `nn.LayerNorm` 默认权重为 1，偏置为 0（这个是正确的）
   - 不同层的初始化不一致可能导致训练不稳定

**问题影响**:

1. **梯度传播问题**：
   - 权重初始化不当会导致梯度在深层网络中消失或爆炸
   - 对于深度图神经网络（6-8 层），这个问题尤其严重
   - 不同层使用不同初始化策略会导致各层学习速度不一致

2. **训练稳定性问题**：
   - 初始权重过大：可能导致早期训练阶段梯度爆炸，损失值 NaN
   - 初始权重过小：可能导致梯度消失，模型难以学习
   - 不一致的初始化：某些层学习快，某些层学习慢，导致训练不稳定

3. **收敛速度问题**：
   - 不合适的初始化会显著增加模型收敛所需的 epoch 数
   - 对于 TDC 数据集这种需要快速实验的场景，这会浪费大量计算资源

4. **图神经网络的特殊性**：
   - 图神经网络需要处理变长输入（不同大小的分子图）
   - 注意力机制需要合适的初始化才能有效学习节点间关系
   - 消息传递层的初始化对模型性能影响很大

**为什么默认初始化不适合深度图神经网络**:

1. **激活函数不匹配**：
   - PyTorch 的 `nn.Linear` 默认使用 Kaiming Uniform，针对 ReLU 优化
   - 但 AEGNNM 模型使用 SiLU（Swish）激活函数
   - SiLU 的梯度特性与 ReLU 不同，需要不同的初始化策略

2. **深度网络的需求**：
   - 深度图神经网络（6-8 层）需要更精细的初始化策略
   - Xavier/Glorot 初始化更适合 tanh/sigmoid 类激活函数
   - 对于 SiLU，可能需要介于 Xavier 和 Kaiming 之间的策略

3. **多模态融合的需求**：
   - 模型融合了图特征、指纹、描述符等多种输入
   - 不同模态的投影层需要一致的初始化，避免某个模态主导训练

**改进建议**:

```python
def init_weights(m):
    """
    统一的权重初始化策略
    针对深度图神经网络和 SiLU 激活函数优化
    """
    if isinstance(m, nn.Linear):
        # 对于 SiLU 激活函数，使用 Xavier/Glorot 初始化更合适
        # 或者使用 He 初始化的变体
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        # 或者使用 Kaiming 初始化的变体（针对 SiLU）
        # nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        # 偏置初始化为 0
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.LayerNorm):
        # LayerNorm 的默认初始化已经是正确的（weight=1, bias=0）
        # 但为了明确性，可以显式设置
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Embedding):
        # 如果使用 Embedding 层，使用正态分布初始化
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        # BatchNorm 的默认初始化通常是合适的
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 在模型创建后应用初始化
model = create_aegnn_model(...)
model.apply(init_weights)

# 对于输出层，可能需要特殊处理
# 例如，对于分类任务，输出层可以使用更小的初始化
if isinstance(model, AEGNNMClassifier):
    if isinstance(model.output_proj[-1], nn.Linear):
        nn.init.xavier_uniform_(model.output_proj[-1].weight, gain=0.1)
```

**针对不同层的特殊初始化建议**:

```python
def init_weights_advanced(model):
    """
    针对不同层使用不同初始化策略的高级版本
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'output_proj' in name or 'output' in name.lower():
                # 输出层：使用较小的初始化，避免初始预测过于极端
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            elif 'embedding' in name.lower():
                # 嵌入层：使用标准初始化
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif 'gate' in name.lower() or 'attention' in name.lower():
                # 门控和注意力层：使用较小的初始化，更保守
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            else:
                # 其他层：标准初始化
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
```

**实施位置**:
- 在 `scripts/train_edmpnn.py` 中，模型创建后（第 4082 行之后）添加初始化
- 或者在 `models/edmpnn_model.py` 的 `AEGNNM.__init__()` 方法末尾添加初始化调用

---

## 🚀 优先改进建议（按影响排序）

### 高优先级（立即实施）

1. **修复描述符归一化的数值稳定性问题**
   - 使用 RobustScaler 或更稳健的方法
   - 添加异常值检测和处理

2. **根据数据集特性动态调整超参数搜索空间**
   - 根据数据集大小、不平衡程度调整模型大小
   - 根据主要指标选择损失函数

3. **优化早停策略**
   - 对极度不平衡数据集增加 patience
   - 调整 AUROC 改善阈值

### 中优先级（近期实施）

4. **添加缺失的超参数优化**
   - Label Smoothing
   - Fingerprint Dropout
   - 更细粒度的 Batch Size 搜索

5. **改进学习率调度**
   - 根据实际训练轮数设置 T_max
   - 根据数据集大小调整 warmup

6. **优化阈值选择策略**
   - 根据数据集特性选择方法
   - 简化回归任务的归一化检测

### 低优先级（长期优化）

7. **增强数据增强**
   - 添加更多增强选项
   - 根据数据集特性选择增强策略

8. **改进权重初始化**
   - 添加显式初始化策略
   - 针对图神经网络优化

---

## 📝 实施检查清单

- [ ] 修复描述符归一化数值稳定性
- [ ] 添加数据集特性检测（大小、不平衡程度）
- [ ] 根据数据集特性动态调整超参数搜索空间
- [ ] 优化早停策略参数
- [ ] 添加 Label Smoothing 到 Optuna 搜索
- [ ] 添加 Fingerprint Dropout 到 Optuna 搜索
- [ ] 改进学习率调度器配置
- [ ] 优化阈值选择逻辑
- [ ] 添加权重初始化策略
- [ ] 添加数据质量检查报告
- [ ] 测试改进后的模型性能

---

## 🔍 调试建议

1. **添加详细的训练日志**
   - 记录每个 epoch 的关键指标
   - 记录梯度范数、学习率变化
   - 记录数据统计信息

2. **可视化训练过程**
   - 绘制损失曲线、指标曲线
   - 可视化注意力权重（如果可能）
   - 分析模型预测分布

3. **对比实验**
   - 对比改进前后的性能
   - 分析哪些改进最有效
   - 记录每个数据集的特殊需求

---

## 📚 参考文献

- TDC Benchmark: https://tdcommons.ai/
- DMPNN Paper: Analyzing Learned Molecular Representations for Property Prediction
- EGNN Paper: E(n) Equivariant Graph Neural Networks
- Focal Loss: Focal Loss for Dense Object Detection
- Class-Balanced Loss: Class-Balanced Loss Based on Effective Number of Samples

