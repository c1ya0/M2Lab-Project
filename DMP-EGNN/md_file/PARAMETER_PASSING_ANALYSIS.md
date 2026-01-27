# 參數傳遞分析報告

## 概述
本報告詳細檢查了訓練和優化流程中各個腳本之間的參數傳遞是否正確。

## 參數傳遞鏈

### 1. 訓練流程：`train_edmpnn_new.sh` → `train_edmpnn_new.py`

#### 1.1 Shell腳本傳遞的參數（train_edmpnn_new.sh）

從 `train_single_seed()` 函數（第368-405行）傳遞的參數：

```bash
--tdc_dataset "${dataset_name}"
--tdc_seed "${seed}"
--model_type "${task_type}"
--use_descriptor
--descriptor_dim 217
--hidden_dim "${hidden_dim}"
--num_layers "${num_layers}"
--num_heads "${num_heads}"
--ffn_expansion_factor "${ffn_expansion_factor:-4}"
--dropout "${dropout}"
--batch_size "${batch_size}"
--gradient_accumulation_steps 1
--grad_clip_norm "${grad_clip_norm}"
--learning_rate "${lr}"
--weight_decay "${weight_decay}"
--drop_path_rate "${drop_path_rate}"
--scheduler_type "${scheduler_type}"
--scheduler_patience 20
--warmup_epochs "${warmup_epochs}"
--min_lr "${min_lr}"
--num_epochs 200
--early_stopping_patience "${early_stopping_patience}"
--use_pre_norm
--dmp_steps "${dmp_steps}"
--activation "${activation}"
--alpha "${alpha:-0.2}"
--aggregation "${aggregation:-mean}"
--use_smart_early_stopping
--smart_early_stopping_max_patience "${smart_early_stopping_max_patience}"
--auroc_improvement_threshold "${auroc_improvement_threshold}"
--log_dir "${log_dir}"
--save_dir "${save_dir}"
```

條件參數：
- `--rotate_aug` (如果 `rotate_aug=true`)
- `--rotation_prob "${rotation_prob}"` (如果 `rotate_aug=true` 且 `rotation_prob>0`)
- `--max_rotation_angle "${max_rotation_angle}"` (如果 `rotate_aug=true` 且 `max_rotation_angle>0`)
- `--descriptor_dropout "${descriptor_dropout}"` (如果 `descriptor_dropout>0`)

損失函數相關（根據 `loss_type`）：
- `--use_focal_loss --focal_alpha "${focal_alpha}" --focal_gamma "${focal_gamma}"` (如果 `loss_type=focal`)
- `--use_class_balanced_focal_loss --focal_alpha "${focal_alpha}" --focal_gamma "${focal_gamma}" --class_balanced_beta "${class_balanced_beta:-0.9999}"` (如果 `loss_type=class_balanced_focal`)
- `--use_bce_for_imbalanced --auto_pos_weight` (如果 `loss_type=bce`)

其他：
- `--enable_manifold_mixup --manifold_mixup_alpha "${mixup_alpha:-0.2}"` (如果 `use_mixup=true`)
- `--label_smoothing "${label_smoothing}"` (如果 `label_smoothing>0`)

#### 1.2 Python腳本接收的參數（train_edmpnn_new.py）

所有參數都在 `main()` 函數中定義（第4358-4493行），並且都能正確接收。

#### 1.3 檢查結果

✅ **所有參數都正確傳遞**

| Shell參數 | Python參數 | 狀態 |
|-----------|-----------|------|
| `--tdc_dataset` | `args.tdc_dataset` | ✅ |
| `--tdc_seed` | `args.tdc_seed` | ✅ |
| `--model_type` | `args.model_type` | ✅ |
| `--use_descriptor` | `args.use_descriptor` | ✅ |
| `--descriptor_dim` | `args.descriptor_dim` | ✅ |
| `--hidden_dim` | `args.hidden_dim` | ✅ |
| `--num_layers` | `args.num_layers` | ✅ |
| `--num_heads` | `args.num_heads` | ✅ |
| `--ffn_expansion_factor` | `args.ffn_expansion_factor` | ✅ |
| `--dropout` | `args.dropout` | ✅ |
| `--batch_size` | `args.batch_size` | ✅ |
| `--gradient_accumulation_steps` | `args.gradient_accumulation_steps` | ✅ |
| `--grad_clip_norm` | `args.grad_clip_norm` | ✅ |
| `--learning_rate` | `args.learning_rate` | ✅ |
| `--weight_decay` | `args.weight_decay` | ✅ |
| `--drop_path_rate` | `args.drop_path_rate` | ✅ |
| `--scheduler_type` | `args.scheduler_type` | ✅ |
| `--scheduler_patience` | `args.scheduler_patience` | ✅ |
| `--warmup_epochs` | `args.warmup_epochs` | ✅ |
| `--min_lr` | `args.min_lr` | ✅ |
| `--num_epochs` | `args.num_epochs` | ✅ |
| `--early_stopping_patience` | `args.early_stopping_patience` | ✅ |
| `--use_pre_norm` | `args.use_pre_norm` | ✅ |
| `--dmp_steps` | `args.dmp_steps` | ✅ |
| `--activation` | `args.activation` | ✅ |
| `--alpha` | `args.alpha` | ✅ |
| `--aggregation` | `args.aggregation` | ✅ |
| `--rotate_aug` | `args.rotate_aug` | ✅ |
| `--rotation_prob` | `args.rotation_prob` | ✅ |
| `--max_rotation_angle` | `args.max_rotation_angle` | ✅ |
| `--descriptor_dropout` | `args.descriptor_dropout` | ✅ |
| `--use_smart_early_stopping` | `args.use_smart_early_stopping` | ✅ |
| `--smart_early_stopping_max_patience` | `args.smart_early_stopping_max_patience` | ✅ |
| `--auroc_improvement_threshold` | `args.auroc_improvement_threshold` | ✅ |
| `--use_focal_loss` | `args.use_focal_loss` | ✅ |
| `--focal_alpha` | `args.focal_alpha` | ✅ |
| `--focal_gamma` | `args.focal_gamma` | ✅ |
| `--use_class_balanced_focal_loss` | `args.use_class_balanced_focal_loss` | ✅ |
| `--class_balanced_beta` | `args.class_balanced_beta` | ✅ |
| `--use_bce_for_imbalanced` | `args.use_bce_for_imbalanced` | ✅ |
| `--auto_pos_weight` | `args.auto_pos_weight` | ✅ |
| `--enable_manifold_mixup` | `args.enable_manifold_mixup` | ✅ |
| `--manifold_mixup_alpha` | `args.manifold_mixup_alpha` | ✅ |
| `--label_smoothing` | `args.label_smoothing` | ✅ |
| `--log_dir` | `args.log_dir` | ✅ |
| `--save_dir` | `args.save_dir` | ✅ |

---

### 2. Optuna優化流程：`optuna_parallel_mod_new.sh` → `optuna_serach_mod_new.py` → `train_edmpnn_new.py`

#### 2.1 Shell腳本傳遞的參數（optuna_parallel_mod_new.sh）

從 `run_optimization_for_seed()` 函數（第514-521行）傳遞給 `optuna_serach_mod_new.py`：

```bash
--dataset "$current_dataset"
--n_trials "$WORKER_TRIALS"
--storage "$STORAGE_URL"
--epochs "$EPOCHS"
--worker_id "$i"
--seed "$seed_num"
```

#### 2.2 Optuna腳本接收的參數（optuna_serach_mod_new.py）

在 `main()` 函數中定義（第862-871行）：
- `--dataset` → `args.dataset`
- `--n_trials` → `args.n_trials`
- `--storage` → `args.storage`
- `--epochs` → `args.epochs`
- `--worker_id` → `args.worker_id`
- `--seed` → `args.seed`

✅ **所有參數都正確傳遞**

#### 2.3 Optuna腳本傳遞給訓練腳本的參數（optuna_serach_mod_new.py）

在 `objective()` 函數中（第489-538行），通過配置文件方式傳遞：

```python
cmd.extend(["--config", trial_config_path])
cmd.extend(["--tdc_dataset", data_arg_path])
cmd.extend(["--tdc_seed", str(seed)])
cmd.extend(["--seed", str(model_init_seed)])
cmd.extend(["--save_dir", seed_save_dir])
cmd.extend(["--log_dir", seed_log_dir])
cmd.extend(["--base_port", str(seed_base_port)])
cmd.extend(["--world_size", "1"])
```

配置文件 `trial_config.json` 包含所有超參數（第383-426行）：
- `hidden_dim`, `num_layers`, `dropout`, `lr`/`learning_rate`
- `weight_decay`, `batch_size`, `grad_clip_norm`
- `num_heads`, `warmup_epochs`, `dmp_steps`
- `scheduler_type`, `min_lr`, `drop_path_rate`
- `activation`, `alpha`, `ffn_expansion_factor`
- `pool_type`/`aggregation`
- `rotate_aug`, `rotation_prob`, `max_rotation_angle`
- `descriptor_dropout`
- `use_mixup`, `mixup_alpha`
- `num_epochs`, `early_stopping_patience`
- `use_smart_early_stopping`, `smart_early_stopping_max_patience`
- `use_descriptor`, `descriptor_dim`
- `use_pre_norm`
- `model_type`, `use_bce_for_imbalanced`, `auto_pos_weight`

#### 2.4 訓練腳本從配置文件加載參數（train_edmpnn_new.py）

在 `main()` 函數中（第4496-4613行），從配置文件加載參數：

✅ **所有參數都正確加載**

| 配置文件鍵 | Python參數 | 狀態 |
|-----------|-----------|------|
| `hidden_dim` | `args.hidden_dim` | ✅ |
| `num_layers` | `args.num_layers` | ✅ |
| `num_heads` | `args.num_heads` | ✅ |
| `ffn_expansion_factor` | `args.ffn_expansion_factor` | ✅ |
| `dropout` | `args.dropout` | ✅ |
| `drop_path_rate` | `args.drop_path_rate` | ✅ |
| `alpha` | `args.alpha` | ✅ |
| `aggregation` 或 `pool_type` | `args.aggregation` | ✅ |
| `activation` | `args.activation` | ✅ |
| `dmp_steps` | `args.dmp_steps` | ✅ |
| `learning_rate` 或 `lr` | `args.learning_rate` | ✅ |
| `weight_decay` | `args.weight_decay` | ✅ |
| `batch_size` | `args.batch_size` | ✅ |
| `grad_clip_norm` | `args.grad_clip_norm` | ✅ |
| `num_epochs` | `args.num_epochs` | ✅ |
| `scheduler_type` | `args.scheduler_type` | ✅ |
| `warmup_epochs` | `args.warmup_epochs` | ✅ |
| `min_lr` | `args.min_lr` | ✅ |
| `early_stopping_patience` | `args.early_stopping_patience` | ✅ |
| `use_smart_early_stopping` | `args.use_smart_early_stopping` | ✅ |
| `smart_early_stopping_max_patience` | `args.smart_early_stopping_max_patience` | ✅ |
| `model_type` | `args.model_type` | ✅ |
| `use_descriptor` | `args.use_descriptor` | ✅ |
| `descriptor_dim` | `args.descriptor_dim` | ✅ |
| `descriptor_dropout` | `args.descriptor_dropout` | ✅ |
| `rotate_aug` | `args.rotate_aug` | ✅ |
| `rotation_prob` | `args.rotation_prob` | ✅ |
| `max_rotation_angle` | `args.max_rotation_angle` | ✅ |
| `use_pre_norm` | `args.use_pre_norm` | ✅ |
| `use_bce_for_imbalanced` | `args.use_bce_for_imbalanced` | ✅ |
| `auto_pos_weight` | `args.auto_pos_weight` | ✅ |
| `use_focal_loss` | `args.use_focal_loss` | ⚠️ 見問題1 |
| `focal_alpha` | `args.focal_alpha` | ⚠️ 見問題1 |
| `focal_gamma` | `args.focal_gamma` | ⚠️ 見問題1 |
| `use_mixup` | `args.enable_manifold_mixup` | ✅ |
| `mixup_alpha` | `args.manifold_mixup_alpha` | ✅ |

---

## 發現的問題

### ⚠️ 問題1：Optuna配置文件中缺少損失函數參數

**位置**：`optuna_serach_mod_new.py` 的 `objective()` 函數（第383-426行）

**問題**：
- Optuna腳本在 `objective()` 函數中根據 `loss_type` 選擇損失函數，但這些參數**沒有保存到配置文件**中
- 配置文件 `trial_config.json` 中缺少：
  - `loss_type`（用於記錄選擇的損失類型）
  - `use_focal_loss`（如果使用Focal Loss）
  - `use_class_balanced_focal_loss`（如果使用Class-Balanced Focal Loss）
  - `focal_alpha`、`focal_gamma`（Focal Loss參數）
  - `class_balanced_beta`（Class-Balanced Loss參數）
  - `label_smoothing`（標籤平滑參數）

**影響**：
- 當使用配置文件訓練時，這些參數不會被正確加載
- 訓練腳本會使用默認值，而不是Optuna優化的值

**建議修復**：
在 `optuna_serach_mod_new.py` 的 `objective()` 函數中，將損失函數相關參數添加到 `trial_config`：

```python
# 在 trial_config["hyperparameters"] 中添加：
"loss_type": loss_type,  # 記錄選擇的損失類型
"use_focal_loss": loss_type in ["focal", "Focal"],
"use_class_balanced_focal_loss": loss_type in ["class_balanced_focal", "ClassBalancedFocal"],
"focal_alpha": focal_alpha if loss_type in ["focal", "Focal", "class_balanced_focal", "ClassBalancedFocal"] else 0.25,
"focal_gamma": focal_gamma if loss_type in ["focal", "Focal", "class_balanced_focal", "ClassBalancedFocal"] else 2.0,
"class_balanced_beta": class_balanced_beta if loss_type in ["class_balanced_focal", "ClassBalancedFocal"] else 0.9999,
"label_smoothing": label_smoothing if label_smoothing > 0 else 0.0,
```

**注意**：Optuna腳本中目前**沒有搜索** `label_smoothing` 參數，這可能需要添加。

---

### ⚠️ 問題2：train_edmpnn_new.sh 中缺少 `f1_improvement_threshold` 參數

**位置**：`train_edmpnn_new.sh` 的 `train_single_seed()` 函數

**問題**：
- Shell腳本傳遞了 `--auroc_improvement_threshold`，但沒有傳遞 `--f1_improvement_threshold`
- Python腳本支持 `--f1_improvement_threshold` 參數（第4454行）
- Optuna配置文件中也沒有包含 `f1_improvement_threshold`

**影響**：
- 如果使用Smart Early Stopping，F1改進閾值會使用默認值（0.001），而不是從Optuna優化中獲取

**建議修復**：
1. 在 `load_optuna_mod_params()` 函數中添加 `f1_improvement_threshold` 的默認值
2. 在 `train_single_seed()` 函數中添加 `--f1_improvement_threshold` 參數傳遞
3. 在 `optuna_serach_mod_new.py` 中添加 `f1_improvement_threshold` 的搜索和保存

---

### ⚠️ 問題3：`train_edmpnn_new.sh` 中硬編碼的參數

**位置**：`train_edmpnn_new.sh` 的 `train_single_seed()` 函數

**問題**：
以下參數在Shell腳本中硬編碼，沒有從Optuna配置中加載：
- `--scheduler_patience 20`（第387行）
- `--num_epochs 200`（第390行）
- `--gradient_accumulation_steps 1`（第381行）

**影響**：
- 這些參數無法通過Optuna優化
- 如果Optuna配置文件中包含這些參數，它們會被忽略

**建議修復**：
1. 從Optuna配置文件中加載這些參數（如果存在）
2. 或者明確說明這些是固定參數，不進行優化

---

### ✅ 問題4：參數名稱一致性檢查

**檢查結果**：所有參數名稱在Shell腳本和Python腳本之間保持一致。

**例外情況**：
- Shell腳本使用 `lr`，Python腳本使用 `learning_rate`，但兩者都支持（第4543-4545行）
- Shell腳本使用 `pool_type`，Python腳本使用 `aggregation`，但兩者都支持（第4533-4535行）
- Shell腳本使用 `use_mixup`，Python腳本使用 `enable_manifold_mixup`，但兩者都支持（第4608-4611行）

這些都是**有意設計的別名**，確保向後兼容性。

---

## 總結

### ✅ 正確傳遞的參數（大部分）

1. **模型架構參數**：`hidden_dim`, `num_layers`, `num_heads`, `ffn_expansion_factor`, `dropout`, `drop_path_rate`, `activation`, `alpha`, `aggregation`, `dmp_steps`
2. **訓練參數**：`learning_rate`, `weight_decay`, `batch_size`, `grad_clip_norm`, `num_epochs`
3. **調度器參數**：`scheduler_type`, `warmup_epochs`, `min_lr`
4. **早停參數**：`early_stopping_patience`, `use_smart_early_stopping`, `smart_early_stopping_max_patience`, `auroc_improvement_threshold`
5. **數據增強參數**：`rotate_aug`, `rotation_prob`, `max_rotation_angle`
6. **特徵參數**：`use_descriptor`, `descriptor_dim`, `descriptor_dropout`
7. **Mixup參數**：`enable_manifold_mixup`, `manifold_mixup_alpha`

### ⚠️ 需要修復的問題

1. **損失函數參數未保存到配置文件**（問題1）
2. **`f1_improvement_threshold` 未傳遞**（問題2）
3. **部分參數硬編碼**（問題3）

### 📋 建議的修復優先級

1. **高優先級**：修復問題1（損失函數參數），因為這會影響模型訓練的正確性
2. **中優先級**：修復問題2（`f1_improvement_threshold`），如果使用Smart Early Stopping
3. **低優先級**：修復問題3（硬編碼參數），如果這些參數需要優化

---

## 驗證建議

1. **運行測試訓練**：使用一個小數據集，驗證所有參數都正確傳遞
2. **檢查日誌**：確認訓練日誌中顯示的參數值與配置文件一致
3. **對比結果**：比較使用命令行參數和配置文件參數的訓練結果是否一致

