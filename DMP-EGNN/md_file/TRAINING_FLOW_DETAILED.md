# train_edmpnn.sh 完整訓練流程詳解

## 📋 總覽

當執行 `./train_edmpnn.sh` 時，腳本會：
1. 初始化環境和配置
2. 解析命令行參數
3. 檢查 TDC 數據集可用性
4. 對每個數據集執行 5-seed **並行訓練**（使用 2 顆 GPU）
   - GPU 0：seeds 1, 3, 5
   - GPU 1：seeds 2, 4
5. 收集結果並計算統計信息
6. 生成最終報告

---

## 🔄 詳細執行流程

### 階段 1: 初始化 (第 1-27 行)

```
1. 激活 Conda 環境
   ├─ 檢查 conda 是否可用
   ├─ 檢查 aegnn_env 環境是否存在
   └─ 如果存在，激活 aegnn_env

2. 顯示啟動信息
   └─ "🧪 AEGNN-M Optimized Training V10 (TDC Multi-Seed with Optuna Best Params)..."

3. 設置顏色變量
   ├─ GREEN, BLUE, YELLOW, RED, NC (No Color)
   └─ 用於終端輸出美化
```

**輸出示例：**
```
🔧 Activating conda environment: aegnn_env
🧪 AEGNN-M Optimized Training V10 (TDC Multi-Seed with Optuna Best Params)...
=================================================================
```

---

### 階段 2: 參數解析 (第 481-619 行)

```
1. 定義所有支持的數據集 (22 個)
   └─ ALL_DATASETS="ames bbb_martins ... vdss_lombardo"

2. 解析命令行參數
   ├─ 如果沒有參數 → 訓練所有 22 個數據集
   ├─ 如果有參數 → 只訓練指定的數據集
   └─ 支持 --exclude/-x 排除特定數據集

3. 驗證數據集名稱
   ├─ 檢查是否在 ALL_DATASETS 中
   ├─ 如果無效，顯示警告並跳過
   └─ 如果有效，加入 SELECTED_DATASETS

4. 應用排除列表
   └─ 從 SELECTED_DATASETS 中移除被排除的數據集
```

**使用示例：**
```bash
# 訓練所有數據集
./train_edmpnn.sh

# 訓練指定數據集
./train_edmpnn.sh ames bbb_martins caco2_wang

# 排除特定數據集
./train_edmpnn.sh --exclude cyp2c9_veith cyp2d6_veith
```

**輸出示例：**
```
📋 No datasets specified, will train all available datasets
📋 Checking available TDC datasets...
```

---

### 階段 3: 數據集檢查循環 (第 621-641 行)

```
對每個 SELECTED_DATASETS 中的數據集：

1. 調用 check_tdc_dataset()
   ├─ 檢查 data/processed_tdc_data/{dataset_name}/ 目錄是否存在
   ├─ 檢查所有 5 個 seeds (1-5) 的數據是否存在
   │  └─ 每個 seed 需要 train.pt, valid.pt, test.pt 三個文件
   ├─ 如果所有 seeds 都存在 → ✅ Found TDC dataset
   ├─ 如果部分 seeds 缺失 → ⚠️ 顯示缺失的 seeds，但繼續訓練
   └─ 如果所有 seeds 都缺失 → ⚠️ 加入 FAILED_DATASETS

2. 如果檢查通過，調用 train_dataset()
```

**輸出示例：**
```
✅ Found TDC dataset: ames
✅ Found TDC dataset: bbb_martins
⚠️  Missing data for seeds: 3 5
   Training will continue with available seeds
⚠️  TDC dataset directory not found: data/processed_tdc_data/missing_dataset
```

---

### 階段 4: 單個數據集訓練流程 (train_dataset 函數)

#### 4.1 超參數加載 (第 193-196 行)

```
調用 load_optuna_mod_params(dataset_name)

1. 讀取 optuna_edmpnn_results/best_trial_info_all.json
   ├─ 查找對應 dataset_name 的配置
   ├─ 提取 best_params
   └─ 如果找不到 → 返回錯誤，跳過該數據集

2. 解析超參數
   ├─ 使用 Python 腳本解析 JSON
   ├─ 設置默認值（如果參數缺失）
   ├─ 處理科學記數法轉換（lr, weight_decay 等）
   └─ 將參數導出為 shell 變量

3. 參數驗證
   ├─ 檢查關鍵參數（lr, weight_decay）是否存在
   └─ 處理 pool_type → aggregation 映射
```

**加載的超參數包括：**
- 模型架構：`hidden_dim`, `num_layers`, `num_heads`, `dmp_steps`, `activation`, `aggregation`
- 訓練參數：`lr`, `weight_decay`, `batch_size`, `dropout`, `grad_clip_norm`
- 調度器：`scheduler_type`, `min_lr`, `warmup_epochs`, `drop_path_rate`
- 正則化：`use_mixup`, `mixup_alpha`, `label_smoothing`
- Loss 函數：`loss_type`, `focal_alpha`, `focal_gamma`

#### 4.2 獲取 Primary Metric (第 198-201 行)

```
調用 get_primary_metric(dataset_name)

1. 讀取 configs/dataset_primary_metrics.yaml
2. 提取 primary_metric 和 metric_type
3. 返回格式："{primary_metric},{metric_type}"
```

**示例：**
- `ames` → `roc_auc,classification`
- `caco2_wang` → `mae,regression`
- `clearance_hepatocyte_az` → `spearman,regression`

#### 4.3 確定任務類型 (第 254-269 行)

```
1. 如果 task_type 已提供 → 直接使用
2. 否則從配置文件獲取 metric_type
   ├─ 如果 metric_type == "regression" → task_type="regressor"
   ├─ 如果 metric_type == "classification" → task_type="classifier"
   └─ 如果 metric_type 未知或缺失 → 使用硬編碼列表判斷（向後兼容）
3. 最終確定：classifier 或 regressor
```

#### 4.4 數據集特定覆蓋 (第 271-282 行)

```
設置默認的 Early Stopping 參數：
├─ early_stopping_patience: 30
├─ smart_early_stopping_max_patience: 50
└─ auroc_improvement_threshold: 0.005

注意：
├─ TDC 數據集使用 Optuna 優化的超參數（從 best_trial_info_all.json）
├─ 所有 TDC 數據集使用其 Optuna 優化參數，無額外覆蓋
└─ 如需為特定 TDC 數據集添加覆蓋，可在該部分添加
```

#### 4.5 顯示訓練配置 (第 291-301 行)

```
輸出訓練配置信息：
├─ Task Type
├─ Primary Metric
├─ 模型架構參數
├─ 訓練參數
├─ 額外配置（Mixup, Loss, Activation, Aggregation）
└─ Early Stopping 配置
```

**輸出示例：**
```
🚀 Starting V10 Training: ames (Optuna Best Params, 5 Seeds - PARALLEL)
   Task Type: classifier
   Primary Metric: roc_auc
   Model: Dim 128, Layers 6, Heads 4, DMP Steps 2
   Config: LR 0.0000889, Batch 32, WD 0.0000000195, Dropout 0.05
   Extras: Mixup=false, Loss=bce, Activation=ELU, Aggregation=sum
   Early Stopping: Patience=30, Max Patience=50
   ⚡ Parallel Mode: Seeds will run concurrently on 2 GPUs
```

#### 4.6 5-Seed 並行訓練循環 (第 320-477 行)

```
⚡ 並行執行模式：5 個 seeds 同時在 2 顆 GPU 上訓練

GPU 分配策略：
├─ GPU 0：seeds 1, 3, 5
└─ GPU 1：seeds 2, 4

執行流程：

1. 創建 train_single_seed() 函數
   └─ 封裝單個 seed 的訓練邏輯

2. 對每個 seed (1, 2, 3, 4, 5) 並行啟動：
   ├─ 檢查該 seed 的數據是否存在
   │  ├─ 檢查 data/processed_tdc_data/{dataset_name}/seed{seed}/ 目錄
   │  ├─ 檢查 train.pt, valid.pt, test.pt 文件
   │  └─ 如果缺失 → ⚠️ 跳過該 seed，記錄到結果文件
   │
   ├─ 創建保存目錄
   │  ├─ checkpoints/{dataset_name}_optuna_final/seed{seed}/
   │  └─ runs/{dataset_name}_optuna_final/seed{seed}/
   │
   ├─ 構建訓練命令（使用 CUDA_VISIBLE_DEVICES 指定 GPU）
   │  ├─ env CUDA_VISIBLE_DEVICES={gpu_id} python3 scripts/train_edmpnn.py
   │  ├─ --tdc_dataset {dataset_name}
   │  ├─ --tdc_seed {seed}
   │  ├─ --model_type {task_type}
   │  ├─ 所有超參數（從 Optuna 結果加載）
   │  └─ 數據集特定覆蓋參數
   │
   ├─ 在背景執行訓練（&）
   │  ├─ 將結果寫入臨時文件
   │  └─ 格式：SUCCESS:{seed}:{score} 或 FAILED:{seed}:{reason}
   │
   └─ 提取測試分數（訓練完成後）
      ├─ 讀取 training_history.json
      ├─ 根據 primary_metric 優先提取對應的測試分數
      │  ├─ primary_metric="roc_auc" → 提取 test_results["roc_auc"]
      │  ├─ primary_metric="pr_auc" → 提取 test_results["pr_auc"]
      │  ├─ primary_metric="mae" → 提取 test_results["mae"]
      │  └─ primary_metric="spearman" → 提取 test_results["spearman"]
      ├─ 如果 primary_metric 不存在，按順序嘗試其他指標
      └─ 寫入結果文件供後續收集

3. 等待所有背景任務完成
   └─ 使用 wait 命令等待所有 PID

4. 收集所有 seeds 的結果
   ├─ 從臨時文件讀取每個 seed 的執行結果
   ├─ 統計成功/失敗數量
   └─ 收集測試分數到 seed_scores 數組
```

**訓練命令示例：**
```bash
python3 scripts/train_edmpnn.py \
  --tdc_dataset ames \
  --tdc_seed 1 \
  --model_type classifier \
  --use_descriptor \
  --descriptor_dim 217 \
  --hidden_dim 128 \
  --num_layers 6 \
  --num_heads 4 \
  --ffn_expansion_factor 4 \
  --dropout 0.05 \
  --batch_size 32 \
  --learning_rate 0.0000889 \
  --weight_decay 0.0000000195 \
  --drop_path_rate 0.15 \
  --scheduler_type step \
  --warmup_epochs 5 \
  --min_lr 0.000000382 \
  --num_epochs 200 \
  --early_stopping_patience 30 \
  --use_pre_norm \
  --dmp_steps 2 \
  --activation ELU \
  --aggregation sum \
  --use_smart_early_stopping \
  --smart_early_stopping_max_patience 50 \
  --auroc_improvement_threshold 0.005 \
  --use_bce_for_imbalanced \
  --auto_pos_weight \
  --log_dir runs/ames_optuna_final/seed1 \
  --save_dir checkpoints/ames_optuna_final/seed1
```

**並行訓練輸出示例：**
```
🚀 Starting V10 Training: ames (Optuna Best Params, 5 Seeds - PARALLEL)
   ⚡ Parallel Mode: Seeds will run concurrently on 2 GPUs
----------------------------------------
🌱 [GPU 0] Training Seed 1 / 5
----------------------------------------
----------------------------------------
🌱 [GPU 1] Training Seed 2 / 5
----------------------------------------
----------------------------------------
🌱 [GPU 0] Training Seed 3 / 5
----------------------------------------
[GPU 0] Executing: CUDA_VISIBLE_DEVICES=0 python3 scripts/train_edmpnn.py ...
[GPU 1] Executing: CUDA_VISIBLE_DEVICES=1 python3 scripts/train_edmpnn.py ...
[GPU 0] Executing: CUDA_VISIBLE_DEVICES=0 python3 scripts/train_edmpnn.py ...
[訓練過程輸出混合顯示，每個訊息都有 [GPU X] 標記...]
⏳ Waiting for all 5 seeds to complete...
✅ [GPU 0] Seed 1 training completed
   [GPU 0] Test Score: 0.8444
✅ [GPU 1] Seed 2 training completed
   [GPU 1] Test Score: 0.8432
✅ [GPU 0] Seed 3 training completed
   [GPU 0] Test Score: 0.8451
⚠️  [GPU 1] Skipping seed 4: data not found
✅ [GPU 0] Seed 5 training completed
   [GPU 0] Test Score: 0.8440
```

**注意：**
- 由於並行執行，不同 seeds 的輸出會交錯顯示
- 每個輸出訊息都有 `[GPU X]` 標記，方便識別
- 所有 seeds 完成後才會顯示最終統計

#### 4.7 結果統計 (第 479-505 行)

```
1. 計算統計信息
   ├─ 成功/失敗的 seed 數量
   ├─ 如果有測試分數 → 計算 mean ± std
   └─ 顯示個別分數

2. 輸出摘要
   └─ 格式：mean ± std
```

**輸出示例：**
```
================================================
📊 Training Summary for ames
================================================
Successful seeds: 5 / 5
Failed seeds: 0 / 5
Test Score (Mean ± Std): 0.8444 ± 0.0012
Individual scores: 0.8432 0.8445 0.8451 0.8440 0.8452
```

---

### 階段 5: train_edmpnn.py 內部流程

當執行 `python3 scripts/train_edmpnn.py` 時：

#### 5.1 數據加載
```
1. 從 data/processed_tdc_data/{dataset}/seed{seed}/ 加載
   ├─ train.pt → 訓練集
   ├─ valid.pt → 驗證集
   └─ test.pt → 測試集

2. 數據格式檢查
   ├─ 如果是 PyG Data 對象列表 → 直接使用
   └─ 如果是舊格式 → 轉換為 Data 對象
```

#### 5.2 模型創建
```
1. 獲取 primary_metric
   ├─ 從 configs/dataset_primary_metrics.yaml 讀取
   └─ 如果是 regression 任務 → 用於選擇 loss function

2. 創建模型
   ├─ 使用 create_aegnn_model()
   ├─ 傳遞 primary_metric（如果是 regression）
   └─ 根據 primary_metric 選擇：
      ├─ spearman → SpearmanLoss
      └─ mae → L1Loss
```

**輸出示例：**
```
📌 Primary metric for clearance_hepatocyte_az: spearman
   Using SpearmanLoss for training (optimized for Spearman correlation)
Creating model...
```

#### 5.3 訓練過程
```
1. DDP 初始化（如果多 GPU）
2. 數據加載器創建
3. 優化器和調度器設置
4. 訓練循環（最多 200 epochs）
   ├─ 每個 epoch：
   │  ├─ 訓練階段
   │  ├─ 驗證階段
   │  ├─ 學習率調度
   │  └─ Early Stopping 檢查
   └─ Smart Early Stopping：
      ├─ 監控 AUROC 改進
      ├─ 動態調整 patience
      └─ 如果改進 < threshold → 延長 patience

5. 測試評估
   ├─ 使用最佳模型
   ├─ 計算所有指標
   └─ 保存結果到 training_history.json
```

#### 5.4 結果保存
```
保存到 checkpoints/{dataset_name}_optuna_final/seed{seed}/:
├─ best_model.pth → 最佳模型權重
├─ training_history.json → 訓練歷史和測試結果
├─ config.json → 訓練配置
└─ training_progress.json → 進度信息（如果啟用）
```

---

### 階段 6: 最終報告 (第 643-659 行)

```
1. 統計所有數據集
   ├─ TOTAL_DATASETS: 處理的數據集總數
   ├─ SUCCESSFUL_DATASETS: 成功完成的數據集
   └─ FAILED_DATASETS: 失敗的數據集列表

2. 輸出最終摘要
   └─ 顯示成功/失敗統計
```

**輸出示例：**
```
==================================
📊 Final Summary
==================================
Total datasets processed: 22
✅ Successful: 20
❌ Failed: 2
Failed datasets: missing_dataset1 missing_dataset2

🎉 All V10 (TDC Multi-Seed with Optuna Best Params) training tasks completed!
```

---

## 📊 完整流程圖

```
執行 train_edmpnn.sh
│
├─ 1. 初始化
│  ├─ 激活 Conda 環境
│  └─ 設置顏色變量
│
├─ 2. 參數解析
│  ├─ 解析命令行參數
│  ├─ 確定要訓練的數據集列表
│  └─ 應用排除列表
│
├─ 3. 數據集檢查循環
│  └─ 對每個數據集：
│     ├─ check_tdc_dataset() → 檢查數據可用性
│     └─ 如果可用 → 進入訓練流程
│
└─ 4. 訓練循環（對每個數據集）
   │
   ├─ 4.1 加載 Optuna 超參數
   │  └─ 從 best_trial_info_all.json 讀取
   │
   ├─ 4.2 獲取 Primary Metric
   │  └─ 從 dataset_primary_metrics.yaml 讀取
   │
   ├─ 4.3 確定任務類型
   │  └─ classifier 或 regressor
   │
   ├─ 4.4 應用數據集特定覆蓋
   │  └─ 調整超參數（如需要）
   │
   ├─ 4.5 顯示配置信息
   │
   └─ 4.6 5-Seed 並行訓練循環
      │
      ├─ 創建 train_single_seed() 函數
      │
      └─ 並行啟動所有 seeds（背景執行）
         │
         ├─ GPU 0: seeds 1, 3, 5
         │  │
         │  ├─ 構建訓練命令（CUDA_VISIBLE_DEVICES=0）
         │  │
         │  ├─ 執行 train_edmpnn.py（背景）
         │  │  │
         │  │  ├─ 加載 TDC 數據
         │  │  ├─ 創建模型（根據 primary_metric 選擇 loss）
         │  │  ├─ 訓練（最多 200 epochs）
         │  │  │  ├─ 訓練階段
         │  │  │  ├─ 驗證階段
         │  │  │  ├─ Early Stopping
         │  │  │  └─ Smart Early Stopping
         │  │  ├─ 測試評估
         │  │  └─ 保存結果
         │  │
         │  └─ 提取測試分數 → 寫入結果文件
         │
         ├─ GPU 1: seeds 2, 4
         │  │
         │  ├─ 構建訓練命令（CUDA_VISIBLE_DEVICES=1）
         │  │
         │  ├─ 執行 train_edmpnn.py（背景）
         │  │  └─ [同上流程]
         │  │
         │  └─ 提取測試分數 → 寫入結果文件
         │
         ├─ 等待所有背景任務完成（wait）
         │
         └─ 收集所有結果並統計
      │
      └─ 4.7 計算統計
         └─ mean ± std

└─ 5. 最終報告
   └─ 顯示所有數據集的成功/失敗統計
```

---

## ⏱️ 時間估算

假設每個數據集：
- 單個 seed 訓練時間：~2-4 小時（取決於數據集大小和 GPU）
- 5 個 seeds（並行執行，2 顆 GPU）：~5-10 小時
  - GPU 0：seeds 1, 3, 5（約 3 個 seeds 的時間）
  - GPU 1：seeds 2, 4（約 2 個 seeds 的時間）
  - 總時間取決於最慢的 GPU（通常是 GPU 0）
- 22 個數據集（順序執行，但每個數據集內部並行）：~110-220 小時（4.5-9 天）

**已實現的優化：**
- ✅ 5 個 seeds 並行執行（使用 2 顆 GPU）
- ✅ 自動 GPU 分配（seeds 1,3,5 → GPU 0；seeds 2,4 → GPU 1）
- ✅ 背景任務管理（自動等待所有任務完成）

**額外建議：**
- 使用 screen/tmux 保持會話
- 監控 GPU 使用情況（`nvidia-smi`）
- 可考慮進一步並行化多個數據集（需要手動管理）

---

## 📁 輸出文件結構

```
checkpoints/
└─ {dataset_name}_optuna_final/
   ├─ seed1/
   │  ├─ best_model.pth
   │  ├─ training_history.json
   │  ├─ config.json
   │  └─ training_progress.json
   ├─ seed2/
   ├─ seed3/
   ├─ seed4/
   └─ seed5/

runs/
└─ {dataset_name}_optuna_final/
   ├─ seed1/  (TensorBoard logs)
   ├─ seed2/
   ├─ seed3/
   ├─ seed4/
   └─ seed5/
```

---

## 🔍 關鍵檢查點

1. **數據可用性檢查**
   - 確保 `data/processed_tdc_data/{dataset}/seed{1-5}/` 存在
   - 每個 seed 目錄必須有 `train.pt`, `valid.pt`, `test.pt`

2. **超參數文件檢查**
   - 確保 `optuna_edmpnn_results/best_trial_info_all.json` 存在
   - 確保每個數據集都有對應的配置

3. **配置文件檢查**
   - 確保 `configs/dataset_primary_metrics.yaml` 存在
   - 確保每個數據集都有 primary_metric 配置

4. **訓練過程監控**
   - 檢查 TensorBoard logs: `runs/{dataset}_optuna_final/seed{seed}/`
   - 檢查訓練歷史: `checkpoints/{dataset}_optuna_final/seed{seed}/training_history.json`

---

## 🎯 成功標誌

訓練成功完成的標誌：
1. ✅ 所有 5 個 seeds 都成功完成
2. ✅ `training_history.json` 包含測試結果
3. ✅ 測試分數被正確提取
4. ✅ 計算出 mean ± std
5. ✅ 最終報告顯示 "Successful: X"

---

## ⚠️ 常見問題

1. **數據集未找到**
   - 檢查 `data/processed_tdc_data/` 目錄
   - 確保數據已預處理

2. **超參數文件缺失**
   - 確保 Optuna 優化已完成
   - 檢查 `optuna_edmpnn_results/best_trial_info_all.json`

3. **訓練失敗**
   - 檢查 GPU 內存是否足夠（並行執行時兩顆 GPU 都會使用）
   - 檢查日誌文件了解錯誤原因
   - 檢查數據格式是否正確
   - 使用 `nvidia-smi` 監控 GPU 使用情況

4. **並行執行相關問題**
   - **輸出訊息混亂**：這是正常的，因為多個 seeds 同時執行，每個訊息都有 `[GPU X]` 標記
   - **GPU 分配不均**：GPU 0 會執行 3 個 seeds，GPU 1 執行 2 個 seeds，這是設計如此
   - **某個 seed 失敗**：其他 seeds 會繼續執行，最後會顯示成功/失敗統計
   - **監控訓練進度**：可以使用 `nvidia-smi` 或 `htop` 監控 GPU 和 CPU 使用情況

5. **測試分數提取失敗**
   - 檢查 `training_history.json` 是否存在
   - 檢查 JSON 格式是否正確
   - 確認 primary_metric 在結果中
   - 注意：腳本會根據 primary_metric 優先提取對應指標（roc_auc, pr_auc, mae, spearman）
   - 如果 primary_metric 不存在，會按順序嘗試其他指標作為備選

