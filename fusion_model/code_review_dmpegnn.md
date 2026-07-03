# 代码审查报告 — DMP-EGNN Fusion Model

**日期**: 2026-04-09  
**审查者**: Claude Code (Sonnet 4.6)  
**范围**: `fusion_model/core/` 和 `fusion_model/train/` 中被修改的 `.py` 文件  
**决定**: **REQUEST CHANGES** — 存在 2 个 HIGH 级别问题

---

## HIGH

### H1 — 跳过 batch 时 avg_loss 分母错误
**文件**: `fusion_model/core/train_utils.py:33` (同样见 `:72`)

**问题**: `avg_loss = total_loss / len(loader)` 使用的是 loader 的总 batch 数作为分母，但循环中凡是 `batch.batch.size(0) < 2` 的 batch 都被 `continue` 跳过、不计入 `total_loss`。当小 batch 频繁出现时，所有 epoch 的 loss 均被系统性低估，Optuna 的超参搜索和 early stopping 的判断基准都会偏移。

**修复**:
```python
def train(model, loader, loss_fn, optimizer, MODEL_TYPE, DEVICE, scheduler=None) -> float:
    model.train()
    total_loss = 0
    num_batches = 0          # <-- 新增
    for batch in loader:
        if batch.batch.size(0) < 2:
            continue
        batch = batch.to(DEVICE)
        output = model_forward(model, batch, MODEL_TYPE)
        loss = loss_fn(output.view(-1), batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        num_batches += 1     # <-- 新增
    avg_loss = total_loss / max(num_batches, 1)   # <-- 修改
    return avg_loss
```
`valid()` 函数中的 `avg_loss = total_loss / len(loader)` (`:72`) 同理需要相同修改。

---

### H2 — `torch.load` 缺少 `weights_only=True`（安全警告 + 未来兼容性）
**文件**: `fusion_model/train/optuna_train.py:527`

**问题**:
```python
torch.save(torch.load(source_path), destination_path)
```
`torch.load` 在 PyTorch ≥ 2.0 中若不指定 `weights_only=True`，会触发 `FutureWarning`；在 PyTorch ≥ 2.6 中默认值已改为 `True`，代码将在新版本中静默失败（无法加载含自定义对象的 checkpoint）。即便此处只保存 `state_dict`，反序列化任意 pickle 也存在潜在安全风险。

**修复**:
```python
torch.save(torch.load(source_path, weights_only=True), destination_path)
```
若 checkpoint 可能包含自定义对象，则改用 `weights_only=False` 并显式注明原因。

---

## MEDIUM

### M1 — `_apply_log1p_to_dataset` 原地修改传入的 dataset
**文件**: `fusion_model/train/seed_train.py:39-54`

**问题**: 函数名以 `_apply_` 开头，实际是原地（in-place）修改 `dataset.labels` 和 `dataset.graphs[i].y`，无返回值、无副作用标注。若调用方在同一进程中复用同一 dataset 对象（例如多 seed 实验或 train/valid 共用引用），会导致 log1p 被重复叠加，训练目标错误且难以追踪。

**修复**: 改为返回新对象，或在文档字符串中明确标注"原地修改，调用前须确保 dataset 未被复用"；函数名加 `_inplace_` 后缀以示警告。推荐方案：
```python
def _apply_log1p_to_dataset(dataset, model_type: str):
    """返回一个 labels/y 已经过 log1p 变换的新 dataset 副本。"""
    import copy
    dataset = copy.deepcopy(dataset)
    if model_type in ("DMPEGNN", "DMPEGNN_MMB_DESC"):
        dataset.labels = [float(np.log1p(l)) for l in dataset.labels]
        for g in dataset.graphs:
            g.y = torch.log1p(g.y)
    else:
        dataset.labels = torch.log1p(dataset.labels)
    return dataset
```

---

### M2 — `best_epoch` 存在未定义风险（防御代码已存在，但逻辑仍有隐患）
**文件**: `fusion_model/train/seed_train.py:407`

**问题**: `best_epoch` 仅在 `is_better` 为 True 时赋值（`:392`）。代码在 `:407` 用 `"best_epoch" in locals()` 做了防御，若从未进入 `is_better` 分支则写入 `None`。但 `None` 被静默存入 summary JSON，调用方无法区分"训练真的未改善"还是"epoch 循环根本未执行"，且日志中不会有任何警告。

**修复**: 在循环前初始化 `best_epoch = None`，并在 summary 写入后显式检查：
```python
best_epoch = None  # 初始化
# ... 训练循环 ...
if best_epoch is None:
    logging.warning("训练结束但 best_epoch 为 None，可能所有 batch 均被跳过或 epoch 数为 0")
```

---

### M3 — `get_model()` 函数在两个模块中近乎完全重复
**文件**: `fusion_model/train/optuna_train.py:86-197` 和 `fusion_model/train/seed_train.py:171-282`

**问题**: 两个函数体逻辑完全一致（seed_train 注释也写明"與 optuna_train.get_model 完全一致"），违反 DRY 原则。日后增加新 model type 或修改参数时，需同步改动两处，极易产生漂移。

**修复**: 将 `get_model` 提取到 `fusion_model/core/model_factory.py`，两个训练脚本均从该模块导入：
```python
# core/model_factory.py
def get_model(args, model_type, task_output_dims, ...):
    ...
```

---

### M4 — `dmpegnn_data_utils.py` 重复 `import os`
**文件**: `fusion_model/core/dmpegnn_data_utils.py:21` 和 `:24`

**问题**: `import os` 出现两次（第 21 行和第 24 行），中间夹着其他 import。虽然不影响运行，但表明该文件曾被手动拼接，且 `isort` / `ruff` 均会报告此问题。

**修复**: 删除重复的 `import os`，将所有标准库 import 整理到文件顶部。

---

### M5 — `_apply_random_rotation` 对每个 graph 串行执行 Python for 循环
**文件**: `fusion_model/core/edmpnn_model_new.py:1030-1034`

**问题**:
```python
for i in range(num_graphs):
    mask = (batch == i)
    rot_matrix = self._get_random_rotation_matrix(...)
    pos_rotated[mask] = (pos[mask] @ rot_matrix).to(dtype)
```
每个 graph 生成独立随机旋转矩阵后依次计算，Python 层循环阻塞 GPU 流水线。在 batch size 较大（如 64）或高频调用（每个 train step 均触发）时，CPU 调度开销显著。

**修复（向量化方案）**:
```python
# 为每个 graph 生成独立旋转矩阵，batched 矩阵乘法
num_graphs = batch.max().item() + 1
# [num_graphs, 3, 3]
rot_matrices = torch.stack([
    self._get_random_rotation_matrix(dtype=dtype, device=device)
    for _ in range(num_graphs)
])
# 将每个节点对应的旋转矩阵广播
R = rot_matrices[batch]          # [N, 3, 3]
pos_rotated = torch.bmm(pos.unsqueeze(1), R).squeeze(1)  # [N, 3]
return pos_rotated.to(dtype)
```
此方案消除 Python 循环，单次 `bmm` 即完成全部旋转。

---

## LOW

### L1 — `return` 语句中 `pos if pos is not None else None` 冗余
**文件**: `fusion_model/core/edmpnn_model_new.py:319`

**问题**:
```python
return out, alpha, pos if pos is not None else None
```
该表达式等价于直接 `return out, alpha, pos`，因为 `pos` 若为 `None` 则 `None if None is not None else None` 仍是 `None`，若非 `None` 则原样返回。条件表达式没有任何实际效果，徒增阅读负担。

**修复**:
```python
return out, alpha, pos
```

---

### L2 — 旧版 `_run_directed_mp` 以三引号字符串形式保留为死代码
**文件**: `fusion_model/core/edmpnn_model_new.py:445-506`

**问题**: 旧实现被包裹在 `'''...'''` 三引号字符串中（不是注释，是字符串字面量）。这是一种常见的"注释掉代码"反模式：Python 解析器仍需解析该字符串，且在某些情况下会被 linter 识别为格式错误的 docstring。版本历史已由 git 保管，死代码应删除。

**修复**: 删除整个三引号块（`:445-506`）。若需保留历史参考，可通过 `git log -p` 查看。

---

## 验证摘要

| # | 文件 | 行号 | 发现确认？ | 严重级别 |
|---|------|------|-----------|---------|
| 1 | `core/train_utils.py` | 33, 72 | 已确认，分母确实用 `len(loader)` 而非已处理 batch 数 | HIGH |
| 2 | `train/seed_train.py` | 39-54 | 已确认，原地修改无副作用标注 | MEDIUM |
| 3 | `train/seed_train.py` | 392, 407 | 已确认，防御代码存在但不够显式 | MEDIUM |
| 4 | `core/edmpnn_model_new.py` | 445-506 | 已确认，三引号死代码块 | LOW |
| 5 | `core/edmpnn_model_new.py` | 518-519 | **安全** — `e_init = e_ij` 是引用，但循环中 `e = e_init + delta` 始终生成新张量，`e_init` 从未被原地 mutate，无 bug | N/A |
| 6 | `core/dmpegnn_data_utils.py` | 21, 24 | 已确认，`import os` 出现两次 | MEDIUM |
| 7 | `train/optuna_train.py` 和 `train/seed_train.py` | 86-197 / 171-282 | 已确认，两函数近乎完全一致 | MEDIUM |
| 8 | `core/edmpnn_model_new.py` | 319 | 已确认，条件表达式冗余 | LOW |
| 9 | `train/optuna_train.py` | 527 | 已确认，`torch.load` 缺少 `weights_only` 参数 | HIGH |
| 10 | `core/edmpnn_model_new.py` | 1030-1034 | 已确认，逐图 Python for 循环，可向量化 | MEDIUM |
| 11 | `core/dmpegnn_dataset.py` | 35 | 已确认，空列表时 `max()` 会 crash（构造函数中 `molecule_indices` 若为空列表则报错）；但当前仅在 `molecule_indices is None` 时才走默认路径（`list(range(len(graphs)))`），若 `graphs` 非空则安全。若调用方传入空列表则会 crash | MEDIUM |

---

## 关于 finding #11（`max(molecule_indices)` 空列表崩溃）补充说明

**文件**: `fusion_model/core/dmpegnn_dataset.py:35`

```python
num_molecules = max(molecule_indices) + 1
```

当 `molecule_indices` 为空列表时，`max()` 抛出 `ValueError: max() arg is an empty sequence`。虽然默认路径（`molecule_indices is None`）通过 `list(range(len(graphs)))` 安全构造，但调用方若显式传入 `molecule_indices=[]` 仍会崩溃，且没有清晰的错误提示。

**修复**:
```python
if not molecule_indices:
    raise ValueError("molecule_indices 不能为空列表")
num_molecules = max(molecule_indices) + 1
```
