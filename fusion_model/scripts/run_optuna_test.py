import os
os.environ["NEMO_LOG_LEVEL"] = "ERROR" # Suppress NeMo logs

# ------ model ------
# model_types = ['DESC', 'GCN', 'MMB', 'MPN', 'MPN_DESC', 'GCN_MMB_DESC', 'MPN_MMB_DESC']
# model_types = ['DESC', 'GCN', 'MMB', 'MPN', 'MPN_DESC', 'GCN_MMB_DESC', 'MPN_MMB_DESC', 'DMPEGNN', 'DMPEGNN_MMB_DESC']
# model_types = ['DESC']
# model_types = ['GCN']
# model_types = ['MMB']
# model_types = ['MPN']
# model_types = ['MPN_DESC']
# model_types = ['MPN_MMB']
# model_types = ['GCN_MMB_DESC']
# model_types = ['MPN_MMB_DESC']
# model_types = ['DMPEGNN_DESC']
# model_types = ['DMPEGNN_MMB_DESC']
# model_types = ['DMPEGNN']
# model_types = ['MMB_DESC']
model_types = ['AEGNN']        
# model_types = ['AEGNN_DESC']   

# ------ dataset ------
# log_transform: True  → Y 是正值且右偏（raw units），測試前套 log1p，test() 以 expm1 還原
# log_transform: False → Y 已是 log scale 或有界百分比，不套任何轉換

dataset_settings = {
    # === A ===
    # 'caco2_wang':                   {'task_type': 'regression',    'loss': 'MAE', 'metric': 'MAE',      'log_transform': False},  # log₁₀ 滲透率，已 log scale
    # 'hia_hou':                    {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    # 'pgp_broccatelli':            {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    # 'bioavailability_ma':         {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    'lipophilicity_astrazeneca':  {'task_type': 'regression',    'loss': 'MAE', 'metric': 'MAE',      'log_transform': False},  # logD，已 log scale
    # 'solubility_aqsoldb':         {'task_type': 'regression',    'loss': 'MAE', 'metric': 'MAE',      'log_transform': False},  # log 溶解度，已 log scale

    # === D ===
    # 'bbb_martins':                {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    # 'ppbr_az':                    {'task_type': 'regression',    'loss': 'MAE', 'metric': 'MAE',      'log_transform': False},  # 血漿蛋白結合率 (%)，有界左偏
    # 'vdss_lombardo':              {'task_type': 'regression',    'loss': 'MAE', 'metric': 'Spearman', 'log_transform': True},   # 分布容積 (L/kg)，極右偏

    # === M ===
    # 'cyp2d6_veith':               {'task_type': 'classification', 'loss': 'BCE', 'metric': 'PR-AUC',   'log_transform': False},
    # 'cyp3a4_veith':               {'task_type': 'classification', 'loss': 'BCE', 'metric': 'PR-AUC',   'log_transform': False},
    # 'cyp2c9_veith':               {'task_type': 'classification', 'loss': 'BCE', 'metric': 'PR-AUC',   'log_transform': False},
    # 'cyp2d6_substrate_carbonmangels': {'task_type': 'classification', 'loss': 'BCE', 'metric': 'PR-AUC',  'log_transform': False},
    # 'cyp3a4_substrate_carbonmangels': {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC', 'log_transform': False},
    # 'cyp2c9_substrate_carbonmangels': {'task_type': 'classification', 'loss': 'BCE', 'metric': 'PR-AUC',  'log_transform': False},

    # === E ===
    # 'half_life_obach':            {'task_type': 'regression',    'loss': 'MAE', 'metric': 'Spearman', 'log_transform': True},   # 半衰期 (hr)，極右偏
    # 'clearance_microsome_az':     {'task_type': 'regression',    'loss': 'MAE', 'metric': 'Spearman', 'log_transform': True},   # 微粒體清除率，右偏
    # 'clearance_hepatocyte_az':    {'task_type': 'regression',    'loss': 'MAE', 'metric': 'Spearman', 'log_transform': True},   # 肝清除率，右偏

    # === T ===
    # 'herg':                       {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    # 'dili':                       {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    # 'ames':                       {'task_type': 'classification', 'loss': 'BCE', 'metric': 'ROC-AUC',  'log_transform': False},
    # 'ld50_zhu':                   {'task_type': 'regression',    'loss': 'MAE', 'metric': 'MAE',      'log_transform': True},   # LD50，正值右偏
}

for model in model_types:
    for data_name, config in dataset_settings.items():
        log_transform_flag = "--log_transform \\" if config.get('log_transform', False) else ""
        cmd = f"""
        python -m test.optuna_test \\
            --model_type {model} \\
            --data_name {data_name} \\
            --task_type {config['task_type']} \\
            --loss_function {config['loss']} \\
            --metric {config['metric']} \\
            --seed_list 1 2 3 4 5 \\
            --num_tasks 1 \\
            {log_transform_flag}
        """
        print(f"Running: {model} on {data_name}")
        os.system(cmd)
