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
# model_types = ['DMPEGNN_MMB_DESC']
model_types = ['MMB_DESC']

# ------ dataset ------
dataset_settings = {
    # === A ===
    # 'caco2_wang':{'task_type': 'regression', 'loss': 'MAE',  'metric': 'MAE'},
    # 'hia_hou':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'pgp_broccatelli':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'bioavailability_ma':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'lipophilicity_astrazeneca':{'task_type': 'regression', 'loss': 'MAE',  'metric': 'MAE'},
    # 'solubility_aqsoldb':{'task_type': 'regression', 'loss': 'MAE',  'metric': 'MAE'},
    
    # === D ===
    # 'bbb_martins':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'ppbr_az':{'task_type': 'regression', 'loss': 'MAE',  'metric': 'MAE'},
    # 'vdss_lombardo':{'task_type': 'regression', 'loss': 'MAE',  'metric': 'Spearman'},

    # # === M ===
    # 'cyp2d6_veith': {'task_type': 'classification', 'loss': 'BCE',  'metric': 'PR-AUC'},
    # 'cyp3a4_veith': {'task_type': 'classification', 'loss': 'BCE',  'metric': 'PR-AUC'},
    # 'cyp2c9_veith': {'task_type': 'classification', 'loss': 'BCE',  'metric': 'PR-AUC'},
    # 'cyp2d6_substrate_carbonmangels': {'task_type': 'classification', 'loss': 'BCE',  'metric': 'PR-AUC'},
    # 'cyp3a4_substrate_carbonmangels': {'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'cyp2c9_substrate_carbonmangels': {'task_type': 'classification', 'loss': 'BCE',  'metric': 'PR-AUC'},

    # === E ===
    # 'half_life_obach': {'task_type': 'regression', 'loss': 'MAE',  'metric': 'Spearman'},
    # 'clearance_microsome_az': {'task_type': 'regression', 'loss': 'MAE',  'metric': 'Spearman'},
    # 'clearance_hepatocyte_az': {'task_type': 'regression', 'loss': 'MAE',  'metric': 'Spearman'},
    
    # === T ===
    # 'herg':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    'dili':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'ames':{'task_type': 'classification', 'loss': 'BCE',  'metric': 'ROC-AUC'},
    # 'ld50_zhu':{'task_type': 'regression', 'loss': 'MAE',  'metric': 'MAE'},
}

for model in model_types:
    for data_name, config in dataset_settings.items():
        cmd = f"""
        python -m test.optuna_test \\
            --model_type {model} \\
            --data_name {data_name} \\
            --task_type {config['task_type']} \\
            --loss_function {config['loss']} \\
            --metric {config['metric']} \\
            --seed_list 1 2 3 4 5 \\
            --num_tasks 1
        """
        print(f"Running: {model} on {data_name}")
        os.system(cmd)