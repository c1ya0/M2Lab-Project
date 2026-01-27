import yaml
import sys
import os

def get_task_type(dataset_name):
    """Get task type (classification/regression) for a dataset.
    This function matches the logic in optuna_serach_mod.py exactly."""
    # Clean dataset name (same as optuna_serach_mod.py)
    clean_name = dataset_name.lower().replace('_dataset.csv', '').replace('.csv', '')
    
    config_path = "configs/dataset_primary_metrics.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'dataset_primary_metrics' in config:
                    dataset_config = config['dataset_primary_metrics'].get(clean_name)
                    if dataset_config and 'metric_type' in dataset_config:
                        return dataset_config['metric_type']
        except Exception as e:
            # Silently fail, use fallback
            pass
    
    # Fallback: Use hardcoded classification datasets list (same as optuna_serach_mod.py)
    classification_datasets = [
        'bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21', 'ames',
        'bbb_martins', 'bioavailability_ma', 'cyp3a4_substrate_carbonmangels',
        'dili', 'herg', 'hia_hou', 'pgp_broccatelli', 'cyp2c9_substrate_carbonmangels',
        'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_veith'
    ]
    if clean_name in classification_datasets:
        return 'classification'
    return 'regression'

def get_primary_metric(dataset_name):
    """Get the primary metric for a dataset from config file.
    This function matches the logic in optuna_serach_mod.py exactly."""
    # Clean dataset name (same as optuna_serach_mod.py)
    clean_name = dataset_name.lower().replace('_dataset.csv', '').replace('.csv', '')
    # Handle possible naming differences (same as optuna_serach_mod.py)
    if clean_name == 'solubility_aqsolb':
        clean_name = 'solubility_aqsoldb'
    
    config_path = "configs/dataset_primary_metrics.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'dataset_primary_metrics' in config:
                    dataset_config = config['dataset_primary_metrics'].get(clean_name)
                    if dataset_config and 'primary_metric' in dataset_config:
                        return dataset_config['primary_metric']
        except Exception as e:
            # Silently fail, use fallback
            pass
    
    # Default: Return default metric based on task type (same as optuna_serach_mod.py)
    task_type = get_task_type(dataset_name)
    if task_type == "classification":
        return "roc_auc"
    else:
        # For regression tasks, default to spearman
        return "spearman"

def get_direction(dataset_name):
    """Get Optuna direction for a dataset.
    This function matches the logic in optuna_serach_mod.py exactly."""
    task_type = get_task_type(dataset_name)
    primary_metric = get_primary_metric(dataset_name)
    
    if task_type == "classification":
        # Classification task: optimize roc_auc and f1, both higher is better
        return "maximize"
    else:
        # Regression task: set direction based on primary metric
        if primary_metric == "spearman":
            # Spearman correlation coefficient: higher is better
            return "maximize"
        elif primary_metric == "mae":
            # MAE: lower is better
            return "minimize"
        else:
            # Default to maximize (if primary metric is not defined)
            return "maximize"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_direction.py <dataset_name>")
        sys.exit(1)
    
    dataset = sys.argv[1]
    direction = get_direction(dataset)
    print(direction)
