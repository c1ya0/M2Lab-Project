# drug_properties
This repository contains code and experiments for **predicting ADMET properties** of molecules by integrating multiple molecular representations:

+ Graph-based representations (e.g., Graph Convolutional Network, Directed Message Passing Neural Network)

+ Large language model (LLM) embeddings from MegaMolBART pretrained on SMILES

+ Molecular descriptors computed from RDKit

- Input: SMILES (Simplified Molecular Input Line Entry System)

- Output: ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)

The framework provides Optuna-based hyperparameter optimization and enables multi-modal fusion for improved predictive performance.

## Project Structure

| Folder            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **core/**         | Core modules: models, dataset preprocessing, tokenizer, utilities           |
| **train/**        | Training pipelines (e.g., `optuna_train.py`, `optuna_train_cv.py`)          |
| **test/**         | Testing pipelines (e.g., `optuna_test.py`, `optuna_test_cv.py`)             |
| **scripts/**      | Entry point scripts for running training and testing                        |
| **analysis/**     | Analysis & visualization (notebooks for CV/TDC results, feature importance) |


## Installation

1. Clone this repository
    ```bash
    git clone git@github.com:m2lab-ntu/drug_properties.git
    cd drug_properties
    ```

2. Create the environment

    + Launch Docker container 
    (using image: nvcr.io/nvidia/clara/megamolbart_v0.2:0.2.3)
    ```bash
    sudo docker run \
        --gpus all \
        --name admet \
        --detach \
        --volume $PWD:/workspace \
        nvcr.io/nvidia/clara/megamolbart_v0.2:0.2.3
    ```
    **以本機使用者身份寫入檔案（避免 results 變成 root 擁有）：** 建立容器時加上 `--user $(id -u):$(id -g)`，或進入容器時用下方「Enter the container（非 root）」方式。

    + Enter the container
    ```bash
    sudo docker exec -it admet bash
    cd /workspace
    ```
    + Enter the container **以非 root 執行**（建議：產生的 results 會屬於你，可直接在 VS Code 編輯）
    ```bash
    sudo docker exec -it --user $(id -u):$(id -g) admet bash
    cd /workspace
    ```

3. Install dependencies

    + Install system packages
    ```bash
    apt update && apt install -y libxrender1 libxext6 libsm6 libx11-6 ffmpeg
    ```

    + Install Python packages
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Prepare Data
Place the TDC datasets under the `data/` directory. 
The training and testing scripts will automatically load data from this folder.

### Training

Before starting training, make sure to specify the desired **model type** and **dataset** in the script arguments.

+ Default training:
```bash
python scripts/run_optuna_train.py
```

+ Cross-validation training:
```bash
python scripts/run_optuna_train_cv.py
```

### Testing

Before starting testing, make sure to specify the desired **model type** and **dataset** in the script arguments.

+ Default testing:
```bash
python scripts/run_optuna_test.py
```

+ Cross-validation testing:
```bash
python scripts/run_optuna_test_cv.py
```