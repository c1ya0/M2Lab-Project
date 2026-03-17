import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# =================== seed ===================
def set_seed(seed):
    torch.manual_seed(seed) # CPU randomness
    np.random.seed(seed) # NumPy randomness
    random.seed(seed) # Python randomness, e.g., random.shuffle
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # CUDA randomness

# =================== time ===================
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

# =================== save config log ===================
def save_config_log(file_path, args):
    with open(file_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k.upper()}: {v}\n")

#   =================== save training log ===================
def save_training_log(file_path, train_time, best_valid_metric, train_losses, valid_losses):
    with open(file_path, "w") as f:
        f.write("=== Training Summary ===\n")
        f.write(f"Training time: {format_time(train_time)}\n")
        f.write(f"Best valid metric: {best_valid_metric:.3f}\n")
        # loss function
        f.write("=== Losses ===\n")
        f.write(f"{'Epoch':>5} | {'Train Loss':>10} | {'Valid Loss':>10}\n")
        f.write("-" * 34 + "\n")
        for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses), 1):
            f.write(f"{epoch:>5} | {train_loss:>10.3f} | {valid_loss:>10.3f}\n")

# =================== save testing log ===================
def save_testing_log(file_path, test_scores, metric, param_count=None, total_param_count=None, model=None, args=None):
    mean = np.mean(test_scores)
    std = np.std(test_scores)
    with open(file_path, "w") as f:
        f.write(f"===== all seed results summary ({metric}) =====\n")
        for i, score in enumerate(test_scores, 1):
            f.write(f"Seed {i}|Test {metric}={score:.3f}\n")
            
        f.write(f"\n===== average result ({metric}) =====\n") 
        f.write(f"Test {metric}(average)={mean:.3f} +- {std:.3f}\n")
        
        if param_count is not None:
            f.write(f"\n===== model parameter count =====\n")
            f.write(f"Total trainable parameters: {param_count:,}\n")
            if total_param_count is not None:
                f.write(f"Total parameters (including frozen): {total_param_count:,}\n")
                
        if args is not None:
            f.write("\n===== experiment configuration (args) =====\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
                
        if model is not None:
            f.write(f"\n===== model architecture =====\n")
            f.write(str(model))
        
# =================== visualization ===================
def plot_loss_curve(train_losses, valid_losses, save_path=None):
    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend() # display the legend
    plt.grid() # display the grid
    if save_path: # if a save path is specified
        plt.savefig(save_path)
    # plt.show() 
    plt.close()


