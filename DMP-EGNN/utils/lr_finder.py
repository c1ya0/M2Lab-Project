"""
Learning Rate Range Test (LR Range Test) Utility
Used to find optimal learning rate range before training
Based on paper: "Cyclical Learning Rates for Training Neural Networks" (Leslie N. Smith, 2015)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import os


class LRFinder:
    """
    Learning Rate Range Test
    
    Find optimal learning rate range before training by gradually increasing learning rate
    and observing loss changes to determine appropriate learning rate range.
    """
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        """
        Args:
            model: Model
            optimizer: Optimizer
            criterion: Loss function
            device: Device
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save original state
        self.original_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        # Record learning rates and losses
        self.lrs = []
        self.losses = []
        self.best_lr = None
        self.smoothed_losses = []
    
    def find_lr(self, 
                train_loader: DataLoader,
                init_lr: float = 1e-7,
                final_lr: float = 10.0,
                beta: float = 0.98,
                num_iter: Optional[int] = None,
                smooth_f: float = 0.05,
                diverge_th: float = 5.0) -> Tuple[float, float]:
        """
        Execute LR Range Test
        
        Args:
            train_loader: Training data loader
            init_lr: Initial learning rate (default: 1e-7)
            final_lr: Final learning rate (default: 10.0)
            beta: Loss smoothing coefficient (default: 0.98)
            num_iter: Number of iterations (if None, use entire dataset)
            smooth_f: Smoothing factor (default: 0.05)
            diverge_th: Divergence threshold (stop test if loss exceeds this multiple of initial loss)
        
        Returns:
            (best_lr, min_lr) tuple, best_lr is optimal learning rate, min_lr is minimum learning rate
        """
        # Reset model and optimizer state
        self.model.load_state_dict(self.original_state['model'])
        self.optimizer.load_state_dict(self.original_state['optimizer'])
        
        # Determine number of iterations
        if num_iter is None:
            num_iter = len(train_loader)
        
        # Calculate learning rate growth factor
        lr_mult = (final_lr / init_lr) ** (1.0 / num_iter)
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = init_lr
        
        # Initialize
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        avg_loss = 0.0
        best_loss = float('inf')
        best_lr = init_lr
        min_lr = init_lr
        
        # Training mode
        self.model.train()
        
        # Iterate
        iterator = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Range Test")
        
        for iteration in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            batch = batch.to(self.device)
            
            # Forward propagation
            pos = getattr(batch, 'pos', None)
            pred, _ = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, pos=pos)
            
            # Calculate loss
            if hasattr(self.model, 'module'):
                # DDP model
                loss = self.model.module.compute_loss(pred, batch.y)
            else:
                # Regular model
                loss = self.model.compute_loss(pred, batch.y)
            
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional, to avoid divergence)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Smooth loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))
            
            # Record
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            self.losses.append(loss.item())
            self.smoothed_losses.append(smoothed_loss)
            
            # Update best learning rate (lowest loss point)
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                best_lr = current_lr
            
            # Update minimum learning rate (point where loss starts to decrease)
            if iteration > 0 and smoothed_loss < self.smoothed_losses[0] * 0.5:
                if min_lr == init_lr:
                    min_lr = current_lr
            
            # Check if diverged
            if smoothed_loss > diverge_th * best_loss:
                print(f"\n⚠️  Loss diverged at iteration {iteration}, stopping LR test")
                break
            
            # Update learning rate (exponential growth)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr * lr_mult
            
            # Update progress bar
            pbar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'loss': f'{smoothed_loss:.4f}',
                'best_lr': f'{best_lr:.2e}'
            })
        
        # Smooth loss curve
        if len(self.smoothed_losses) > 0:
            smoothed = self._smooth_losses(self.smoothed_losses, smooth_f)
            self.smoothed_losses = smoothed
        
        # Find optimal learning rate (point with fastest loss decrease)
        optimal_lr = self._find_optimal_lr()
        if optimal_lr is None:
            optimal_lr = best_lr
        
        self.best_lr = optimal_lr
        
        return optimal_lr, min_lr
    
    def _smooth_losses(self, losses: List[float], smooth_f: float) -> List[float]:
        """Smooth loss curve"""
        smoothed = []
        for i, loss in enumerate(losses):
            if i == 0:
                smoothed.append(loss)
            else:
                smoothed.append(smooth_f * loss + (1 - smooth_f) * smoothed[-1])
        return smoothed
    
    def _find_optimal_lr(self) -> Optional[float]:
        """
        Find optimal learning rate
        Use method of finding point with fastest gradient descent
        """
        if len(self.smoothed_losses) < 10:
            return None
        
        # Calculate gradient of loss (negative gradient indicates descent speed)
        losses = np.array(self.smoothed_losses)
        lrs = np.array(self.lrs)
        
        # Calculate gradient in log space
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)
        
        # Find point with most negative gradient (fastest descent)
        min_grad_idx = np.argmin(gradients)
        
        # Ensure it's not a boundary point
        if min_grad_idx > 0 and min_grad_idx < len(lrs) - 1:
            return lrs[min_grad_idx]
        
        return None
    
    def plot(self, save_path: Optional[str] = None, skip_start: int = 10, skip_end: int = 5):
        """
        Plot learning rate range test results
        
        Args:
            save_path: Save path (if None, display)
            skip_start: Number of iterations to skip at start (avoid initial instability)
            skip_end: Number of iterations to skip at end (avoid divergence part)
        """
        if len(self.lrs) == 0:
            print("⚠️  No data to plot. Run find_lr() first.")
            return
        
        # Skip start and end parts
        start_idx = skip_start
        end_idx = len(self.lrs) - skip_end if skip_end > 0 else len(self.lrs)
        
        lrs_plot = self.lrs[start_idx:end_idx]
        losses_plot = self.losses[start_idx:end_idx]
        smoothed_plot = self.smoothed_losses[start_idx:end_idx] if len(self.smoothed_losses) > 0 else losses_plot
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw loss and smoothed loss
        ax.plot(lrs_plot, losses_plot, alpha=0.3, color='blue', label='Raw Loss')
        ax.plot(lrs_plot, smoothed_plot, color='blue', label='Smoothed Loss', linewidth=2)
        
        # Mark optimal learning rate
        if self.best_lr is not None:
            best_idx = min(range(len(lrs_plot)), key=lambda i: abs(lrs_plot[i] - self.best_lr))
            ax.axvline(x=self.best_lr, color='red', linestyle='--', 
                      label=f'Optimal LR: {self.best_lr:.2e}')
            ax.plot(self.best_lr, smoothed_plot[best_idx], 'ro', markersize=10)
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Range Test', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved LR range test plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def reset(self):
        """Reset model and optimizer state"""
        self.model.load_state_dict(self.original_state['model'])
        self.optimizer.load_state_dict(self.original_state['optimizer'])


def find_optimal_lr(model, train_loader, device='cuda', 
                   init_lr=1e-7, final_lr=10.0, num_iter=None) -> Tuple[float, float]:
    """
    Convenience function: Find optimal learning rate
    
    Args:
        model: Model
        train_loader: Training data loader
        device: Device
        init_lr: Initial learning rate
        final_lr: Final learning rate
        num_iter: Number of iterations (if None, use entire dataset)
    
    Returns:
        (best_lr, min_lr) tuple
    """
    # Create optimizer (temporary, only for LR test)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    
    # Create LR Finder
    lr_finder = LRFinder(model, optimizer, None, device=device)
    
    # Execute LR Range Test
    best_lr, min_lr = lr_finder.find_lr(
        train_loader,
        init_lr=init_lr,
        final_lr=final_lr,
        num_iter=num_iter
    )
    
    return best_lr, min_lr


if __name__ == "__main__":
    # Test example
    print("LR Range Test Utility")
    print("Usage:")
    print("  from utils.lr_finder import LRFinder, find_optimal_lr")
    print("  best_lr, min_lr = find_optimal_lr(model, train_loader, device='cuda')")
    print("  print(f'Optimal LR: {best_lr:.2e}, Min LR: {min_lr:.2e}')")


