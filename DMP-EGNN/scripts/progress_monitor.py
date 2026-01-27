"""
Efficient progress monitoring tool
Provides multiple progress monitoring mechanisms for Optuna pruning
"""

import os
import json
import time
import fcntl
from pathlib import Path
from typing import Dict, Optional, Tuple
from multiprocessing import Queue, Process
import threading


class JSONProgressMonitor:
    """
    Method 1: JSON progress file (recommended)
    Pros: Simple, efficient, easy to debug, doesn't require much modification to training script
    Cons: Training script needs to periodically write progress file
    """
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self._lock_file = f"{progress_file}.lock"
    
    def write_progress(self, epoch: int, val_loss: float, train_loss: Optional[float] = None):
        """Training script calls this method to write progress"""
        progress_data = {
            'epoch': epoch,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'timestamp': time.time()
        }
        
        # Use file lock to ensure atomic write
        try:
            with open(self._lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    with open(self.progress_file, 'w') as f:
                        json.dump(progress_data, f)
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Failed to write progress: {e}")
    
    def read_progress(self) -> Optional[Dict]:
        """Read current progress"""
        if not os.path.exists(self.progress_file):
            return None
        
        try:
            with open(self._lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
                try:
                    with open(self.progress_file, 'r') as f:
                        return json.load(f)
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            return None
    
    def get_latest_epoch_and_loss(self) -> Optional[Tuple[int, float]]:
        """Get latest epoch and validation loss"""
        progress = self.read_progress()
        if progress:
            return progress.get('epoch'), progress.get('val_loss')
        return None


class QueueProgressMonitor:
    """
    Method 2: Inter-process communication Queue (efficient but requires training script modification)
    Pros: Real-time, efficient, no file I/O needed
    Cons: Requires training script modification to support callbacks
    """
    
    def __init__(self):
        self.queue = Queue()
        self._latest_progress = None
        self._monitor_thread = None
        self._stop_monitoring = False
    
    def start_monitoring(self):
        """Start monitoring thread"""
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while not self._stop_monitoring:
            try:
                if not self.queue.empty():
                    self._latest_progress = self.queue.get_nowait()
            except Exception:
                pass
            time.sleep(0.1)
    
    def put_progress(self, epoch: int, val_loss: float, train_loss: Optional[float] = None):
        """Training script calls this method to report progress"""
        try:
            self.queue.put_nowait({
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'timestamp': time.time()
            })
        except Exception:
            pass
    
    def get_latest_epoch_and_loss(self) -> Optional[Tuple[int, float]]:
        """Get latest epoch and validation loss"""
        if self._latest_progress:
            return self._latest_progress.get('epoch'), self._latest_progress.get('val_loss')
        return None


class LogFileProgressMonitor:
    """
    Method 3: Parse log files (slowest but no training script modification needed)
    Pros: No training script modification needed
    Cons: Slow, requires parsing text, may miss updates
    """
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self._last_position = 0
    
    def get_latest_epoch_and_loss(self) -> Optional[Tuple[int, float]]:
        """Parse log file to get latest epoch and validation loss"""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                # Move to last read position
                f.seek(self._last_position)
                lines = f.readlines()
                self._last_position = f.tell()
                
                # Parse from newest to oldest
                epoch = None
                val_loss = None
                for line in reversed(lines):
                    # Look for validation loss
                    if 'val_loss' in line.lower() or 'validation loss' in line.lower():
                        # Extract numbers (this is a simple example, may need adjustment based on actual log format)
                        import re
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if len(numbers) >= 2:
                            epoch = int(numbers[0])
                            val_loss = float(numbers[1])
                            break
                
                if epoch is not None and val_loss is not None:
                    return epoch, val_loss
        except Exception:
            pass
        
        return None


class TensorBoardProgressMonitor:
    """
    Method 4: Read TensorBoard event files (moderate speed, no training script modification needed)
    Pros: No training script modification needed, works with existing TensorBoard logs
    Cons: Requires TensorBoard library, somewhat complex
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._last_update_time = 0
        self._cached_progress = None
    
    def get_latest_epoch_and_loss(self) -> Optional[Tuple[int, float]]:
        """Read TensorBoard event files to get latest epoch and validation loss"""
        # Check if cache is still fresh (1 second)
        current_time = time.time()
        if current_time - self._last_update_time < 1.0 and self._cached_progress:
            return self._cached_progress
        
        try:
            from tensorboard.backend.event_processing import event_accumulator
            
            if not os.path.exists(self.log_dir):
                return None
            
            # Find newest event file
            event_files = []
            for root, dirs, files in os.walk(self.log_dir):
                for file in files:
                    if file.startswith('events.out.tfevents'):
                        event_files.append(os.path.join(root, file))
            
            if not event_files:
                return None
            
            # Read newest file
            newest_file = max(event_files, key=os.path.getmtime)
            ea = event_accumulator.EventAccumulator(newest_file)
            ea.Reload()
            
            # Try to read validation loss
            if 'val_loss' in ea.Tags()['scalars']:
                events = ea.Scalars('val_loss')
                if events:
                    latest_event = events[-1]
                    epoch = latest_event.step
                    val_loss = latest_event.value
                    
                    self._cached_progress = (epoch, val_loss)
                    self._last_update_time = current_time
                    return epoch, val_loss
        except Exception:
            pass
        
        return None


# Example usage
if __name__ == "__main__":
    # Test JSONProgressMonitor
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    progress_file = os.path.join(temp_dir, "training_progress.json")
    
    monitor = JSONProgressMonitor(progress_file)
    
    # Simulate writing progress
    for epoch in range(5):
        monitor.write_progress(epoch, val_loss=0.5 - epoch * 0.05, train_loss=0.6 - epoch * 0.06)
        time.sleep(0.1)
        
        # Read progress
        result = monitor.get_latest_epoch_and_loss()
        if result:
            print(f"Epoch {result[0]}, Val Loss: {result[1]:.4f}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nTest complete!")
