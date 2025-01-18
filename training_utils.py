import time
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingMonitor:
    def __init__(self, save_dir='training_logs', save_every=50):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_every = save_every
        
        self.losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.tokens_per_sec = []
        self.start_time = time.time()
        
        self.log_file = self.save_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    def update(self, step, loss, lr, grad_norm, tokens_per_sec):
        """Update training metrics."""
        self.losses.append(float(loss))
        self.learning_rates.append(float(lr))
        self.grad_norms.append(float(grad_norm))
        self.tokens_per_sec.append(float(tokens_per_sec))
        
        # Save periodically
        if step % self.save_every == 0:
            self.save_logs()
            self.plot_metrics()
    
    def save_logs(self):
        """Save training logs to file."""
        logs = {
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'grad_norms': self.grad_norms,
            'tokens_per_sec': self.tokens_per_sec,
            'total_training_time': time.time() - self.start_time
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f)
    
    def plot_metrics(self):
        """Create and save training plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        ax1.plot(self.losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        
        # Plot learning rate
        ax2.plot(self.learning_rates)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('LR')
        
        # Plot gradient norm
        ax3.plot(self.grad_norms)
        ax3.set_title('Gradient Norm')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Norm')
        
        # Plot tokens per second
        ax4.plot(self.tokens_per_sec)
        ax4.set_title('Training Speed')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Tokens/sec')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()

def save_checkpoint(model, optimizer, step, loss, save_dir='checkpoints'):
    """Save a training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = save_dir / f'checkpoint_step_{step}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = save_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['loss']
