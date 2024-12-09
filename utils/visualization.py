import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

class TrainingVisualizer:
    def __init__(self, save_dir: str):
        """Initialize training visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Set style
        plt.style.use('seaborn')
        
    def update_history(self, metrics: Dict[str, float], lr: float):
        """Update training history with new metrics."""
        self.history['train_loss'].append(metrics['train_loss'])
        self.history['train_acc'].append(metrics['train_acc'])
        self.history['val_loss'].append(metrics['val_loss'])
        self.history['val_acc'].append(metrics['val_acc'])
        self.history['learning_rates'].append(lr)
        
    def plot_training_history(self):
        """Plot training history including loss, accuracy, and learning rate curves."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot(epochs, self.history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(self.save_dir / 'roc_curve.png')
        plt.close()
        
    def plot_learning_rate_schedule(self, optimizer: torch.optim.Optimizer, num_epochs: int, steps_per_epoch: int):
        """Plot expected learning rate schedule."""
        total_steps = num_epochs * steps_per_epoch
        lrs = []
        
        # Get initial learning rate
        init_lr = optimizer.param_groups[0]['lr']
        
        # Create steps array
        steps = np.linspace(0, total_steps, 100)
        
        # Plot learning rate curve
        plt.figure(figsize=(10, 5))
        plt.plot(steps, [init_lr * (1 - x/total_steps) for x in steps])
        plt.title('Expected Learning Rate Schedule')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(self.save_dir / 'lr_schedule.png')
        plt.close()
        
    def plot_prediction_distribution(self, probabilities: np.ndarray, labels: np.ndarray):
        """Plot distribution of prediction probabilities."""
        plt.figure(figsize=(10, 6))
        
        # Plot distributions for each class
        for i, label in enumerate(['Real', 'Fake']):
            mask = labels == i
            sns.kdeplot(probabilities[mask], label=f'True {label}')
            
        plt.title('Distribution of Prediction Probabilities')
        plt.xlabel('Probability of Fake')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'prediction_distribution.png')
        plt.close()
        
    def plot_metrics_comparison(self, metrics_list: List[Dict[str, float]], model_names: List[str]):
        """Plot comparison of metrics across different models."""
        metrics_df = pd.DataFrame(metrics_list, index=model_names)
        
        plt.figure(figsize=(12, 6))
        metrics_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison.png')
        plt.close()
        
    def save_training_summary(self, metrics: Dict[str, float], model_name: str):
        """Save training summary as text file."""
        summary_path = self.save_dir / f'{model_name}_training_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Final Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nBest Validation Metrics:\n")
            best_epoch = np.argmin(self.history['val_loss'])
            f.write(f"Epoch: {best_epoch + 1}\n")
            f.write(f"Loss: {self.history['val_loss'][best_epoch]:.4f}\n")
            f.write(f"Accuracy: {self.history['val_acc'][best_epoch]:.4f}\n") 