import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, Any

class TrainingVisualizer:
    """Class for creating and saving training visualizations."""
    
    def __init__(self, save_dir: str):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        # Load existing history if available
        history_file = self.save_dir / 'training_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.history = json.load(f)
                logging.info(f"Loaded existing training history from {history_file}")
            except Exception as e:
                logging.warning(f"Could not load training history: {str(e)}")
    
    def update_training_plots(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update training plots with new metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        # Update history
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(float(train_metrics['loss']))
        self.history['train_acc'].append(float(train_metrics['accuracy']))
        self.history['val_loss'].append(float(val_metrics['loss']))
        self.history['val_acc'].append(float(val_metrics['accuracy']))
        
        # Save updated history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Create plots
        self._plot_metrics()
        
    def _plot_metrics(self):
        """Create and save training metric plots."""
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss
        ax1.plot(self.history['epochs'], self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['epochs'], self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['epochs'], self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['epochs'], self.history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()
        
    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
    def save_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray):
        """Save ROC curve plot.
        
        Args:
            y_true: True labels
            y_scores: Predicted probabilities
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save
        plt.savefig(self.save_dir / 'roc_curve.png')
        plt.close() 