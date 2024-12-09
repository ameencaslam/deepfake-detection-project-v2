import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
import os

class TrainingVisualizer:
    def __init__(self, save_dir: Path):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots and results
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
        
    def save_training_summary(self, metrics: dict, model_name: str):
        """Save training summary and plots."""
        # Save metrics
        metrics_file = self.save_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Create plots
        self.plot_confusion_matrix(
            np.array(metrics['all_labels']),
            np.array(metrics['all_preds'])
        )
        self.plot_roc_curve(
            np.array(metrics['all_labels']),
            np.array(metrics['all_probs'])
        )
        self.plot_prediction_distribution(
            np.array(metrics['all_probs']),
            np.array(metrics['all_labels'])
        )
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
    def plot_roc_curve(self, y_true, y_score):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / 'roc_curve.png')
        plt.close()
        
    def plot_prediction_distribution(self, probabilities, labels):
        """Plot distribution of prediction probabilities."""
        plt.figure(figsize=(10, 6))
        
        # Plot distributions for each class
        for i, label in enumerate(['Real', 'Fake']):
            mask = labels == i
            if np.any(mask):
                sns.kdeplot(probabilities[mask], label=f'{label} Images',
                          fill=True, alpha=0.5)
        
        plt.xlabel('Prediction Probability (Fake)')
        plt.ylabel('Density')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.savefig(self.save_dir / 'prediction_distribution.png')
        plt.close() 