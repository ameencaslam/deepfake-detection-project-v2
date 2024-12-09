import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
import warnings
from dataclasses import dataclass, field
import gc

@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    style: str = 'seaborn'
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 300
    font_size: int = 10
    color_palette: str = 'deep'
    interactive: bool = True
    dark_mode: bool = False
    
    def apply(self):
        """Apply plot configuration."""
        plt.style.use(self.style)
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['font.size'] = self.font_size
        sns.set_palette(self.color_palette)
        
        if self.dark_mode:
            plt.style.use('dark_background')

@dataclass
class VisualizationMetadata:
    """Metadata for visualizations."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: str = ''
    epoch: int = 0
    dataset_size: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

class TrainingVisualizer:
    def __init__(self,
                 save_dir: Union[str, Path],
                 config: Optional[PlotConfig] = None,
                 cache_size: int = 1000):
        """Initialize training visualizer.
        
        Args:
            save_dir: Directory to save plots
            config: Plot configuration
            cache_size: Number of data points to cache
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PlotConfig()
        self.config.apply()
        self.logger = logging.getLogger(__name__)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'gradient_norms': []
        }
        
        # Initialize cache
        self.cache_size = cache_size
        self.prediction_cache = {
            'true': [],
            'pred': [],
            'prob': []
        }
        
        # Initialize metadata
        self.metadata = VisualizationMetadata()
        
    def update_history(self, metrics: Dict[str, float], lr: float, grad_norm: Optional[float] = None):
        """Update training history with new metrics."""
        self.history['train_loss'].append(metrics['train_loss'])
        self.history['train_acc'].append(metrics['train_acc'])
        self.history['val_loss'].append(metrics['val_loss'])
        self.history['val_acc'].append(metrics['val_acc'])
        self.history['learning_rates'].append(lr)
        if grad_norm is not None:
            self.history['gradient_norms'].append(grad_norm)
            
        # Update metadata
        self.metadata.metrics = metrics
        self.metadata.epoch += 1
        
    def update_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """Update prediction cache."""
        # Maintain cache size
        if len(self.prediction_cache['true']) >= self.cache_size:
            for key in self.prediction_cache:
                self.prediction_cache[key] = self.prediction_cache[key][-self.cache_size:]
                
        self.prediction_cache['true'].extend(y_true)
        self.prediction_cache['pred'].extend(y_pred)
        self.prediction_cache['prob'].extend(y_prob)
        
    def plot_training_history(self, interactive: bool = True):
        """Plot training history including loss, accuracy, and learning rate curves."""
        if not self.history['train_loss']:
            return
            
        if interactive:
            self._plot_training_history_plotly()
        else:
            self._plot_training_history_matplotlib()
            
    def _plot_training_history_plotly(self):
        """Create interactive training history plot with plotly."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Loss', 'Accuracy', 'Learning Rate'),
            vertical_spacing=0.1
        )
        
        # Add loss traces
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['train_loss'],
                      name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['val_loss'],
                      name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Add accuracy traces
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['train_acc'],
                      name='Training Accuracy', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['val_acc'],
                      name='Validation Accuracy', line=dict(color='red')),
            row=2, col=1
        )
        
        # Add learning rate trace
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['learning_rates'],
                      name='Learning Rate', line=dict(color='green')),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text='Training History'
        )
        
        # Save plot
        fig.write_html(self.save_dir / 'training_history.html')
        fig.write_image(self.save_dir / 'training_history.png', scale=2)
        
    def _plot_training_history_matplotlib(self):
        """Create static training history plot with matplotlib."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
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
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if self.config.interactive:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels or ['Negative', 'Positive'],
                y=labels or ['Negative', 'Positive'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={'size': 20},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label'
            )
            
            fig.write_html(self.save_dir / 'confusion_matrix.html')
            fig.write_image(self.save_dir / 'confusion_matrix.png', scale=2)
            
        else:
            plt.figure(figsize=self.config.figsize)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels or ['Negative', 'Positive'],
                       yticklabels=labels or ['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.save_dir / 'confusion_matrix.png')
            plt.close()
            
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {roc_auc:.2f})',
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )
            
            fig.write_html(self.save_dir / 'roc_curve.html')
            fig.write_image(self.save_dir / 'roc_curve.png', scale=2)
            
        else:
            plt.figure(figsize=self.config.figsize)
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
            
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                name=f'PR curve (AUC = {pr_auc:.2f})',
                mode='lines'
            ))
            
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                showlegend=True
            )
            
            fig.write_html(self.save_dir / 'pr_curve.html')
            fig.write_image(self.save_dir / 'pr_curve.png', scale=2)
            
        else:
            plt.figure(figsize=self.config.figsize)
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(self.save_dir / 'pr_curve.png')
            plt.close()
            
    def plot_prediction_distribution(self, probabilities: np.ndarray, labels: np.ndarray):
        """Plot distribution of prediction probabilities."""
        if self.config.interactive:
            fig = go.Figure()
            
            for i, label in enumerate(['Real', 'Fake']):
                mask = labels == i
                fig.add_trace(go.Histogram(
                    x=probabilities[mask],
                    name=f'True {label}',
                    opacity=0.75,
                    nbinsx=50
                ))
                
            fig.update_layout(
                title='Distribution of Prediction Probabilities',
                xaxis_title='Probability of Fake',
                yaxis_title='Count',
                barmode='overlay'
            )
            
            fig.write_html(self.save_dir / 'prediction_distribution.html')
            fig.write_image(self.save_dir / 'prediction_distribution.png', scale=2)
            
        else:
            plt.figure(figsize=self.config.figsize)
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
            
    def plot_feature_importance(self, importance_scores: np.ndarray,
                              feature_names: Optional[List[str]] = None):
        """Plot feature importance scores."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance_scores))]
            
        if self.config.interactive:
            fig = go.Figure(go.Bar(
                x=feature_names,
                y=importance_scores,
                text=importance_scores.round(3),
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                showlegend=False
            )
            
            fig.write_html(self.save_dir / 'feature_importance.html')
            fig.write_image(self.save_dir / 'feature_importance.png', scale=2)
            
        else:
            plt.figure(figsize=self.config.figsize)
            sns.barplot(x=feature_names, y=importance_scores)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'feature_importance.png')
            plt.close()
            
    def plot_learning_rate_schedule(self, optimizer: torch.optim.Optimizer,
                                  num_epochs: int, steps_per_epoch: int):
        """Plot expected learning rate schedule."""
        total_steps = num_epochs * steps_per_epoch
        init_lr = optimizer.param_groups[0]['lr']
        steps = np.linspace(0, total_steps, 100)
        
        if self.config.interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=[init_lr * (1 - x/total_steps) for x in steps],
                mode='lines'
            ))
            
            fig.update_layout(
                title='Expected Learning Rate Schedule',
                xaxis_title='Training Step',
                yaxis_title='Learning Rate',
                yaxis_type='log'
            )
            
            fig.write_html(self.save_dir / 'lr_schedule.html')
            fig.write_image(self.save_dir / 'lr_schedule.png', scale=2)
            
        else:
            plt.figure(figsize=self.config.figsize)
            plt.plot(steps, [init_lr * (1 - x/total_steps) for x in steps])
            plt.title('Expected Learning Rate Schedule')
            plt.xlabel('Training Step')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(self.save_dir / 'lr_schedule.png')
            plt.close()
            
    def plot_metrics_comparison(self, metrics_list: List[Dict[str, float]],
                              model_names: List[str]):
        """Plot comparison of metrics across different models."""
        df = pd.DataFrame(metrics_list, index=model_names)
        
        if self.config.interactive:
            fig = go.Figure()
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=model_names,
                    y=df[metric],
                    text=df[metric].round(3),
                    textposition='auto'
                ))
                
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group'
            )
            
            fig.write_html(self.save_dir / 'model_comparison.html')
            fig.write_image(self.save_dir / 'model_comparison.png', scale=2)
            
        else:
            plt.figure(figsize=(12, 6))
            df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].plot(kind='bar')
            plt.title('Model Performance Comparison')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.save_dir / 'model_comparison.png')
            plt.close()
            
    def save_training_summary(self, metrics: Dict[str, float], model_name: str):
        """Save training summary as text and JSON."""
        # Update metadata
        self.metadata.model_name = model_name
        self.metadata.metrics.update(metrics)
        
        # Save as text
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
            
        # Save as JSON
        json_path = self.save_dir / f'{model_name}_training_summary.json'
        summary_dict = {
            'metadata': self.metadata.__dict__,
            'history': self.history,
            'final_metrics': metrics
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary_dict, f, indent=4)
            
    def cleanup(self):
        """Clean up resources."""
        plt.close('all')
        gc.collect()
        
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup() 