"""Evaluation module for deepfake detection models."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

from core import load_model
from utils.dataset import DeepfakeDataset
from utils.visualization import TrainingVisualizer
from config.base_config import Config

def evaluate(config: Config) -> None:
    """Evaluate a model with the given configuration."""
    logger = logging.getLogger('deepfake')
    
    # Setup evaluation components
    model = load_model(config)
    dataset = DeepfakeDataset(config.data, split='test')
    visualizer = TrainingVisualizer(config.paths.results)
    
    # Evaluation loop
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataset:
        # Forward pass
        outputs = model(batch['image'])
        predictions = outputs.argmax(dim=1)
        
        # Store results
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    # Get detailed classification report
    class_report = classification_report(
        all_labels, 
        all_preds,
        target_names=['Real', 'Fake'],
        output_dict=True
    )
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info("\nClassification Report:")
    logger.info(pd.DataFrame(class_report).transpose())
    
    # Save visualizations
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        classes=['Real', 'Fake']
    )
    visualizer.finalize()