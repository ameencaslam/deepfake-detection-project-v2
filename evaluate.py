import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from models.model_registry import get_model, MODEL_REGISTRY
from utils.dataset import DeepfakeDataset
from utils.hardware import HardwareManager
from config.paths import get_checkpoint_dir, get_data_dir
from config.base_config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    
    # Get default paths from config
    default_data = get_data_dir()
    default_checkpoint_dir = get_checkpoint_dir()
    
    # Basic arguments with defaults
    parser.add_argument('--model', type=str, default='efficientnet',
                       choices=list(MODEL_REGISTRY.keys()),
                       help=f'Model architecture to use. Available: {", ".join(MODEL_REGISTRY.keys())}')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint. If not specified, will use latest from checkpoint directory')
    parser.add_argument('--data_path', type=str, default=default_data,
                       help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=Config().training.batch_size,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=Config().data.image_size,
                       help='Input image size')
    
    args = parser.parse_args()
    
    # If checkpoint not specified, find latest for the model
    if args.checkpoint is None:
        checkpoint_dir = Path(default_checkpoint_dir) / args.model
        if not checkpoint_dir.exists():
            raise ValueError(f"No checkpoints found for model {args.model} in {checkpoint_dir}")
        
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Get latest checkpoint by modification time
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        args.checkpoint = str(latest)
        print(f"Using latest checkpoint: {args.checkpoint}")
    
    return args

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def compute_metrics(predictions, labels, probabilities):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc_roc = roc_auc_score(labels, probabilities)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    
    # Add confusion matrix counts
    true_positives = ((predictions == 1) & (labels == 1)).sum()
    true_negatives = ((predictions == 0) & (labels == 0)).sum()
    false_positives = ((predictions == 1) & (labels == 0)).sum()
    false_negatives = ((predictions == 0) & (labels == 1)).sum()
    
    metrics.update({
        'true_positives': int(true_positives),
        'true_negatives': int(true_negatives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'total_samples': len(labels)
    })
    
    return metrics

def main():
    args = parse_args()
    
    # Initialize hardware
    hw_manager = HardwareManager()
    hw_manager.print_hardware_info()
    
    # Create config for model initialization
    config = Config()
    config.model.architecture = args.model
    
    # Load model
    model = get_model(args.model, config=config.model.__dict__)
    checkpoint = torch.load(args.checkpoint, map_location=hw_manager.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(hw_manager.device)
    
    print(f"\nEvaluating {args.model} model")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    
    # Get data loader
    test_loader = DeepfakeDataset.get_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=config.data.num_workers,
        image_size=args.image_size,
        train=False
    )
    
    # Evaluate
    predictions, labels, probabilities = evaluate(model, test_loader, hw_manager.device)
    metrics = compute_metrics(predictions, labels, probabilities)
    
    # Print detailed results
    print("\nEvaluation Results:")
    print(f"Total Samples: {metrics['total_samples']}")
    print("\nMetrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

if __name__ == '__main__':
    main() 