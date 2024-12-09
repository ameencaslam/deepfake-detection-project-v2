import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import json

class DeepfakeDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 split: str = 'train',
                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 seed: int = 42):
        """
        Initialize Deepfake Detection Dataset.
        
        Args:
            data_dir: Root directory containing 'real' and 'fake' folders
            transform: Image transformations
            split: One of ['train', 'val', 'test']
            split_ratio: (train_ratio, val_ratio, test_ratio)
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Get all image paths and labels
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        real_images = [(os.path.join('real', img), 0) for img in os.listdir(real_dir)
                      if img.endswith(('.jpg', '.png', '.jpeg'))]
        fake_images = [(os.path.join('fake', img), 1) for img in os.listdir(fake_dir)
                      if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        all_images = real_images + fake_images
        
        # Split dataset
        train_ratio, val_ratio, test_ratio = split_ratio
        assert abs(sum(split_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1"
        
        # First split into train and temp
        train_images, temp_images = train_test_split(
            all_images,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=[x[1] for x in all_images]
        )
        
        # Split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_images, test_images = train_test_split(
            temp_images,
            train_size=val_ratio_adjusted,
            random_state=seed,
            shuffle=True,
            stratify=[x[1] for x in temp_images]
        )
        
        # Select appropriate split
        if split == 'train':
            self.images = train_images
        elif split == 'val':
            self.images = val_images
        elif split == 'test':
            self.images = test_images
        else:
            raise ValueError(f"Invalid split: {split}")
            
        # Save split information
        self.split_info = {
            'train_size': len(train_images),
            'val_size': len(val_images),
            'test_size': len(test_images),
            'total_size': len(all_images),
            'class_distribution': {
                'real': len(real_images),
                'fake': len(fake_images)
            }
        }
        
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.images[idx]
        img_path = os.path.join(self.data_dir, img_path)
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
    def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
        """Get default transforms for training and validation."""
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        return {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        }
        
    def save_split_info(self, save_path: str):
        """Save dataset split information to JSON file."""
        with open(save_path, 'w') as f:
            json.dump(self.split_info, f, indent=4)
            
    @staticmethod
    def create_dataloaders(data_dir: str,
                          batch_size: int,
                          num_workers: int,
                          image_size: int = 224,
                          split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                          seed: int = 42) -> Dict[str, DataLoader]:
        """Create DataLoaders for train, validation, and test sets."""
        # Get transforms
        transforms_dict = DeepfakeDataset.get_transforms(image_size)
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            datasets[split] = DeepfakeDataset(
                data_dir=data_dir,
                transform=transforms_dict[split],
                split=split,
                split_ratio=split_ratio,
                seed=seed
            )
            
        # Create dataloaders
        dataloaders = {}
        for split in ['train', 'val', 'test']:
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
            
        return dataloaders 