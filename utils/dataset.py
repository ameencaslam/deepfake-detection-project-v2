import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import random
import logging

class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection."""
    
    def __init__(self,
                 data_path: str,
                 image_size: int = 224,
                 train: bool = True,
                 transform = None,
                 **kwargs):
        """Initialize dataset.
        
        Args:
            data_path: Path to dataset root
            image_size: Size to resize images to
            train: Whether this is training set
            transform: Optional transform to apply
        """
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.train = train
        
        # Get image paths and labels
        real_dir = os.path.join(data_path, 'real')
        fake_dir = os.path.join(data_path, 'fake')
        
        # Get all image paths and labels
        real_images = [(os.path.join('real', img), 0) for img in os.listdir(real_dir)
                      if img.endswith(('.png', '.jpg', '.jpeg'))]
        fake_images = [(os.path.join('fake', img), 1) for img in os.listdir(fake_dir)
                      if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Combine and shuffle
        all_images = real_images + fake_images
        random.shuffle(all_images)
        
        # Load or create train/val split
        split_file = os.path.join(data_path, 'split_info.json')
        if os.path.exists(split_file):
            # Load existing split
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            train_images = split_info['train']
            val_images = split_info['val']
            logging.info(f"Loaded existing split: {len(train_images)} train, {len(val_images)} val")
        else:
            # Create new split
            split_idx = int(len(all_images) * 0.8)  # 80% train, 20% val
            train_images = all_images[:split_idx]
            val_images = all_images[split_idx:]
            
            # Save split info
            split_info = {
                'train': train_images,
                'val': val_images,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': len(all_images),
                'train_size': len(train_images),
                'val_size': len(val_images)
            }
            with open(split_file, 'w') as f:
                json.dump(split_info, f, indent=4)
            logging.info(f"Created new split: {len(train_images)} train, {len(val_images)} val")
        
        # Set images based on mode
        self.images = train_images if train else val_images
        
        # Set up transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of item to get
            
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.images[idx]
        img_path = os.path.join(self.data_path, img_path)
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    @staticmethod
    def get_dataloader(data_path: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: int = 224,
                      train: bool = True,
                      **kwargs) -> DataLoader:
        """Get data loader.
        
        Args:
            data_path: Path to dataset root
            batch_size: Batch size
            num_workers: Number of workers for loading
            image_size: Size to resize images to
            train: Whether this is training set
            
        Returns:
            DataLoader instance
        """
        # Create dataset
        dataset = DeepfakeDataset(
            data_path=data_path,
            image_size=image_size,
            train=train,
            **kwargs
        )
        
        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=train
        )
        
        return loader

# Update exports at the end of the file
__all__ = ['DeepfakeDataset', 'get_dataloader', 'create_dataloaders'] 