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
        
        # Try to load existing split
        split_info_path = os.path.join(data_dir, 'split_info.json')
        if os.path.exists(split_info_path):
            try:
                with open(split_info_path, 'r') as f:
                    saved_split = json.load(f)
                
                # Verify the dataset hasn't changed
                if saved_split['total_size'] == len(all_images) and \
                   saved_split['class_distribution']['real'] == len(real_images) and \
                   saved_split['class_distribution']['fake'] == len(fake_images):
                    
                    # Use saved split information
                    print(f"Using existing dataset split from {split_info_path}")
                    train_size = saved_split['train_size']
                    val_size = saved_split['val_size']
                    test_size = saved_split['test_size']
                    
                    # Sort images to ensure consistent order
                    all_images.sort(key=lambda x: x[0])
                    
                    # Split according to saved sizes
                    train_images = all_images[:train_size]
                    val_images = all_images[train_size:train_size + val_size]
                    test_images = all_images[train_size + val_size:]
                    
                    self.split_info = saved_split
                    
                else:
                    print("Dataset has changed, creating new split")
                    train_images, val_images, test_images = self._create_split(
                        all_images, split_ratio, seed
                    )
            except Exception as e:
                print(f"Error loading split info: {str(e)}, creating new split")
                train_images, val_images, test_images = self._create_split(
                    all_images, split_ratio, seed
                )
        else:
            train_images, val_images, test_images = self._create_split(
                all_images, split_ratio, seed
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
            
    def _create_split(self, all_images: List[Tuple[str, int]], 
                    split_ratio: Tuple[float, float, float],
                    seed: int) -> Tuple[List[Tuple[str, int]], ...]:
        """Create a new dataset split."""
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
        
        # Save split information
        self.split_info = {
            'train_size': len(train_images),
            'val_size': len(val_images),
            'test_size': len(test_images),
            'total_size': len(all_images),
            'class_distribution': {
                'real': len([x for x in all_images if x[1] == 0]),
                'fake': len([x for x in all_images if x[1] == 1])
            }
        }
        
        return train_images, val_images, test_images
        
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
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Add timestamp and additional metadata
        split_info = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': self.data_dir,
            'split_ratio': {
                'train': self.split_info['train_size'] / self.split_info['total_size'],
                'val': self.split_info['val_size'] / self.split_info['total_size'],
                'test': self.split_info['test_size'] / self.split_info['total_size']
            },
            **self.split_info
        }
        
        # Save to JSON file
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=4)
            
        # Try to backup using project manager
        try:
            from manage import ProjectManager
            project_manager = ProjectManager()
            project_manager.backup()  # This will include the split info file
        except Exception as e:
            print(f"Note: Could not backup split info: {str(e)}")
            
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
            
        # Save split info in dataset directory
        split_info_path = os.path.join(data_dir, 'split_info.json')
        datasets['train'].save_split_info(split_info_path)
        
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

    @staticmethod
    def get_dataloader(data_path: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: int = 224,
                      train: bool = False) -> DataLoader:
        """Get a single dataloader for evaluation or training.
        
        Args:
            data_path: Path to data directory containing 'real' and 'fake' subdirectories
            batch_size: Batch size for the dataloader
            num_workers: Number of workers for data loading
            image_size: Size of input images
            train: Whether this is for training (affects transforms and shuffling)
        """
        # Get appropriate transform
        transforms_dict = DeepfakeDataset.get_transforms(image_size)
        transform = transforms_dict['train'] if train else transforms_dict['val']
        
        # Create dataset
        dataset = DeepfakeDataset(
            data_dir=data_path,
            transform=transform,
            split='train' if train else 'test',  # Use test split for evaluation
            split_ratio=(0.7, 0.15, 0.15)  # Default split ratio
        )
        
        # Create and return dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,  # Shuffle only for training
            num_workers=num_workers,
            pin_memory=True
        )

# Update exports at the end of the file
__all__ = ['DeepfakeDataset', 'get_dataloader', 'create_dataloaders'] 