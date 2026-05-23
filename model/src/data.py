"""Data loading and preprocessing for Breaking-Fake model training."""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np


class BreakingFakeDataset(Dataset):
    """
    Custom dataset for AI-generated vs. authentic images.
    
    Expected folder structure:
        data/
          raw/
            ai_images/          (AI-generated, label=1)
            real_images/        (Real photographs, label=0)
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to data/raw folder
            transform: Torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self.default_transform()
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Label 0: AI-generated
        ai_dir = self.data_dir / 'ai_images'
        if ai_dir.exists():
            ai_paths = list(ai_dir.glob('*.jpg')) + list(ai_dir.glob('*.png')) + list(ai_dir.glob('*.jpeg'))
            for img_path in ai_paths:
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # Label 1: Real
        real_dir = self.data_dir / 'real_images'
        if real_dir.exists():
            real_paths = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png')) + list(real_dir.glob('*.jpeg'))
            for img_path in real_paths:
                self.images.append(str(img_path))
                self.labels.append(1)
        
        # Diagnostics
        print(f"Data dir: {self.data_dir} | ai_exists={ai_dir.exists()} | real_exists={real_dir.exists()}")
        print(f"Loaded {len(self.images)} images")
        if len(self.images) == 0:
            print(f"Warning: No images found in {self.data_dir}. Looked into {ai_dir} and {real_dir}.")
    
    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            # Return a blank image on error
            img = torch.zeros(3, 224, 224)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create train, val, and test DataLoaders.
    
    Args:
        data_dir: Path to data/raw folder
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = BreakingFakeDataset(data_dir)

    labels = np.asarray(dataset.labels)
    train_indices = []
    val_indices = []
    test_indices = []
    rng = np.random.default_rng(42)

    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
