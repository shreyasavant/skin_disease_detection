import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_datasets(dataset_dir, image_size=224, batch_size=32, seed=42):
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Define transforms (no augmentation
    # currently matching our CNN model)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    train_ds_full = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    class_names = train_ds_full.classes
    
    # Split into train/validation
    val_size = int(0.2 * len(train_ds_full))
    train_size = len(train_ds_full) - val_size
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size])
    
    # Load test dataset
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"\n===> Dataset ready — Classes: {class_names}")
    return train_loader, val_loader, test_loader, class_names