"""
Utility functions for data loading and preprocessing.
Contains transforms, dataset loaders, and device detection.
"""
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# ImageNet normalization constants for ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    """
    Automatically detect and return the best available device (CUDA or CPU).
    Prints detailed information about the selected device.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU (No CUDA GPU detected)")
        print(f"   To enable GPU: Install PyTorch with CUDA support")
        print(f"   Command: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    return device


def get_transform():
    """
    Get the standard ResNet preprocessing transform.
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline for ResNet-18
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_datasets(data_dir="../data/chestxrays"):
    """
    Load training and testing datasets from the specified directory.
    
    Args:
        data_dir (str): Path to the data directory containing train/ and test/ folders
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    data_path = Path(data_dir)
    transform = get_transform()
    
    train_dataset = ImageFolder(data_path / "train", transform=transform)
    test_dataset = ImageFolder(data_path / "test", transform=transform)
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes}")
    
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size=32, train_shuffle=True):
    """
    Create DataLoaders for training and testing.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        batch_size (int): Batch size for training
        train_shuffle (bool): Whether to shuffle training data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # Single batch for evaluation
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader
