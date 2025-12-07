"""
Training script with DATA AUGMENTATION for best accuracy.
This version adds random transformations to training images.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path

from model import create_model, save_model
from utils import get_device, IMAGENET_MEAN, IMAGENET_STD


def set_seed(seed=101010):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_augmented_transforms():
    """
    Get training transforms with data augmentation.
    Augmentation helps the model generalize better.
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),                    # Rotate ±10 degrees
        transforms.RandomHorizontalFlip(p=0.5),           # Flip 50% of images
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # Random crop and resize
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, test_transform


def load_augmented_datasets(data_dir="../data/chestxrays"):
    """Load datasets with augmentation for training."""
    data_path = Path(data_dir)
    
    train_transform, test_transform = get_augmented_transforms()
    
    train_dataset = ImageFolder(data_path / "train", transform=train_transform)
    test_dataset = ImageFolder(data_path / "test", transform=test_transform)
    
    print(f"Train dataset: {len(train_dataset)} images (with augmentation)")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes}")
    
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    """Create DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def unfreeze_layer4(model):
    """Unfreeze layer4 for fine-tuning."""
    print("Unfreezing layer4...")
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    return running_loss / len(train_loader)


def evaluate_model(model, test_loader, device):
    """Evaluate the model."""
    model.eval()
    
    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=2, average="macro").to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = accuracy_metric(all_preds, all_labels).item()
    f1 = f1_metric(all_preds, all_labels).item()
    
    return accuracy, f1


def main():
    """Main training function with augmentation."""
    print("=" * 60)
    print("X-Ray Training with DATA AUGMENTATION")
    print("=" * 60)
    
    set_seed(101010)
    device = get_device()
    
    # Load datasets with augmentation
    print("\nLoading datasets with augmentation...")
    train_dataset, test_dataset = load_augmented_datasets()
    
    print("\nCreating data loaders...")
    batch_size = 12  # Optimized for MX350 (2GB VRAM)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=batch_size)
    
    print("\nCreating model...")
    model = create_model(num_classes=2, freeze_layers=True)
    model = unfreeze_layer4(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    num_epochs = 20
    print(f"\nTraining for {num_epochs} epochs with augmentation...")
    print("=" * 60)
    
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc, test_f1 = evaluate_model(model, test_loader, device)
        
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  Test F1-Score: {test_f1:.3f}")
        
        scheduler.step(avg_loss)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best accuracy!")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    
    print(f"\nFINAL RESULTS:")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test F1-Score: {test_f1:.3f}")
    
    save_model(model, "saved_model_augmented.pth")
    print("\nModel saved as: saved_model_augmented.pth")


if __name__ == "__main__":
    main()
