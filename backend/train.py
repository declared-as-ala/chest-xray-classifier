"""
Training script for the X-ray classification model.
Trains ResNet-18 for 3 epochs and evaluates on test set.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score

from model import create_model, save_model
from utils import get_device, load_datasets, create_dataloaders


def set_seed(seed=101010):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        tuple: (accuracy, f1_score)
    """
    model.eval()
    
    # Initialize metrics
    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=2, average="macro").to(device)
    
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Collect predictions and labels
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_metric(all_preds, all_labels).item()
    f1 = f1_metric(all_preds, all_labels).item()
    
    return accuracy, f1


def main():
    """Main training function."""
    print("=" * 60)
    print("X-Ray Classification Model Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(101010)
    
    # Get device
    device = get_device()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset = load_datasets()
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=32,
        train_shuffle=True
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=2, freeze_layers=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize the final layer parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Training configuration
    num_epochs = 3
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    
    # Print results (rounded to 3 decimals)
    print(f"\nTest Accuracy: {test_acc:.3f}")
    print(f"Test F1-Score: {test_f1:.3f}")
    
    # Save the model
    print("\nSaving model...")
    save_model(model, "saved_model.pth")
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
