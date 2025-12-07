"""
ResNet-18 model definition for binary X-ray classification.
Contains model creation, layer freezing, and save/load functions.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path


def create_model(num_classes=2, freeze_layers=True):
    """
    Create a ResNet-18 model for binary classification.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for NORMAL/PNEUMONIA)
        freeze_layers (bool): Whether to freeze all layers except the final FC layer
        
    Returns:
        torch.nn.Module: ResNet-18 model with modified final layer
    """
    # Load pre-trained ResNet-18 with ImageNet weights
    print("Loading ResNet-18 with ImageNet pre-trained weights...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all layers if specified
    if freeze_layers:
        print("Freezing all layers except final FC layer...")
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer for binary classification
    # ResNet-18's FC layer has 512 input features
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    print(f"Modified final FC layer: {num_features} -> {num_classes} classes")
    print(f"Trainable parameters in final layer: {sum(p.numel() for p in model.fc.parameters())}")
    
    return model


def save_model(model, save_path="saved_model.pth"):
    """
    Save the model's state dictionary to a file.
    
    Args:
        model (torch.nn.Module): Model to save
        save_path (str): Path where to save the model weights
    """
    save_path = Path(save_path)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path.absolute()}")


def load_model(model_path="saved_model.pth", device="cpu"):
    """
    Load a trained model from a file.
    
    Args:
        model_path (str): Path to the saved model weights
        device (str or torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model architecture
    model = create_model(num_classes=2, freeze_layers=False)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_path.absolute()}")
    return model


def get_class_names():
    """
    Get the class names for the X-ray dataset.
    
    Returns:
        list: List of class names
    """
    return ["NORMAL", "PNEUMONIA"]
