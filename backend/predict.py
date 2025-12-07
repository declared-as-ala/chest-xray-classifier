"""
Prediction module for single image inference.
Handles image preprocessing and model inference.
"""
import torch
from PIL import Image
from pathlib import Path

from model import load_model, get_class_names
from utils import get_transform, get_device


def predict_image(image_path, model_path="saved_model.pth", device=None):
    """
    Predict the class of a single X-ray image.
    
    Args:
        image_path (str or Path): Path to the image file
        model_path (str): Path to the saved model weights
        device (str or torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary with 'prediction' (class name) and 'confidence' (float)
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    # Load the trained model
    model = load_model(model_path, device)
    
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Get prediction
    prediction, confidence = run_inference(model, image, device)
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }


def predict_from_pil(pil_image, model_path="saved_model.pth", device=None):
    """
    Predict the class of a PIL Image object.
    
    Args:
        pil_image (PIL.Image): PIL Image object
        model_path (str): Path to the saved model weights
        device (str or torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary with 'prediction' (class name) and 'confidence' (float)
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    # Load the trained model
    model = load_model(model_path, device)
    
    # Preprocess the PIL image
    transform = get_transform()
    
    # Convert to RGB if needed (in case of grayscale or RGBA)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    image = transform(pil_image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Get prediction
    prediction, confidence = run_inference(model, image, device)
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image file.
    
    Args:
        image_path (str or Path): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_transform()
    image = transform(image)
    
    return image


def run_inference(model, image_tensor, device):
    """
    Run inference on a preprocessed image tensor.
    
    Args:
        model (torch.nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor (batch of 1)
        device (torch.device): Device to run on
        
    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class and confidence
        confidence, predicted_idx = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        
        # Get class name
        class_names = get_class_names()
        predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidence


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_image(image_path)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
