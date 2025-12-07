"""
FastAPI server for X-ray classification.
Provides REST API endpoint for image prediction.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path

from predict import predict_from_pil
from utils import get_device


# Initialize FastAPI app
app = FastAPI(
    title="X-Ray Classification API",
    description="Binary classification of chest X-rays (NORMAL vs PNEUMONIA)",
    version="1.0.0"
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and device
MODEL_PATH = Path(__file__).parent / "saved_model.pth"
DEVICE = None


@app.on_event("startup")
async def startup_event():
    """Initialize device on startup."""
    global DEVICE
    DEVICE = get_device()
    print(f"API server started. Using device: {DEVICE}")
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please run 'python train.py' first to train the model.")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "X-Ray Classification API",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_exists = MODEL_PATH.exists()
    return {
        "status": "healthy" if model_exists else "model_not_found",
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded X-ray image.
    
    Args:
        file (UploadFile): Uploaded image file
        
    Returns:
        dict: Prediction result with class name and confidence
        
    Example response:
        {
            "prediction": "NORMAL",
            "confidence": 0.982
        }
    """
    # Validate model exists
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first using 'python train.py'"
        )
    
    try:
        # Read and open image
        contents = await file.read()
        
        # Try to open as image - this validates it's actually an image
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()  # Verify it's a valid image
            # Re-open because verify() closes the file
            image = Image.open(io.BytesIO(contents))
        except Exception as img_error:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(img_error)}"
            )
        
        # Run prediction
        result = predict_from_pil(
            pil_image=image,
            model_path=str(MODEL_PATH),
            device=DEVICE
        )
        
        # Round confidence to 3 decimal places
        result["confidence"] = round(result["confidence"], 3)
        
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        await file.close()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
