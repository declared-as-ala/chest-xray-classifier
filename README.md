# X-Ray Classification System

A complete full-stack application for classifying chest X-rays as NORMAL or PNEUMONIA using deep learning (ResNet-18 transfer learning) with a FastAPI backend and React frontend.

## 🎯 Project Overview
<img width="1903" height="865" alt="Capture d&#39;écran 2025-12-07 205556" src="https://github.com/user-attachments/assets/7fd868af-0e13-4ca6-a1e8-7813f69aa7ad" />

This application provides an AI-powered diagnostic tool that:
- **Backend**: Uses PyTorch and ResNet-18 (pre-trained on ImageNet) for binary classification
- **Frontend**: Modern React interface with shadcn/ui components for image upload and result visualization
- **Transfer Learning**: Fine-tunes only the final layer for 3 epochs on chest X-ray data
- **REST API**: FastAPI provides a `/predict` endpoint for real-time inference

## 📁 Project Structure

```
pytorch/
│
├── backend/                    # FastAPI + PyTorch backend
│   ├── main.py                # FastAPI server with /predict endpoint
│   ├── model.py               # ResNet-18 model definition and save/load
│   ├── train.py               # Training script (3 epochs)
│   ├── predict.py             # Single image prediction logic
│   ├── utils.py               # Data loading, transforms, device detection
│   ├── requirements.txt       # Python dependencies
│   └── saved_model.pth        # Trained model weights (generated after training)
│
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── components/ui/     # shadcn/ui components
│   │   ├── App.jsx            # Main application component
│   │   ├── api.js             # API integration layer
│   │   └── index.css          # Tailwind CSS styles
│   ├── package.json
│   └── vite.config.js
│
├── data/                       # Dataset (not included in repo)
│   └── chestxrays/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── x-rays_sample.png          # Sample X-ray for testing
└── README.md
```

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **npm** or **yarn**

### Backend Setup

1. **Navigate to backend folder**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (required before running the API):
   ```bash
   python train.py
   ```

   This will:
   - Load the chest X-ray dataset
   - Fine-tune ResNet-18 for 3 epochs
   - Save the trained model to `saved_model.pth`
   - Print test accuracy and F1-score (rounded to 3 decimals)

4. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend folder**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## 🎮 Usage

1. **Train the model** (first time only):
   ```bash
   cd backend
   python train.py
   ```

2. **Start the backend server**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

3. **Start the frontend** (in a new terminal):
   ```bash
   cd frontend
   npm run dev
   ```

4. **Open your browser** to `http://localhost:5173`

5. **Upload an X-ray image** and click "Predict" to get the classification result

## 🔌 API Documentation

### Endpoints

#### `POST /predict`

Classify a chest X-ray image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (key: `file`)

**Response**:
```json
{
  "prediction": "NORMAL",
  "confidence": 0.982
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@x-rays_sample.png"
```

#### `GET /health`

Check server and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_path": "c:\\path\\to\\saved_model.pth",
  "model_exists": true,
  "device": "cuda"
}
```

## 🧠 Model Architecture

- **Base Model**: ResNet-18 (pre-trained on ImageNet)
- **Fine-tuning Strategy**: 
  - All layers frozen except final FC layer
  - Final layer modified: 512 → 2 classes (NORMAL, PNEUMONIA)
- **Training**:
  - 3 epochs
  - Adam optimizer (lr=0.001)
  - CrossEntropyLoss
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 🎨 Frontend Features

- **Premium UI**: Built with shadcn/ui components and Tailwind CSS
- **Image Upload**: Drag & drop or click to select
- **Real-time Preview**: See uploaded image before prediction
- **Confidence Display**: Visual progress bar and percentage
- **Error Handling**: User-friendly error messages
- **Responsive Design**: Works on desktop and mobile
- **Loading States**: Animated spinner during prediction

## 📊 Model Performance

After training for 3 epochs (as configured), the model outputs:
- **Test Accuracy**: Displayed to 3 decimal places
- **Test F1-Score**: Displayed to 3 decimal places

*Note: For better performance, see recommendations below.*

## 💡 Recommendations for Improvement

### Accuracy Improvements
1. **Increase Training Epochs**: Train for 10-20 epochs instead of 3
2. **Unfreeze More Layers**: Unfreeze the last 2-3 residual blocks of ResNet-18
3. **Data Augmentation**: Add random rotations, flips, and brightness adjustments
4. **Learning Rate Schedule**: Use ReduceLROnPlateau or CosineAnnealingLR
5. **Larger Dataset**: Collect more training samples if possible

### Transfer Learning Optimizations
- **Different Optimizers**: Try SGD with momentum (0.9) or AdamW
- **Discriminative Learning Rates**: Lower LR for frozen layers, higher for final layer
- **Ensemble Methods**: Train multiple models and average predictions
- **Class Weighting**: Handle class imbalance with weighted loss

### Frontend Enhancements
- **Batch Upload**: Allow multiple images at once
- **History**: Save and display previous predictions
- **Model Info**: Display model version and training metrics
- **Dark Mode Toggle**: Manual theme switching
- **Export Results**: Download predictions as JSON or PDF

### Deployment Options

#### Backend Deployment (Render / Railway / Heroku)
1. Add `Procfile`:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
2. Add `runtime.txt`:
   ```
   python-3.10
   ```
3. Deploy to Render:
   - Connect GitHub repository
   - Set build command: `pip install -r backend/requirements.txt`
   - Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

#### Frontend Deployment (Vercel / Netlify)
1. Build for production:
   ```bash
   npm run build
   ```
2. Deploy to Vercel:
   ```bash
   npm i -g vercel
   vercel
   ```
3. Update API URL in `src/api.js` to your deployed backend URL

## 🛠️ Technologies Used

### Backend
- **PyTorch** - Deep learning framework
- **torchvision** - Pre-trained models and transforms
- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **Pillow** - Image processing
- **torchmetrics** - Metrics calculation

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Utility-first CSS
- **shadcn/ui** - Component library
- **Axios** - HTTP client

## 📝 Best Practices Implemented

✅ **Clean Separation of Concerns**: Backend modules clearly separated (model, training, prediction, API)  
✅ **No Hard-coded Paths**: Uses `pathlib` for cross-platform compatibility  
✅ **Proper Inference Mode**: Uses `torch.no_grad()` and `model.eval()`  
✅ **Device Auto-detection**: Automatically uses GPU if available  
✅ **CORS Configuration**: Enables frontend-backend communication  
✅ **Error Handling**: Comprehensive error messages  
✅ **Type Hints & Docstrings**: Well-documented code  
✅ **Responsive UI**: Mobile-first design approach  

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure you've run `python train.py` before starting the API server
- Check that `saved_model.pth` exists in the `backend/` folder

### CORS Errors
- Ensure the backend server is running on `http://localhost:8000`
- Check that CORS middleware is properly configured in `main.py`

### Frontend Not Connecting
- Verify the API_BASE_URL in `src/api.js` matches your backend URL
- Check that both servers are running

### Training Issues
- Ensure dataset is in the correct structure: `data/chestxrays/train/` and `data/chestxrays/test/`
- Each should have `NORMAL/` and `PNEUMONIA/` subfolders
- Verify you have enough disk space for model weights (~45 MB)

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using PyTorch, FastAPI, and React**
