# Backend Technical Documentation - How It Works

## 📚 Table of Contents

1. [Overview](#overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [How Training Works - Step by Step](#how-training-works---step-by-step)
4. [How Prediction Works - Step by Step](#how-prediction-works---step-by-step)
5. [Transfer Learning Explained](#transfer-learning-explained)
6. [File-by-File Breakdown](#file-by-file-breakdown)
7. [The Complete Data Flow](#the-complete-data-flow)

---

## Overview

This backend is a **PyTorch-based deep learning system** that uses **transfer learning** with ResNet-18 to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. It consists of:

- 🧠 **Model**: Pre-trained ResNet-18 modified for binary classification
- 📊 **Training Pipeline**: Fine-tunes only the last layer for 3 epochs
- 🔮 **Prediction Service**: Loads trained model and makes predictions
- 🌐 **API Server**: FastAPI REST endpoint for frontend integration

---

## Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────┘

[1] Data Loading (utils.py)
    ├── Load Images from data/chestxrays/train & test
    ├── Apply ResNet transforms (resize, normalize)
    └── Create DataLoaders (batches of images)
           ↓
[2] Model Creation (model.py)
    ├── Download ResNet-18 pre-trained on ImageNet
    ├── Freeze all 18 layers (4 residual blocks + initial conv)
    ├── Replace final FC layer: 512 inputs → 2 outputs
    └── Only this final layer will be trained
           ↓
[3] Training Loop (train.py)
    ├── For 3 epochs:
    │   ├── Forward pass: images → model → predictions
    │   ├── Calculate loss: CrossEntropyLoss
    │   ├── Backward pass: compute gradients
    │   └── Update weights: only final layer parameters
    ├── Evaluate on test set
    └── Save model to saved_model.pth
           ↓
[4] Prediction (predict.py)
    ├── Load saved_model.pth
    ├── Preprocess new image (same transforms)
    ├── Forward pass: image → model → prediction
    └── Return class name + confidence
           ↓
[5] API Server (main.py)
    ├── FastAPI receives HTTP request with image
    ├── Call predict.py functions
    └── Return JSON: {"prediction": "NORMAL", "confidence": 0.982}
```

---

## How Training Works - Step by Step

### 🎯 What Happens When You Run `python train.py`

Let me explain the **complete training process** from start to finish:

### **Step 1: Initialization & Setup**

```python
# Set random seeds for reproducibility
set_seed(101010)
```

**What happens:**
- Sets random seeds in PyTorch, NumPy, and Python's random module
- This ensures that every time you train, you get the same results (same initialization, same data shuffling)
- **Why:** For scientific reproducibility and debugging

---

### **Step 2: Device Detection**

```python
device = get_device()
```

**What happens:**
- Checks if NVIDIA GPU (CUDA) is available
- If yes, uses GPU for faster training (10-50x faster)
- If no, falls back to CPU
- Prints: `Using device: cuda` or `Using device: cpu`

**In your case:** You likely see `cuda` since training was fast

---

### **Step 3: Load Dataset**

```python
train_dataset, test_dataset = load_datasets()
```

**What happens inside `load_datasets()`:**

1. **Reads directory structure:**
   ```
   data/chestxrays/train/
       ├── NORMAL/       (folder 0)
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       └── PNEUMONIA/    (folder 1)
           ├── image1.jpg
           └── ...
   ```

2. **ImageFolder automatically:**
   - Assigns label `0` to all images in `NORMAL/` folder
   - Assigns label `1` to all images in `PNEUMONIA/` folder
   - Creates list: `[(image_path, label), (image_path, label), ...]`

3. **Applies transforms to each image:**
   ```python
   transforms.ToTensor()  # Convert PIL Image to tensor [0-1]
   transforms.Normalize(  # Normalize using ImageNet statistics
       mean=[0.485, 0.456, 0.406],  # R, G, B channels
       std=[0.229, 0.224, 0.225]
   )
   ```

**Why these specific numbers?**
- ResNet-18 was originally trained on ImageNet dataset
- ImageNet has mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
- We use the **same normalization** so our X-rays are in the same "scale" as ImageNet images
- This helps the pre-trained weights work better on our data

**Output:**
```
Train dataset: 5216 images
Test dataset: 624 images
Classes: ['NORMAL', 'PNEUMONIA']
```

---

### **Step 4: Create DataLoaders**

```python
train_loader, test_loader = create_dataloaders(
    train_dataset, test_dataset, batch_size=32
)
```

**What happens:**

1. **DataLoader shuffles and batches the data:**
   - Instead of processing 5216 images one by one
   - Groups them into batches of 32 images
   - Total batches: 5216 ÷ 32 ≈ 163 batches per epoch

2. **Why batches?**
   - **Memory efficiency**: Can't fit all 5216 images in GPU memory at once
   - **Faster training**: GPU processes 32 images in parallel
   - **Better gradients**: Averaging loss over 32 images gives smoother gradient updates

3. **Shuffling:**
   - `shuffle=True` for training → randomizes order every epoch
   - `shuffle=False` for testing → consistent evaluation

---

### **Step 5: Create Model**

```python
model = create_model(num_classes=2, freeze_layers=True)
model = model.to(device)
```

**What happens inside `create_model()`:**

#### **5.1: Download Pre-trained ResNet-18**

```python
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

**What downloads:**
- File: `resnet18-f37072fd.pth` (44.7 MB)
- Location: `C:\Users\Ala\.cache\torch\hub\checkpoints\`
- Contains: 11,689,512 pre-trained parameters

**ResNet-18 Architecture:**
```
Input Image (3 x 224 x 224)
    ↓
[Initial Convolution]
    ↓
[Residual Block 1] - layer1 (2 residual blocks)
    ↓
[Residual Block 2] - layer2 (2 residual blocks)
    ↓
[Residual Block 3] - layer3 (2 residual blocks)
    ↓
[Residual Block 4] - layer4 (2 residual blocks)
    ↓
[Global Average Pooling]
    ↓
[Fully Connected Layer] - 512 → 1000 classes (original)
```

#### **5.2: Freeze All Layers**

```python
for param in model.parameters():
    param.requires_grad = False
```

**What this does:**
- Sets `requires_grad=False` for ALL 11,689,512 parameters
- **Meaning:** These weights will NOT be updated during training
- **Why:** We trust the pre-trained features from ImageNet
- **Benefit:** Much faster training, needs less data

**Frozen layers:**
- ❄️ Initial convolution layer
- ❄️ layer1 (all parameters frozen)
- ❄️ layer2 (all parameters frozen)
- ❄️ layer3 (all parameters frozen)
- ❄️ layer4 (all parameters frozen)

#### **5.3: Replace Final Layer**

```python
num_features = model.fc.in_features  # 512
model.fc = nn.Linear(512, 2)  # NEW LAYER
```

**What happens:**
1. **Original final layer:**
   - Input: 512 features from layer4
   - Output: 1000 classes (ImageNet categories)
   - Parameters: 512 × 1000 + 1000 = 513,000

2. **New final layer:**
   - Input: 512 features from layer4 (unchanged)
   - Output: 2 classes (NORMAL, PNEUMONIA)
   - Parameters: 512 × 2 + 2 = **1,026 trainable parameters**
   - **Initialized randomly** (not pre-trained)

3. **This new layer has `requires_grad=True` by default**
   - ✅ Only these 1,026 parameters will be trained
   - This is why training is so fast!

**Output:**
```
Loading ResNet-18 with ImageNet pre-trained weights...
Freezing all layers except final FC layer...
Modified final FC layer: 512 -> 2 classes
Trainable parameters in final layer: 1026
```

---

### **Step 6: Define Loss Function & Optimizer**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

#### **6.1: CrossEntropyLoss**

**What it does:**
- Combines Softmax + Negative Log Likelihood
- For 2 classes, it calculates:

```python
# Model outputs: [logit_normal, logit_pneumonia]
# Example: [2.3, -0.5]

# Step 1: Softmax converts to probabilities
probabilities = softmax([2.3, -0.5])
# = [0.948, 0.052]  (sums to 1.0)

# Step 2: If true label is NORMAL (0):
loss = -log(0.948) = 0.053  (low loss, good!)

# Step 2: If true label is PNEUMONIA (1):
loss = -log(0.052) = 2.956  (high loss, bad!)
```

**Why use it:**
- Standard for classification tasks
- Automatically handles probability conversion
- Penalizes confident wrong predictions heavily

#### **6.2: Adam Optimizer**

```python
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

**What it optimizes:**
- **Only** the 1,026 parameters in `model.fc`
- Uses Adam algorithm (Adaptive Moment Estimation)
- Learning rate = 0.001

**How Adam works:**
1. Maintains a moving average of gradients (momentum)
2. Maintains a moving average of squared gradients (RMSprop)
3. Adapts learning rate for each parameter individually
4. Generally converges faster than basic SGD

---

### **Step 7: Training Loop (3 Epochs)**

```python
for epoch in range(3):
    avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
```

**What happens in EACH epoch:**

#### **7.1: Epoch Overview**

An epoch = one complete pass through the entire training dataset

- Total images: 5,216
- Batch size: 32
- Batches per epoch: 163

#### **7.2: Training One Epoch - Detailed**

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set to training mode (enables dropout, batchnorm tracking)
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # inputs.shape: [32, 3, 224, 224] = 32 images
        # labels.shape: [32] = 32 labels (0 or 1)
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # STEP 1: Zero gradients
        optimizer.zero_grad()
        
        # STEP 2: Forward pass
        outputs = model(inputs)
        # outputs.shape: [32, 2] = 32 predictions for 2 classes
        
        # STEP 3: Calculate loss
        loss = criterion(outputs, labels)
        # Single number representing average loss for this batch
        
        # STEP 4: Backward pass (calculate gradients)
        loss.backward()
        # Computes gradient for all 1,026 parameters in final layer
        
        # STEP 5: Update weights
        optimizer.step()
        # Updates the 1,026 parameters using Adam algorithm
```

**Let me explain each step:**

##### **7.2.1: Forward Pass - What Really Happens**

When you call `outputs = model(inputs)`:

1. **Input batch:** `[32, 3, 224, 224]` (32 images, 3 RGB channels, 224x224 pixels)

2. **Through frozen layers:**
   ```
   Input → conv1 → bn1 → relu → maxpool
   → layer1 (frozen) → learned features
   → layer2 (frozen) → learned features
   → layer3 (frozen) → learned features
   → layer4 (frozen) → learned features
   → avgpool → [32, 512] feature vectors
   ```

3. **Through trainable final layer:**
   ```
   [32, 512] → Linear(512, 2) → [32, 2]
   ```

4. **Output:** `[32, 2]` = 32 predictions, each with 2 logits (raw scores)

**Example output for one image:**
```python
outputs[0] = [2.3, -0.5]
# 2.3 = score for NORMAL
# -0.5 = score for PNEUMONIA
# After softmax: [0.948, 0.052] → predicts NORMAL with 94.8% confidence
```

##### **7.2.2: Loss Calculation**

```python
loss = criterion(outputs, labels)
```

- Takes all 32 predictions and 32 true labels
- Calculates CrossEntropyLoss for each
- Returns average loss: e.g., `0.6756`

##### **7.2.3: Backward Pass**

```python
loss.backward()
```

**What happens:**
- PyTorch automatically calculates gradients using **backpropagation**
- For each of the 1,026 parameters in final layer, computes:
  ```
  gradient = ∂loss/∂parameter
  ```
- Tells us: "If we increase this parameter by a tiny amount, how much does loss change?"
- **Frozen layers:** No gradients computed (saves time and memory)

##### **7.2.4: Weight Update**

```python
optimizer.step()
```

**What happens:**
For each of the 1,026 parameters:
```python
parameter_new = parameter_old - learning_rate × gradient
```

With Adam, it's more sophisticated:
```python
momentum = 0.9 × momentum + 0.1 × gradient
variance = 0.999 × variance + 0.001 × gradient²
parameter_new = parameter_old - lr × momentum / sqrt(variance)
```

**Result:** Parameters move in direction that reduces loss

---

### **Step 8: Epoch Progress**

During training you see:

```
Epoch [1/3]
  Batch [5/10], Loss: 0.6756
  Batch [10/10], Loss: 0.4565
Epoch [1/3] - Average Loss: 0.6779
```

**What this means:**

- **Batch 5 loss: 0.6756** → Average loss for images 129-160
- **Batch 10 loss: 0.4565** → Average loss for images 289-320
- **Average loss: 0.6779** → Average over ALL 163 batches

**Why does loss decrease?**
- Early batches: Model is still learning → high loss
- Later batches: Model has learned some patterns → lower loss
- Epoch 1 → Epoch 2 → Epoch 3: Loss keeps decreasing

**Your actual training:**
```
Epoch 1: 0.6779 → Starting to learn
Epoch 2: 0.4841 → Learning patterns
Epoch 3: 0.3851 → Converging
```

---

### **Step 9: Evaluation on Test Set**

```python
test_acc, test_f1 = evaluate_model(model, test_loader, device)
```

**What happens:**

```python
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # torch.max returns (values, indices)
        # indices are the predicted classes (0 or 1)
```

**Metrics calculated:**

1. **Accuracy:**
   ```
   accuracy = (correct predictions) / (total predictions)
   Your result: 0.770 = 77.0%
   Meaning: 480 out of 624 test images classified correctly
   ```

2. **F1-Score:**
   ```
   F1 = 2 × (precision × recall) / (precision + recall)
   Your result: 0.769 = 76.9%
   Meaning: Balanced performance on both classes
   ```

**Output:**
```
Test Accuracy: 0.770
Test F1-Score: 0.769
```

---

### **Step 10: Save Model**

```python
save_model(model, "saved_model.pth")
```

**What gets saved:**
```python
torch.save(model.state_dict(), "saved_model.pth")
```

**Contents of `saved_model.pth`:**
- **NOT** the entire model architecture
- **ONLY** the parameter values (weights and biases)
- All 11,690,538 parameters (frozen + trainable)
- File size: ~45 MB

**Why save state_dict instead of entire model?**
- More flexible: Can load into different model architectures
- Smaller file size
- Better compatibility across PyTorch versions

---

## How Prediction Works - Step by Step

### 🔮 What Happens When You Call `/predict` API

Let's trace a single prediction from upload to result:

### **Step 1: Image Upload (Frontend)**

```javascript
// User selects file
const file = selectedImage  // e.g., chest_xray.jpg

// Send to backend
const formData = new FormData()
formData.append('file', file)
axios.post('/predict', formData)
```

---

### **Step 2: API Receives Request (main.py)**

```python
@app.post("/predict")
async def predict(file: UploadFile):
    # Read uploaded bytes
    contents = await file.read()
    
    # Open as PIL Image
    image = Image.open(io.BytesIO(contents))
    
    # Call prediction function
    result = predict_from_pil(image, "saved_model.pth", device)
```

---

### **Step 3: Load Model (predict.py)**

```python
model = load_model("saved_model.pth", device)
```

**What happens:**
1. Creates fresh ResNet-18 architecture
2. Loads saved parameters from `saved_model.pth`
3. Sets to **evaluation mode**: `model.eval()`
4. Moves to device (GPU or CPU)

**Evaluation mode changes:**
- Disables dropout (if any)
- Fixes batch normalization statistics
- Ensures consistent predictions

---

### **Step 4: Preprocess Image**

```python
# Convert to RGB if needed
if image.mode != 'RGB':
    image = image.convert('RGB')

# Apply same transforms as training
transform = get_transform()  # ToTensor + Normalize
image_tensor = transform(image)
image_batch = image_tensor.unsqueeze(0)  # Add batch dimension
```

**Image transformation:**
```
Original Image (PIL)
    ↓
Convert to tensor [0-1] → shape: [3, H, W]
    ↓
Normalize with ImageNet stats
    ↓
Add batch dimension → shape: [1, 3, H, W]
    ↓
Move to device (GPU/CPU)
```

**Why add batch dimension?**
- Model expects input shape: `[batch_size, 3, height, width]`
- We have 1 image → batch_size=1
- `[3, 224, 224]` → `[1, 3, 224, 224]`

---

### **Step 5: Forward Pass (Inference)**

```python
with torch.no_grad():
    outputs = model(image_tensor)
```

**What `torch.no_grad()` does:**
- Disables gradient calculation
- Saves memory (50% reduction)
- Faster inference
- We don't need gradients for prediction

**Forward pass:**
```
Input: [1, 3, 224, 224]
    ↓
ResNet frozen layers → [1, 512] features
    ↓
Final FC layer (trained) → [1, 2] logits
    ↓
Output: [2.3, -0.5]  (example)
```

---

### **Step 6: Convert to Probabilities**

```python
probabilities = torch.nn.functional.softmax(outputs, dim=1)
```

**Softmax formula:**
```python
softmax([2.3, -0.5]) = [e^2.3 / (e^2.3 + e^-0.5), 
                         e^-0.5 / (e^2.3 + e^-0.5)]
                     = [0.948, 0.052]
```

**Result:**
- Probability of NORMAL: 94.8%
- Probability of PNEUMONIA: 5.2%
- **Sum = 100%**

---

### **Step 7: Get Prediction**

```python
confidence, predicted_idx = torch.max(probabilities, 1)
# confidence: 0.948
# predicted_idx: 0 (NORMAL)

class_names = ["NORMAL", "PNEUMONIA"]
predicted_class = class_names[predicted_idx]
# "NORMAL"
```

---

### **Step 8: Return Result**

```python
return {
    "prediction": "NORMAL",
    "confidence": 0.948
}
```

**API returns JSON:**
```json
{
  "prediction": "NORMAL",
  "confidence": 0.948
}
```

---

## Transfer Learning Explained

### 🤔 Why Transfer Learning?

**Problem:** Training a deep neural network from scratch requires:
- Millions of labeled images
- Weeks of GPU training time
- Huge computational cost

**Solution:** Use a model already trained on millions of images!

### How ResNet-18 Helps

**ImageNet pre-training:**
- Trained on 1.2 million images
- 1000 categories (dogs, cats, cars, planes, etc.)
- Learned to detect:
  - **Low-level features:** Edges, corners, textures
  - **Mid-level features:** Shapes, patterns
  - **High-level features:** Object parts

### Feature Reuse

**Key insight:** Features learned on ImageNet transfer to X-rays!

```
Layer 1 (Early):
  - Detects: Horizontal edges, vertical edges, diagonal lines
  - Useful for: Finding rib outlines, lung boundaries

Layer 2:
  - Detects: Textures, simple patterns
  - Useful for: Normal lung texture vs. infiltrates

Layer 3:
  - Detects: Complex shapes
  - Useful for: Heart shape, rib cage structure

Layer 4:
  - Detects: High-level patterns
  - Useful for: Overall lung appearance

Final Layer (We Train This):
  - Combines all features → NORMAL or PNEUMONIA decision
```

### Why Freeze Layers?

**Benefits:**
1. **Fast training:** Only 1,026 parameters to update vs. 11 million
2. **Less data needed:** Pre-trained features are already good
3. **Prevents overfitting:** Frozen weights can't overfit to small dataset
4. **Better generalization:** Leverages ImageNet knowledge

**Your Results:**
- 3 epochs, ~10 minutes → 77% accuracy
- From scratch: Would need 50+ epochs, hours of training, might not reach 77%

---

## File-by-File Breakdown

### 1. `utils.py` - Data Utilities

**Purpose:** Handle all data loading and preprocessing

**Key Functions:**

```python
get_device()
# Returns: torch.device("cuda") or torch.device("cpu")
# Automatically detects GPU availability

get_transform()
# Returns: Compose([ToTensor(), Normalize(...)])
# Standard ResNet preprocessing

load_datasets(data_dir)
# Returns: (train_dataset, test_dataset)
# ImageFolder automatically assigns labels from folder names

create_dataloaders(train_dataset, test_dataset, batch_size=32)
# Returns: (train_loader, test_loader)
# Batches data for efficient training
```

---

### 2. `model.py` - Model Architecture

**Purpose:** Define, create, save, and load the ResNet-18 model

**Key Functions:**

```python
create_model(num_classes=2, freeze_layers=True)
# 1. Downloads ResNet-18 with ImageNet weights
# 2. Freezes all layers except final FC
# 3. Replaces FC: 512 → num_classes
# Returns: Modified ResNet-18

save_model(model, path)
# Saves model.state_dict() to file

load_model(path, device)
# 1. Creates model architecture
# 2. Loads saved weights
# 3. Sets to eval() mode
# Returns: Trained model ready for inference

get_class_names()
# Returns: ["NORMAL", "PNEUMONIA"]
```

---

### 3. `train.py` - Training Pipeline

**Purpose:** Complete training workflow

**Main Function:**

```python
def main():
    # 1. Setup
    set_seed(101010)
    device = get_device()
    
    # 2. Load data
    train_dataset, test_dataset = load_datasets()
    train_loader, test_loader = create_dataloaders(...)
    
    # 3. Create model
    model = create_model(num_classes=2, freeze_layers=True)
    model = model.to(device)
    
    # 4. Define training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 5. Training loop
    for epoch in range(3):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 6. Evaluation
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    
    # 7. Save model
    save_model(model, "saved_model.pth")
```

**Helper Functions:**

```python
train_epoch(model, train_loader, criterion, optimizer, device)
# One complete pass through training data
# Returns: average loss

evaluate_model(model, test_loader, device)
# Calculates accuracy and F1-score on test set
# Returns: (accuracy, f1_score)
```

---

### 4. `predict.py` - Inference

**Purpose:** Make predictions on single images

**Key Functions:**

```python
predict_from_pil(pil_image, model_path, device)
# 1. Load trained model
# 2. Preprocess image
# 3. Run forward pass
# 4. Convert to probabilities
# 5. Return prediction and confidence

run_inference(model, image_tensor, device)
# Core inference logic
# Returns: (class_name, confidence)
```

---

### 5. `main.py` - FastAPI Server

**Purpose:** REST API for frontend integration

**Endpoints:**

```python
@app.get("/")
# Returns API info

@app.get("/health")
# Returns server health status

@app.post("/predict")
# Main endpoint:
# 1. Receives uploaded image
# 2. Validates file type
# 3. Calls predict_from_pil()
# 4. Returns JSON result
```

---

## The Complete Data Flow

### Training Flow

```
[User runs: python train.py]
           ↓
[1] Load Data
    data/chestxrays/train/ → 5216 images
    data/chestxrays/test/  → 624 images
           ↓
[2] Create DataLoaders
    Batch size: 32
    Train batches: 163
           ↓
[3] Load ResNet-18
    Download pre-trained weights (44.7 MB)
    Freeze layers 1-4
    Replace final layer: 512→2
           ↓
[4] Training Loop (3 epochs)
    For each batch:
      • Forward pass
      • Calculate loss
      • Backprop gradients (only final layer)
      • Update weights (1,026 parameters)
           ↓
[5] Evaluate
    Test Accuracy: 77.0%
    Test F1-Score: 76.9%
           ↓
[6] Save Model
    Saved to: saved_model.pth (45 MB)
```

---

### Prediction Flow

```
[User uploads image via frontend]
           ↓
[Frontend sends POST /predict]
           ↓
[FastAPI receives request]
           ↓
[Load saved_model.pth]
    • Create ResNet-18 architecture
    • Load trained parameters
    • Set to eval() mode
           ↓
[Preprocess image]
    • Convert to RGB
    • ToTensor() → [0, 1]
    • Normalize with ImageNet stats
    • Add batch dimension
           ↓
[Forward Pass]
    Image → Frozen Layers → Features (512)
    Features → Trained FC Layer → Logits [2]
           ↓
[Softmax]
    Logits → Probabilities [0.948, 0.052]
           ↓
[Return Result]
    {
      "prediction": "NORMAL",
      "confidence": 0.948
    }
           ↓
[Frontend displays result]
```

---

## 🎓 Key Takeaways

1. **Transfer Learning** = Use pre-trained ImageNet features + train only final layer
2. **Fast Training** = 3 epochs × 163 batches = 489 weight updates total
3. **Small Dataset OK** = 5,216 images enough because we reuse ImageNet features
4. **77% Accuracy in 10 minutes** = Much better than training from scratch
5. **Frozen Layers** = Features transfer from natural images to X-rays
6. **Only 1,026 parameters trained** = Why training is so fast

---

## 📊 Training Statistics

**Your Training Results:**
```
Dataset:
  - Training images: 5,216
  - Test images: 624
  - Classes: 2 (NORMAL, PNEUMONIA)

Model:
  - Total parameters: 11,690,538
  - Frozen parameters: 11,689,512 (99.99%)
  - Trainable parameters: 1,026 (0.01%)

Training:
  - Epochs: 3
  - Batch size: 32
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss function: CrossEntropyLoss

Results:
  - Final training loss: 0.3851
  - Test accuracy: 77.0%
  - Test F1-score: 76.9%
  - Model size: ~45 MB
```

---

**Questions? Check the main README.md or walkthrough.md for more details!**
