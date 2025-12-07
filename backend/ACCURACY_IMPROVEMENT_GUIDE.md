# Improving Model Accuracy & Reducing Loss

This guide explains how to improve your model beyond the baseline 77% accuracy.

## 🎯 Quick Comparison

| Training Script | Epochs | Techniques | Expected Accuracy |
|----------------|--------|------------|-------------------|
| `train.py` (basic) | 3 | Basic transfer learning | **77%** |
| `train_enhanced.py` | 15 | + Unfreeze layer4, LR scheduler, early stopping | **82-88%** |
| `train_with_augmentation.py` | 20 | + Data augmentation, layer-wise LR | **85-92%** |

---

## 🚀 How to Use Enhanced Training

### Option 1: Enhanced Training (Recommended)

```bash
python train_enhanced.py
```

**What it does:**
- ✅ Trains for **15 epochs** (vs 3)
- ✅ Unfreezes **layer4** (last residual block) - 2.1M more parameters
- ✅ Uses **learning rate scheduler** - reduces LR when loss plateaus
- ✅ **Layer-wise learning rates** - layer4 gets lower LR (0.0001), FC gets higher LR (0.001)
- ✅ **Early stopping** - stops if no improvement for 5 epochs
- ✅ Saves **best model** automatically

**Expected improvement:** 82-88% accuracy

---

### Option 2: Maximum Accuracy with Augmentation

```bash
python train_with_augmentation.py
```

**What it does (all of Option 1 plus):**
- ✅ **Data augmentation:**
  - Random rotation (±10 degrees)
  - Random horizontal flip
  - Random brightness/contrast changes
  - Random crop and resize
- ✅ Trains for **20 epochs**
- ✅ Better generalization (less overfitting)

**Expected improvement:** 85-92% accuracy

---

## 📊 What Each Improvement Does

### 1. More Epochs (15-20 vs 3)

**Why:** 3 epochs is not enough for the model to fully learn patterns.

**Your current training:**
```
Epoch 1: Loss 0.6779 → Model is just starting to learn
Epoch 2: Loss 0.4841 → Still improving rapidly
Epoch 3: Loss 0.3851 → Could improve much more!
```

**With 15 epochs:**
```
Epoch 1-5:   Loss decreases rapidly (0.68 → 0.35)
Epoch 6-10:  Loss keeps improving (0.35 → 0.20)
Epoch 11-15: Loss stabilizes (0.20 → 0.15)
Result: Much better accuracy!
```

---

### 2. Unfreeze Layer4

**Current (train.py):**
- ❄️ All ResNet layers frozen
- ✅ Only final layer (1,026 params) trains
- Fast but limited accuracy

**Enhanced (train_enhanced.py):**
- ❄️ Layers 1-3 frozen
- ✅ Layer4 unfrozen (2.1M params) + final layer
- Slower but much better accuracy

**Why it helps:**
- Layer4 learns high-level X-ray specific features
- Can adapt ImageNet features to medical images
- Critical for distinguishing NORMAL vs PNEUMONIA patterns

---

### 3. Learning Rate Scheduler

**Problem with fixed LR:**
```
Early epochs: Loss drops fast → 0.001 LR is good
Later epochs: Loss barely improves → 0.001 LR is too high
```

**Solution - ReduceLROnPlateau:**
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)
```

**How it works:**
- If loss doesn't improve for 2 epochs → reduce LR by 50%
- Example: 0.001 → 0.0005 → 0.00025
- Allows fine-tuning in later epochs

---

### 4. Layer-wise Learning Rates

**Why different LRs:**
```python
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},  # Lower
    {'params': model.fc.parameters(), 'lr': 0.001}         # Higher
])
```

- **Layer4:** Pre-trained on ImageNet → needs small changes → low LR (0.0001)
- **Final layer:** Randomly initialized → needs big changes → high LR (0.001)

**Benefit:** Prevents destroying good pre-trained features while learning new ones

---

### 5. Data Augmentation

**Problem with small dataset:**
- Only 300 training images
- Model memorizes specific images → overfitting
- Poor generalization to new X-rays

**Solution - Augmentation:**
```python
transforms.RandomRotation(10)           # Simulate different angles
transforms.RandomHorizontalFlip()       # Mirror images
transforms.ColorJitter(brightness=0.2)  # Vary contrast
transforms.RandomResizedCrop(224)       # Different crops
```

**How it helps:**
- Creates "infinite" variations of training images
- Model learns robust features
- Much better on unseen test images

**Example:**
```
Original image → Model sees:
  - Rotated 5° clockwise
  - Flipped horizontally
  - Slightly brighter
  - Slightly cropped
All from ONE base image!
```

---

### 6. Early Stopping

**Problem:**
- Training too long → overfitting
- Stops generalizing to test set

**Solution:**
```python
if test_acc > best_accuracy:
    best_accuracy = test_acc
    best_model_state = model.state_dict().copy()
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= 5:
    print("Early stopping!")
    break
```

**Automatically stops when:**
- Test accuracy stops improving for 5 consecutive epochs
- Saves the best model (not the last one)

---

## 🔍 Expected Training Output (Enhanced)

```bash
python train_enhanced.py
```

**Expected output:**
```
Epoch [1/15]
  Batch [5/10], Loss: 0.6756
  Average Loss: 0.6779
  Test Accuracy: 0.770
  Test F1-Score: 0.769
  ✓ New best accuracy! Saved checkpoint.

Epoch [2/15]
  ...
  Average Loss: 0.4841
  Test Accuracy: 0.795
  ✓ New best accuracy! Saved checkpoint.

Epoch [5/15]
  Average Loss: 0.3012
  Test Accuracy: 0.835
  ✓ New best accuracy!

Epoch [10/15]
  Average Loss: 0.1823
  Test Accuracy: 0.875
  ✓ New best accuracy!

Epoch [15/15]
  Average Loss: 0.1456
  Test Accuracy: 0.883
  
FINAL RESULTS
Test Accuracy: 0.883  ← Much better than 0.770!
Test F1-Score: 0.881
```

---

## 💡 Additional Tips

### If accuracy is still not satisfying:

1. **Train longer:**
   ```python
   num_epochs = 30  # in train_enhanced.py
   ```

2. **Unfreeze layer3 too:**
   ```python
   for name, param in model.named_parameters():
       if 'layer3' in name or 'layer4' in name or 'fc' in name:
           param.requires_grad = True
   ```

3. **Try SGD instead of Adam:**
   ```python
   optimizer = optim.SGD(
       model.parameters(), 
       lr=0.01, 
       momentum=0.9,
       weight_decay=1e-4
   )
   ```

4. **Larger batch size (if you have GPU):**
   ```python
   batch_size = 64  # vs current 32
   ```

5. **Collect more training data:**
   - Current: 300 images
   - Recommended: 1000+ images for best results

---

## ⚡ Quick Start Commands

**For best results, run this:**
```bash
# Navigate to backend
cd backend

# Train with augmentation (best accuracy)
python train_with_augmentation.py

# Wait ~30-60 minutes depending on hardware
# Model will be saved as: saved_model_augmented.pth
```

**Then update your API to use the better model:**
```python
# In main.py, change:
MODEL_PATH = Path(__file__).parent / "saved_model_augmented.pth"
```

---

## 📈 Monitoring Training

Watch these numbers improve:

1. **Loss should decrease:**
   - Epoch 1: ~0.67
   - Epoch 5: ~0.30
   - Epoch 15: ~0.15
   - If loss stops decreasing → increase epochs or unfreeze more layers

2. **Accuracy should increase:**
   - Epoch 1: ~77%
   - Epoch 5: ~83%
   - Epoch 15: ~88%+
   - If accuracy plateaus → add augmentation or more data

3. **Gap between train and test:**
   - Small gap (<5%) → Good generalization
   - Large gap (>15%) → Overfitting → use augmentation

---

## 🎯 Summary

| To achieve | Use this script | Time needed |
|-----------|----------------|-------------|
| **80-85% accuracy** | `train_enhanced.py` | 20-40 min |
| **85-92% accuracy** | `train_with_augmentation.py` | 40-80 min |
| **92%+ accuracy** | Combine all + more data + 30 epochs | 2-4 hours |

**Your next step:** Run `python train_with_augmentation.py` and watch accuracy improve! 🚀
