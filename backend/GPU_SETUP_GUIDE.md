# GPU Setup Guide for NVIDIA MX350 on Windows

Complete guide to enable CUDA GPU training on your NVIDIA GeForce MX350.

## 🎯 Your Hardware

- **CPU**: Intel Core with Intel Iris Xe Graphics
- **GPU**: NVIDIA GeForce MX350 (2GB VRAM)
- **NVIDIA Driver**: 581.29
- **CUDA Version**: 13.0
- **Current Issue**: PyTorch uses CPU instead of GPU

---

## 📋 Step-by-Step GPU Setup

### Step 1: Check Current PyTorch CUDA Status

First, let's see if your current PyTorch installation supports CUDA:

```bash
cd backend
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output if CUDA is NOT working:**
```
PyTorch version: 2.x.x+cpu
CUDA available: False
CUDA version: N/A
```

---

### Step 2: Uninstall Current PyTorch (CPU-only version)

```bash
pip uninstall torch torchvision torchaudio
```

**Confirm 'yes' when prompted.**

---

### Step 3: Install CUDA-Enabled PyTorch

For **CUDA 12.4** (closest stable version to your CUDA 13.0):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Note:** CUDA 13.0 is not yet fully supported by PyTorch. CUDA 12.4 is backward compatible and will work perfectly with your driver 581.29.

**This will download:**
- PyTorch with CUDA 12.4 support (~2.5 GB)
- torchvision with CUDA support
- torchaudio with CUDA support

**Installation time:** 5-10 minutes depending on internet speed

---

### Step 4: Verify GPU Detection

After installation completes, run:

```bash
python -c "import torch; print('='*60); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print('='*60)"
```

**Expected output (SUCCESS):**
```
============================================================
PyTorch version: 2.5.1+cu124
CUDA available: True
CUDA version: 12.4
GPU device: NVIDIA GeForce MX350
Number of GPUs: 1
============================================================
```

---

### Step 5: Test GPU with Simple Tensor

```bash
python -c "import torch; device = torch.device('cuda'); x = torch.randn(1000, 1000, device=device); print(f'Tensor created on: {x.device}'); print('GPU is working! 🚀')"
```

**Expected output:**
```
Tensor created on: cuda:0
GPU is working! 🚀
```

---

## 🔧 Code Updates for GPU Training

### Update 1: Enhanced `get_device()` Function

I'll update `utils.py` to provide more detailed GPU information.

### Update 2: Optimal Batch Size for MX350 (2GB VRAM)

**MX350 Memory Guidelines:**

| Model | Recommended Batch Size | Memory Usage |
|-------|----------------------|--------------|
| ResNet-18 (frozen layers) | **16-24** | ~1.2-1.5 GB |
| ResNet-18 (layer4 unfrozen) | **8-12** | ~1.5-1.8 GB |

**For your training scripts:**
- `train.py` (only FC layer): **batch_size=24**
- `train_enhanced.py` (layer4 unfrozen): **batch_size=12**
- `train_with_augmentation.py` (layer4 unfrozen): **batch_size=12**

**Why smaller batches:**
- MX350 has only 2GB VRAM
- Need memory for model weights + gradients + activations
- Batch size 12 is safe and won't cause OOM errors

---

## ⚡ Expected Speed Improvements

### CPU (Current):
- Time per epoch: ~200-300 seconds
- Total training (20 epochs): ~80-100 minutes

### GPU (MX350):
- Time per epoch: ~15-25 seconds
- Total training (20 epochs): ~5-8 minutes

**Speed up: 10-15x faster! 🚀**

---

## 🎯 Quick Setup Commands (Copy-Paste All)

Open PowerShell in your backend directory and run these commands one by one:

```powershell
# Step 1: Check current PyTorch
python -c "import torch; print(f'Current: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Step 2: Uninstall CPU-only PyTorch
pip uninstall -y torch torchvision torchaudio

# Step 3: Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 4: Verify GPU
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Step 5: Test GPU with tensor
python -c "import torch; x = torch.randn(1000, 1000, device='cuda'); print('GPU Working! Device:', x.device)"
```

---

## 🔍 Troubleshooting

### Issue 1: "CUDA out of memory"

**Solution:** Reduce batch size in training scripts:

```python
# In train_with_augmentation.py, line ~165
batch_size = 8  # Reduce from 32 to 8
```

### Issue 2: "CUDA driver version is insufficient"

**Solution:** Your driver 581.29 is fine. This error shouldn't occur.

### Issue 3: Still using CPU after installation

**Solution:** Check if you're in the correct virtual environment:

```bash
# Activate your venv
venv\Scripts\activate

# Reinstall PyTorch in venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 📊 After GPU Setup - Training Commands

### Run with GPU (recommended batch sizes):

```bash
# Basic training (3 epochs, fast test)
python train.py

# Enhanced training (15 epochs, good accuracy)
python train_enhanced.py

# Best accuracy (20 epochs, augmentation)
python train_with_augmentation.py
```

**The scripts will automatically use GPU now!**

---

## 🎯 Monitoring GPU Usage

### During Training:

Open a **new PowerShell window** and run:

```bash
nvidia-smi
```

**You should see:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 581.29       Driver Version: 581.29       CUDA Version: 13.0    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P0    N/A /  N/A |   1200MiB /  2048MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

**What to look for:**
- **Memory-Usage**: Should be ~1.2-1.5 GB during training
- **GPU-Util**: Should be 80-100% during forward/backward passes
- **Temp**: Should stay under 85°C (MX350 is fine up to 90°C)

---

## 🚀 Next Steps After Setup

1. **Run GPU verification** (Step 4 above)
2. **Update batch sizes** in training scripts (I'll do this for you)
3. **Run training** with `python train_with_augmentation.py`
4. **Watch 10-15x speedup!**

---

## ⚠️ Important Notes

1. **Driver Compatibility**: Your driver 581.29 supports CUDA 12.x and 13.x - you're good!
2. **VRAM Limitation**: With 2GB, you can train ResNet-18 but not larger models (ResNet-50, etc.)
3. **Batch Size**: Start with 12, reduce to 8 if you get OOM errors
4. **Mixed Precision**: Not recommended for MX350 (complicates things without much benefit)
5. **Multi-GPU**: You have 1 GPU, scripts are already configured for single GPU

---

## 📈 Expected Results After GPU Setup

### Before (CPU):
```
Epoch [1/20]
  Batch [5/10], Loss: 0.6756
  ...
Epoch time: 250 seconds ⏱️
Total: ~83 minutes for 20 epochs
```

### After (GPU):
```
Epoch [1/20]
  Batch [5/10], Loss: 0.6756
  ...
Epoch time: 18 seconds ⚡
Total: ~6 minutes for 20 epochs
```

**That's 13x faster!** 🚀

---

## 🎯 Summary

Run these commands in order:

```bash
# 1. Check current setup
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. Uninstall CPU version
pip uninstall -y torch torchvision torchaudio

# 3. Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

# 5. Train with GPU
python train_with_augmentation.py
```

**You're ready to train at lightning speed!** ⚡🚀
