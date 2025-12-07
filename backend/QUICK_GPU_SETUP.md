# 🚀 QUICK SETUP - Copy & Paste These Commands

## Step 1: Check Current PyTorch (in your venv)
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Step 2: Uninstall CPU-only PyTorch
```bash
pip uninstall -y torch torchvision torchaudio
```

## Step 3: Install CUDA-enabled PyTorch (for CUDA 12.4/13.0)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Step 4: Verify GPU Detection
```bash
python check_gpu.py
```

**Expected output:**
```
============================================================
✅ GPU SETUP IS READY!
============================================================

Your training will use:
🚀 NVIDIA GeForce MX350
⚡ CUDA 12.4
💾 2.00 GB VRAM

Recommended batch sizes for your GPU:
  - train.py (frozen layers): batch_size=24
  - train_enhanced.py (layer4 unfrozen): batch_size=12
  - train_with_augmentation.py (layer4 unfrozen): batch_size=12
```

## Step 5: Train with GPU (10-15x faster!)
```bash
python train_with_augmentation.py
```

---

## 📊 Expected Speed Improvement

| Configuration | Time per Epoch | Total (20 epochs) |
|--------------|----------------|-------------------|
| **CPU** (current) | ~250 seconds | ~83 minutes |
| **GPU** (MX350) | ~18 seconds | ~6 minutes |

**Speedup: 13-15x faster!** ⚡

---

## ⚠️ If You Get "CUDA out of memory"

Reduce batch size in the script:

```python
# Edit line 163 in train_with_augmentation.py
batch_size = 8  # Reduce from 12 to 8
```

---

## 🎯 Files I Updated for GPU:

1. ✅ `utils.py` - Enhanced `get_device()` with GPU info
2. ✅ `train_enhanced.py` - Batch size 12 (GPU optimized)
3. ✅ `train_with_augmentation.py` - Batch size 12 (GPU optimized)
4. ✅ Created `check_gpu.py` - GPU verification script
5. ✅ Created `GPU_SETUP_GUIDE.md` - Complete guide

---

## 🔍 Monitor GPU During Training

Open a **separate PowerShell** window:

```bash
nvidia-smi -l 1
```

This shows GPU usage, temperature, and memory every 1 second.

---

**Start here:** Copy Step 2, 3, 4, and 5 commands into your PowerShell! 🚀
