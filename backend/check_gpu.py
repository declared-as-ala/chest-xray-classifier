"""
GPU Verification Script
Checks if CUDA is properly installed and GPU is detected.
"""
import torch

print("=" * 70)
print("GPU VERIFICATION TEST")
print("=" * 70)

# 1. PyTorch Version
print(f"\n1. PyTorch Version: {torch.__version__}")

# 2. CUDA Availability
cuda_available = torch.cuda.is_available()
print(f"2. CUDA Available: {cuda_available}")

if cuda_available:
    # 3. CUDA Version
    print(f"3. CUDA Version: {torch.version.cuda}")
    
    # 4. Number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"4. Number of GPUs: {gpu_count}")
    
    # 5. GPU Details
    for i in range(gpu_count):
        print(f"\n   GPU {i} Details:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   - Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"   - Compute Capability: {props.major}.{props.minor}")
    
    # 6. Test GPU with tensor operation
    print(f"\n5. Testing GPU with tensor operation...")
    try:
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y  # Matrix multiplication on GPU
        print(f"   ✅ GPU tensor operation successful!")
        print(f"   ✅ Result tensor device: {z.device}")
        print(f"   ✅ Result tensor shape: {z.shape}")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ GPU SETUP IS READY!")
    print("=" * 70)
    print("\nYour training will use:")
    print(f"🚀 {torch.cuda.get_device_name(0)}")
    print(f"⚡ CUDA {torch.version.cuda}")
    print(f"💾 {props.total_memory / (1024**3):.2f} GB VRAM")
    print("\nRecommended batch sizes for your GPU:")
    print("  - train.py (frozen layers): batch_size=24")
    print("  - train_enhanced.py (layer4 unfrozen): batch_size=12")
    print("  - train_with_augmentation.py (layer4 unfrozen): batch_size=12")
    
else:
    print("\n" + "=" * 70)
    print("❌ NO GPU DETECTED - USING CPU")
    print("=" * 70)
    print("\nTo enable GPU training:")
    print("\n1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Install CUDA-enabled PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("\n3. Verify installation:")
    print("   python check_gpu.py")
    print("\nSee GPU_SETUP_GUIDE.md for detailed instructions.")

print("\n" + "=" * 70)
