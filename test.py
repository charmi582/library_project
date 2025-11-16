import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  > CUDA 版本 (PyTorch 所編譯): {torch.version.cuda}")
    print(f"  > 偵測到的 GPU 數量: {torch.cuda.device_count()}")
    print(f"  > 目前 GPU 名稱: {torch.cuda.get_device_name(0)}")
else:
    print("!! 警告: PyTorch 無法偵測到可用的 CUDA 裝置 !!")
    print("   請檢查：")
    print("   1. 是否安裝了 NVIDIA 驅動程式？ (可在終端機輸入 'nvidia-smi')")
    print("   2. PyTorch 是否安裝了 'CPU' 版本？ (應安裝 GPU 版本)")