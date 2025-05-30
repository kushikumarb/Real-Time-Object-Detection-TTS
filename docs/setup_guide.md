# Advanced Setup Guide

## GPU Acceleration

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Custom Models

1. Download models from Ultralytics:

```bash
bash scripts/download_model.sh yolov8s.pt
```

2. Run with custom model:

```bash
python src/main.py --model models/yolov8s.pt
```
