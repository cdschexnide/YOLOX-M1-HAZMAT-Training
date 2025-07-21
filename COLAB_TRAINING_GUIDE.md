# YOLOX Hazmat Detection Training on Google Colab

This guide provides step-by-step instructions for training your YOLOX hazmat detection model on Google Colab, avoiding the thermal throttling issues experienced on M1 Mac.

## Expected Performance Improvement

| Platform | Training Time | Batch Size | Thermal Issues |
|----------|---------------|------------|----------------|
| **M1 Pro MacBook** | 3-4 days | 2-4 | Yes (throttling) |
| **M1 Max Mac Studio** | 8+ hours | 2-4 | Yes (throttling) |
| **Google Colab (Free)** | 1-2 hours | 32-64 | No |
| **Google Colab Pro** | 30-60 minutes | 64+ | No |

## Prerequisites

1. **Google Account** - for accessing Colab
2. **Google Drive** - for storing dataset and results
3. **Dataset Upload** - hazmat VOC dataset uploaded to Drive

## Step 1: Upload Dataset to Google Drive

1. **Create folder structure** in Google Drive:
   ```
   MyDrive/
   └── hazmat_dataset/
       └── VOCdevkit/
           └── VOC2007/
               ├── Annotations/     (2,429 XML files)
               ├── JPEGImages/      (2,429 JPG files)
               └── ImageSets/Main/  (train.txt, val.txt, test.txt)
   ```

2. **Upload your VOC dataset** to this exact structure
3. **Verify file counts** match your local dataset

## Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
3. For faster training: Consider **Colab Pro** for V100/A100 GPUs

## Step 3: Setup Methods (Choose One)

### Method A: Automated Setup (Recommended)

1. **Upload and run** the automated setup notebook:
   ```python
   # Upload YOLOX_Hazmat_Colab_Training.ipynb to Colab
   # Run all cells in order
   ```

2. **Follow the notebook** - it handles all dependency conflicts automatically

### Method B: Manual Setup

1. **Clone repository**:
   ```python
   !git clone https://github.com/YOUR_USERNAME/YOLOX-M1-HAZMAT-Training.git
   %cd YOLOX-M1-HAZMAT-Training
   ```

2. **Run setup script**:
   ```python
   !python colab_setup.py
   ```

3. **Start training**:
   ```python
   !python train_hazmat_m1.py -f exps/hazmat/yolox_s_hazmat_colab.py -b 32
   ```

## Step 4: Monitor Training

**Expected output**:
```
epoch: 1/25, iter: 10/52, mem: 0Mb, iter_time: 0.8s, data_time: 0.1s, 
total_loss: 25.3, iou_loss: 4.2, l1_loss: 0.0, conf_loss: 19.8, cls_loss: 1.3, 
lr: 1.234e-05, size: 640, ETA: 45:23
```

**Key metrics to watch**:
- **iter_time**: Should be ~0.5-1.5 seconds (much faster than M1 Mac)
- **total_loss**: Should decrease over time
- **ETA**: Should be 1-2 hours for 25 epochs

## Step 5: Handle Session Timeouts

**Colab sessions timeout after ~12 hours (Free) or ~24 hours (Pro)**

If training is interrupted:
1. **Reconnect to runtime**
2. **Re-run setup cells**
3. **Resume training** from last checkpoint:
   ```python
   !python train_hazmat_m1.py \
       -f exps/hazmat/yolox_s_hazmat_colab.py \
       -b 32 \
       --resume \
       -c YOLOX_outputs/yolox_s_hazmat_colab/latest_ckpt.pth
   ```

## Step 6: Download Results

**After training completes**:
1. **Checkpoints** saved to `YOLOX_outputs/`
2. **Backup to Drive**:
   ```python
   !cp -r YOLOX_outputs/* /content/drive/MyDrive/yolox_results/
   ```
3. **Download** locally via Drive or Colab file browser

## Troubleshooting

### Common Issues & Solutions

**Issue**: "No module named 'yolox'"
```python
# Solution: Reinstall YOLOX
!pip install -e . --no-deps
```

**Issue**: NumPy 2.0 conflicts
```python
# Solution: Force NumPy 1.x
!pip install --no-deps numpy==1.24.3
```

**Issue**: CUDA out of memory
```python
# Solution: Reduce batch size
# Change -b 32 to -b 16 or -b 8
```

**Issue**: Dataset not found
```python
# Solution: Check Google Drive path
!ls /content/drive/MyDrive/hazmat_dataset/VOCdevkit/VOC2007/
```

### Dependency Conflicts Prevention

The setup automatically handles these common conflicts:
- ✅ **NumPy 1.x vs 2.x** - Forces 1.x compatibility
- ✅ **PyTorch CUDA versions** - Uses Colab-compatible versions  
- ✅ **OpenCV conflicts** - Installs compatible version
- ✅ **ONNX issues** - Skips problematic packages initially

## Performance Optimization

### GPU Memory Optimization
```python
# For T4 (16GB): batch_size=32
# For V100 (32GB): batch_size=64  
# For A100 (40GB+): batch_size=96
```

### Training Speed Tips
1. **Enable mixed precision**: `--fp16` flag
2. **Use image caching**: `--cache` flag  
3. **Optimal batch size**: Match GPU memory
4. **Colab Pro**: 2-3x faster GPUs

### Cost Optimization
- **Free Colab**: T4 GPU, ~12h sessions, sufficient for most training
- **Colab Pro ($10/month)**: V100/A100 GPUs, ~24h sessions, 2-3x faster

## Expected Results

**After 25 epochs (~1-2 hours)**:
- **Functional hazmat detection model**
- **12 hazmat classes** properly classified
- **Good accuracy** for proof of concept
- **Ready for testing** on new images

**Model capabilities**:
- Detects hazmat signs in images
- Classifies into 12 categories:
  - explosive, flammable, oxidizer, dangerous-when-wet
  - poison, spontaneously-combustible, radioactive, corrosive
  - non-flammable-gas, flammable-solid, infectious-substance, organic-peroxide
- Provides bounding boxes and confidence scores

## Next Steps

1. **Test trained model** on sample images
2. **Evaluate performance** on validation set
3. **Deploy model** for inference
4. **Fine-tune** with additional epochs if needed (50-100 for production)

---

🎉 **Success!** You've trained a hazmat detection model 5-10x faster than on M1 Mac, avoiding thermal throttling issues entirely!