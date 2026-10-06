# Quick Start: ResNet50 with Pretrained Weights

Test Spiking-PGD on ResNet50 using pretrained weights from [PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) (93.65% accuracy).

## 🚀 Super Quick Start (2 commands)

```bash
# 1. Setup (installs package and downloads weights automatically)
python download_pretrained_resnet50.py

# 2. Run test (uses pretrained weights automatically)
python test_resnet50_spiking.py --rho 0.02 --num_samples 1000
```

**That's it!** The script will automatically download the pretrained ResNet50 weights (~91 MB) on first run.

## 📦 What Gets Installed

- **Package**: `cifar10_models` from [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)
- **Model**: ResNet50 pretrained on CIFAR-10
- **Accuracy**: 93.65% on CIFAR-10 test set
- **Parameters**: 23.5M parameters (~91 MB)

## 🎯 Usage Examples

### Example 1: Quick Test (1000 samples, ~1 minute)

```bash
python test_resnet50_spiking.py \
    --rho 0.02 \
    --steps 20 \
    --num_samples 1000
```

**Expected Output:**
```
Benign accuracy: ~93.65%
Robust accuracy: ~40-50%
Average precision: ~65%
Cost reduction: ~35%
```

### Example 2: Full Test (10,000 samples, ~10 minutes)

```bash
python test_resnet50_spiking.py \
    --rho 0.02 \
    --steps 20
```

### Example 3: High Efficiency (more reuse)

```bash
python test_resnet50_spiking.py \
    --rho 0.03 \
    --steps 20 \
    --num_samples 1000
```

**Expected:** ~40% precision (60% cost reduction)

### Example 4: Strong Attack

```bash
python test_resnet50_spiking.py \
    --rho 0.02 \
    --epsilon 16 \
    --steps 50
```

### Example 5: Run All Experiments

```bash
bash scripts/run_resnet50_experiments.sh
```

Runs 13 different configurations automatically!

## 🔧 Key Parameters

| Parameter | Default | Description | Recommended Values |
|-----------|---------|-------------|-------------------|
| `--rho` | 0.02 | Reuse threshold | 0.01 (strong), 0.02 (balanced), 0.03 (efficient) |
| `--epsilon` | 8 | Attack budget (0-255) | 4 (small), 8 (standard), 16 (large) |
| `--steps` | 20 | Attack iterations | 10 (quick), 20 (standard), 50 (strong) |
| `--num_samples` | 10000 | Test samples | 1000 (quick test), 10000 (full) |

## 📊 Expected Results

### ResNet50 (93.65% clean accuracy)

| rho | Precision | Robust Acc | Cost Reduction |
|-----|-----------|------------|----------------|
| 0.01 | ~95% | ~42% | ~5% |
| 0.02 | ~65% | ~40% | ~35% |
| 0.03 | ~40% | ~38% | ~60% |

*Note: Actual results may vary slightly based on random initialization*

## 🆚 Comparison: Manual Checkpoint vs Auto-Download

### Old Way (Manual Checkpoint)
```bash
# Download/train model manually
# Save checkpoint file
python test_resnet50_spiking.py --checkpoint /path/to/checkpoint.pth
```

### New Way (Automatic)
```bash
# Automatically downloads pretrained weights
python test_resnet50_spiking.py --rho 0.02
```

**Advantages:**
- ✅ No manual download needed
- ✅ Verified pretrained weights (93.65% accuracy)
- ✅ One-line command
- ✅ Automatic installation

## 🔍 Behind the Scenes

When you run `test_resnet50_spiking.py`:

1. **Check for `cifar10_models` package**
   - If not found, automatically install from GitHub
   
2. **Load pretrained weights**
   - Downloads ResNet50 weights (~91 MB) on first run
   - Cached for future use
   
3. **Apply data preprocessing**
   - Normalize with CIFAR-10 mean/std
   - `mean=[0.4914, 0.4822, 0.4465]`
   - `std=[0.2471, 0.2435, 0.2616]`
   
4. **Run Spiking-PGD attack**
   - Adaptive activation reuse
   - Track computational cost

## 💻 Use in Your Own Code

```python
# Import pretrained model
from cifar10_models.resnet import resnet50
from spiking import SpikingModel
import torch

# Load pretrained ResNet50
net = resnet50(pretrained=True)
net.eval()

# Wrap with spiking mechanism
spiking_model = SpikingModel(model=net, rho=0.02)
spiking_model.eval()

# Test on your data (must be normalized!)
# mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
output = spiking_model(normalized_images)

# Check computational cost
precision = spiking_model.get_precision()
print(f"Used {precision['overall']*100:.1f}% of full computation")
```

## 🐛 Troubleshooting

### Issue: "ImportError: No module named cifar10_models"

**Solution:** Run the setup script
```bash
python download_pretrained_resnet50.py
```

Or install manually:
```bash
pip install git+https://github.com/huyvnphan/PyTorch_CIFAR10.git
```

### Issue: "SSL certificate error" during download

**Solution:** Use pip with trusted host
```bash
pip install --trusted-host github.com \
    git+https://github.com/huyvnphan/PyTorch_CIFAR10.git
```

### Issue: Weights download is slow

**Cause:** First-time download of ~91 MB weights

**Solution:** Be patient, or download manually from:
- https://github.com/huyvnphan/PyTorch_CIFAR10/releases

### Issue: Out of memory

**Solution:** Reduce batch size or test samples
```bash
python test_resnet50_spiking.py \
    --batch_size 50 \
    --num_samples 500
```

## 📚 Additional Resources

- **Original Repository**: https://github.com/huyvnphan/PyTorch_CIFAR10
- **Paper Citation**: See repository README
- **Other Models**: VGG, DenseNet, MobileNet also available

## 🎓 Comparison with ResNet18

| Metric | ResNet18 | ResNet50 | Difference |
|--------|----------|----------|------------|
| Clean Acc | ~95% | ~93.65% | -1.35% |
| Parameters | 11.2M | 23.5M | 2.1× |
| Robust Acc (PGD-20) | ~42% | ~40% | -2% |
| Spiking Savings | 30-40% | 30-40% | Similar |

**Key Insight**: ResNet50 benefits similarly from Spiking-PGD despite being larger!

## ✅ Verify Installation

```bash
python -c "from cifar10_models.resnet import resnet50; \
           m = resnet50(pretrained=True); \
           print('✓ ResNet50 loaded successfully!')"
```

## 🚀 Next Steps

1. **Run quick test** → `python test_resnet50_spiking.py --num_samples 1000`
2. **Try different rho** → Compare 0.01, 0.02, 0.03
3. **Full evaluation** → Remove `--num_samples` for all 10K images
4. **Batch experiments** → `bash scripts/run_resnet50_experiments.sh`
5. **Use in your code** → See code example above

---

**Ready to test!** 🎯

The pretrained model will be automatically downloaded on first run (one-time ~91 MB download).

For more details, see [`resnet50.md`](resnet50.md).


