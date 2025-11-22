# Quick Start Guide

Get started with Spiking-PGD in 5 minutes! ⚡

## 1. Installation (1 minute)

```bash
cd spiking_attack
pip install torch torchvision numpy matplotlib
```

## 2. Run Your First Attack (2 minutes)

### Option A: Use the script

```bash
# Run attack with default settings (rho=0.02, epsilon=8/255, 20 steps)
python spiking_attack.py \
    --dataset cifar10 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 20
```

### Option B: Python code

```python
from models import resnet18
from spiking import SpikingModel
import torch

# 1. Load your model
net = resnet18(num_classes=10)
net.load_state_dict(torch.load('checkpoint.pth')['net'])

# 2. Wrap with SpikingModel
spiking_model = SpikingModel(model=net, rho=0.02)
spiking_model.eval()

# 3. Create attacker
from spiking_attack import SpikingPGDAttack
adversary = SpikingPGDAttack(
    model=spiking_model,
    epsilon=8/255,
    k=20,
    alpha=2/255
)

# 4. Generate adversarial examples
adv_images = adversary.perturb(clean_images, labels)

# 5. Check computational cost
precision = spiking_model.get_precision()
print(f"Computational cost: {precision['overall']*100:.1f}%")
```

## 3. Compare with Baseline (2 minutes)

```bash
python compare_with_baseline.py \
    --rho 0.02 \
    --steps 20 \
    --num_batches 10
```

**Expected output:**
```
Standard PGD:  Accuracy=20.0%, Time=0.523s, Cost: 100.0%
Spiking PGD:   Accuracy=21.0%, Time=0.341s, Cost: 65.0%
Speedup: 1.53x, Cost Reduction: 35.0%
```

## 4. Visualize the Mechanism (Optional)

```bash
python visualize_spiking.py --steps 20 --num_samples 100
```

This creates several plots showing:
- Precision over iterations
- Layer-wise precision patterns
- Cost vs threshold curves

## Key Parameters Explained

### `--rho` (Threshold for Relative Change)

Controls when to recompute vs reuse:

| Value | Cost | Attack Strength | When to Use |
|-------|------|----------------|-------------|
| 0.01 | 95% | Very Strong | Maximum effectiveness |
| 0.02 | 65% | Strong | **RECOMMENDED** |
| 0.03 | 35% | Moderate | High efficiency |
| 0.05 | 15% | Weak | Extreme efficiency |

**Rule of thumb**: Start with 0.02, increase if attack is too weak, decrease if you need more savings.

### `--epsilon` (Attack Budget)

L∞ perturbation bound:

| Value | Description |
|-------|-------------|
| 4 | Small perturbation (harder to attack) |
| 8 | Standard CIFAR-10 setting |
| 16 | Large perturbation |

### `--steps` (Number of Iterations)

More steps = stronger attack, but more computation:

| Value | Use Case |
|-------|----------|
| 10 | Quick evaluation / Training |
| 20 | Standard evaluation |
| 50 | Strong evaluation |

## Common Use Cases

### Use Case 1: Evaluate Model Robustness

```bash
python spiking_attack.py \
    --dataset cifar10 \
    --file_name your_checkpoint \
    --epoch_eval 150 \
    --rho 0.02 \
    --steps 20
```

### Use Case 2: Adversarial Training

```bash
python spiking_train.py \
    --dataset cifar10 \
    --epochs 200 \
    --rho_initial 0.1 \
    --schedule exponential \
    --lambda_decay 5.0
```

### Use Case 3: Study Efficiency-Effectiveness Trade-off

```bash
# Test different thresholds
for rho in 0.01 0.02 0.03 0.05; do
    python spiking_attack.py --rho $rho --steps 20
done
```

### Use Case 4: Reproduce Paper Results

```bash
# Use default settings - they match the paper
bash run_attack.sh
```

## Understanding the Output

When you run an attack, you'll see:

```
========== Iteration 5/20 ==========
conv2d: COMPUTE (relative change >= 0.02)
linear: REUSE (relative change < 0.02)
...
Loss: 2.3451

============================================================
Overall Precision: 67.50%
Layer-wise Precision: ['85.0%', '75.0%', '60.0%', ...]
============================================================
```

**What this means:**
- `COMPUTE`: Layer did full forward computation (change was large)
- `REUSE`: Layer reused previous output (change was small)
- `Overall Precision: 67.50%`: Used 67.5% of full computation
- **Savings**: 32.5% computation saved!

## Troubleshooting

### Problem: Precision is always 100%

**Solution**: Increase rho (try 0.03 or 0.05)

### Problem: Attack is too weak

**Solution**: Decrease rho (try 0.01) or increase steps

### Problem: Out of memory

**Solution**: 
1. Reduce batch size
2. Call `model.reset_state()` between batches
3. Use smaller model

### Problem: No speedup observed

**Solution**:
1. Test on larger batches
2. Use CPU profiling (GPU kernel launch overhead may dominate)
3. Ensure rho is not too low (< 0.02)

## Next Steps

1. **Read the full README**: `README.md`
2. **Understand the architecture**: `ARCHITECTURE.md`
3. **See what changed from quantization**: `SUMMARY.md`
4. **Experiment with different settings**:
   ```bash
   bash run_attack.sh  # Runs multiple configurations
   ```

## Cheat Sheet

```bash
# Basic attack (default settings)
python spiking_attack.py

# High efficiency (more savings)
python spiking_attack.py --rho 0.03

# High effectiveness (minimal savings)
python spiking_attack.py --rho 0.01

# Adversarial training (recommended)
python spiking_train.py --schedule exponential --lambda_decay 5.0

# Compare with standard PGD
python compare_with_baseline.py --num_batches 10

# Visualize behavior
python visualize_spiking.py

# Run all experiments
bash run_attack.sh
```

## Getting Help

1. Check `README.md` for detailed documentation
2. Read `ARCHITECTURE.md` for technical details
3. Look at code comments in `spiking/spiking_layer.py`
4. Open an issue on GitHub

---

**That's it!** You're ready to use Spiking-PGD. Start experimenting and enjoy the efficiency gains! 🚀

