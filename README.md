# Spiking Attack: Fine-Grained Iterative Adversarial Attacks

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](19984_Fine_Grained_Iterative_A.pdf)

This repository implements **Spiking-PGD**, a fine-grained iterative adversarial attack algorithm based on adaptive activation reuse. The implementation is based on the paper:

> **Fine-Grained Iterative Adversarial Attacks WITH LIMITED COMPUTATION BUDGET**

## 📖 Overview

### The Problem
Traditional iterative adversarial attacks (e.g., PGD) require substantial computational resources:
- Each iteration needs both forward and backward passes through the entire network
- PGD-20 costs ~40× more than natural inference
- Adversarial training with PGD-10 costs ~10× more than natural training

### Our Solution: Spiking Mechanism
Instead of treating all layers uniformly at every iteration, we introduce **adaptive activation reuse**:

1. **Selective Recomputation**: Only recompute layer activations when the relative change exceeds a threshold ρ (rho)
   ```
   if ||a_t - a_{t-1}|| / ||a_t|| >= ρ:
       compute full forward pass
   else:
       reuse previous activation
   ```

2. **Virtual Surrogate Gradient**: Preserve informative backward signals even when reusing activations
   - For Conv2d: `grad_input = conv_transpose(grad_output, weight)`
   - For Linear: `grad_input = grad_output @ weight`

3. **Fine-Grained Control**: Adaptive computation at both iteration-wise and layer-wise levels

### Key Benefits
- ✅ **Maintains attack effectiveness** with comparable success rates
- ✅ **Significantly reduces computational cost** (30-70% of original budget)
- ✅ **Expands efficiency-effectiveness Pareto frontier**
- ✅ **Applicable to adversarial training** with minimal accuracy degradation

## 🏗️ Repository Structure

```
spiking_attack/
├── spiking/                    # Core spiking mechanism implementation
│   ├── __init__.py            # Module exports
│   ├── spiking_layer.py       # SpikingModule with adaptive reuse
│   └── spiking_model.py       # SpikingModel wrapper and ThresholdScheduler
├── models/                     # Model architectures (ResNet, etc.)
├── spiking_attack.py          # Main attack evaluation script
├── spiking_train.py           # Adversarial training with Spiking-PGD
├── run_attack.sh              # Example script for running attacks
├── run_train.sh               # Example script for adversarial training
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spiking_attack.git
cd spiking_attack

# Install dependencies
pip install torch torchvision numpy
```

### Run Spiking-PGD Attack

Evaluate a pretrained model under Spiking-PGD attack:

```bash
python spiking_attack.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epsilon 8 \
    --steps 20 \
    --step_size 2 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150
```

**Parameters:**
- `--rho`: Threshold for relative activation change (default: 0.02)
  - Lower ρ → more reuse → lower cost but potentially weaker attack
  - Higher ρ → more computation → higher cost but stronger attack
- `--epsilon`: L∞ perturbation budget (0-255 range)
- `--steps`: Number of PGD iterations
- `--step_size`: Step size for gradient ascent

### Adversarial Training with Spiking-PGD

Train a robust model using Spiking-PGD for adversarial example generation:

```bash
python spiking_train.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --rho_initial 0.1 \
    --schedule exponential \
    --lambda_decay 5.0 \
    --exp_name spiking_at
```

**Threshold Scheduling:**
- `--schedule constant`: Fixed threshold throughout training
- `--schedule exponential`: Decay from ρ₀ to 0 following:
  ```
  ρ(t) = ρ₀ × (e^{-λt/N} - e^{-λ}) / (1 - e^{-λ})
  ```
  where t is current epoch, N is total epochs, λ is decay rate

## 📊 Key Components

### 1. SpikingModule (`spiking/spiking_layer.py`)

The core layer that implements adaptive activation reuse:

```python
from spiking import SpikingModule

# Replace standard Conv2d/Linear with SpikingModule
spiking_layer = SpikingModule(
    org_module=nn.Conv2d(3, 64, kernel_size=3),
    rho=0.02  # threshold
)

# Enable spiking mechanism
spiking_layer.set_spiking_state(True)

# Get computational cost (precision)
precision = spiking_layer.get_precision()  # 1.0 = full computation, 0.0 = full reuse
```

**Key Methods:**
- `should_recompute(activation)`: Decide whether to compute or reuse
- `virtual_surrogate_hook(grad)`: Compute gradient when activation was reused
- `get_precision()`: Return percentage of full computations performed

### 2. SpikingModel (`spiking/spiking_model.py`)

Wraps a standard model with spiking mechanism:

```python
from spiking import SpikingModel

# Wrap your model
net = resnet18(num_classes=10)
spiking_model = SpikingModel(model=net, rho=0.02)

# Control spiking state
spiking_model.set_spiking_state(True)  # Enable adaptive reuse
spiking_model.reset_state()            # Reset between attacks

# Get overall precision
stats = spiking_model.get_precision()
print(f"Overall precision: {stats['overall']:.2%}")
```

### 3. ThresholdScheduler (`spiking/spiking_model.py`)

Schedule threshold ρ during adversarial training:

```python
from spiking import ThresholdScheduler

# Exponential decay schedule
scheduler = ThresholdScheduler(
    schedule_type='exponential',
    rho_initial=0.1,
    total_epochs=200,
    lambda_decay=5.0
)

# Get threshold for current epoch
rho = scheduler.get_threshold(epoch=50)
spiking_model.set_threshold(rho)
```

### 4. Spiking-PGD Attack (`spiking_attack.py`)

Complete PGD attack with adaptive reuse:

```python
class SpikingPGDAttack:
    def __init__(self, model, epsilon=8/255, k=20, alpha=2/255):
        self.model = model  # SpikingModel
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha

    def perturb(self, x_natural, y):
        # Reset state before attack
        self.model.reset_state()
        self.model.set_spiking_state(True)
        
        # Iterative attack with adaptive reuse
        x = x_natural + random_noise
        for i in range(self.k):
            loss = F.cross_entropy(self.model(x), y)
            grad = autograd.grad(loss, x)[0]
            x = x + self.alpha * sign(grad)
            x = clip(x, x_natural - self.epsilon, x_natural + self.epsilon)
        
        return x
```

## 📈 Expected Results

Based on the paper (CIFAR-10, ResNet-18):

| Method | Attack Steps | Precision (%) | Attack Success Rate (%) |
|--------|-------------|---------------|------------------------|
| Standard PGD-20 | 20 | 100.0 | 80.2 |
| Spiking-PGD (ρ=0.01) | 20 | ~99.0 | ~80.0 |
| Spiking-PGD (ρ=0.02) | 20 | ~65.0 | ~79.5 |
| Spiking-PGD (ρ=0.03) | 20 | ~32.0 | ~78.0 |

**Precision** = Average percentage of full computations performed (lower is cheaper)

### Adversarial Training Results

| Method | Clean Acc (%) | Robust Acc (%) | Effective Precision (%) |
|--------|--------------|----------------|------------------------|
| Standard PGD-AT | 80.18 | 48.75 | 100.0 |
| Spiking-AT (λ=5.0) | 80.46 | 48.76 | 71.6 |
| Spiking-AT (λ=3.0) | 81.09 | 48.88 | 55.7 |

Spiking-AT achieves comparable robustness with **30-45% cost reduction**!

## 🔬 Methodology Deep Dive

### Mathematical Formulation

**Coarse-grained optimization** (standard PGD):
```
max_{S ∈ {0,...,T}} L(x_S)
subject to: Σ_{t=1}^S C_t ≤ C_total
```
Only controls number of iterations S.

**Fine-grained optimization** (Spiking-PGD):
```
max_{δ ∈ {0,1}^{T×L}} L(x_T(δ))
subject to: Σ_{t=1}^T Σ_{l=1}^L δ_{t,l} C_{t,l} ≤ C_total
```
Controls computation at each layer l and iteration t via mask δ.

**Proposition 4.1**: Fine-grained control achieves better or equal objective value:
```
V_coarse ≤ V_fine
```

### Adaptive Reuse Criterion

At iteration t, for layer l with input activation a_t:

```python
relative_change = ||a_t - a_{t-1}|| / ||a_t||

if relative_change >= ρ:
    # Significant change → recompute
    o_t = layer(a_t)
else:
    # Small change → reuse
    o_t = o_{t-1}  # reuse previous output
    grad = virtual_surrogate_gradient(grad_output)
```

### Virtual Surrogate Gradient

When forward pass reuses activation, backward pass still needs gradients:

```python
# Forward: o_t = o_{t-1} (reused)
# Backward: need ∂L/∂a_t even though we didn't compute o_t = f(a_t)

# Solution: Use chain rule with stored weights
if layer is Conv2d:
    ∂L/∂a_t = conv_transpose(∂L/∂o_t, W)
elif layer is Linear:
    ∂L/∂a_t = (∂L/∂o_t) @ W^T
```

This preserves gradient information while saving forward computation!

## 🎯 Tuning Guide

### Choosing Threshold ρ

| ρ Value | Computation Cost | Attack Strength | Use Case |
|---------|-----------------|----------------|----------|
| 0.0 | Lowest | Weakest | Baseline (extreme reuse) |
| 0.01 | ~99% of full | Very strong | Minimal savings, maximal strength |
| 0.02 | ~65% of full | Strong | **Recommended balance** |
| 0.03 | ~32% of full | Moderate | High efficiency, acceptable strength |
| 0.05+ | ~15% of full | Weak | Too aggressive reuse |

**Rule of thumb**: Start with ρ=0.02 for attack evaluation, ρ=0.1 for adversarial training.

### Threshold Scheduling (Training)

**Constant schedule:**
- Simple, predictable
- Fixed computational cost throughout training
- Use for quick experiments

**Exponential decay:**
- Starts with high ρ (more reuse, faster training in early epochs)
- Ends with low ρ (less reuse, stronger attacks in late epochs)
- Better final robustness with lower average cost
- **Recommended for production**

**λ parameter:**
- λ=1.0: Slow decay, more reuse
- λ=5.0: **Recommended balance**
- λ=10.0: Fast decay, approaches standard training


## 🤝 Acknowledgments

This implementation is based on:
- Paper: "Fine-Grained Iterative Adversarial Attacks WITH LIMITED COMPUTATION BUDGET"
- Original quantization-based exploration (this repository's history)
- Standard PGD attack framework

## 📄 License

This project is for research purposes. Please check the license file for details.

## 🐛 Issues and Contributions

If you encounter any issues or have suggestions for improvements, please:
1. Check existing issues
2. Open a new issue with detailed description
3. Submit pull requests for bug fixes or enhancements

---

**Key Insight**: By adaptively reusing activations based on relative change, Spiking-PGD achieves the sweet spot between computational efficiency and attack effectiveness! 🎯
