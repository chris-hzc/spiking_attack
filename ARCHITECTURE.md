# Architecture Documentation

## Code Organization

This document explains the architecture of the Spiking Attack implementation and how different components interact.

## Module Hierarchy

```
spiking_attack/
│
├── spiking/                          # Core spiking mechanism
│   ├── spiking_layer.py             # Layer-level adaptive reuse
│   ├── spiking_model.py             # Model wrapper and scheduling
│   └── __init__.py                  # Module exports
│
├── models/                           # Neural network architectures
│   ├── resnet.py                    # ResNet variants
│   └── ...                          # Other architectures
│
├── quant/                            # Legacy quantization code (reference)
│   └── ...
│
├── spiking_attack.py                # Attack evaluation script
├── spiking_train.py                 # Adversarial training script
├── compare_with_baseline.py         # Comparison with standard PGD
└── test_quant.py                    # Original quantization-based code
```

## Core Components

### 1. SpikingModule (`spiking/spiking_layer.py`)

**Purpose**: Wraps Conv2d/Linear layers to implement adaptive activation reuse.

**Key Attributes**:
- `rho`: Threshold for relative activation change
- `pre_activation`: Previous input activation (stored)
- `pre_output`: Previous output (stored)
- `pre_grad`: Previous gradient (for virtual surrogate)
- `precision_list`: Track computation decisions

**Key Methods**:

```python
def should_recompute(self, current_activation) -> bool:
    """
    Decision function: compute or reuse?
    Returns True if ||a_t - a_{t-1}|| / ||a_t|| >= rho
    """
    
def virtual_surrogate_hook(self, grad):
    """
    Compute gradient when activation was reused.
    Uses stored weights to compute grad_input from grad_output.
    """

def forward(self, input):
    """
    Main forward pass with adaptive computation:
    1. Check if should recompute
    2. Either compute full forward or reuse previous output
    3. Attach gradient hooks if reusing
    4. Store activation and output for next iteration
    """
```

**Computation Flow**:

```
Iteration t:
┌─────────────────────────────────────────────────────────┐
│ Input: a_t                                              │
│                                                         │
│ Decision: ||a_t - a_{t-1}|| / ||a_t|| >= rho ?        │
│                                                         │
│   YES (large change)         NO (small change)         │
│   │                          │                          │
│   ├─> Full computation       ├─> Reuse o_{t-1}        │
│   │    o_t = layer(a_t)      │    o_t = o_{t-1}       │
│   │    precision += 1         │    precision += 0       │
│   │                          │                          │
│   │    Standard gradient     │    Virtual gradient     │
│   │                          │    (via surrogate)      │
│                                                         │
│ Store: pre_activation = a_t, pre_output = o_t         │
└─────────────────────────────────────────────────────────┘
```

### 2. SpikingModel (`spiking/spiking_model.py`)

**Purpose**: Wraps entire model with spiking mechanism, provides high-level control.

**Key Methods**:

```python
def spiking_module_refactor(self, module, rho):
    """
    Recursively replace Conv2d/Linear with SpikingModule.
    Also moves ReLU into SpikingModule for efficiency.
    """

def set_spiking_state(self, use_spiking: bool):
    """Enable/disable spiking for all layers"""

def set_threshold(self, rho: float):
    """Update threshold for all layers"""

def reset_state(self):
    """Clear stored activations (call before each attack)"""

def get_precision(self):
    """Get precision statistics across all layers"""
```

**Usage Pattern**:

```python
# Setup
model = SpikingModel(base_model, rho=0.02)

# Before attack
model.reset_state()
model.set_spiking_state(True)

# During attack
for step in range(attack_steps):
    loss = criterion(model(x), y)
    grad = compute_gradient(loss, x)
    x = x + alpha * sign(grad)

# Get statistics
precision = model.get_precision()  # e.g., {'overall': 0.65, 'layer_wise': [0.5, 0.7, ...]}
```

### 3. ThresholdScheduler (`spiking/spiking_model.py`)

**Purpose**: Schedule threshold ρ during adversarial training.

**Scheduling Strategies**:

**Constant Schedule**:
```
rho(t) = rho_0  for all t
```

**Exponential Decay** (recommended):
```
rho(t) = rho_0 × (e^{-λt/N} - e^{-λ}) / (1 - e^{-λ})

where:
- rho_0: initial threshold (e.g., 0.1)
- λ: decay rate (e.g., 5.0)
- N: total epochs (e.g., 200)
- t: current epoch

Properties:
- rho(0) = rho_0
- rho(N) = 0
- Smooth decay
```

**Usage**:

```python
scheduler = ThresholdScheduler(
    schedule_type='exponential',
    rho_initial=0.1,
    total_epochs=200,
    lambda_decay=5.0
)

for epoch in range(200):
    rho = scheduler.get_threshold(epoch)
    model.set_threshold(rho)
    train_one_epoch(model)
```

## Attack Pipeline

### Spiking-PGD Attack Flow

```
1. Initialization
   ├─> Load pretrained model
   ├─> Wrap with SpikingModel(model, rho)
   └─> Create SpikingPGDAttack(spiking_model, epsilon, k, alpha)

2. For each batch:
   ├─> Reset model state
   │   ├─> clear pre_activation, pre_output
   │   └─> reset precision tracking
   │
   ├─> Enable spiking mechanism
   │
   ├─> Generate adversarial example:
   │   │
   │   ├─> x = x_clean + random_noise
   │   │
   │   └─> For each iteration i:
   │       ├─> Forward: logits = model(x)
   │       │   └─> Each layer decides: compute or reuse
   │       │
   │       ├─> Backward: grad = ∂loss/∂x
   │       │   └─> Virtual surrogate for reused layers
   │       │
   │       └─> Update: x = x + alpha × sign(grad)
   │
   └─> Evaluate on clean model (transfer attack)
```

### Adversarial Training Flow

```
1. Setup
   ├─> Create base model
   ├─> Wrap with SpikingModel
   ├─> Create ThresholdScheduler
   └─> Create optimizer

2. For each epoch:
   │
   ├─> Update threshold: rho = scheduler.get_threshold(epoch)
   │   └─> model.set_threshold(rho)
   │
   ├─> For each batch:
   │   │
   │   ├─> Generate adversarial examples with Spiking-PGD
   │   │   ├─> Enable spiking
   │   │   ├─> Attack with adaptive reuse
   │   │   └─> Track precision
   │   │
   │   ├─> Train on adversarial examples
   │   │   ├─> Disable spiking (standard forward)
   │   │   ├─> Compute loss on adv examples
   │   │   └─> Update model weights
   │   │
   │   └─> Log precision statistics
   │
   └─> Evaluate on clean and robust test sets
```

## Gradient Flow

### Standard Forward-Backward (no reuse)

```
Forward:
input ──> layer(input) ──> output

Backward:
grad_input <── layer.backward(grad_output) <── grad_output
```

### Spiking Forward-Backward (with reuse)

```
Forward (reuse):
input ──X──> (skip computation) ──> output = pre_output

Backward (virtual surrogate):
grad_input <── virtual_gradient(grad_output, weights) <── grad_output
              ↑
              └─ Uses stored weights to compute gradient
```

**Virtual Surrogate Implementation**:

For Conv2d:
```python
grad_input = torch.nn.grad.conv2d_input(
    input_size=input.shape,
    weight=stored_weight,
    grad_output=grad_output,
    stride=stride,
    padding=padding,
    ...
)
```

For Linear:
```python
grad_input = grad_output @ stored_weight
```

This ensures gradient information flows correctly even when forward pass was skipped!

## Memory Management

**Stored State per Layer**:
- `pre_activation`: size = input size (e.g., [batch, channels, H, W])
- `pre_output`: size = output size (e.g., [batch, channels, H', W'])
- `pre_grad`: size = grad_output size (only when used)

**Memory Cost**:
- Standard PGD: O(1) per iteration (no state storage)
- Spiking-PGD: O(L) where L = number of layers (one activation + output per layer)
- Additional cost is typically < 2× model parameters

**Optimization Tips**:
- Call `reset_state()` after each attack to free memory
- Use `torch.no_grad()` when storing activations
- `.clone()` is necessary to avoid gradient computation on stored tensors

## Performance Characteristics

### Computational Complexity

**Standard PGD (T iterations, L layers)**:
- Forward: T × L × C_layer
- Backward: T × L × C_layer
- Total: 2TLC_layer

**Spiking-PGD (precision = p)**:
- Forward: T × L × p × C_layer (only fraction p computed)
- Backward: T × L × C_layer (gradient always computed, but virtual gradient is cheaper)
- Total: ≈ TLC_layer × (p + 1)

**Speedup**:
```
Standard / Spiking = 2 / (p + 1)

Example with p = 0.3 (30% precision):
Speedup = 2 / 1.3 ≈ 1.54×
```

### Threshold Selection Guide

| rho | Expected Precision | Attack Strength | Use Case |
|-----|-------------------|----------------|----------|
| 0.00 | ~5% | Weak | Baseline experiment |
| 0.01 | ~95% | Very Strong | Minimal savings |
| 0.02 | ~65% | Strong | **Recommended** |
| 0.03 | ~35% | Moderate | High efficiency |
| 0.05 | ~15% | Weak | Too aggressive |

**Empirical Observation** (from paper):
- Layer-wise precision varies: early layers change less than later layers
- Precision decreases over iterations: later iterations have more reuse
- Optimal rho depends on attack strength required

## Extension Points

### Adding New Attack Algorithms

To adapt other attacks (MI-FGSM, C&W, etc.):

```python
class SpikingMIFGSM:
    def __init__(self, model, ...):
        self.model = model  # SpikingModel
        self.momentum = 0
        
    def perturb(self, x_natural, y):
        self.model.reset_state()
        self.model.set_spiking_state(True)
        
        for i in range(self.k):
            # Compute gradient with spiking
            grad = compute_gradient(...)
            
            # Momentum update (MI-FGSM specific)
            self.momentum = mu * self.momentum + grad / torch.norm(grad, p=1)
            
            # Update x
            x = x + alpha * sign(self.momentum)
```

### Custom Threshold Schedules

```python
class CustomScheduler(ThresholdScheduler):
    def get_threshold(self, epoch):
        # Implement custom logic
        # Example: stepwise decay
        if epoch < 100:
            return 0.05
        elif epoch < 150:
            return 0.02
        else:
            return 0.01
```

### Layer-Specific Thresholds

Modify `SpikingModel`:

```python
def set_threshold_layerwise(self, rho_list):
    """Set different threshold for each layer"""
    spiking_modules = [m for m in self.model.modules() 
                       if isinstance(m, SpikingModule)]
    for module, rho in zip(spiking_modules, rho_list):
        module.set_threshold(rho)
```

## Debugging Tips

### Verify Adaptive Reuse

```python
# Enable detailed logging
for module in model.modules():
    if isinstance(module, SpikingModule):
        module.debug = True  # Add this attribute to print decisions

# Check precision distribution
precision = model.get_precision()
print(f"Layer-wise: {precision['layer_wise']}")
print(f"Overall: {precision['overall']}")
```

### Check Gradient Flow

```python
# Before spiking
x.requires_grad = True
loss = criterion(model(x), y)
grad_standard = torch.autograd.grad(loss, x, retain_graph=True)[0]

# With spiking
model.set_spiking_state(True)
loss = criterion(model(x), y)
grad_spiking = torch.autograd.grad(loss, x)[0]

# Compare
print(f"Gradient diff: {torch.norm(grad_standard - grad_spiking)}")
```

### Profile Performance

```python
import time

# Standard PGD
start = time.time()
adv_standard = standard_pgd.perturb(x, y)
time_standard = time.time() - start

# Spiking PGD
start = time.time()
adv_spiking = spiking_pgd.perturb(x, y)
time_spiking = time.time() - start

print(f"Speedup: {time_standard / time_spiking:.2f}x")
```

## Common Issues and Solutions

**Issue 1**: Precision is always 100%
- **Cause**: Threshold rho is too low
- **Solution**: Increase rho (try 0.02 or higher)

**Issue 2**: Attack is too weak
- **Cause**: Threshold rho is too high, too much reuse
- **Solution**: Decrease rho (try 0.02 or lower)

**Issue 3**: Gradient becomes NaN
- **Cause**: Virtual surrogate gradient computation error
- **Solution**: Check weight storage, ensure `.clone()` is used

**Issue 4**: Memory leak during attack
- **Cause**: Not calling `reset_state()` between attacks
- **Solution**: Call `model.reset_state()` before each `perturb()`

**Issue 5**: No speedup observed
- **Cause**: GPU kernel launch overhead dominates
- **Solution**: Test on larger batches or use CPU profiling

---

For more details, see the paper and code comments in each module.

