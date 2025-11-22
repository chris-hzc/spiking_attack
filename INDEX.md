# Repository Index

Complete guide to all files in this repository.

## 📁 Directory Structure

```
spiking_attack/
├── spiking/                    # Core spiking mechanism [NEW]
├── models/                     # Neural network architectures
├── quant/                      # Legacy quantization code [REFERENCE]
├── data/                       # Downloaded datasets (auto-created)
├── checkpoint/                 # Training checkpoints (auto-created)
├── log/                        # Evaluation logs (auto-created)
├── Scripts (Python)            # Main executable scripts
├── Scripts (Shell)             # Batch experiment scripts
└── Documentation               # Guides and references
```

---

## 🔥 Quick Access

### Getting Started
1. **First time?** → [`QUICK_START.md`](#quick_startmd) (5 min guide)
2. **Need overview?** → [`README.md`](#readmemd) (comprehensive guide)
3. **Want details?** → [`ARCHITECTURE.md`](#architecturemd) (technical docs)
4. **Migrating from quant?** → [`SUMMARY.md`](#summarymd) (comparison)

### Running Code
- **Attack evaluation** → [`spiking_attack.py`](#spiking_attackpy)
- **Adversarial training** → [`spiking_train.py`](#spiking_trainpy)
- **Compare methods** → [`compare_with_baseline.py`](#compare_with_baselinepy)
- **Visualize** → [`visualize_spiking.py`](#visualize_spikingpy)

---

## 📂 Detailed File Listing

### Core Implementation (`spiking/`)

#### `spiking/spiking_layer.py` 
**Purpose**: Layer-level adaptive activation reuse  
**Size**: ~250 lines  
**Key Classes**:
- `SpikingModule`: Wraps Conv2d/Linear with adaptive computation
  - `should_recompute()`: Decision function
  - `virtual_surrogate_hook()`: Virtual gradient computation
  - `forward()`: Adaptive forward pass

**Key Functions**:
- `set_threshold_all_layers()`: Update threshold for all layers
- `reset_state_all_layers()`: Clear stored activations
- `get_model_precision()`: Get computation statistics

**When to read**: Understanding core mechanism

---

#### `spiking/spiking_model.py`
**Purpose**: Model wrapper and threshold scheduling  
**Size**: ~160 lines  
**Key Classes**:
- `SpikingModel`: Wraps entire model with spiking mechanism
  - `spiking_module_refactor()`: Replace layers with SpikingModule
  - `set_spiking_state()`: Enable/disable mechanism
  - `set_threshold()`: Update threshold
  - `get_precision()`: Get statistics

- `ThresholdScheduler`: Schedule threshold during training
  - `get_threshold(epoch)`: Compute threshold for epoch
  - Supports constant and exponential decay

**When to read**: Using spiking model in your code

---

#### `spiking/__init__.py`
**Purpose**: Module exports  
**Size**: ~20 lines  
**Exports**: All public APIs from spiking module

**When to read**: Understanding available imports

---

### Main Scripts

#### `spiking_attack.py`
**Purpose**: Evaluate model under Spiking-PGD attack  
**Size**: ~330 lines  
**Main Components**:
- `SpikingPGDAttack`: Attack implementation
- `test()`: Evaluation loop
- Metrics: benign accuracy, robust accuracy, precision

**Usage**:
```bash
python spiking_attack.py --rho 0.02 --steps 20 --epsilon 8
```

**When to use**: Evaluating model robustness

---

#### `spiking_train.py`
**Purpose**: Adversarial training with Spiking-PGD  
**Size**: ~350 lines  
**Main Components**:
- `SpikingPGDTraining`: Attack for training
- `train(epoch)`: Training loop
- `test_clean(epoch)`: Clean accuracy evaluation
- `test_robust(epoch)`: Robust accuracy evaluation

**Usage**:
```bash
python spiking_train.py --schedule exponential --lambda_decay 5.0
```

**When to use**: Training robust models

---

#### `compare_with_baseline.py`
**Purpose**: Compare Spiking-PGD vs Standard PGD  
**Size**: ~280 lines  
**Main Components**:
- `StandardPGD`: Baseline attack
- `SpikingPGD`: Spiking attack
- Side-by-side comparison with timing

**Usage**:
```bash
python compare_with_baseline.py --rho 0.02 --num_batches 10
```

**When to use**: Demonstrating efficiency gains

---

#### `visualize_spiking.py`
**Purpose**: Generate visualizations of spiking behavior  
**Size**: ~380 lines  
**Generates**:
- `precision_vs_iteration.png`: Overall precision over time
- `layerwise_precision_heatmap.png`: Per-layer patterns
- `precision_vs_rho.png`: Cost vs threshold curve
- `loss_vs_iteration.png`: Attack loss curves
- `layerwise_precision_distribution.png`: Final iteration stats
- `spiking_summary.png`: Comprehensive summary

**Usage**:
```bash
python visualize_spiking.py --steps 20 --num_samples 100
```

**When to use**: Understanding mechanism behavior

---

#### `test_quant.py` [LEGACY]
**Purpose**: Original quantization-based attack  
**Size**: ~330 lines  
**Status**: Preserved for reference and comparison

**When to use**: Comparing with old approach

---

### Shell Scripts

#### `run_attack.sh`
**Purpose**: Run multiple attack experiments  
**Experiments**:
1. Vary rho (0.01, 0.02, 0.03)
2. Vary epsilon (4, 8, 16)
3. Vary steps (10, 20, 50)

**Usage**:
```bash
bash run_attack.sh
```

**When to use**: Systematic evaluation

---

#### `run_train.sh`
**Purpose**: Run multiple training experiments  
**Experiments**:
1. Constant threshold schedules
2. Exponential decay schedules (λ=3,5,8)
3. Standard PGD baseline

**Usage**:
```bash
bash run_train.sh
```

**When to use**: Adversarial training experiments

---

#### `run.sh` [LEGACY]
**Purpose**: Original quantization experiment script  
**Status**: Preserved for reference

---

### Documentation

#### `README.md`
**Purpose**: Main documentation  
**Size**: ~500 lines  
**Contents**:
- Overview and motivation
- Installation and quick start
- Repository structure
- Key components explanation
- Expected results
- Methodology deep dive
- Tuning guide
- Citation

**When to read**: First time using repository

---

#### `QUICK_START.md`
**Purpose**: 5-minute getting started guide  
**Size**: ~300 lines  
**Contents**:
- Installation (1 min)
- First attack (2 min)
- Comparison (2 min)
- Key parameters explained
- Common use cases
- Troubleshooting

**When to read**: Want to get started quickly

---

#### `ARCHITECTURE.md`
**Purpose**: Technical architecture documentation  
**Size**: ~600 lines  
**Contents**:
- Module hierarchy
- Core components detailed explanation
- Attack pipeline
- Gradient flow diagrams
- Memory management
- Performance characteristics
- Extension points
- Debugging tips

**When to read**: Understanding implementation details

---

#### `SUMMARY.md`
**Purpose**: Comparison between quantization and spiking  
**Size**: ~450 lines  
**Contents**:
- What changed
- File structure changes
- Implementation comparison
- Parameter changes
- Conceptual mapping
- Migration guide
- Advantages of spiking approach

**When to read**: Migrating from quantization approach

---

#### `CHANGELOG.md`
**Purpose**: Version history and changes  
**Size**: ~400 lines  
**Contents**:
- [2.0.0] Major rewrite (quantization → spiking)
- [1.0.0] Original quantization version
- Detailed change lists
- Migration instructions

**When to read**: Understanding what changed

---

#### `INDEX.md`
**Purpose**: This file - repository navigation  
**Size**: You're reading it!  
**Contents**:
- Complete file listing
- Purpose and size of each file
- When to read/use each file

**When to read**: Finding specific files

---

### Model Architectures (`models/`)

#### `models/resnet.py`
**Purpose**: ResNet variants for CIFAR and ImageNet  
**Models**: ResNet-18, ResNet-34, ResNet-50

---

#### `models/mobilenetv2.py`
**Purpose**: MobileNetV2 architecture

---

#### `models/mnasnet.py`
**Purpose**: MnasNet architecture

---

#### `models/regnet.py`
**Purpose**: RegNet architecture

---

#### `models/utils.py`
**Purpose**: Utility functions for models

---

#### `models/__init__.py`
**Purpose**: Model exports

---

### Legacy Quantization (`quant/`)

All files in this directory are from the original quantization-based approach and are preserved for reference.

#### `quant/quant_layer.py`
**Purpose**: Quantized layer implementation  
**Key Classes**: `QuantModule`, `UniformAffineQuantizer`

---

#### `quant/quant_model.py`
**Purpose**: Quantized model wrapper  
**Key Classes**: `QuantModel`

---

#### `quant/quant_block.py`
**Purpose**: Quantized blocks (e.g., ResNet blocks)

---

#### `quant/fold_bn.py`
**Purpose**: Fold batch normalization into conv layers

---

#### `quant/adaptive_rounding.py`
**Purpose**: Adaptive rounding for quantization

---

#### `quant/layer_recon.py`
**Purpose**: Layer-wise reconstruction

---

#### `quant/block_recon.py`
**Purpose**: Block-wise reconstruction

---

#### `quant/data_utils.py`
**Purpose**: Data utilities

---

#### `quant/__init__.py`
**Purpose**: Quantization module exports

---

## 🗺️ Navigation Guide

### By Task

| Task | Files to Read |
|------|--------------|
| **First time user** | QUICK_START.md → README.md |
| **Run attack** | spiking_attack.py, run_attack.sh |
| **Train model** | spiking_train.py, run_train.sh |
| **Understand mechanism** | spiking/spiking_layer.py → ARCHITECTURE.md |
| **Migrate from quant** | SUMMARY.md → spiking/spiking_model.py |
| **Debug issues** | ARCHITECTURE.md (Debugging section) |
| **Extend code** | ARCHITECTURE.md (Extension section) |
| **Visualize results** | visualize_spiking.py |
| **Compare methods** | compare_with_baseline.py |

### By Expertise Level

**Beginner** (just want to use it):
1. QUICK_START.md
2. spiking_attack.py
3. README.md

**Intermediate** (want to understand):
1. README.md
2. spiking/spiking_layer.py
3. ARCHITECTURE.md

**Advanced** (want to extend):
1. ARCHITECTURE.md
2. All files in spiking/
3. SUMMARY.md (for comparison context)

---

## 📊 File Statistics

| Category | Count | Total Lines |
|----------|-------|-------------|
| Core Implementation | 3 | ~430 |
| Main Scripts | 5 | ~1,620 |
| Shell Scripts | 3 | ~150 |
| Documentation | 6 | ~2,650 |
| Models | 5 | ~800 |
| Legacy Quant | 9 | ~1,500 |
| **Total** | **31** | **~7,150** |

---

## 🔍 Search Index

### By Keyword

- **Adaptive reuse**: spiking_layer.py (should_recompute)
- **Virtual gradient**: spiking_layer.py (virtual_surrogate_hook)
- **Threshold scheduling**: spiking_model.py (ThresholdScheduler)
- **Precision tracking**: spiking_layer.py (get_precision)
- **Attack evaluation**: spiking_attack.py
- **Adversarial training**: spiking_train.py
- **Comparison**: compare_with_baseline.py, SUMMARY.md
- **Visualization**: visualize_spiking.py
- **Migration guide**: SUMMARY.md
- **Debugging**: ARCHITECTURE.md
- **Extension**: ARCHITECTURE.md

---

## 📞 Getting Help

1. **Quick answer**: Check QUICK_START.md
2. **Detailed explanation**: Check README.md
3. **Technical details**: Check ARCHITECTURE.md
4. **Comparison with old code**: Check SUMMARY.md
5. **Can't find something**: Use this INDEX.md

---

**Last Updated**: 2025-11-22  
**Repository Version**: 2.0.0  
**Total Files**: 31

