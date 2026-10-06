#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
# Experiment script for ResNet50 with Spiking-PGD on CIFAR-10
# Uses pretrained weights from https://github.com/huyvnphan/PyTorch_CIFAR10

echo "=========================================="
echo "ResNet50 Spiking-PGD Experiments"
echo "Using pretrained weights from PyTorch_CIFAR10"
echo "=========================================="
echo ""

# Check if cifar10_models is installed
python -c "import cifar10_models" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  cifar10_models package not found!"
    echo ""
    echo "Installing from GitHub repository..."
    python download_pretrained_resnet50.py
    
    # Check again
    python -c "import cifar10_models" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ Installation failed!"
        echo "Please install manually:"
        echo "  pip install git+https://github.com/huyvnphan/PyTorch_CIFAR10.git"
        exit 1
    fi
fi

echo "✓ cifar10_models package found"
echo ""

# Experiment 1: Test different rho values
echo "=========================================="
echo "Experiment 1: Varying rho (threshold)"
echo "=========================================="

# rho = 0.01 (minimal reuse, high cost)
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.01 \
    --epsilon 8 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# rho = 0.02 (balanced) - RECOMMENDED
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# rho = 0.03 (aggressive reuse, low cost)
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.03 \
    --epsilon 8 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# rho = 0.05 (very aggressive)
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.05 \
    --epsilon 8 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# Experiment 2: Test different attack budgets
echo ""
echo "=========================================="
echo "Experiment 2: Varying epsilon (attack budget)"
echo "=========================================="

# Small perturbation
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 4 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# Standard perturbation
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# Large perturbation
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 16 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# Experiment 3: Test different attack steps
echo ""
echo "=========================================="
echo "Experiment 3: Varying attack steps"
echo "=========================================="

# Quick attack
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 10 \
    --num_samples 1000 \
    --save_log

# Standard attack
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 20 \
    --num_samples 1000 \
    --save_log

# Strong attack
python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 50 \
    --num_samples 1000 \
    --save_log

# Full test set evaluation (optional, takes longer)
echo ""
echo "=========================================="
echo "Experiment 4: Full test set (optional)"
echo "=========================================="
echo "Testing on full 10000 samples with rho=0.02..."

python test_resnet50_spiking.py \
    --use_pretrained 1 \
    --rho 0.02 \
    --epsilon 8 \
    --steps 20 \
    --save_log

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in: log/resnet50_spiking/"

