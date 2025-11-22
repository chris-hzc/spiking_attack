#!/bin/bash
# Example script for running Spiking-PGD attack evaluation

# Experiment 1: Standard attack with different rho values
echo "=========================================="
echo "Experiment 1: Varying rho (threshold)"
echo "=========================================="

# rho = 0.01 (minimal reuse, ~99% computation)
python spiking_attack.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epsilon 8 \
    --steps 20 \
    --step_size 2 \
    --rho 0.01 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# rho = 0.02 (balanced, ~65% computation) - RECOMMENDED
python spiking_attack.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epsilon 8 \
    --steps 20 \
    --step_size 2 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# rho = 0.03 (aggressive reuse, ~32% computation)
python spiking_attack.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epsilon 8 \
    --steps 20 \
    --step_size 2 \
    --rho 0.03 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# Experiment 2: Different attack budgets
echo "=========================================="
echo "Experiment 2: Varying epsilon (attack budget)"
echo "=========================================="

# Epsilon = 4/255
python spiking_attack.py \
    --dataset cifar10 \
    --epsilon 4 \
    --steps 20 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# Epsilon = 8/255 (standard)
python spiking_attack.py \
    --dataset cifar10 \
    --epsilon 8 \
    --steps 20 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# Epsilon = 16/255
python spiking_attack.py \
    --dataset cifar10 \
    --epsilon 16 \
    --steps 20 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# Experiment 3: Different number of attack steps
echo "=========================================="
echo "Experiment 3: Varying attack steps"
echo "=========================================="

# 10 steps
python spiking_attack.py \
    --dataset cifar10 \
    --steps 10 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# 20 steps (standard)
python spiking_attack.py \
    --dataset cifar10 \
    --steps 20 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

# 50 steps
python spiking_attack.py \
    --dataset cifar10 \
    --steps 50 \
    --rho 0.02 \
    --file_name pgd_adversarial_training \
    --epoch_eval 150

echo "All experiments completed!"

