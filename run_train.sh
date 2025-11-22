#!/bin/bash
# Example script for adversarial training with Spiking-PGD

# Experiment 1: Constant threshold schedule
echo "=========================================="
echo "Experiment 1: Constant threshold schedule"
echo "=========================================="

# Constant rho = 0.02
python spiking_train.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --rho_initial 0.02 \
    --schedule constant \
    --exp_name spiking_at_constant_0.02

# Constant rho = 0.03
python spiking_train.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --rho_initial 0.03 \
    --schedule constant \
    --exp_name spiking_at_constant_0.03

# Experiment 2: Exponential decay schedule (RECOMMENDED)
echo "=========================================="
echo "Experiment 2: Exponential decay schedule"
echo "=========================================="

# Lambda = 3.0 (slower decay)
python spiking_train.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --rho_initial 0.1 \
    --schedule exponential \
    --lambda_decay 3.0 \
    --exp_name spiking_at_exp_lambda3

# Lambda = 5.0 (balanced, RECOMMENDED)
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
    --exp_name spiking_at_exp_lambda5

# Lambda = 8.0 (faster decay)
python spiking_train.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --rho_initial 0.1 \
    --schedule exponential \
    --lambda_decay 8.0 \
    --exp_name spiking_at_exp_lambda8

# Experiment 3: Standard PGD-AT baseline (no spiking, rho=0)
echo "=========================================="
echo "Experiment 3: Standard PGD-AT baseline"
echo "=========================================="

python spiking_train.py \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --epsilon 8 \
    --steps 10 \
    --step_size 2 \
    --rho_initial 0.0 \
    --schedule constant \
    --exp_name standard_pgd_at_baseline

echo "All training experiments completed!"

