"""
Spiking-PGD Attack: Fine-Grained Iterative Adversarial Attacks
Based on the paper: Fine-Grained Iterative Adversarial Attacks WITH LIMITED COMPUTATION BUDGET

This script implements the Spiking-PGD attack algorithm that adaptively reuses
layer activations when relative changes are below a threshold, significantly
reducing computation cost while maintaining attack effectiveness.
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import copy
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import sys
import numpy as np
import argparse
import time

from models import *
from spiking import SpikingModel, ThresholdScheduler

time_str = time.strftime('%Y-%m-%d-%H-%M')

# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser(
    description='Spiking-PGD Attack Evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Dataset and model parameters
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'cifar100', 'imagenet'],
                    help='dataset to use')
parser.add_argument('--arch', default='resnet18', type=str, 
                    help='model architecture')
parser.add_argument('--epoch_eval', default=150, type=int, 
                    help='epoch checkpoint to evaluate')
parser.add_argument('--file_name', default='pgd_adversarial_training', type=str,
                    help='checkpoint file name')

# Spiking attack parameters
parser.add_argument('--rho', default=0.02, type=float,
                    help='threshold for relative activation change (default: 0.02)')
parser.add_argument('--schedule', default='constant', type=str,
                    choices=['constant', 'exponential'],
                    help='threshold scheduling strategy')
parser.add_argument('--lambda_decay', default=5.0, type=float,
                    help='decay rate for exponential schedule')

# Attack parameters
parser.add_argument('--epsilon', default=8, type=int,
                    help='attack epsilon (L-inf bound) in range [0, 255]')
parser.add_argument('--steps', default=20, type=int,
                    help='number of PGD attack steps')
parser.add_argument('--step_size', default=2, type=int,
                    help='step size for each attack iteration (in range [0, 255])')

# Experiment settings
parser.add_argument('--batch_size', default=100, type=int,
                    help='batch size for evaluation')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of data loading workers')

args = parser.parse_args()

# ============================================================================
# Device and Data Setup
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
if args.dataset == 'cifar10':
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
elif args.dataset == 'cifar100':
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
elif args.dataset == 'imagenet':
    _, test_dataset = get_tinyimagenet_loader(
        train_batch_size=128, test_batch_size=args.batch_size
    )

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, 
    shuffle=False, num_workers=args.num_workers
)

# ============================================================================
# Model Setup
# ============================================================================
num_classes_dict = {'cifar10': 10, 'cifar100': 100, 'imagenet': 200}
num_classes = num_classes_dict[args.dataset]

# Create base model
if args.dataset == 'imagenet':
    net = resnet18_imagenet(num_classes=num_classes)
else:
    net = resnet18(num_classes=num_classes)

net = net.to(device)
net.eval()
cudnn.benchmark = True

# Load checkpoint
if args.dataset == 'cifar10':
    if args.file_name == 'pgd_adversarial_training':
        checkpoint_path = (
            './checkpoint_from_scratch/checkpoint_adjustednonedecay0.0_alpha0.0_epsilon8/'
            f'{args.file_name}_{args.epoch_eval}'
        )
    else:
        checkpoint_path = './checkpoint_from_scratch/basic_training'
elif args.dataset == 'cifar100':
    checkpoint_path = (
        f'log_train/cifar100/checkpoint_adjustednonedecay0.0_alpha0.0_epsilon8/'
        f'{args.file_name}_{args.epoch_eval}'
    )
elif args.dataset == 'imagenet':
    checkpoint_path = (
        f'log_train/imagenet/checkpoint_adjustednonedecay0.0_alpha0.0_epsilon8/'
        f'{args.file_name}_{args.epoch_eval}'
    )

checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])

# Create backbone (clean model for transfer attack evaluation)
backbone = copy.deepcopy(net)
backbone.eval()

# Create spiking model for attack
spiking_model = SpikingModel(model=net, rho=args.rho)
spiking_model.to(device)
spiking_model.eval()

# ============================================================================
# Spiking-PGD Attack Implementation
# ============================================================================
class SpikingPGDAttack:
    """
    Spiking-PGD: PGD attack with adaptive activation reuse.
    
    At each iteration:
    1. Enable spiking mechanism for forward and backward pass
    2. For each layer, compute only if relative change >= rho
    3. Otherwise reuse previous activation (forward) and virtual gradient (backward)
    4. Track precision (percentage of full computations)
    
    Args:
        model: SpikingModel wrapped neural network
        epsilon: L-infinity perturbation budget (normalized to [0,1])
        k: Number of PGD iterations
        alpha: Step size for each iteration (normalized to [0,1])
        random_start: Whether to start from random perturbation
    """
    
    def __init__(self, model, epsilon=8/255, k=20, alpha=2/255, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start

    def perturb(self, x_natural, y):
        """
        Generate adversarial examples using Spiking-PGD.
        
        Args:
            x_natural: Clean input images [batch_size, C, H, W]
            y: True labels [batch_size]
            
        Returns:
            x_adv: Adversarial examples [batch_size, C, H, W]
        """
        x = x_natural.detach()
        
        # Random initialization
        if self.random_start:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        # Reset model state before attack
        self.model.reset_state()
        self.model.reset_precision_tracking()
        
        # Enable spiking mechanism
        self.model.set_spiking_state(True)
        
        # Iterative attack
        for i in range(self.k):
            print(f"\n========== Iteration {i+1}/{self.k} ==========")
            
            x.requires_grad_()
            
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            
            print(f"Loss: {loss.item():.4f}")
            
            # Compute gradient
            loss.backward()
            grad = x.grad
            
            # PGD update
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), 
                         x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        # Print precision statistics
        precision_stats = self.model.get_precision()
        print(f"\n{'='*60}")
        print(f"Overall Precision: {precision_stats['overall']*100:.2f}%")
        print(f"Layer-wise Precision: {[f'{p*100:.1f}%' for p in precision_stats['layer_wise']]}")
        print(f"{'='*60}\n")
        
        return x


# ============================================================================
# Create Attacker
# ============================================================================
adversary = SpikingPGDAttack(
    model=spiking_model,
    epsilon=args.epsilon / 255,
    k=args.steps,
    alpha=args.step_size / 255,
    random_start=True
)

criterion = nn.CrossEntropyLoss()

# ============================================================================
# Evaluation Function
# ============================================================================
def test():
    """
    Evaluate model under Spiking-PGD attack.
    
    Metrics:
    1. Benign accuracy (clean images on backbone model)
    2. Transfer attack accuracy (adversarial images on backbone model)
    3. Overall precision (average percentage of full computations)
    """
    print('\n' + '='*60)
    print('SPIKING-PGD ATTACK EVALUATION')
    print('='*60)
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.arch}')
    print(f'Epsilon: {args.epsilon}/255')
    print(f'Steps: {args.steps}')
    print(f'Step Size: {args.step_size}/255')
    print(f'Threshold (rho): {args.rho}')
    print('='*60 + '\n')
    
    spiking_model.eval()
    backbone.eval()
    
    benign_correct = 0
    adv_correct_transfer = 0
    total = 0
    
    all_precisions = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        
        # ====== Benign Accuracy ======
        with torch.no_grad():
            outputs = backbone(inputs)
            loss = criterion(outputs, targets)
        
        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'\n--- Batch {batch_idx} ---')
            print(f'Current benign accuracy: {predicted.eq(targets).sum().item() / targets.size(0) * 100:.2f}%')
        
        # ====== Generate Adversarial Examples ======
        adv = adversary.perturb(inputs, targets)
        
        # Get precision for this batch
        precision = spiking_model.get_precision()['overall']
        all_precisions.append(precision)
        
        # ====== Transfer Attack (evaluate on backbone) ======
        with torch.no_grad():
            adv_outputs = backbone(adv)
            loss = criterion(adv_outputs, targets)
        
        _, predicted = adv_outputs.max(1)
        adv_correct_transfer += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Current transfer adversarial accuracy: {predicted.eq(targets).sum().item() / targets.size(0) * 100:.2f}%')
            print(f'Cumulative benign accuracy: {100. * benign_correct / total:.2f}%')
            print(f'Cumulative transfer adversarial accuracy: {100. * adv_correct_transfer / total:.2f}%')
            print(f'Average precision: {np.mean(all_precisions) * 100:.2f}%')
    
    # ====== Final Results ======
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(f'Total samples: {total}')
    print(f'Benign accuracy: {100. * benign_correct / total:.2f}%')
    print(f'Transfer adversarial accuracy: {100. * adv_correct_transfer / total:.2f}%')
    print(f'Attack success rate: {100. * (1 - adv_correct_transfer / total):.2f}%')
    print(f'Average precision (computational cost): {np.mean(all_precisions) * 100:.2f}%')
    print('='*60 + '\n')
    
    return {
        'benign_acc': 100. * benign_correct / total,
        'adv_acc': 100. * adv_correct_transfer / total,
        'precision': np.mean(all_precisions) * 100
    }

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    # Create log directory
    log_dir = f'log/{args.dataset}/{args.file_name}_{args.epoch_eval}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_file = (f'{log_dir}/spiking_steps{args.steps}_eps{args.epsilon}_'
                f'stepsize{args.step_size}_rho{args.rho}_{time_str}.log')
    sys.stdout = open(log_file, 'w', buffering=1)
    
    # Print configuration
    print('='*60)
    print('CONFIGURATION')
    print('='*60)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('='*60 + '\n')
    
    # Run evaluation
    results = test()
    
    # Close log file
    sys.stdout.close()
    
    print(f'\nResults saved to: {log_file}')

