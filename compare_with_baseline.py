"""
Comparison script: Spiking-PGD vs Standard PGD

This script runs both Spiking-PGD and standard PGD attacks on the same model
and compares their effectiveness and computational cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
import argparse
import time
import numpy as np

from models import *
from spiking import SpikingModel

# ============================================================================
# Arguments
# ============================================================================
parser = argparse.ArgumentParser(description='Compare Spiking-PGD with Standard PGD')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--file_name', default='pgd_adversarial_training', type=str)
parser.add_argument('--epoch_eval', default=150, type=int)
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--steps', default=20, type=int)
parser.add_argument('--step_size', default=2, type=int)
parser.add_argument('--rho', default=0.02, type=float)
parser.add_argument('--num_batches', default=10, type=int, 
                    help='number of batches to test (for quick comparison)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Load Data and Model
# ============================================================================
transform_test = transforms.Compose([transforms.ToTensor()])

if args.dataset == 'cifar10':
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    num_classes = 10
elif args.dataset == 'cifar100':
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    num_classes = 100

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, num_workers=4
)

# Load model
net = resnet18(num_classes=num_classes)
net = net.to(device)
net.eval()

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

checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])

# Create spiking model
spiking_model = SpikingModel(model=copy.deepcopy(net), rho=args.rho)
spiking_model.to(device)
spiking_model.eval()

# ============================================================================
# Standard PGD Attack
# ============================================================================
class StandardPGD:
    def __init__(self, model, epsilon, k, alpha):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
    
    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        
        for _ in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = x.grad
            
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), 
                         x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        return x

# ============================================================================
# Spiking PGD Attack
# ============================================================================
class SpikingPGD:
    def __init__(self, model, epsilon, k, alpha):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
    
    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        
        self.model.reset_state()
        self.model.reset_precision_tracking()
        self.model.set_spiking_state(True)
        
        for _ in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = x.grad
            
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), 
                         x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        return x

# ============================================================================
# Comparison
# ============================================================================
epsilon = args.epsilon / 255
alpha = args.step_size / 255

standard_pgd = StandardPGD(net, epsilon, args.steps, alpha)
spiking_pgd = SpikingPGD(spiking_model, epsilon, args.steps, alpha)

print('\n' + '='*70)
print('COMPARISON: STANDARD PGD vs SPIKING-PGD')
print('='*70)
print(f'Dataset: {args.dataset}')
print(f'Model: {args.arch}')
print(f'Epsilon: {args.epsilon}/255')
print(f'Steps: {args.steps}')
print(f'Spiking threshold (rho): {args.rho}')
print(f'Testing on {args.num_batches} batches')
print('='*70 + '\n')

criterion = nn.CrossEntropyLoss()

# Statistics
stats = {
    'standard': {'correct': 0, 'total': 0, 'time': []},
    'spiking': {'correct': 0, 'total': 0, 'time': [], 'precision': []}
}

for batch_idx, (inputs, targets) in enumerate(test_loader):
    if batch_idx >= args.num_batches:
        break
    
    inputs, targets = inputs.to(device), targets.to(device)
    
    print(f'\nBatch {batch_idx + 1}/{args.num_batches}')
    print('-' * 70)
    
    # Standard PGD
    start_time = time.time()
    adv_standard = standard_pgd.perturb(inputs, targets)
    standard_time = time.time() - start_time
    
    with torch.no_grad():
        outputs = net(adv_standard)
        _, predicted = outputs.max(1)
        standard_correct = predicted.eq(targets).sum().item()
    
    stats['standard']['correct'] += standard_correct
    stats['standard']['total'] += targets.size(0)
    stats['standard']['time'].append(standard_time)
    
    print(f'Standard PGD: Accuracy={standard_correct/targets.size(0)*100:.2f}%, '
          f'Time={standard_time:.3f}s')
    
    # Spiking PGD
    start_time = time.time()
    adv_spiking = spiking_pgd.perturb(inputs, targets)
    spiking_time = time.time() - start_time
    
    precision = spiking_model.get_precision()['overall']
    
    spiking_model.set_spiking_state(False)
    spiking_model.reset_state()
    with torch.no_grad():
        outputs = spiking_model(adv_spiking)
        _, predicted = outputs.max(1)
        spiking_correct = predicted.eq(targets).sum().item()
    
    stats['spiking']['correct'] += spiking_correct
    stats['spiking']['total'] += targets.size(0)
    stats['spiking']['time'].append(spiking_time)
    stats['spiking']['precision'].append(precision)
    
    print(f'Spiking PGD:  Accuracy={spiking_correct/targets.size(0)*100:.2f}%, '
          f'Time={spiking_time:.3f}s, Precision={precision*100:.2f}%')
    print(f'Speedup: {standard_time/spiking_time:.2f}x')

# ============================================================================
# Final Summary
# ============================================================================
print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

standard_acc = 100. * stats['standard']['correct'] / stats['standard']['total']
spiking_acc = 100. * stats['spiking']['correct'] / stats['spiking']['total']
standard_time = np.mean(stats['standard']['time'])
spiking_time = np.mean(stats['spiking']['time'])
avg_precision = np.mean(stats['spiking']['precision']) * 100

print(f'\nStandard PGD:')
print(f'  - Robust Accuracy: {standard_acc:.2f}%')
print(f'  - Average Time: {standard_time:.3f}s per batch')
print(f'  - Computational Cost: 100.0%')

print(f'\nSpiking-PGD (rho={args.rho}):')
print(f'  - Robust Accuracy: {spiking_acc:.2f}%')
print(f'  - Average Time: {spiking_time:.3f}s per batch')
print(f'  - Computational Cost: {avg_precision:.2f}%')

print(f'\nComparison:')
print(f'  - Accuracy Difference: {spiking_acc - standard_acc:+.2f}%')
print(f'  - Speedup: {standard_time/spiking_time:.2f}x')
print(f'  - Cost Reduction: {100 - avg_precision:.2f}%')

print('\n' + '='*70)

# Effectiveness-Efficiency Trade-off
if abs(spiking_acc - standard_acc) <= 1.0:
    print('✓ Spiking-PGD maintains attack effectiveness (< 1% difference)')
else:
    print(f'⚠ Spiking-PGD has {abs(spiking_acc - standard_acc):.2f}% accuracy difference')

if avg_precision < 70:
    print(f'✓ Spiking-PGD significantly reduces cost ({100-avg_precision:.1f}% reduction)')
else:
    print(f'⚠ Limited cost reduction ({100-avg_precision:.1f}%)')

print('='*70 + '\n')

