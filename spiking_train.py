"""
Adversarial Training with Spiking-PGD
Based on the paper: Fine-Grained Iterative Adversarial Attacks WITH LIMITED COMPUTATION BUDGET

This script implements adversarial training using Spiking-PGD to generate adversarial
examples during training, with support for threshold scheduling to balance computational
cost and robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import sys
import argparse
import time
import numpy as np

from models import *
from spiking import SpikingModel, ThresholdScheduler

time_str = time.strftime('%Y-%m-%d-%H-%M')

# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser(
    description='Adversarial Training with Spiking-PGD',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Dataset and model
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'],
                    help='dataset')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='model architecture')

# Training hyperparameters
parser.add_argument('--epochs', default=200, type=int,
                    help='number of training epochs')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['cosine', 'step'],
                    help='learning rate schedule')

# Attack parameters for training
parser.add_argument('--epsilon', default=8, type=int,
                    help='attack epsilon (L-inf bound) in range [0, 255]')
parser.add_argument('--steps', default=10, type=int,
                    help='number of PGD attack steps during training')
parser.add_argument('--step_size', default=2, type=int,
                    help='step size for each attack iteration')

# Spiking parameters
parser.add_argument('--rho_initial', default=0.1, type=float,
                    help='initial threshold for relative activation change')
parser.add_argument('--schedule', default='exponential', type=str,
                    choices=['constant', 'exponential'],
                    help='threshold scheduling strategy')
parser.add_argument('--lambda_decay', default=5.0, type=float,
                    help='decay rate for exponential schedule')

# Experiment settings
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--save_freq', default=50, type=int,
                    help='checkpoint saving frequency (epochs)')
parser.add_argument('--exp_name', default='spiking_at', type=str,
                    help='experiment name')

args = parser.parse_args()

# ============================================================================
# Device and Data Setup
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
if args.dataset == 'cifar10':
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    num_classes = 10
elif args.dataset == 'cifar100':
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=args.num_workers
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100,
    shuffle=False, num_workers=args.num_workers
)

# ============================================================================
# Model, Optimizer, Scheduler Setup
# ============================================================================
print('Building model...')
net = resnet18(num_classes=num_classes)
net = net.to(device)
cudnn.benchmark = True

# Create spiking model for adversarial training
spiking_model = SpikingModel(model=net, rho=args.rho_initial)
spiking_model.to(device)

# Threshold scheduler
threshold_scheduler = ThresholdScheduler(
    schedule_type=args.schedule,
    rho_initial=args.rho_initial,
    total_epochs=args.epochs,
    lambda_decay=args.lambda_decay
)

# Optimizer
optimizer = optim.SGD(
    spiking_model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)

# Learning rate scheduler
if args.lr_schedule == 'cosine':
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
elif args.lr_schedule == 'step':
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

criterion = nn.CrossEntropyLoss()

# ============================================================================
# Spiking-PGD for Training
# ============================================================================
class SpikingPGDTraining:
    """Spiking-PGD attack for adversarial training"""
    
    def __init__(self, model, epsilon=8/255, k=10, alpha=2/255):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha

    def perturb(self, x_natural, y):
        """Generate adversarial examples during training"""
        x = x_natural.detach()
        
        # Random initialization
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, 0, 1)
        
        # Reset model state
        self.model.reset_state()
        
        # Enable spiking mechanism
        self.model.set_spiking_state(True)
        
        for _ in range(self.k):
            x.requires_grad_()
            
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            
            loss.backward()
            grad = x.grad
            
            # PGD update
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon),
                         x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        return x.detach()


adversary = SpikingPGDTraining(
    model=spiking_model,
    epsilon=args.epsilon / 255,
    k=args.steps,
    alpha=args.step_size / 255
)

# ============================================================================
# Training and Testing Functions
# ============================================================================
def train(epoch):
    """Train for one epoch"""
    print(f'\n[ Epoch {epoch+1}/{args.epochs} - Training ]')
    
    spiking_model.train()
    
    # Update threshold based on schedule
    rho = threshold_scheduler.get_threshold(epoch)
    spiking_model.set_threshold(rho)
    
    train_loss = 0
    correct = 0
    total = 0
    
    precisions = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        spiking_model.reset_precision_tracking()
        adv_inputs = adversary.perturb(inputs, targets)
        
        # Get precision for this batch
        precision = spiking_model.get_precision()['overall']
        precisions.append(precision)
        
        # Disable spiking for training forward pass
        spiking_model.set_spiking_state(False)
        spiking_model.reset_state()
        
        # Forward pass on adversarial examples
        optimizer.zero_grad()
        outputs = spiking_model(adv_inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}/{len(train_loader)} | '
                  f'Loss: {train_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.2f}% | '
                  f'Precision: {np.mean(precisions)*100:.2f}% | '
                  f'Rho: {rho:.4f}')
    
    avg_precision = np.mean(precisions) * 100
    print(f'Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.3f} | '
          f'Train Acc: {100.*correct/total:.2f}% | '
          f'Avg Precision: {avg_precision:.2f}%')
    
    return train_loss/len(train_loader), 100.*correct/total, avg_precision


def test_clean(epoch):
    """Test on clean examples"""
    print(f'\n[ Epoch {epoch+1}/{args.epochs} - Clean Test ]')
    
    spiking_model.eval()
    spiking_model.set_spiking_state(False)
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = spiking_model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f'Clean Test Acc: {acc:.2f}%')
    
    return test_loss/len(test_loader), acc


def test_robust(epoch, steps=20):
    """Test robustness against PGD attack"""
    print(f'\n[ Epoch {epoch+1}/{args.epochs} - Robust Test (PGD-{steps}) ]')
    
    spiking_model.eval()
    
    # Use standard PGD for evaluation (no spiking)
    spiking_model.set_spiking_state(False)
    
    correct = 0
    total = 0
    
    epsilon = args.epsilon / 255
    alpha = args.step_size / 255
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x = torch.clamp(x, 0, 1)
        
        for _ in range(steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = spiking_model(x)
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            grad = x.grad
            
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, 0, 1)
        
        with torch.no_grad():
            outputs = spiking_model(x)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f'Robust Test Acc (PGD-{steps}): {acc:.2f}%')
    
    return acc


# ============================================================================
# Main Training Loop
# ============================================================================
if __name__ == '__main__':
    # Create checkpoint directory
    checkpoint_dir = f'checkpoint/{args.exp_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    log_dir = f'log_train/{args.dataset}/{args.exp_name}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/train_{time_str}.log'
    
    # Print configuration
    print('='*60)
    print('ADVERSARIAL TRAINING WITH SPIKING-PGD')
    print('='*60)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print(f'Threshold Scheduler: {threshold_scheduler}')
    print('='*60 + '\n')
    
    # Training loop
    best_clean_acc = 0
    best_robust_acc = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'clean_acc': [],
        'robust_acc': [],
        'precision': []
    }
    
    for epoch in range(args.epochs):
        # Training
        train_loss, train_acc, precision = train(epoch)
        
        # Testing
        _, clean_acc = test_clean(epoch)
        robust_acc = test_robust(epoch, steps=20)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['clean_acc'].append(clean_acc)
        history['robust_acc'].append(robust_acc)
        history['precision'].append(precision)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            state = {
                'net': spiking_model.model.state_dict(),
                'epoch': epoch,
                'clean_acc': clean_acc,
                'robust_acc': robust_acc,
                'history': history
            }
            checkpoint_path = f'{checkpoint_dir}/epoch_{epoch+1}.pth'
            torch.save(state, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        
        # Track best accuracies
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
        
        print(f'\nBest Clean Acc: {best_clean_acc:.2f}% | '
              f'Best Robust Acc: {best_robust_acc:.2f}%\n')
        print('-'*60)
    
    print('\n' + '='*60)
    print('TRAINING COMPLETED')
    print('='*60)
    print(f'Best Clean Accuracy: {best_clean_acc:.2f}%')
    print(f'Best Robust Accuracy: {best_robust_acc:.2f}%')
    print('='*60)

