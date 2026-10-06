"""
Test Spiking-PGD Attack on ResNet50 with CIFAR-10
Using Pretrained Model from https://github.com/huyvnphan/PyTorch_CIFAR10

This script evaluates ResNet50 robustness under Spiking-PGD attack with
adaptive activation reuse mechanism using publicly available pretrained weights.
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import sys
import numpy as np
import argparse
import time
from tqdm import tqdm

from spiking import SpikingModel

# ============================================================================
# Configuration
# ============================================================================
parser = argparse.ArgumentParser(
    description='Test Spiking-PGD on ResNet50 (CIFAR-10) with pretrained weights',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Model and dataset
parser.add_argument('--use_pretrained', default=1, type=int,
                    help='use pretrained weights from PyTorch_CIFAR10 repo (1=yes, 0=no)')
parser.add_argument('--batch_size', default=100, type=int,
                    help='test batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of data loading workers')

# Spiking attack parameters
parser.add_argument('--rho', default=0.02, type=float,
                    help='threshold for relative activation change')
parser.add_argument('--epsilon', default=8, type=int,
                    help='attack epsilon in range [0, 255]')
parser.add_argument('--steps', default=20, type=int,
                    help='number of PGD attack steps')
parser.add_argument('--step_size', default=2, type=int,
                    help='step size per iteration in range [0, 255]')
parser.add_argument('--random_start', default=1, type=int,
                    help='use random initialization (1=yes, 0=no)')

# Evaluation settings
parser.add_argument('--num_samples', default=None, type=int,
                    help='number of samples to test (default: all 10000)')
parser.add_argument('--verbose', action='store_true',
                    help='print detailed per-batch results')
parser.add_argument('--save_log', action='store_true',
                    help='save results to log file')

args = parser.parse_args()

# ============================================================================
# Setup
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Data transforms (matching PyTorch_CIFAR10 preprocessing)
# Images are normalized with CIFAR-10 mean and std
# Reference: https://github.com/huyvnphan/PyTorch_CIFAR10
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2471, 0.2435, 0.2616]
    )
])

# Load CIFAR-10 test set
print('Loading CIFAR-10 test dataset...')
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=args.batch_size,
    shuffle=False, 
    num_workers=args.num_workers
)

print(f'Test dataset size: {len(test_dataset)}')

# ============================================================================
# Load Pretrained ResNet50 Model
# ============================================================================
print('\n' + '='*70)
print('Loading Pretrained ResNet50 Model from PyTorch_CIFAR10')
print('='*70)

if args.use_pretrained:
    print('Attempting to load pretrained weights from huyvnphan/PyTorch_CIFAR10...')
    print('Source: https://github.com/huyvnphan/PyTorch_CIFAR10')
    
    try:
        # Try to import from cifar10_models package
        from cifar10_models.resnet import resnet50
        net = resnet50(pretrained=True)
        print('✓ Successfully loaded pretrained ResNet50 from cifar10_models')
        print('  Model accuracy on CIFAR-10: ~93.65% (from repository)')
        
    except ImportError:
        print('\n⚠️  cifar10_models package not found!')
        print('Installing from GitHub repository...')
        
        import subprocess
        try:
            # Install the package
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'git+https://github.com/huyvnphan/PyTorch_CIFAR10.git'
            ])
            print('✓ Installation completed!')
            
            # Import again after installation
            from cifar10_models.resnet import resnet50
            net = resnet50(pretrained=True)
            print('✓ Successfully loaded pretrained ResNet50')
            
        except Exception as e:
            print(f'\n✗ Failed to install cifar10_models: {e}')
            print('\nFalling back to untrained model...')
            from models import resnet50 as local_resnet50
            net = local_resnet50(num_classes=10)
            print('⚠️  Using UNTRAINED model - results will not be meaningful!')
else:
    # Use local model without pretrained weights
    print('Using local ResNet50 model without pretrained weights...')
    from models import resnet50 as local_resnet50
    net = local_resnet50(num_classes=10)
    print('⚠️  Model is UNTRAINED - results will not be meaningful!')

net = net.to(device)
net.eval()
cudnn.benchmark = True

print('✓ Model setup completed!')

# ============================================================================
# Wrap with SpikingModel
# ============================================================================
print('\n' + '='*70)
print('Creating Spiking Model')
print('='*70)

spiking_model = SpikingModel(model=net, rho=args.rho)
spiking_model.to(device)
spiking_model.eval()

print(f'Threshold (rho): {args.rho}')
print('✓ Spiking model created!')

# ============================================================================
# Spiking-PGD Attack Implementation
# ============================================================================
class SpikingPGDAttack:
    """
    Spiking-PGD attack for ResNet50 evaluation.
    """
    
    def __init__(self, model, epsilon, k, alpha, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start

    def perturb(self, x_natural, y, verbose=False):
        """
        Generate adversarial examples.
        
        Args:
            x_natural: Clean images [batch_size, 3, 32, 32]
            y: True labels [batch_size]
            verbose: Print per-iteration info
            
        Returns:
            x_adv: Adversarial examples
        """
        x = x_natural.detach()
        
        # Random initialization
        if self.random_start:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        # Reset model state
        self.model.reset_state()
        self.model.reset_precision_tracking()
        self.model.set_spiking_state(True)
        
        # Iterative attack
        for i in range(self.k):
            x.requires_grad_()
            
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            
            if verbose and (i == 0 or i == self.k - 1 or (i + 1) % 5 == 0):
                print(f'  Iteration {i+1}/{self.k}: Loss = {loss.item():.4f}')
            
            # Compute gradient
            loss.backward()
            grad = x.grad
            
            # PGD update (in normalized space)
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), 
                         x_natural + self.epsilon)
            
            # Note: We keep x in normalized space [-inf, inf] during attack
            # The normalization bounds are not clipped to [0,1]
        
        return x


# ============================================================================
# Create Attacker
# ============================================================================
print('\n' + '='*70)
print('Creating Spiking-PGD Attacker')
print('='*70)

epsilon = args.epsilon / 255.0
alpha = args.step_size / 255.0

adversary = SpikingPGDAttack(
    model=spiking_model,
    epsilon=epsilon,
    k=args.steps,
    alpha=alpha,
    random_start=bool(args.random_start)
)

print(f'Attack Configuration:')
print(f'  - Epsilon: {args.epsilon}/255 = {epsilon:.4f}')
print(f'  - Steps: {args.steps}')
print(f'  - Step size: {args.step_size}/255 = {alpha:.4f}')
print(f'  - Random start: {bool(args.random_start)}')
print(f'  - Note: Attack performed in normalized image space')
print('✓ Attacker created!')

# ============================================================================
# Evaluation Function
# ============================================================================
def evaluate():
    """
    Evaluate ResNet50 under Spiking-PGD attack.
    """
    print('\n' + '='*70)
    print('EVALUATION START')
    print('='*70)
    print(f'Model: ResNet50 (Pretrained on CIFAR-10)')
    print(f'Source: https://github.com/huyvnphan/PyTorch_CIFAR10')
    print(f'Dataset: CIFAR-10 Test Set (10,000 images)')
    print(f'Attack: Spiking-PGD')
    print(f'Threshold (rho): {args.rho}')
    print('='*70 + '\n')
    
    spiking_model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Statistics
    benign_correct = 0
    adv_correct = 0
    total = 0
    
    all_precisions = []
    all_losses = []
    batch_times = []
    
    # Progress bar
    num_batches = len(test_loader)
    if args.num_samples:
        num_batches = min(num_batches, (args.num_samples + args.batch_size - 1) // args.batch_size)
    
    pbar = tqdm(enumerate(test_loader), total=num_batches, 
                desc='Testing', unit='batch')
    
    for batch_idx, (inputs, targets) in pbar:
        # Limit number of samples if specified
        if args.num_samples and total >= args.num_samples:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        batch_total = targets.size(0)
        
        # ====== Benign Accuracy ======
        spiking_model.set_spiking_state(False)
        with torch.no_grad():
            outputs = spiking_model(inputs)
        
        _, predicted = outputs.max(1)
        benign_batch_correct = predicted.eq(targets).sum().item()
        benign_correct += benign_batch_correct
        
        # ====== Generate Adversarial Examples ======
        start_time = time.time()
        
        verbose_batch = args.verbose and (batch_idx % 10 == 0)
        if verbose_batch:
            print(f'\n--- Batch {batch_idx} ---')
        
        adv = adversary.perturb(inputs, targets, verbose=verbose_batch)
        
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        # Get precision
        precision_stats = spiking_model.get_precision()
        precision = precision_stats['overall']
        all_precisions.append(precision)
        
        # ====== Adversarial Accuracy ======
        spiking_model.set_spiking_state(False)
        spiking_model.reset_state()
        
        with torch.no_grad():
            adv_outputs = spiking_model(adv)
            loss = criterion(adv_outputs, targets)
        
        _, predicted = adv_outputs.max(1)
        adv_batch_correct = predicted.eq(targets).sum().item()
        adv_correct += adv_batch_correct
        
        all_losses.append(loss.item())
        total += batch_total
        
        # Update progress bar
        pbar.set_postfix({
            'Benign': f'{100.*benign_correct/total:.1f}%',
            'Robust': f'{100.*adv_correct/total:.1f}%',
            'Prec': f'{precision*100:.1f}%'
        })
        
        # Verbose output
        if verbose_batch:
            print(f'Benign Acc: {benign_batch_correct/batch_total*100:.2f}%')
            print(f'Robust Acc: {adv_batch_correct/batch_total*100:.2f}%')
            print(f'Precision: {precision*100:.2f}%')
            print(f'Time: {batch_time:.3f}s')
    
    pbar.close()
    
    # ====== Final Results ======
    benign_acc = 100. * benign_correct / total
    robust_acc = 100. * adv_correct / total
    attack_success_rate = 100. * (1 - adv_correct / total)
    avg_precision = np.mean(all_precisions) * 100
    avg_loss = np.mean(all_losses)
    avg_time = np.mean(batch_times)
    
    print('\n' + '='*70)
    print('FINAL RESULTS')
    print('='*70)
    print(f'Model: ResNet50 on CIFAR-10')
    print(f'Total samples tested: {total}')
    print(f'\nAccuracy:')
    print(f'  - Benign accuracy: {benign_acc:.2f}%')
    print(f'  - Robust accuracy (under attack): {robust_acc:.2f}%')
    print(f'  - Attack success rate: {attack_success_rate:.2f}%')
    print(f'\nComputational Cost:')
    print(f'  - Average precision: {avg_precision:.2f}%')
    print(f'  - Cost reduction: {100 - avg_precision:.2f}%')
    print(f'\nPerformance:')
    print(f'  - Average attack loss: {avg_loss:.4f}')
    print(f'  - Average time per batch: {avg_time:.3f}s')
    print(f'  - Total time: {sum(batch_times):.2f}s')
    
    # Layer-wise precision (sample from last batch)
    if len(precision_stats['layer_wise']) > 0:
        print(f'\nLayer-wise Precision (last batch):')
        layer_precisions = [f'{p*100:.1f}%' for p in precision_stats['layer_wise']]
        print(f'  {layer_precisions}')
    
    print('='*70 + '\n')
    
    # Save results
    results = {
        'model': 'ResNet50',
        'dataset': 'CIFAR-10',
        'checkpoint': args.checkpoint,
        'attack': 'Spiking-PGD',
        'rho': args.rho,
        'epsilon': args.epsilon,
        'steps': args.steps,
        'samples_tested': total,
        'benign_accuracy': benign_acc,
        'robust_accuracy': robust_acc,
        'attack_success_rate': attack_success_rate,
        'avg_precision': avg_precision,
        'cost_reduction': 100 - avg_precision,
        'avg_loss': avg_loss,
        'avg_time_per_batch': avg_time
    }
    
    return results


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    # Setup logging if requested
    if args.save_log:
        log_dir = 'log/resnet50_spiking'
        os.makedirs(log_dir, exist_ok=True)
        
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = (f'{log_dir}/resnet50_rho{args.rho}_eps{args.epsilon}_'
                   f'steps{args.steps}_{time_str}.log')
        
        # Redirect stdout to log file
        sys.stdout = open(log_file, 'w', buffering=1)
        print(f'Log file: {log_file}\n')
    
    # Print configuration
    print('='*70)
    print('CONFIGURATION')
    print('='*70)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('='*70 + '\n')
    
    # Run evaluation
    try:
        results = evaluate()
        
        # Print summary
        print('\n' + '='*70)
        print('SUMMARY')
        print('='*70)
        print(f'ResNet50 on CIFAR-10 with Spiking-PGD (rho={args.rho}):')
        print(f'  Benign: {results["benign_accuracy"]:.2f}%')
        print(f'  Robust: {results["robust_accuracy"]:.2f}%')
        print(f'  Precision: {results["avg_precision"]:.2f}%')
        print(f'  Cost Reduction: {results["cost_reduction"]:.2f}%')
        print('='*70)
        
        # Success message
        print('\n✓ Evaluation completed successfully!')
        
    except KeyboardInterrupt:
        print('\n\nEvaluation interrupted by user.')
    except Exception as e:
        print(f'\n\nError during evaluation: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if args.save_log:
            sys.stdout.close()
            print(f'\nResults saved to: {log_file}')

