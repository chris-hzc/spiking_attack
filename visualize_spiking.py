"""
Visualization Script for Spiking Mechanism

This script generates visualizations to understand the spiking behavior:
1. Activation change over iterations
2. Layer-wise precision patterns
3. Threshold sensitivity analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from models import *
from spiking import SpikingModel

# ============================================================================
# Setup
# ============================================================================
parser = argparse.ArgumentParser(description='Visualize Spiking Mechanism')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--file_name', default='pgd_adversarial_training', type=str)
parser.add_argument('--epoch_eval', default=150, type=int)
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--steps', default=20, type=int)
parser.add_argument('--step_size', default=2, type=int)
parser.add_argument('--num_samples', default=100, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
transform_test = transforms.Compose([transforms.ToTensor()])
if args.dataset == 'cifar10':
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    num_classes = 10
    
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.num_samples, shuffle=False
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

checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])

# ============================================================================
# Collect Data for Visualization
# ============================================================================
def run_attack_with_tracking(model, x_natural, y, epsilon, steps, alpha):
    """Run attack and track activation changes"""
    x = x_natural.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x = torch.clamp(x, 0, 1)
    
    model.reset_state()
    model.set_spiking_state(True)
    
    # Track per-iteration data
    iteration_data = []
    
    for i in range(steps):
        # Get all SpikingModule layers
        spiking_layers = [m for m in model.model.modules() 
                         if isinstance(m, type(model.model).__bases__[0])]
        
        # Compute forward
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        
        loss.backward()
        grad = x.grad
        
        # Collect precision data
        precision_stats = model.get_precision()
        
        iteration_data.append({
            'iteration': i,
            'loss': loss.item(),
            'layer_wise_precision': precision_stats['layer_wise'].copy(),
            'overall_precision': precision_stats['overall']
        })
        
        # PGD update
        x = x.detach() + alpha * torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
        x = torch.clamp(x, 0, 1)
    
    return iteration_data

# Get one batch
inputs, targets = next(iter(test_loader))
inputs, targets = inputs.to(device), targets.to(device)

# ============================================================================
# Experiment 1: Track precision over iterations for different rho
# ============================================================================
print("Experiment 1: Precision over iterations for different rho values")

rho_values = [0.01, 0.02, 0.03, 0.05]
epsilon = args.epsilon / 255
alpha = args.step_size / 255

all_results = {}

for rho in rho_values:
    print(f"Testing rho={rho}...")
    spiking_model = SpikingModel(model=net, rho=rho)
    spiking_model.to(device)
    spiking_model.eval()
    
    results = run_attack_with_tracking(
        spiking_model, inputs, targets, epsilon, args.steps, alpha
    )
    all_results[rho] = results

# ============================================================================
# Visualization 1: Overall Precision vs Iteration
# ============================================================================
plt.figure(figsize=(12, 6))

for rho in rho_values:
    iterations = [r['iteration'] for r in all_results[rho]]
    precisions = [r['overall_precision'] * 100 for r in all_results[rho]]
    plt.plot(iterations, precisions, marker='o', label=f'rho={rho}')

plt.xlabel('Attack Iteration', fontsize=12)
plt.ylabel('Precision (%)', fontsize=12)
plt.title('Overall Precision vs Attack Iteration', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_vs_iteration.png', dpi=150)
print("Saved: precision_vs_iteration.png")

# ============================================================================
# Visualization 2: Layer-wise Precision Heatmap (for rho=0.02)
# ============================================================================
rho_selected = 0.02
results = all_results[rho_selected]

# Extract layer-wise precision matrix
num_layers = len(results[0]['layer_wise_precision'])
precision_matrix = np.zeros((args.steps, num_layers))

for i, r in enumerate(results):
    precision_matrix[i, :] = np.array(r['layer_wise_precision']) * 100

plt.figure(figsize=(14, 6))
plt.imshow(precision_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
plt.colorbar(label='Precision (%)')
plt.xlabel('Attack Iteration', fontsize=12)
plt.ylabel('Layer Index', fontsize=12)
plt.title(f'Layer-wise Precision Heatmap (rho={rho_selected})', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('layerwise_precision_heatmap.png', dpi=150)
print("Saved: layerwise_precision_heatmap.png")

# ============================================================================
# Visualization 3: Cumulative Precision vs Rho
# ============================================================================
rho_range = np.linspace(0.005, 0.1, 20)
cumulative_precisions = []

print("\nExperiment 2: Precision vs threshold (rho)")

for rho in rho_range:
    print(f"Testing rho={rho:.4f}...", end='\r')
    spiking_model = SpikingModel(model=net, rho=rho)
    spiking_model.to(device)
    spiking_model.eval()
    
    results = run_attack_with_tracking(
        spiking_model, inputs[:10], targets[:10], epsilon, args.steps, alpha
    )
    
    # Average precision across all iterations
    avg_precision = np.mean([r['overall_precision'] for r in results]) * 100
    cumulative_precisions.append(avg_precision)

plt.figure(figsize=(10, 6))
plt.plot(rho_range, cumulative_precisions, marker='o', linewidth=2, markersize=6)
plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Full Computation')
plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Savings')
plt.axvline(x=0.02, color='g', linestyle='--', alpha=0.5, label='Recommended (rho=0.02)')
plt.xlabel('Threshold (rho)', fontsize=12)
plt.ylabel('Average Precision (%)', fontsize=12)
plt.title('Computational Cost vs Threshold', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_vs_rho.png', dpi=150)
print("\nSaved: precision_vs_rho.png")

# ============================================================================
# Visualization 4: Loss Curve Comparison
# ============================================================================
plt.figure(figsize=(12, 6))

for rho in rho_values:
    iterations = [r['iteration'] for r in all_results[rho]]
    losses = [r['loss'] for r in all_results[rho]]
    plt.plot(iterations, losses, marker='o', label=f'rho={rho}')

plt.xlabel('Attack Iteration', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.title('Attack Loss vs Iteration (Different Thresholds)', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_vs_iteration.png', dpi=150)
print("Saved: loss_vs_iteration.png")

# ============================================================================
# Visualization 5: Layer-wise Precision Distribution
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Layer-wise Precision Distribution at Final Iteration', 
             fontsize=16, fontweight='bold')

for idx, rho in enumerate(rho_values):
    ax = axes[idx // 2, idx % 2]
    
    final_precisions = all_results[rho][-1]['layer_wise_precision']
    layer_indices = np.arange(len(final_precisions))
    
    bars = ax.bar(layer_indices, np.array(final_precisions) * 100, 
                  color='steelblue', alpha=0.7, edgecolor='black')
    
    # Color bars by precision
    for i, bar in enumerate(bars):
        if final_precisions[i] < 0.3:
            bar.set_color('green')
        elif final_precisions[i] < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer Index', fontsize=10)
    ax.set_ylabel('Precision (%)', fontsize=10)
    ax.set_title(f'rho={rho}', fontsize=12)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('layerwise_precision_distribution.png', dpi=150)
print("Saved: layerwise_precision_distribution.png")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for rho in rho_values:
    results = all_results[rho]
    avg_precision = np.mean([r['overall_precision'] for r in results]) * 100
    final_loss = results[-1]['loss']
    
    print(f"\nrho = {rho:.3f}:")
    print(f"  Average Precision: {avg_precision:.2f}%")
    print(f"  Cost Reduction: {100 - avg_precision:.2f}%")
    print(f"  Final Loss: {final_loss:.4f}")
    
    # Layer variation
    layer_precisions = results[-1]['layer_wise_precision']
    print(f"  Layer Precision Range: [{min(layer_precisions)*100:.1f}%, "
          f"{max(layer_precisions)*100:.1f}%]")

print("\n" + "="*70)
print("All visualizations saved!")
print("="*70)

# ============================================================================
# Create Summary Figure
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Precision over iterations
ax1 = fig.add_subplot(gs[0, :])
for rho in rho_values:
    iterations = [r['iteration'] for r in all_results[rho]]
    precisions = [r['overall_precision'] * 100 for r in all_results[rho]]
    ax1.plot(iterations, precisions, marker='o', label=f'rho={rho}', linewidth=2)
ax1.set_xlabel('Attack Iteration', fontsize=11)
ax1.set_ylabel('Precision (%)', fontsize=11)
ax1.set_title('A) Overall Precision vs Iteration', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Heatmap
ax2 = fig.add_subplot(gs[1, :])
im = ax2.imshow(precision_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
ax2.set_xlabel('Attack Iteration', fontsize=11)
ax2.set_ylabel('Layer Index', fontsize=11)
ax2.set_title(f'B) Layer-wise Precision Heatmap (rho={rho_selected})', 
              fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Precision (%)')

# Subplot 3: Precision vs rho
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(rho_range, cumulative_precisions, marker='o', linewidth=2, color='steelblue')
ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Full')
ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
ax3.axvline(x=0.02, color='g', linestyle='--', alpha=0.5, label='Rec.')
ax3.set_xlabel('Threshold (rho)', fontsize=11)
ax3.set_ylabel('Avg Precision (%)', fontsize=11)
ax3.set_title('C) Cost vs Threshold', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Subplot 4: Loss curves
ax4 = fig.add_subplot(gs[2, 1])
for rho in rho_values:
    iterations = [r['iteration'] for r in all_results[rho]]
    losses = [r['loss'] for r in all_results[rho]]
    ax4.plot(iterations, losses, marker='o', label=f'rho={rho}', linewidth=2)
ax4.set_xlabel('Attack Iteration', fontsize=11)
ax4.set_ylabel('Loss', fontsize=11)
ax4.set_title('D) Loss vs Iteration', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

fig.suptitle('Spiking Attack Mechanism Visualization', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('spiking_summary.png', dpi=150, bbox_inches='tight')
print("\nSaved comprehensive summary: spiking_summary.png")

