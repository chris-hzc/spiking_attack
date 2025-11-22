"""
Spiking Layer Implementation for Fine-Grained Iterative Adversarial Attacks
Based on the paper: Fine-Grained Iterative Adversarial Attacks WITH LIMITED COMPUTATION BUDGET

Key Features:
- Adaptive activation reuse based on relative change threshold (rho)
- Virtual surrogate gradient for backward pass when reusing activations
- Full-precision computation (no quantization)
- Layer-wise precision tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import warnings


class StraightThrough(nn.Module):
    """Identity module for pass-through operations"""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class SpikingModule(nn.Module):
    """
    Spiking Module that adaptively reuses activations based on relative change.
    
    Core mechanism:
    - Computes activation only when relative change exceeds threshold rho
    - Otherwise reuses previous activation (forward) and uses virtual gradient (backward)
    - Tracks precision as percentage of full computations performed
    
    Args:
        org_module: Original nn.Conv2d or nn.Linear layer
        rho: Threshold for relative activation change (default: 0.02)
        se_module: Optional Squeeze-and-Excitation module
    """
    
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], 
                 rho: float = 0.02, se_module=None):
        super(SpikingModule, self).__init__()
        
        # Store original layer configuration
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride, 
                padding=org_module.padding,
                dilation=org_module.dilation, 
                groups=org_module.groups
            )
            self.fwd_func = F.conv2d
            self.layer_type = 'conv2d'
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            self.layer_type = 'linear'
            
        # Store weights and biases
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        
        # Spiking mechanism parameters
        self.rho = rho  # Threshold for relative activation change
        self.use_spiking = False  # Enable/disable spiking mechanism
        
        # Storage for previous activations and outputs
        self.pre_activation = None  # Previous input activation
        self.pre_output = None      # Previous output
        self.pre_grad = None        # Previous gradient for virtual surrogate
        
        # Activation function
        self.activation_function = StraightThrough()
        self.se_module = se_module
        
        # Precision tracking
        self.precision_list = []  # Track computation decisions (1=compute, 0=reuse)
        
        self.extra_repr = org_module.extra_repr

    def should_recompute(self, current_activation: torch.Tensor) -> bool:
        """
        Determine whether to recompute or reuse based on relative change.
        
        Returns True if relative change exceeds threshold rho:
            ||a_t - a_{t-1}|| / ||a_t|| >= rho
        """
        if self.pre_activation is None:
            return True  # First iteration always computes
        
        activation_diff = current_activation - self.pre_activation
        diff_norm = torch.norm(activation_diff)
        act_norm = torch.norm(current_activation)
        
        # Avoid division by zero
        if act_norm < 1e-10:
            return False
        
        relative_change = diff_norm / act_norm
        
        # Debug output
        # print(f"Layer {self.layer_type}: relative_change={relative_change:.6f}, threshold={self.rho}")
        
        return relative_change >= self.rho

    def save_grad_hook(self, grad):
        """Hook to save gradient from pre_output for virtual surrogate"""
        self.pre_grad = grad.clone()

    def virtual_surrogate_hook(self, grad):
        """
        Virtual surrogate gradient: compute gradient w.r.t. input using chain rule
        even though we reused the activation in forward pass.
        
        For Conv2d: grad_input = conv_transpose(grad_output, weight)
        For Linear: grad_input = grad_output @ weight
        """
        if self.pre_grad is None:
            return  # No saved gradient, use standard backprop
        
        if self.layer_type == 'conv2d':
            # Compute virtual gradient for conv2d using grad.conv2d_input
            grad_est = torch.nn.grad.conv2d_input(
                input_size=grad.shape,
                weight=self.org_weight,
                grad_output=self.pre_grad,
                **self.fwd_kwargs
            )
            grad.copy_(grad_est)
        elif self.layer_type == 'linear':
            # Compute virtual gradient for linear layer
            grad_est = self.pre_grad @ self.org_weight
            grad.copy_(grad_est)

    def forward(self, input: torch.Tensor):
        """
        Forward pass with adaptive computation.
        
        Decision logic:
        1. If relative change >= rho: perform full computation
        2. Otherwise: reuse previous output, attach virtual gradient hook
        """
        weight = self.org_weight
        bias = self.org_bias
        
        # Spiking mechanism (adaptive reuse)
        if self.use_spiking and self.pre_activation is not None:
            recompute = self.should_recompute(input)
            
            if recompute:
                # Full computation
                out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
                self.precision_list.append(1)  # Computed
                print(f"{self.layer_type}: COMPUTE (relative change >= {self.rho})")
            else:
                # Reuse previous output
                # Create dependency on input for gradient flow
                input_clone = input.clone()
                out = input_clone.reshape(-1)[0] * 0 + self.pre_output
                
                # Attach hooks for virtual surrogate gradient
                if input.requires_grad:
                    self.pre_output.requires_grad_()
                    self.pre_output.register_hook(self.save_grad_hook)
                    input_clone.register_hook(self.virtual_surrogate_hook)
                
                self.precision_list.append(0)  # Reused
                print(f"{self.layer_type}: REUSE (relative change < {self.rho})")
            
            # Store current activation and output for next iteration
            with torch.no_grad():
                self.pre_activation = input.clone()
                self.pre_output = out.clone()
                
        else:
            # Standard forward pass (no spiking)
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            
            if self.use_spiking:
                # First iteration: initialize storage
                with torch.no_grad():
                    self.pre_activation = input.clone()
                    self.pre_output = out.clone()
                self.precision_list.append(1)
        
        # Apply SE module if present
        if self.se_module is not None:
            out = self.se_module(out)
        
        # Apply activation function (ReLU, etc.)
        out = self.activation_function(out)
        
        return out

    def set_spiking_state(self, use_spiking: bool = True):
        """Enable or disable spiking mechanism"""
        self.use_spiking = use_spiking

    def set_threshold(self, rho: float):
        """Update threshold for relative change"""
        self.rho = rho

    def reset_state(self):
        """Reset stored activations and outputs (call before new attack iteration)"""
        self.pre_activation = None
        self.pre_output = None
        self.pre_grad = None

    def get_precision(self):
        """
        Calculate precision as percentage of full computations performed.
        
        Returns:
            float: precision in range [0, 1], where 1 means all computations performed
        """
        if len(self.precision_list) == 0:
            return 1.0
        return sum(self.precision_list) / len(self.precision_list)

    def reset_precision_tracking(self):
        """Reset precision tracking for new batch"""
        self.precision_list = []


def set_threshold_all_layers(model, rho: float):
    """Set threshold rho for all SpikingModule layers in model"""
    for module in model.modules():
        if isinstance(module, SpikingModule):
            module.set_threshold(rho)


def reset_state_all_layers(model):
    """Reset state for all SpikingModule layers (call before new attack)"""
    for module in model.modules():
        if isinstance(module, SpikingModule):
            module.reset_state()


def get_model_precision(model):
    """
    Get overall precision across all SpikingModule layers.
    
    Returns:
        dict with 'layer_wise' and 'overall' precision
    """
    precisions = []
    for module in model.modules():
        if isinstance(module, SpikingModule):
            precisions.append(module.get_precision())
    
    if len(precisions) == 0:
        return {'layer_wise': [], 'overall': 1.0}
    
    return {
        'layer_wise': precisions,
        'overall': sum(precisions) / len(precisions)
    }


def reset_precision_tracking_all_layers(model):
    """Reset precision tracking for all layers"""
    for module in model.modules():
        if isinstance(module, SpikingModule):
            module.reset_precision_tracking()

