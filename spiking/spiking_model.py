"""
Spiking Model for Fine-Grained Iterative Adversarial Attacks

Wraps a standard neural network model with SpikingModule layers that implement
adaptive activation reuse based on the paper methodology.
"""

import torch
import torch.nn as nn
from spiking.spiking_layer import SpikingModule, StraightThrough
from quant.fold_bn import search_fold_and_remove_bn


class SpikingModel(nn.Module):
    """
    Wraps a standard model with spiking mechanism for adaptive computation.
    
    Key features:
    - Replaces Conv2d and Linear layers with SpikingModule
    - Maintains ReLU activation functions within SpikingModule
    - Supports threshold scheduling (constant or exponential decay)
    
    Args:
        model: Base nn.Module to wrap
        rho: Initial threshold for relative change (default: 0.02)
    """
    
    def __init__(self, model: nn.Module, rho: float = 0.02):
        super().__init__()
        
        # Fold batch normalization into conv layers for cleaner computation
        search_fold_and_remove_bn(model)
        
        self.model = model
        self.rho = rho
        
        # Replace standard layers with SpikingModule
        self.spiking_module_refactor(self.model, rho)

    def spiking_module_refactor(self, module: nn.Module, rho: float):
        """
        Recursively replace Conv2d and Linear layers with SpikingModule.
        
        Args:
            module: nn.Module to refactor
            rho: Threshold for relative activation change
        """
        prev_spiking_module = None
        
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                # Replace with SpikingModule
                spiking_layer = SpikingModule(child_module, rho=rho)
                setattr(module, name, spiking_layer)
                prev_spiking_module = spiking_layer
                
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                # Move activation function into previous SpikingModule
                if prev_spiking_module is not None:
                    prev_spiking_module.activation_function = child_module
                    setattr(module, name, StraightThrough())
                    
            elif isinstance(child_module, StraightThrough):
                continue
                
            else:
                # Recursively process child modules
                self.spiking_module_refactor(child_module, rho)

    def set_spiking_state(self, use_spiking: bool = True):
        """
        Enable or disable spiking mechanism for all layers.
        
        Args:
            use_spiking: If True, enable adaptive reuse; if False, use standard forward
        """
        for m in self.model.modules():
            if isinstance(m, SpikingModule):
                m.set_spiking_state(use_spiking)

    def set_threshold(self, rho: float):
        """
        Update threshold rho for all SpikingModule layers.
        
        Args:
            rho: New threshold value
        """
        self.rho = rho
        for m in self.model.modules():
            if isinstance(m, SpikingModule):
                m.set_threshold(rho)

    def reset_state(self):
        """Reset all stored activations and outputs (call before each attack)"""
        for m in self.model.modules():
            if isinstance(m, SpikingModule):
                m.reset_state()

    def reset_precision_tracking(self):
        """Reset precision tracking for all layers (call before each batch)"""
        for m in self.model.modules():
            if isinstance(m, SpikingModule):
                m.reset_precision_tracking()

    def get_precision(self):
        """
        Get precision statistics across all layers.
        
        Returns:
            dict: {
                'layer_wise': list of per-layer precision values,
                'overall': mean precision across all layers
            }
        """
        precisions = []
        for m in self.model.modules():
            if isinstance(m, SpikingModule):
                precisions.append(m.get_precision())
        
        if len(precisions) == 0:
            return {'layer_wise': [], 'overall': 1.0}
        
        return {
            'layer_wise': precisions,
            'overall': sum(precisions) / len(precisions)
        }

    def forward(self, input):
        """Standard forward pass through the model"""
        return self.model(input)


class ThresholdScheduler:
    """
    Scheduler for threshold rho during adversarial training.
    
    Supports two strategies:
    1. Constant: rho remains fixed
    2. Exponential decay: rho decays from rho_0 to 0 over training
    
    Exponential decay formula (from paper Appendix B):
        rho(t) = rho_0 * (e^{-lambda*t/N} - e^{-lambda}) / (1 - e^{-lambda})
        where rho(0) = rho_0 and rho(N) = 0
    """
    
    def __init__(self, schedule_type: str = 'constant', 
                 rho_initial: float = 0.1,
                 total_epochs: int = 200,
                 lambda_decay: float = 5.0):
        """
        Args:
            schedule_type: 'constant' or 'exponential'
            rho_initial: Initial threshold value
            total_epochs: Total number of training epochs (for exponential)
            lambda_decay: Decay rate parameter (for exponential)
        """
        self.schedule_type = schedule_type
        self.rho_0 = rho_initial
        self.N = total_epochs
        self.lambda_decay = lambda_decay

    def get_threshold(self, epoch: int) -> float:
        """
        Get threshold value for given epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            float: threshold value rho
        """
        if self.schedule_type == 'constant':
            return self.rho_0
        
        elif self.schedule_type == 'exponential':
            import math
            t = epoch
            N = self.N
            lam = self.lambda_decay
            rho_0 = self.rho_0
            
            # rho(t) = rho_0 * (e^{-lambda*t/N} - e^{-lambda}) / (1 - e^{-lambda})
            numerator = math.exp(-lam * t / N) - math.exp(-lam)
            denominator = 1 - math.exp(-lam)
            
            rho_t = rho_0 * numerator / denominator
            
            # Ensure non-negative
            return max(0.0, rho_t)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def __repr__(self):
        if self.schedule_type == 'constant':
            return f"ThresholdScheduler(constant, rho={self.rho_0})"
        else:
            return f"ThresholdScheduler(exponential, rho_0={self.rho_0}, lambda={self.lambda_decay})"

