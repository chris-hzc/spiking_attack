"""
Spiking Attack Module

Implementation of fine-grained iterative adversarial attacks with adaptive computation
based on the paper: Fine-Grained Iterative Adversarial Attacks WITH LIMITED COMPUTATION BUDGET
"""

from spiking.spiking_layer import (
    SpikingModule,
    StraightThrough,
    set_threshold_all_layers,
    reset_state_all_layers,
    get_model_precision,
    reset_precision_tracking_all_layers
)
from spiking.spiking_model import SpikingModel, ThresholdScheduler

__all__ = [
    'SpikingModule',
    'SpikingModel',
    'ThresholdScheduler',
    'StraightThrough',
    'set_threshold_all_layers',
    'reset_state_all_layers',
    'get_model_precision',
    'reset_precision_tracking_all_layers'
]

