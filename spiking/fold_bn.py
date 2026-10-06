"""
Batch-norm folding utilities.

Folding BN into the preceding Conv2d/Linear makes every spiking layer a single
affine map A(a) = W a + b, which is what the linearity argument behind spiking
forward computation (Sec. 4.2 of the paper) relies on.
"""

import torch
import torch.nn as nn


class StraightThrough(nn.Module):
    """Identity module used to replace layers that have been folded away."""

    def forward(self, input):
        return input


@torch.no_grad()
def fold_bn_into_layer(layer: nn.Module, bn: nn.BatchNorm2d):
    """Fold the running statistics and affine parameters of ``bn`` into ``layer``."""
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std if bn.affine else 1.0 / std
    shift = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)

    view = [-1] + [1] * (layer.weight.dim() - 1)
    layer.weight.mul_(scale.view(view))
    if layer.bias is None:
        layer.bias = nn.Parameter(torch.zeros_like(bn.running_mean))
    layer.bias.mul_(scale).add_(shift - bn.running_mean * scale)


def search_fold_and_remove_bn(model: nn.Module):
    """
    Recursively fold every BatchNorm that directly follows a Conv2d/Linear
    (in ``named_children`` order) and replace it with :class:`StraightThrough`.

    The model should be in eval mode with trained running statistics.
    """
    model.eval()
    prev = None
    for name, child in model.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)) and prev is not None:
            fold_bn_into_layer(prev, child)
            setattr(model, name, StraightThrough())
        elif isinstance(child, (nn.Conv2d, nn.Linear)):
            prev = child
        elif isinstance(child, StraightThrough):
            continue
        else:
            search_fold_and_remove_bn(child)
            prev = None
