#!/usr/bin/env python3
import torch

from avalanche.training.plugins import SupervisedPlugin


class GradClippingPlugin(SupervisedPlugin):
    """
    Gradient clipping plugin, clips the gradient to the desired norm
    """

    def __init__(self, clipgrad=10000):
        super().__init__()
        self.clipgrad = clipgrad

    def after_backward(self, strategy, **kwargs):
        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), self.clipgrad)
