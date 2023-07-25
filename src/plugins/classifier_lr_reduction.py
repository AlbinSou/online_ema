import copy
from collections import defaultdict

import numpy as np
import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


def reset_optimizer(optimizer, model, lr, lower_ratio):
    """Reset the optimizer to update the list of learnable parameters.

    .. warning::
        This function fails if the optimizer uses multiple parameter groups.

    :param optimizer:
    :param model:
    :return:
    """
    optimizer.state = defaultdict(dict)

    # Split in backbone parameters and classifier parameters
    backbone_parameters = []
    classifier_parameters = []

    last_layer_name = list(model.named_parameters())[-1][0].split(".")[0]
    for (n, p) in model.named_parameters():
        if last_layer_name in n:
            classifier_parameters.append(p)
        else:
            backbone_parameters.append(p)

    optimizer.param_groups[0]["params"] = backbone_parameters
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["params"] = classifier_parameters
    optimizer.param_groups[1]["lr"] = lr * lower_ratio


class LowerLRClassifierPlugin(SupervisedPlugin):
    """
    At eval time, use the ensemble (for old task data) of old model
    from a few iterations back and current model
    """

    def __init__(self, lower_ratio: float = 0.1, clip_norm=1.0):
        super().__init__()
        self.lower_ratio = lower_ratio
        self.clip_norm = clip_norm

    def before_training_exp(self, strategy, **kwargs):
        strategy.optimizer.param_groups.append(
            copy.deepcopy(strategy.optimizer.param_groups[0])
        )
        reset_optimizer(
            strategy.optimizer,
            strategy.model,
            strategy.optimizer.param_groups[0]["lr"],
            self.lower_ratio,
        )

    def after_backward(self, strategy, **kwargs):
        last_layer_name = list(strategy.model.named_parameters())[-1][0].split(".")[0]
        # Clip gradient norm of distinct tasks parameters to the same norm
        # (so that they are alloweed to move the same)
        classifier_parameters = getattr(strategy.model, last_layer_name).parameters()
        current_experience = strategy.experience.current_experience
        num_classes = len(np.unique(strategy.experience.dataset.targets))
        num_classes_per_experience = num_classes // (current_experience + 1)

        for p in classifier_parameters:
            for t in range(current_experience + 1):
                # bias
                if len(p.shape) == 1:
                    task_grad = p.grad[
                        t
                        * num_classes_per_experience : (t + 1)
                        * num_classes_per_experience
                    ]
                    task_grad.copy_(task_grad / torch.norm(task_grad)) * self.clip_norm
                # Classifier
                else:
                    task_grad = p.grad[
                        t
                        * num_classes_per_experience : (t + 1)
                        * num_classes_per_experience,
                        :,
                    ]
                    task_grad.copy_(task_grad / torch.norm(task_grad)) * self.clip_norm
