#!/usr/bin/env python3
import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn as nn
from avalanche.evaluation import GenericPluginMetric, Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metrics.mean import Mean
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from torch import Tensor

if TYPE_CHECKING:
    from avalanche.training.strategies import SupervisedTemplate


class FIMTracePluginMetric(PluginMetric[float]):
    """
    Compute the trace of the FIM on current task every few iterations
    """

    def __init__(self, update_every: int = 5):
        super().__init__()
        self.importances = None
        self.update_every = update_every

    def update(self, strategy) -> Tensor:
        self.importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )

    def result(self) -> Tensor:
        return float(self.compute_trace(self.importances))

    def reset(self):
        pass

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=False, add_task=k
                )
                metrics.append(MetricValue(
                    self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=False, add_task=True
            )
            return MetricValue(self, metric_name, metric_value, plot_x_position)

    def after_training_iteration(self, strategy, **kwargs):
        current_iter = strategy.clock.train_iterations
        if current_iter % self.update_every == 0:
            self.update(strategy)
        return self._package_result(strategy)

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size
    ):
        """
        Compute EWC importance matrix for each parameter
        """
        training = model.training
        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        if training:
            model.train()
        return importances

    def compute_trace(self, importances):
        trace = 0.
        for name, imp in importances:
            trace += torch.sum(imp)
        return trace

    def __str__(self):
        return "FIM_Trace"
