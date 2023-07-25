import copy
from collections import defaultdict
from typing import TYPE_CHECKING, List
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metrics.mean import Mean
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from avalanche.training.strategies import SupervisedTemplate

def _sum_two_grads(gradient_1, gradient_2):
    if gradient_1 is None:
        return gradient_2
    new_grad = []
    for g1, g2 in itertools.zip_longest(gradient_1, gradient_2, fillvalue=None):
        if g1 is None:
            new_grad.append(g2)
        else:
            new_grad.append(g1 + g2)
    return new_grad


def _filter_params(params):
    return [p for p in params if p.requires_grad]


@torch.no_grad()
def _compute_l2_norm(vector):
    norm = 0.0
    for v in vector:
        if v is None:
            continue
        norm += torch.sum(torch.square(v))
    norm = torch.sqrt(norm)
    return norm.detach()


@torch.no_grad()
def _compute_l2_diff(vector_small, vector_big, per_layer=True):
    norm = 0.0
    per_layer_dict = {}
    for (ns, vs), (nb, vb) in zip(vector_small, vector_big):
        if vs is None or vb is None:
            continue
        if vs.numel() != vb.numel():
            vb = vb[: vs.shape[0]]
        per_layer_norm = torch.sum(torch.square(vs - vb))
        per_layer_dict[ns] = float(torch.sqrt(per_layer_norm).detach())
        norm += per_layer_norm
    norm = torch.sqrt(norm)
    if per_layer:
        return per_layer_dict
    else:
        return float(norm.detach())


@torch.no_grad()
def _compute_cosine_diff(vector_small, vector_big, per_layer=True):
    norm = 0.0
    per_layer_dict = {}
    for (ns, vs), (nb, vb) in zip(vector_small, vector_big):
        if vs is None or vb is None:
            continue
        if vs.numel() != vb.numel():
            vb = vb[: vs.shape[0]]
        per_layer_norm = torch.nn.functional.cosine_similarity(
            vs.view(-1), vb.view(-1), dim=0
        )
        per_layer_dict[ns] = float(per_layer_norm.detach())
        norm += per_layer_norm
    if per_layer:
        return per_layer_dict
    else:
        return float(norm.detach())


@torch.no_grad()
def compute_sparsity(vector):
    norm_2 = 0.0
    norm_1 = 0.0
    total = 0
    for v in vector:
        if v is None:
            continue
        total += v.numel()
        norm_2 += torch.sum(torch.square(v))
        norm_1 += torch.sum(torch.abs(v))

    norm_2 = torch.sqrt(norm_2)
    sparsity = (np.sqrt(total) - norm_1 / norm_2) / (np.sqrt(total) - 1)
    return float(sparsity.detach())


@torch.no_grad()
def _sum(parameters_list):
    sum_of_parameters = parameters_list[0]
    for parameters in parameters_list[1:]:
        for summed_param, param in zip(sum_of_parameters, parameters):
            if summed_param is None or param is None:
                continue
            summed_param.copy_(summed_param + param)
    return sum_of_parameters


def get_flat(parameters):
    return torch.cat([p.view(-1) for p in parameters])


class SparsityMetric(Metric[float]):
    """
    Standalone gradient norm metric, keeps a dictionnary of norm (one per task)
    """

    def __init__(self):
        """
        Creates an instance of the Norm metric
        """
        super().__init__()
        self.sparsity = dict()

    def update(self, task_gradient: List[Tensor], task_total: Tensor, tid: int) -> None:
        """
        Update the running norm

        :return: None.
        """
        sparsity = compute_sparsity(task_gradient)
        self.sparsity[tid] = sparsity

    def result(self, task_label=None) -> float:
        """
        Retrieves the norm result

        :return: The average norm per task
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {k: v for k, v in self.sparsity.items()}
        else:
            return {task_label: self.sparsity[task_label]}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.sparsity = dict()


class GradProjMetric(Metric[float]):
    """
    Keeps the value of the scalar product between new gradient and old gradient
    """

    def __init__(self):
        """
        Creates an instance of the Norm metric
        """
        super().__init__()
        self.projection = None

    def update(self, old_task_gradient, new_task_gradient) -> None:
        """
        Update the running norm
        """
        flat_old = get_flat(old_task_gradient)
        flat_new = get_flat(new_task_gradient)
        proj = F.cosine_similarity(flat_old, flat_new[: len(flat_old)], dim=0)
        self.projection = float(proj.detach())

    def result(self) -> float:
        """
        Retrieves the norm result

        :return: The average norm per task
        """
        return self.projection

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.projection = None


class GradNormMetric(Metric[float]):
    """
    Standalone gradient norm metric, keeps a dictionnary of norm (one per task)
    """

    def __init__(self):
        """
        Creates an instance of the Norm metric
        """
        self._mean_norm = defaultdict(Mean)

    def update(self, task_gradient: List[Tensor], task_total: Tensor, tid: int) -> None:
        """
        Update the running norm

        :return: None.
        """
        task_norm = _compute_l2_norm(task_gradient)
        self._mean_norm[tid].update(task_norm, task_total)

    def result(self, task_label=None) -> float:
        """
        Retrieves the norm result

        :return: The average norm per task
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {k: v.result() for k, v in self._mean_norm.items()}
        else:
            return {task_label: self._mean_norm[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.

        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_norm = defaultdict(Mean)
        else:
            self._mean_norm[task_label].reset()


class GradProjPluginMetric(PluginMetric[float]):
    """
    Compute the gradient projection on old task grad
    """

    def __init__(self):
        super().__init__()
        self._norm = GradProjMetric()

    def update(self, strategy) -> Tensor:
        num_classes_per_task = len(np.unique(strategy.experience.dataset.targets))
        current_task = strategy.clock.train_exp_counter

        old_task_gradients = []
        current_task_gradient = None

        for tid in range(current_task + 1):
            selector = torch.logical_and(
                strategy.mb_y >= num_classes_per_task * tid,
                strategy.mb_y < num_classes_per_task * (tid + 1),
            )
            task_output = strategy.mb_output[selector]
            task_targets = strategy.mb_y[selector]
            task_loss = strategy._criterion(task_output, task_targets)
            ratio = len(task_targets) / len(strategy.mb_y)
            task_gradient = torch.autograd.grad(
                task_loss * ratio,
                _filter_params(strategy.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )

            if tid == current_task:
                current_task_gradient = task_gradient
            else:
                old_task_gradients.append(task_gradient)

        if len(old_task_gradients) == 0:
            return
        old_task_gradient = _sum(old_task_gradients)
        self._norm.update(old_task_gradient, current_task_gradient)

    def result(self) -> Tensor:
        return self._norm.result()

    def reset(self) -> None:
        self._norm.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=False, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=False, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def before_backward(self, strategy, **kwargs):
        self.reset()
        self.update(strategy)
        return self._package_result(strategy)

    def __str__(self):
        return "GradProjMetric"


class GradNormPluginMetric(PluginMetric[float]):
    """
    Compute the gradient norm at each backward pass
    """

    def __init__(self):
        """
        Compute the gradient norm (per task) at each backward pass
        """
        super().__init__()
        self._norm = GradNormMetric()

    def update(self, strategy) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """

        # This will not work in the class incremental learning setting
        # for tid in torch.unique(strategy.mb_task_id):
        #    task_output = strategy.mb_output[strategy.mb_task_id == tid]
        #    task_targets = strategy.mb_y[strategy.mb_task_id == tid]
        #    task_loss = strategy._criterion(task_output, task_targets)
        #    task_gradient = torch.autograd.grad(task_loss,
        #                                        strategy.model.parameters(),
        #                                        retain_graph=True)
        #    task_total = len(task_output)
        #    self._norm.update(task_gradient, task_total, int(tid))

        num_classes_per_task = len(np.unique(strategy.experience.dataset.targets))
        current_task = strategy.clock.train_exp_counter

        for tid in range(current_task + 1):
            selector = torch.logical_and(
                strategy.mb_y >= num_classes_per_task * tid,
                strategy.mb_y < num_classes_per_task * (tid + 1),
            )
            task_output = strategy.mb_output[selector]
            task_targets = strategy.mb_y[selector]
            task_loss = strategy._criterion(task_output, task_targets)
            task_gradient = torch.autograd.grad(
                task_loss,
                _filter_params(strategy.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            task_total = len(task_output)
            self._norm.update(task_gradient, task_total, int(tid))

    def result(self) -> Tensor:
        """
        Retrieves the current grad norm

        :return: L2 grad norm
        """
        return self._norm.result()

    def reset(self) -> None:
        """
        This metric is resetted at every computation

        :return: None.
        """
        self._norm.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=False, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=False, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def before_backward(self, strategy, **kwargs):
        self.reset()
        self.update(strategy)
        return self._package_result(strategy)

    def __str__(self):
        return "GradNormMetric"


class GradNormWithDistPluginMetric(PluginMetric[float]):
    """
    Compute the gradient norm at each backward pass
    """

    def __init__(self):
        """
        Compute the gradient norm (per task) at each backward pass
        """
        super().__init__()
        self.stability_norm = GradNormMetric()
        self.plasticity_norm = GradNormMetric()

    def update(self, strategy) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """
        num_classes_per_task = len(np.unique(strategy.experience.dataset.targets))
        current_task = strategy.clock.train_exp_counter

        #################################
        #  Compute Plasticity Gradient  #
        #################################
        
        selector = torch.logical_and(
            strategy.mb_y >= num_classes_per_task * current_task,
            strategy.mb_y < num_classes_per_task * (current_task + 1),
        )
        task_output = strategy.mb_output[selector]
        task_targets = strategy.mb_y[selector]
        task_loss = strategy._criterion(task_output, task_targets)
        task_gradient = torch.autograd.grad(
            task_loss,
            _filter_params(strategy.model.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        self.plasticity_norm.update(task_gradient, 1, 0)

        ################################
        #  Compute Stability Gradient  #
        ################################
        stab_grad = None
        
        for tid in range(current_task):
            selector = torch.logical_and(
                strategy.mb_y >= num_classes_per_task * tid,
                strategy.mb_y < num_classes_per_task * (tid + 1),
            )
            task_output = strategy.mb_output[selector]
            task_targets = strategy.mb_y[selector]
            task_loss = strategy._criterion(task_output, task_targets)
            task_gradient = torch.autograd.grad(
                task_loss,
                _filter_params(strategy.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            stab_grad = _sum_two_grads(stab_grad, task_gradient)


        if hasattr(strategy, "distill_loss"):
            distillation_gradient = torch.autograd.grad(
                strategy.distill_loss,
                _filter_params(strategy.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            stab_grad = _sum_two_grads(stab_grad, distillation_gradient)


        if stab_grad is not None:
            self.stability_norm.update(stab_grad, 1, 0)


    def result(self) -> Tensor:
        """
        Retrieves the current grad norm

        :return: L2 grad norm
        """
        return self.stability_norm.result(), self.plasticity_norm.result()

    def reset(self) -> None:
        """
        This metric is resetted at every computation

        :return: None.
        """
        self.stability_norm.reset()
        self.plasticity_norm.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_stability, metric_plasticity = self.result()
        plot_x_position = strategy.clock.train_iterations
        metric_name = get_metric_name(
            self, strategy, add_experience=False, add_task=True
        )
        metric_name_plasticity = "Plasticity" + metric_name
        metric_name_stability = "Stability" + metric_name
        return [MetricValue(self, metric_name_plasticity, metric_plasticity, plot_x_position), 
                MetricValue(self, metric_name_stability, metric_stability, plot_x_position)]

    def before_backward(self, strategy, **kwargs):
        self.reset()
        self.update(strategy)
        return self._package_result(strategy)

    def __str__(self):
        return "GradNormMetric"


class ParameterNorm(Metric[float]):
    """
    Standalone parameter norm metric, records the norm of current model
    """

    def __init__(self):
        """
        Creates an instance of the Norm metric
        """
        self._mean_norm = Mean()

    def update(self, model: nn.Module) -> None:
        """
        Update the running norm

        :return: None.
        """
        weight_norm = _compute_l2_norm(model.parameters())
        self._mean_norm.update(weight_norm, 1)

    def result(self) -> float:
        """
        Retrieves the norm result

        :return: The average norm per task
        """
        return self._mean_norm.result()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._mean_norm.reset()


class ParameterNormPluginMetric(PluginMetric[float]):
    """
    Compute the parameter norm at each backward pass
    """

    def __init__(self):
        """
        Compute the gradient norm (per task) at each backward pass
        """
        super().__init__()
        self._norm = ParameterNorm()

    def update(self, strategy) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """
        self._norm.update(strategy.model)

    def result(self) -> Tensor:
        """
        Retrieves the current grad norm

        :return: L2 grad norm
        """
        return self._norm.result()

    def reset(self) -> None:
        """
        This metric is resetted at every computation

        :return: None.
        """
        self._norm.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations
        metric_name = get_metric_name(
            self, strategy, add_experience=False, add_task=False
        )
        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def after_training_iteration(self, strategy, **kwargs):
        self.reset()
        self.update(strategy)
        return self._package_result(strategy)

    def __str__(self):
        return "ParameterNormMetric"


class GradDistributionPlugin(SupervisedPlugin):
    """Records per-task grad distribution after each backward pass"""

    def __init__(self, tb_logger):
        super().__init__()
        self.tb_logger = tb_logger.writer

    def compute_per_task_grad(self, strategy) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """

        # This will not work in the class incremental learning setting
        # for tid in torch.unique(strategy.mb_task_id):
        #    task_output = strategy.mb_output[strategy.mb_task_id == tid]
        #    task_targets = strategy.mb_y[strategy.mb_task_id == tid]
        #    task_loss = strategy._criterion(task_output, task_targets)
        #    task_gradient = torch.autograd.grad(task_loss,
        #                                        strategy.model.parameters(),
        #                                        retain_graph=True)
        #    task_total = len(task_output)
        #    self._norm.update(task_gradient, task_total, int(tid))

        gradients = dict()

        num_classes_per_task = len(np.unique(strategy.experience.dataset.targets))
        current_task = strategy.clock.train_exp_counter

        for tid in range(current_task + 1):
            selector = torch.logical_and(
                strategy.mb_y >= num_classes_per_task * tid,
                strategy.mb_y < num_classes_per_task * (tid + 1),
            )
            task_output = strategy.mb_output[selector]
            task_targets = strategy.mb_y[selector]
            task_loss = strategy._criterion(task_output, task_targets)
            task_gradient = torch.autograd.grad(
                task_loss,
                _filter_params(strategy.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )

            gradients[tid] = task_gradient

        return gradients

    def before_backward(self, strategy, **kwargs):
        gradients = self.compute_per_task_grad(strategy)
        for (tid, grad) in gradients.items():
            for (n, p), g in zip(strategy.model.named_parameters(), grad):
                self.tb_logger.add_histogram(
                    f"{tid}_{n}",
                    g.data.view(-1),
                    global_step=strategy.clock.train_iterations,
                )


class ParameterShiftPlugin(SupervisedPlugin):
    """Records parameter shift from the previous checkpoint"""

    def __init__(self, tb_logger, mode="cosine"):
        super().__init__()
        assert mode in ["cosine", "l2"]
        self.tb_logger = tb_logger.writer
        self.old_model = None
        self.mode = mode

        self._metric_dict = {}

    def after_training_iteration(self, strategy, **kwargs):
        if self.old_model is None:
            return
        if self.mode == "cosine":
            self._metric_dict = _compute_cosine_diff(
                self.old_model.named_parameters(), strategy.model.named_parameters()
            )
        elif self.mode == "l2":
            self._metric_dict = _compute_l2_diff(
                self.old_model.named_parameters(), strategy.model.named_parameters()
            )

        self.tb_logger.add_scalars(
            "weight_shift",
            self._metric_dict,
            global_step=strategy.clock.train_iterations,
        )

    def after_training_exp(self, strategy, **kwargs):
        self.old_model = copy.deepcopy(strategy.model)


class GradSparsityPluginMetric(PluginMetric[float]):
    """
    Compute the gradient norm at each backward pass
    """

    def __init__(self):
        """
        Compute the gradient norm (per task) at each backward pass
        """
        super().__init__()
        self._norm = SparsityMetric()

    def update(self, strategy) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """

        # This will not work in the class incremental learning setting
        # for tid in torch.unique(strategy.mb_task_id):
        #    task_output = strategy.mb_output[strategy.mb_task_id == tid]
        #    task_targets = strategy.mb_y[strategy.mb_task_id == tid]
        #    task_loss = strategy._criterion(task_output, task_targets)
        #    task_gradient = torch.autograd.grad(task_loss,
        #                                        strategy.model.parameters(),
        #                                        retain_graph=True)
        #    task_total = len(task_output)
        #    self._norm.update(task_gradient, task_total, int(tid))

        num_classes_per_task = len(np.unique(strategy.experience.dataset.targets))
        current_task = strategy.clock.train_exp_counter

        for tid in range(current_task + 1):
            selector = torch.logical_and(
                strategy.mb_y >= num_classes_per_task * tid,
                strategy.mb_y < num_classes_per_task * (tid + 1),
            )
            task_output = strategy.mb_output[selector]
            task_targets = strategy.mb_y[selector]
            task_loss = strategy._criterion(task_output, task_targets)
            task_gradient = torch.autograd.grad(
                task_loss,
                _filter_params(strategy.model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            task_total = len(task_output)
            self._norm.update(task_gradient, task_total, int(tid))

    def result(self) -> Tensor:
        """
        Retrieves the current grad norm

        :return: L2 grad norm
        """
        return self._norm.result()

    def reset(self) -> None:
        """
        This metric is resetted at every computation

        :return: None.
        """
        self._norm.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=False, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=False, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def before_backward(self, strategy, **kwargs):
        self.reset()
        self.update(strategy)
        return self._package_result(strategy)

    def __str__(self):
        return "GradSparsityMetric"


__all__ = [
    "GradNormPluginMetric",
    "ParameterNormPluginMetric",
    "GradDistributionPlugin",
    "GradSparsityPluginMetric",
    "GradProjPluginMetric",
]
