#!/usr/bin/env python3
################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from collections import defaultdict
from typing import Dict, List

import torch
from torch import Tensor

from avalanche.evaluation import GenericPluginMetric, Metric, PluginMetric
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean


class DictMean(Metric[float]):
    """
    The standalone General metric. This is a general metric
    used to compute more specific ones.

    Instances of this metric keeps the running average scalar
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average scalar
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return a scalar value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the scalar metric.

        By default this metric in its initial state will return a scalar
        value of 0. The metric can be updated by using the `update` method
        while the running scalar can be retrieved using the `result` method.
        """
        self._mean_scalar = defaultdict(Mean)
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, scalar: Tensor, patterns: int, task_label: int) -> None:
        """
        Update the running scalar given the scalar Tensor and the minibatch size.

        :param scalar: The scalar Tensor. Different reduction types don't affect
            the result.
        :param patterns: The number of patterns in the minibatch.
        :param task_label: the task label associated to the current experience
        :return: None.
        """
        self._mean_scalar[task_label].update(torch.mean(scalar), weight=patterns)

    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running average scalar per pattern.

        Calling this method will not change the internal state of the metric.
        :param task_label: None to return metric values for all the task labels.
            If an int, return value only for that task label
        :return: The running scalar, as a float.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {k: v.result() for k, v in self._mean_scalar.items()}
        else:
            return {task_label: self._mean_scalar[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.

        :param task_label: None to reset all metric values. If an int,
            reset metric value corresponding to that task label.
        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_scalar = defaultdict(Mean)
        else:
            self._mean_scalar[task_label].reset()


class GeneralPluginMetric(GenericPluginMetric[float]):
    def __init__(self, scalar_name, reset_at, emit_at, mode):
        self._scalar = DictMean()
        self.scalar_name = scalar_name
        super(GeneralPluginMetric, self).__init__(self._scalar, reset_at, emit_at, mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == "stream" or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == "stream" or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]
        if hasattr(strategy, self.scalar_name):
            scalar = getattr(strategy, self.scalar_name)
        else:
            scalar = torch.tensor(0.)
        self._scalar.update(
            scalar, patterns=len(strategy.mb_y), task_label=task_label
        )


class MinibatchGeneral(GeneralPluginMetric):
    """
    The minibatch scalar metric.
    This plugin metric only works at training time.

    This metric computes the average scalar over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochGeneral` instead.
    """

    def __init__(self, scalar_name):
        """
        Creates an instance of the MinibatchGeneral metric.
        """
        super(MinibatchGeneral, self).__init__(
            scalar_name, reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return f"{self.scalar_name}_MB"


class EpochGeneral(GeneralPluginMetric):
    """
    The average scalar over a single training epoch.
    This plugin metric only works at training time.

    The scalar will be logged after each training epoch by computing
    the scalar on the predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, scalar_name):
        """
        Creates an instance of the EpochGeneral metric.
        """

        super(EpochGeneral, self).__init__(
            scalar_name, reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return f"{self.scalar_name}_Epoch"


class RunningEpochGeneral(GeneralPluginMetric):
    """
    The average scalar across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the scalar averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, scalar_name):
        """
        Creates an instance of the RunningEpochGeneral metric.
        """

        super(RunningEpochGeneral, self).__init__(
            scalar_name, reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return f"{self.scalar_name}_Epoch"


class ExperienceGeneral(GeneralPluginMetric):
    """
    At the end of each experience, this metric reports
    the average scalar over all patterns seen in that experience.
    This plugin metric only works at eval time.
    """

    def __init__(self, scalar_name):
        """
        Creates an instance of ExperienceGeneral metric
        """
        super(ExperienceGeneral, self).__init__(
            scalar_name, reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return f"{self.scalar_name}_Exp"


class StreamGeneral(GeneralPluginMetric):
    """
    At the end of the entire stream of experiences, this metric reports the
    average scalar over all patterns seen in all experiences.
    This plugin metric only works at eval time.
    """

    def __init__(self, scalar_name):
        """
        Creates an instance of StreamGeneral metric
        """
        super(StreamGeneral, self).__init__(
            scalar_name, reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return f"{self.scalar_name}_Stream"


def general_metrics(
    scalar_to_be_logged: str,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param scalar_to_be_logged: name of the strategy 
                                scalar field to be logged
    :param minibatch: If True, will return a metric able to log
        the minibatch scalar at training time.
    :param epoch: If True, will return a metric able to log
        the epoch scalar at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch scalar at training time.
    :param experience: If True, will return a metric able to log
        the scalar on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the scalar averaged over the entire evaluation stream of experiences.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchGeneral(scalar_to_be_logged))

    if epoch:
        metrics.append(EpochGeneral(scalar_to_be_logged))

    if epoch_running:
        metrics.append(RunningEpochGeneral(scalar_to_be_logged))

    if experience:
        metrics.append(ExperienceGeneral(scalar_to_be_logged))

    if stream:
        metrics.append(StreamGeneral(scalar_to_be_logged))

    return metrics


__all__ = [
    "DictMean",
    "MinibatchGeneral",
    "EpochGeneral",
    "RunningEpochGeneral",
    "ExperienceGeneral",
    "StreamGeneral",
    "general_metrics",
]
