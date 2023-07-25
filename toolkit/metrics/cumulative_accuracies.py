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

from typing import List, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict


class CumulativeAccuracy(Metric[float]):
    """ Keeps a dictionnary of cumulative Accuracy for each experience """

    def __init__(self):
        self._mean_accuracy = defaultdict(lambda: Mean())

    @torch.no_grad()
    def update(
        self,
        classes_splits,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        for t, classes in classes_splits.items():

            # Only compute Accuracy for classes that are in classes set
            if len(set(true_y.cpu().numpy()).intersection(classes)) == 0:
                continue

            # Transform predicted logits to label using cumulative Accuracy
            # method (only taking logits accumulated from previous experiences)
            logits_exp = predicted_y[:, list(classes)]

            if len(true_y) != len(predicted_y):
                raise ValueError("Size mismatch for true_y and predicted_y tensors")

            # Check if logits or labels
            if len(logits_exp.shape) > 1:
                # Logits -> transform to labels
                prediction = torch.max(logits_exp, 1)[1]

            if len(true_y.shape) > 1:
                # Logits -> transform to labels
                true_y = torch.max(true_y, 1)[1]

            true_positives = float(torch.sum(torch.eq(prediction, true_y)))
            total_patterns = len(true_y)
            self._mean_accuracy[t].update(
                true_positives / total_patterns, total_patterns
            )

    def result(self) -> float:
        """Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
        if len(self._mean_accuracy) == 0:
            return None
        return {t: self._mean_accuracy[t].result() for t in self._mean_accuracy}

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        for t in self._mean_accuracy:
            self._mean_accuracy[t].reset()


class CumulativeAccuracyPluginMetric(GenericPluginMetric[float]):

    def __init__(self, reset_at="stream", emit_at="stream", mode="eval"):
        """
        Creates the CumulativeAccuracy plugin
        """
        self._accuracy = CumulativeAccuracy()
        self.num_task = 0 
        self.classes_seen_so_far = set()
        self.classes_splits = {}
        super(CumulativeAccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def before_eval_exp(self, strategy, **kwargs):
        super().before_eval_exp(strategy, **kwargs)
        new_classes = set(strategy.experience.classes_in_this_experience)
        if len(new_classes.union(self.classes_seen_so_far)) > len(self.classes_seen_so_far):
            # New classes have been introduced
            self.classes_seen_so_far = self.classes_seen_so_far.union(new_classes)
            self.classes_splits.update({self.num_task: self.classes_seen_so_far})
            self.num_task += 1


    def reset(self, strategy=None) -> None:
        self._metric.reset()

    def result(self, strategy=None) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._accuracy.update(self.classes_splits, strategy.mb_output, strategy.mb_y)

    def __repr__(self):
        return "CumulativeAccuracy"


__all__ = ["CumulativeAccuracyPluginMetric"]
