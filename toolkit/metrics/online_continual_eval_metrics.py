#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, TypeVar

import torch

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metrics.accuracy import TaskAwareAccuracy
from avalanche.evaluation.metrics import Mean

if TYPE_CHECKING:
    from avalanche.training import SupervisedTemplate

TResult = TypeVar("TResult")


class TrackerPluginMetric(PluginMetric[TResult]):
    """General Tracker Plugin for Continual Evaluation.
    Implements (optional) resetting after training iteration.
    """

    def __init__(self, name, metric, reset_at="iteration"):
        """Emits and updates metrics at each iteration"""
        super().__init__()
        self._metric = metric
        self.name = name

        # Mode is train
        assert reset_at in {"iteration", "never"}  # Not at stream
        self._reset_at = reset_at

    # Basic methods
    def reset(self, strategy=None) -> None:
        """Default behavior metric."""
        self._metric.reset()

    def result(self, strategy=None):
        """Default behavior metric."""
        return self._metric.result()

    def update(self, strategy=None, **kwargs):
        """(Optional) Template method to overwrite by subclass.
        Subclass can define own update methods instead.
        """
        pass

    def after_eval(self, strategy: "SupervisedTemplate") -> None:
        """Pass to evaluator plugin."""
        return self._package_result(strategy)

    def before_eval(self, strategy: "SupervisedTemplate") -> None:
        if self._reset_at == "iteration":
            self.reset(strategy)

    def _package_result(self, strategy: "SupervisedTemplate", x_pos=None) -> "MetricResult":
        metric_value = self.result(strategy)
        add_exp = False
        plot_x_position = (
            strategy.clock.train_iterations if x_pos is None else x_pos
        )  # Allows pre-update step at -1

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        """Task label is determined by subclass, not current name. (e.g. Accuracy returns dict of per-task results.)"""
        reset_map = {"iteration": "MB", "never": "STREAM"}
        assert self._reset_at in reset_map
        return f"TRACK_{reset_map[self._reset_at]}_{self.name}"


class WindowedForgettingPluginMetric(TrackerPluginMetric[float]):
    """For metric definition, see original paper: https://arxiv.org/abs/2205.13452"""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self._current_acc = TaskAwareAccuracy()  # Per-task acc
        super().__init__(
            name=f"F{self.window_size}", metric=self._current_acc, reset_at="iteration"
        )

        self.acc_window: Dict[int, deque] = defaultdict(deque)
        self.max_forgetting: Dict[int, float] = defaultdict(float)

    def reset(self, strategy) -> None:
        """Only current acc is reset (each iteration), not the window"""
        self._current_acc.reset()

    def result(self, strategy=None) -> Dict[int, float]:
        return self.max_forgetting  # Always return all task results

    def update_current_task_acc(self, strategy):
        self._current_acc.update(
            strategy.mb_output, strategy.mb_y, task_labels=strategy.experience.current_experience
        )

    def update_task_window(self, strategy):
        new_acc_dict: Dict[int, float] = self._current_acc.result(
            task_label=strategy.experience.current_experience
        )
        new_acc = new_acc_dict[strategy.experience.current_experience]

        # Add to window
        task_acc_window = self.acc_window[strategy.experience.current_experience]
        task_acc_window.append(new_acc)
        if len(task_acc_window) > self.window_size:
            task_acc_window.popleft()

        # Update forgetting
        self.max_forgetting[strategy.experience.current_experience] = max(
            self.max_forgetting[strategy.experience.current_experience],
            self.max_consec_delta_from_window(task_acc_window),
        )
        assert len(task_acc_window) <= self.window_size

    def max_consec_delta_from_window(self, window) -> float:
        """Return max A_i - A_j for i<j in the window."""
        if len(window) <= 1:
            return 0
        max_delta = float("-inf")
        max_found_acc = float("-inf")
        for idx, val in enumerate(window):
            if val < max_found_acc:  # Delta can only increase if higher
                continue
            max_found_acc = val
            for other_idx in range(idx + 1, len(window)):  # Deltas with next ones
                other_val = window[other_idx]
                delta = self.delta(val, other_val)

                if delta > max_delta:
                    max_delta = delta
        return max_delta

    @staticmethod
    def delta(first_val, next_val):
        """May overwrite to define increase/decrease.
        For forgetting we look for the largest decrease."""
        return first_val - next_val

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        self.update_current_task_acc(strategy)

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> None:
        self.update_task_window(strategy)


class WindowedPlasticityPluginMetric(WindowedForgettingPluginMetric):
    """For metric definition, see original paper: https://arxiv.org/abs/2205.13452"""

    def __init__(self, window_size):
        super().__init__(window_size)
        self.name = f"P{self.window_size}"  # overwrite name

    @staticmethod
    def delta(first_val, next_val):
        """Largest increase."""
        return next_val - first_val


class TaskTrackingMINAccuracyPluginMetric(TrackerPluginMetric[float]):
    """The accuracy measured per iteration. The minimum accuracy is updated (or created) for tasks that are not
    currently learning. Returns a dictionary of available Acc Minima of all tasks.

    Average over dictionary values to obtain the Average Minimum Accuracy.
    For metric definition, see original paper: https://arxiv.org/abs/2205.13452
    """

    def __init__(self):
        self._current_acc = TaskAwareAccuracy()
        self.min_acc_tasks: dict = defaultdict(lambda: float("inf"))
        super().__init__(name="acc_MIN", metric=self._current_acc, reset_at="iteration")

    def result(self, strategy=None) -> Dict[int, float]:
        return {task: min_acc for task, min_acc in self.min_acc_tasks.items()}

    def update(self, strategy, **kwargs):
        """Loss is updated externally from common stat collector."""
        self._current_acc.update(
            strategy.mb_output, strategy.mb_y, task_labels=strategy.experience.current_experience
        )

    def update_acc_minimum(self, strategy):
        """Update minimum."""
        current_learning_task = strategy.clock.current_training_meta_task
        current_acc_results: Dict[int, float] = self._current_acc.result()
        for task, task_result in current_acc_results.items():
            if task != current_learning_task:  # Not for current learning task
                self.min_acc_tasks[task] = min(self.min_acc_tasks[task], task_result)

    def after_eval_iteration(self, strategy: 'SupervisedTemplate') -> None:
        self.update(strategy)

    def after_eval_exp(self, strategy: 'SupervisedTemplate'):
        self.update_acc_minimum(strategy)


class WCACCPluginMetric(TrackerPluginMetric[float]):
    """Avg over minimum accuracies previous tasks and current accuracy at this training step."""

    def __init__(self, min_acc_plugin: TaskTrackingMINAccuracyPluginMetric):
        self._current_acc = TaskAwareAccuracy()
        self.min_acc_plugin = min_acc_plugin
        self.WCACC = None
        super().__init__(
            name="WCACC", metric=self._current_acc, reset_at="iteration"
        )  # Reset current_acc at iteration

    def result(self, strategy=None) -> dict:
        return {0: self.WCACC}

    def update(self, strategy, **kwargs):
        """Update current acc"""
        current_learning_task = strategy.clock.current_training_meta_task
        if current_learning_task == strategy.experience.current_experience:
            self._current_acc.update(
                strategy.mb_output, strategy.mb_y, task_labels=strategy.experience.current_experience
            )

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        """Update current task acc"""
        self.update(strategy)

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        """Update final metric."""
        self.update_WCACC(strategy)

    def update_WCACC(self, strategy: "SupervisedTemplate"):
        avg_list = []
        current_learning_task = strategy.clock.current_training_meta_task

        if (
            current_learning_task != strategy.experience.current_experience
        ):  # Only update once on current task step
            return

        current_learning_task_acc: float = self._current_acc.result()[
            current_learning_task
        ]
        avg_list.append(current_learning_task_acc)

        # Min-ACC results of OTHER tasks
        min_acc_results: Dict[int, float] = self.min_acc_plugin.result()
        if len(min_acc_results) > 0:
            avg_list.extend(
                [
                    min_acc
                    for task_id, min_acc in min_acc_results.items()
                    if task_id != current_learning_task
                ]
            )

        self.WCACC = torch.mean(torch.tensor(avg_list)).item()


class AAAPluginMetric:
    """ Average accuracy averaged across iterations """

    def __init__(self, average_acc_metric):
        self.average_metric = average_acc_metric
        self._metric = Mean()
        self.last_iteration = 0

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        """Update final metric."""
        self.update(strategy)

    def update(self, strategy, **kwargs):
        current_learning_task = strategy.clock.current_training_meta_task
        if (
            current_learning_task != strategy.experience.current_experience
        ):  
            # Only update once on current task step so that we have seen all the tasks 
            return
        metric_at_this_iteration = self.average_metric.result()
        self._metric.update(metric_at_this_iteration, strategy.clock.train_iterations - self.last_iteration)
        self.last_iteration = strategy.clock.train_iterations

    def result(self) -> dict:
        return {0: self._metric.result()}

    def __str__(self):
        return f"AAA_Stream"

    def _package_result(self, strategy: "SupervisedTemplate", x_pos=None) -> "MetricResult":
        metric_value = self.result()
        add_exp = False
        plot_x_position = (
            strategy.clock.train_iterations if x_pos is None else x_pos
        )  # Allows pre-update step at -1

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def after_eval(self, strategy, **kwargs):
        """Pass to evaluator plugin."""
        return self._package_result(strategy)
