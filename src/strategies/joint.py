#!/usr/bin/env python3
from typing import Iterable, Sequence, Optional, Union, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


def cross_entropy(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """Calculates cross-entropy with temperature scaling"""
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


class Joint(SupervisedTemplate):
    def __init__(
        self,
        num_tasks: int,
        model: Module,
        optimizer: Optimizer,
        criterion,
        reset_weights = False,
        fast=False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every=-1,
        peval_mode="epoch",
    ):
        """Joint Strategy,
            accumulates the dataset and only train
            during the last task

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop.
                if -1: no evaluation during training.
                if  0: calls `eval` after the final epoch of each training
                    experience.
                if >0: calls `eval` every `eval_every` epochs and at the end
                    of all the epochs for a single experience.
        :param fast: wether to run in fast mode (only train on last exp)
        """

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )

        self.reset_weights = reset_weights
        self.dataset = None  # cumulative dataset
        self.last_experience = num_tasks - 1
        self.fast = fast
        if self.fast:
            print("Running in Fast Mode")

    def train_dataset_adaptation(self, **kwargs):
        """
        Concatenates all the previous experiences.
        """
        if self.dataset is None:
            self.dataset = self.experience.dataset
        else:
            self.dataset = AvalancheConcatDataset(
                [self.dataset, self.experience.dataset]
            )
        self.adapted_dataset = self.dataset

    def _train_exp(self, experience, eval_streams=None, **kwargs):
        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]

        # Reset network weights
        if self.reset_weights:
            for m in self.model.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

        if (
            self.fast
            and experience.current_experience == self.last_experience
            or not self.fast
        ):
            for _ in range(self.train_epochs):
                self._before_training_epoch(**kwargs)

                if self._stop_training:  # Early stopping
                    self._stop_training = False
                    break

                self.training_epoch(**kwargs)
                self._after_training_epoch(**kwargs)

