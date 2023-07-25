#!/usr/bin/env python3

import copy
from typing import TYPE_CHECKING

import torch

from avalanche.benchmarks.utils import concat_datasets
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


class ReplayLikeMIR(SupervisedPlugin):
    """
    Replay plugin but working like MIR for fair comparison
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size_mem: int = None,
    ):
        """
        mem_size: int       : Fixed memory size
        batch_size_mem: int : Size of the batch sampled from
                              the bigger subsample batch
        """
        super().__init__()
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_backward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return
        samples_x, samples_y, samples_tid = next(self.replay_loader)
        samples_x, samples_y, samples_tid = (
            samples_x.to(strategy.device),
            samples_y.to(strategy.device),
            samples_tid.to(strategy.device),
        )
        replay_output = avalanche_forward(strategy.model, 
                                          samples_x, samples_tid)
        replay_loss = strategy._criterion(replay_output, samples_y)
        strategy.loss += replay_loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        # Exclude classes that were in the last batch
        buffer = concat_datasets(
            [
                self.storage_policy.buffer_groups[key].buffer
                for key, _ in self.storage_policy.buffer_groups.items()
                if int(key) not in torch.unique(strategy.mb_y).cpu()
            ]
        )
        if len(buffer) > 0:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer, batch_size=self.batch_size_mem, shuffle=True
                )
            )
        else:
            self.replay_loader = None
