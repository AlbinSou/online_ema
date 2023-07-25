#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import copy
import random
from pprint import pprint
from typing import TYPE_CHECKING

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.benchmarks.utils import concat_datasets

if TYPE_CHECKING:
    pass


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


class ConcatReplayPlugin(SupervisedPlugin):
    """Replay Plugin that works closer from the one used in MIR and ER-ACE.
    A separate loader is kept. However, it still keeps the advantage from replay as implemented
    in Avalanche, in that it performs only one forward pass with data from both replay and current task,
    leading to cleaner batch norm statistics adaptation I guess ? At least I saw differences from the
    results of one method vs another """

    def __init__(
        self, mem_size: int = 200, batch_size_mem: int = 10, adaptive_size=True
    ):
        super().__init__()
        self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.storage_policy = ClassBalancedBuffer(
            self.mem_size, adaptive_size=adaptive_size
        )
        self.replay_loader = None

    def before_training_exp(self, strategy, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        else:
            self.replay_loader = None

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)

    def before_forward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid = next(self.replay_loader)
        batch_x, batch_y, batch_tid = (
            batch_x.to(strategy.device),
            batch_y.to(strategy.device),
            batch_tid.to(strategy.device),
        )

        strategy.mbatch[0], strategy.mbatch[1], strategy.mbatch[2] = (
            torch.cat((strategy.mbatch[0], batch_x)),
            torch.cat((strategy.mbatch[1], batch_y)),
            torch.cat((strategy.mbatch[2], batch_tid)),
        )
