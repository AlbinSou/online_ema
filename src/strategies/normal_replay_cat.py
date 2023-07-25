#!/usr/bin/env python3
from typing import List, Optional, Sequence

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from src.plugins.replay import StoragePlugin
from toolkit.utils import cycle


class NormalReplayCat(SupervisedTemplate):
    """Normal replay where the replay batch, of
    the same size than the batch, is concatenated
    to the current batch"""

    def __init__(
        self,
        memory_size: int,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every=-1,
        peval_mode="epoch",
    ):

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

        # Modified
        self.memory_size = memory_size

        self.replay_plugin = StoragePlugin(mem_size=memory_size)
        self.plugins.append(self.replay_plugin)

    def training_epoch(self, **kwargs):
        """Training epoch."""
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            if self.experience.current_experience > 0:
                br_x, br_y, br_t = next(self.replay_loader)
                br_x, br_y, br_t = (
                    br_x.to(self.device),
                    br_y.to(self.device),
                    br_t.to(self.device),
                )
                self.mbatch[0] = torch.cat((self.mbatch[0], br_x))
                self.mbatch[1] = torch.cat((self.mbatch[1], br_y))
                self.mbatch[2] = torch.cat((self.mbatch[2], br_t))

            self.optimizer.zero_grad()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            # Modified
            self.loss = self.criterion()

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _train_exp(self, experience, eval_streams=None, **kwargs):
        """Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        self.make_optimizer()

        self._before_training_exp(**kwargs)

        # Modified
        if self.experience.current_experience > 0:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    self.replay_plugin.storage_policy.buffer,
                    batch_size=self.train_mb_size,
                    shuffle=True,
                )
            )

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)

        self._after_training_exp(**kwargs)
