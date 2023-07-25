import operator
import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F

from avalanche.models import avalanche_forward
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from src.plugins import MeanEvaluation
from toolkit.utils import evaluating
from avalanche.evaluation.metric_results import MetricValue


class EarlyStoppingPlugin(SupervisedPlugin):
    """Early stopping and model checkpoint plugin.

    The plugin checks a metric and stops the training loop when the accuracy
    on the metric stopped progressing for `patience` epochs.
    After training, the best model's checkpoint is loaded.

    .. warning::
        The plugin checks the metric value, which is updated by the strategy
        during the evaluation. This means that you must ensure that the
        evaluation is called frequently enough during the training loop.

        For example, if you set `patience=1`, you must also set `eval_every=1`
        in the `BaseTemplate`, otherwise the metric won't be updated after
        every epoch/iteration. Similarly, `peval_mode` must have the same
        value.

    Slightly modified version that restores both the strategy model
    (training model), but also potentially other models (i.e running model used for evaluation)

    """

    def __init__(
        self,
        patience: int,
        val_stream_name: str,
        metric_name: str = "Top1_Acc_Stream",
        mode: str = "max",
        peval_mode: str = "epoch",
        margin: float = 0.0,
    ):
        """Init.

        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
            metrics. The corresponding stream will be used to keep track of the
            evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
            reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
            given metric should me maximized (resp. minimized).
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            early stopping should happen after `patience`
            epochs or iterations (Default='epoch').
        :param margin: a minimal margin of improvements required to be
            considered best than a previous one. It should be an float, the
            default value is 0. That means that any improvement is considered
            better.
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.patience = patience

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        assert type(margin) == float
        self.margin = margin

        self.metric_name = metric_name
        self.metric_key = f"{self.metric_name}/eval_phase/" f"{self.val_stream_name}"

        if mode not in ("max", "min"):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == "max" else operator.lt

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step = None

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None

        # Check for mean eval plugin
        for plugin in strategy.plugins:
            if isinstance(plugin, MeanEvaluation):
                self.mean_eval_plugin = plugin

    def before_training_exp(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None

    def before_training_iteration(self, strategy, **kwargs):
        if self.peval_mode == "iteration":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return
            curr_step = self._get_strategy_counter(strategy)
            if curr_step - self.best_step >= self.patience:
                self._set_best_state(strategy)
                strategy.stop_training()

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return
            curr_step = self._get_strategy_counter(strategy)
            if curr_step - self.best_step >= self.patience:
                self._set_best_state(strategy)
                strategy.stop_training()

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        names = [k for k in res.keys() if k.startswith(self.metric_key)]
        if len(names) == 0:
            return None

        full_name = names[-1]
        val_acc = res.get(full_name)
        if self.best_val is None:
            warnings.warn(
                f"Metric {self.metric_name} used by the EarlyStopping plugin "
                f"is not computed yet. EarlyStopping will not be triggered."
            )
        if self.best_val is None or self.operator(val_acc, self.best_val):
            self.best_state = self._get_best_state(strategy)
            if self.best_val is None:
                self.best_val = val_acc
                self.best_step = 0
                return None

            if self.operator(float(val_acc - self.best_val), self.margin):
                self.best_step = self._get_strategy_counter(strategy)
                self.best_val = val_acc

        return self.best_val

    def _get_strategy_counter(self, strategy):
        if self.peval_mode == "epoch":
            return strategy.clock.train_exp_epochs
        elif self.peval_mode == "iteration":
            return strategy.clock.train_exp_iterations
        else:
            raise ValueError("Invalid `peval_mode`:", self.peval_mode)

    def _get_best_state(self, strategy):
        if self.mean_eval_plugin is not None:
            return deepcopy(strategy.model.state_dict()), deepcopy(
                self.mean_eval_plugin.running_model.state_dict()
            )
        else:
            return deepcopy(strategy.model.state_dict())

    def _set_best_state(self, strategy):
        if self.mean_eval_plugin is not None:
            strategy.model.load_state_dict(self.best_state[0])
            self.mean_eval_plugin.running_model.load_state_dict(self.best_state[1])
        else:
            strategy.model.load_state_dict(self.best_state)


@torch.no_grad()
def eval_model_loss(model, dataset, batch_size, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    loss = 0
    total = 0
    with evaluating(model) as dmodel:
        for mb_x, mb_y, mb_task_id in loader:
            mb_x, mb_y, mb_task_id = (
                mb_x.to(device),
                mb_y.to(device),
                mb_task_id.to(device),
            )
            out = avalanche_forward(dmodel, mb_x, mb_task_id)
            loss += F.cross_entropy(out, mb_y)
            total += len(mb_x)
    return loss / total


class MemoryEarlyStopping(SupervisedPlugin):
    def __init__(
        self,
        memory_size: int,
        patience: int,
        peval_mode: str = "iteration",
        optional_plugin=None,
    ):
        """Init.
        :param patience: Number of epochs to wait before stopping the training.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            early stopping should happen after `patience`
            epochs or iterations (Default='epoch').
        """
        super().__init__()
        self.patience = patience

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step = None

        self.storage_policy = ClassBalancedBuffer(memory_size, adaptive_size=True)

        self.mean_eval_plugin = optional_plugin

    def before_training_exp(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None
        self.storage_policy.update(strategy, **kwargs)

    def before_training_iteration(self, strategy, **kwargs):
        if self.peval_mode == "iteration":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return
            if self._get_strategy_counter(strategy) - self.best_step >= self.patience:
                self._set_state(strategy)
                strategy.stop_training()

    def after_training_iteration(self, strategy, **kwargs):
        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "best_memory_validation",
                float(self.best_val.cpu().numpy()),
                x_plot=strategy.clock.train_iterations,
            )
        )

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return
            if self._get_strategy_counter(strategy) - self.best_step >= self.patience:
                self._set_state(strategy)
                strategy.stop_training()

            
    def _model_for_eval(self, strategy):
        if self.mean_eval_plugin is not None:
            return self.mean_eval_plugin.running_model
        else:
            return strategy.model

    def _update_best(self, strategy):
        if len(self.storage_policy.buffer) == 0:
            return None

        val_loss = eval_model_loss(
            self._model_for_eval(strategy),
            self.storage_policy.buffer,
            strategy.train_mb_size,
            strategy.device,
        )

        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "current_memory_validation",
                float(val_loss.cpu().numpy()),
                x_plot=strategy.clock.train_iterations,
            )
        )

        if self.best_val is None or val_loss < self.best_val:
            self.best_state = self._get_state(strategy)
            if self.best_val is None:
                self.best_val = val_loss
                self.best_step = 0
                return None

            if val_loss < self.best_val:
                self.best_step = self._get_strategy_counter(strategy)
                self.best_val = val_loss

        return self.best_val

    def _get_strategy_counter(self, strategy):
        if self.peval_mode == "epoch":
            return strategy.clock.train_exp_epochs
        elif self.peval_mode == "iteration":
            return strategy.clock.train_exp_iterations

    def _get_state(self, strategy):
        if self.mean_eval_plugin is not None:
            return deepcopy(strategy.model.state_dict()), deepcopy(
                self.mean_eval_plugin.running_model.state_dict()
            )
        else:
            return deepcopy(strategy.model.state_dict())

    def _set_state(self, strategy):
        if self.mean_eval_plugin is not None:
            strategy.model.load_state_dict(self.best_state[0])
            self.mean_eval_plugin.running_model.load_state_dict(self.best_state[1])
        else:
            strategy.model.load_state_dict(self.best_state)

    @property
    def model_to_use_at_eval(self):
        if self.best_state is None:
            return self.mean_eval_plugin.running_model
        else:
            model = deepcopy(self.mean_eval_plugin.running_model)
            model.load_state_dict(self.best_state[1])
        return model

    def __repr__(self):
        return f"best_validation"

class RetainBestValModel(SupervisedPlugin):
    def __init__(
        self,
        memory_size: int,
        peval_mode: str = "iteration",
        optional_plugin=None,
        replace: bool = False,
    ):
        """
        Not an early stopping plugin, just train on everything 
        and retain the best ema model at each experience, using small set aside 
        memory validation buffer
        """
        super().__init__()

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.replace = replace

        self.storage_policy = ClassBalancedBuffer(memory_size, adaptive_size=True)

        self.mean_eval_plugin = optional_plugin
        self.training_model = None

    def before_training_exp(self, strategy, **kwargs):
        # Not for online learning
        #self.best_state = None
        #self.best_val = None
        self.storage_policy.update(strategy, **kwargs)

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None

    def before_training_iteration(self, strategy, **kwargs):
        if self.peval_mode == "iteration":
            ub = self._update_best(strategy)
            if ub is None or self.best_val is None:
                return

    def after_training_iteration(self, strategy, **kwargs):
        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "best_memory_validation",
                float(self.best_val.cpu().numpy()),
                x_plot=strategy.clock.train_iterations,
            )
        )

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return

    def _model_for_eval(self, strategy):
        if self.mean_eval_plugin is not None:
            return self.mean_eval_plugin.running_model
        else:
            return strategy.model

    def _update_best(self, strategy):
        if len(self.storage_policy.buffer) == 0:
            return None

        val_loss = eval_model_loss(
            self._model_for_eval(strategy),
            self.storage_policy.buffer,
            strategy.train_mb_size,
            strategy.device,
        )

        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "current_memory_validation",
                float(val_loss.cpu().numpy()),
                x_plot=strategy.clock.train_iterations,
            )
        )

        if self.best_val is None or val_loss < self.best_val:
            self.best_state = self._get_state(strategy)
            if self.best_val is None:
                self.best_val = val_loss
                self.best_step = 0
                return None

            if val_loss < self.best_val:
                self.best_val = val_loss

        return self.best_val

    def _get_state(self, strategy):
        if self.mean_eval_plugin is not None:
            return deepcopy(self.mean_eval_plugin.running_model)
        else:
            return deepcopy(strategy.model)

    def before_eval(self, strategy, **kwargs):
        if not self.replace:
            return
        self.training_model = strategy.model
        strategy.model = self.best_state
        strategy.model.eval()

    def after_eval(self, strategy, **kwargs):
        if not self.replace:
            return
        strategy.model = self.training_model
        strategy.model.train()

    @property
    def model_to_use_at_eval(self):
        if self.best_state is None:
            return self.mean_eval_plugin.running_model
        else:
            model = self.best_state
        return model

    def __repr__(self):
        return f"best_validation"
