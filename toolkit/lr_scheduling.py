from copy import deepcopy

from avalanche.training.plugins import SupervisedPlugin

from torch.utils.tensorboard import SummaryWriter


class ReduceLROnPlateauPlugin(SupervisedPlugin):
    """
    Learning Rate Scheduler Plugin that works similarly to Pytorch's ReduceLROnPlateau,
    additionaly, the plugins retains the best model (w.r.t validation stream) at each epoch

    This plugin manages learning rate scheduling inside of a strategy using its
    own scheduling routine derived from Pytorch ReduceLROnPlateau scheduler 

    """

    def __init__(self, optimizer, patience: int, val_stream_name: str, 
    metric_name: str = 'Top1_Acc_Stream', lr_factor: float=0.3, lr_min: float=1e-6,
    reset_scheduler=True, reset_lr=True):
        """
        Creates a ``LRSchedulerPlugin`` instance.

        :param scheduler: a learning rate scheduler that can be updated through
            a step() method and can be reset by setting last_epoch=0
        :param reset_scheduler: If True, the scheduler is reset at the end of
            the experience.
            Defaults to True.
        :param reset_lr: If True, the optimizer learning rate is reset to its
            original value.
            Default to True.
        """
        super().__init__()

        self.optimizer = optimizer

        # Scheduling part
        self.reset_scheduler = reset_scheduler
        self.reset_lr = reset_lr
        self.patience = patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.base_lrs = self.current_lrs

        # Retain best part
        self.val_stream_name = val_stream_name
        self.metric_name = metric_name
        self.metric_key = f'{self.metric_name}/eval_phase/' \
                          f'{self.val_stream_name}'

    @property
    def current_lrs(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def before_training_exp(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.lr_patience = 0

        # Reset last metrics to not fetch val accuracy from last task
        strategy.evaluator.reset_last_metrics()

    def before_training_epoch(self, strategy, **kwargs):
        # We use before_training_epoch so that validation comes from last epoch
        # Perform a scheduler step
        res = strategy.evaluator.get_last_metrics()
        if len(res) == 0:
            val_acc = 0
        else:
            val_acc = [val for k, val in res.items() if self.metric_key in k].pop()

        if self.best_val is None or val_acc > self.best_val:
            # Retain best values
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc

            # Reset Patience
            self.lr_patience = 0
        else:
            self.lr_patience += 1

        # Decrease learning rates by factor and load best models
        param_groups = self.optimizer.param_groups

        if self.lr_patience > self.patience:
            for group, lr in zip(param_groups, self.current_lrs):
                group['lr'] = lr*self.lr_factor
                if group['lr'] < self.lr_min:
                    strategy.model.load_state_dict(self.best_state)
                    strategy.stop_training()
                    break
            self.lr_patience = 0

            # Load best model
            strategy.model.load_state_dict(deepcopy(self.best_state))

    def after_training_exp(self, strategy, **kwargs):

        # Scheduler part
        param_groups = self.optimizer.param_groups
        base_lrs = self.base_lrs

        if self.reset_lr:
            for group, lr in zip(param_groups, base_lrs):
                group['lr'] = lr

        if self.reset_scheduler:
            self.lr_patience = 0

        # Retain Best part
        strategy.model.load_state_dict(deepcopy(self.best_state))

