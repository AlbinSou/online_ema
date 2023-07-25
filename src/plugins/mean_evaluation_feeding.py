import copy


from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from toolkit.dataset import model_adaptation

class MeanEvaluationSelfFeeding(SupervisedPlugin):
    """
    Keeps in memory a running average of the
    model and uses it for evaluation

    After every training iteration, replaces the content of
    the current model parameters with the current EMA model
    """

    def __init__(self, momentum=0.99, update_at="iteration", replace=True):
        super().__init__()
        self.running_model = None
        self.momentum = momentum
        self.replace = replace

        assert update_at in ["experience", "iteration"]
        self.update_at = update_at

        # Dummy pointer used to not lose training model
        self.training_model = None

    def before_training(self, strategy, **kwargs):
        if self.running_model is None:
            self.running_model = copy.deepcopy(strategy.model)

    def before_eval_exp(self, strategy, **kwargs):
        if not self.replace:
            model_adaptation(self.running_model, strategy.experience)
            self.running_model.to(strategy.device)

    def before_training_exp(self, strategy, **kwargs):
        model_adaptation(self.running_model, strategy.experience)
        self.running_model.to(strategy.device)

    def before_eval(self, strategy, **kwargs):
        if not self.replace:
            return
        self.training_model = strategy.model
        strategy.model = copy.deepcopy(self.running_model)
        strategy.model.eval()

    def after_eval(self, strategy, **kwargs):
        if not self.replace:
            return
        strategy.model = self.training_model
        strategy.model.train()

    def after_training_iteration(self, strategy, **kwargs):
        if self.update_at == "iteration":
            self.update_running_model(strategy.model)
            strategy.model.load_state_dict(self.running_model.state_dict())

    def after_training_exp(self, strategy, **kwargs):
        if self.update_at == "experience":
            self.update_running_model(strategy.model)
            strategy.model.load_state_dict(self.running_model.state_dict())

    def update_running_model(self, model):
        for (kr, pr), (kc, pc) in zip(
            self.running_model.state_dict().items(), model.state_dict().items()
        ):
            if "active" in kr:
                pr.copy_(pc)
                continue
            pr.copy_(self.momentum * pr + (1 - self.momentum) * pc)

    @property
    def model_to_use_at_eval(self):
        return self.running_model

    def __repr__(self):
        return f"ema_{self.momentum}"
