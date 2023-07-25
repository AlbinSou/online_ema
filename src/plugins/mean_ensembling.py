import copy


from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from toolkit.dataset import model_adaptation
import numpy as np

def uniform_weighting(wold, t, lmb=1):
    return 1

def linear_weighting(wold, t, lmb=1):
    return wold + 1

def log_weighting(wold, t, lmb=1):
    return wold + np.log(t)

def quadratic_weighting(wold, t, lmb=1):
    return wold + t**2

def ema_weighting(wold, t, lmb=0.995):
    return (1/lmb)*wold

class MeanEnsemblingEvaluation(SupervisedPlugin):
    """
    Keeps in memory a running average of the
    model and uses it for evaluation

    General class that can use several weighting scheme
    """

    def __init__(self, update_at="iteration", replace=True, weighting="quadratic"):
        super().__init__()
        self.running_model = None
        self.running_sum = None
        self.total = 0
        self.total_weight = 0
        self.replace = replace

        assert weighting in ["uniform", "linear", "log", "quadratic", "ema"]
        assert update_at in ["experience", "iteration"]
        self.update_at = update_at

        self.weighting_function = globals()[weighting + "_weighting"]
        self.previous_weight = 0

        # Dummy pointer used to not lose training model
        self.training_model = None

    def before_training(self, strategy, **kwargs):
        if self.running_sum is None:
            self.running_sum = copy.deepcopy(strategy.model)
            self.running_model = self.running_sum

    def before_eval_exp(self, strategy, **kwargs):
        if not self.replace:
            model_adaptation(self.running_sum, strategy.experience)
            self.running_sum.to(strategy.device)

    def before_training_exp(self, strategy, **kwargs):
        model_adaptation(self.running_sum, strategy.experience)
        self.running_sum.to(strategy.device)

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
            self.update_running_sum(strategy.model)

    def after_training_exp(self, strategy, **kwargs):
        if self.update_at == "experience":
            self.update_running_sum(strategy.model)

    def update_running_sum(self, model):
        self.total += 1

        weight = self.weighting_function(self.previous_weight, self.total)
        self.previous_weight = weight
        self.total_weight += weight

        for (ks, ps), (kc, pc) in zip(
            self.running_sum.state_dict().items(), model.state_dict().items()
        ):
            if "active" in ks:
                ps.copy_(pc)
                continue
            ps.copy_(ps + weight*pc)

        # Update running model
        self.running_model = copy.deepcopy(self.running_sum)
        for (ks, ps), (kr, pr) in zip(
            self.running_sum.state_dict().items(), self.running_model.state_dict().items()
        ):
            if "active" in kr:
                pr.copy_(ps)
                continue
            pr.copy_(ps/self.total_weight)


    @property
    def model_to_use_at_eval(self):
        return self.running_model

    def __repr__(self):
        return f"mean_ensemble"
