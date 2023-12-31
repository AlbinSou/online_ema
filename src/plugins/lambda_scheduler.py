from avalanche.evaluation.metric_results import MetricValue
from avalanche.training.plugins import SupervisedPlugin
from toolkit.utils import linear_schedule


class LambdaScheduler(SupervisedPlugin):
    """
    Updates the regularization parameter of a plugin after each experience,
    """

    def __init__(
        self,
        plugin,
        scheduled_key: str,
        num_warmup_iterations: int,
        schedule_by: str = "epoch",
        start_value=0.0,
        end_value=1.0,
        scheduling_function=linear_schedule,
        reset_at=None,
    ):
        """
        :param plugin: plugin object
        :param scheduled_key: key to schedule
        :param schedule_by: epoch or iteration
        """
        self.plugin = plugin
        self.key = scheduled_key
        self.schedule_by = schedule_by
        self.total_data = 0
        self.reset_at = reset_at

        self.offset = 0

        assert schedule_by in ["iteration", "epoch", "data", "experience"]

        if plugin is None:
            print(f"Scheduling Strategy key {self.key}")
        else:
            assert hasattr(plugin, self.key)
            print(f"Scheduling plugin {plugin.__class__.__name__} key {self.key}")

        self.scheduler = scheduling_function(
            num_warmup_iterations, start_value, end_value
        )
        self._set_plugin_attr(self.scheduler(0))


    def before_training_epoch(self, strategy, **kwargs):
        if self.schedule_by != "epoch":
            return
        counter = strategy.clock.train_exp_epochs
        value = self.scheduler(counter)
        self._set_plugin_attr(value, strategy)

    def _set_plugin_attr(self, value, strategy=None):
        if self.plugin is not None:
            setattr(self.plugin, self.key, value)
        else:
            if strategy is not None:
                setattr(strategy, self.key, value)

        if strategy is not None:
            strategy.evaluator.publish_metric_value(
                MetricValue(
                    "Metric",
                    f"{self.key}",
                    value,
                    x_plot=strategy.clock.train_iterations,
                )
            )


    def before_training_iteration(self, strategy, **kwargs):
        if self.schedule_by != "iteration":
            return
        counter = strategy.clock.train_iterations - self.offset
        value = self.scheduler(counter)
        self._set_plugin_attr(value, strategy)

    def after_training_exp(self, strategy, **kwargs):
        if self.reset_at == "experience" and self.schedule_by == "iteration":
            self.offset = strategy.clock.train_iterations

        #if self.schedule_by != "data":
        #    return
        #self.total_data += len(strategy.experience.dataset)
        #value = self.scheduler(self.total_data)
        if self.schedule_by != "experience":
            return
        value = self.scheduler(strategy.experience.current_experience)
        self._set_plugin_attr(value, strategy)
