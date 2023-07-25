import collections
import copy
import itertools
import os

import ray
from torch.optim import SGD
from typing import List, Dict, Tuple
import torch

# Specify required resources for an actor.
from avalanche.benchmarks import SplitMNIST
from avalanche.core import SupervisedPlugin
from avalanche.logging import TensorboardLogger
from toolkit.json_logger import ParallelJSONLogger
from avalanche.models import SimpleMLP
from avalanche.training import OnlineNaive
from avalanche.training.plugins import EvaluationPlugin

# change this if you want more/less resource per worker
@ray.remote(num_cpus=4, num_gpus=0.5)
class EvaluationActor(object):
    """Parallel Evaluation Actor.

    Methods called on different actors can execute in parallel, and methods
    called on the same actor are executed serially in the order that they
    are called. Methods on the same actor will share state with one another,
    as shown below.  """

    def __init__(self, logdir="actor_log_dir", metrics=[], **strat_args):
        """Constructor.

        Remember to pass an evaluator to the model. Use different logdir for each actor.

        :param strat_args:
        """

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        filename_json = os.path.join(logdir, "logs.json")
        fp = open(filename_json, "w")
        fp.close()

        self.json_logger = ParallelJSONLogger(filename_json)

        evaluator = EvaluationPlugin(
            *metrics,
            loggers=[
                TensorboardLogger(tb_log_dir=logdir),
                self.json_logger,
            ],
        )

        peval_args = {"evaluator": evaluator}

        # NOTE: we need a stateful actor to keep the same logger for each evaluation
        # step and to serialize the eval calls.
        # This could be a queue for each momentum value
        self.strat = OnlineNaive(model=None, optimizer=None, **peval_args, **strat_args)

    def eval(self, model, clock, stream, **kwargs):
        self.strat.model = model
        self.strat.clock = clock
        self.strat.eval(stream, **kwargs)

    def write_files(self):
        self.json_logger.update_json()


class ParallelEval(SupervisedPlugin):
    """Schedules periodic evaluation during training.

    This plugin is automatically configured and added by the BaseTemplate.
    """

    def __init__(
        self,
        plugins,
        metrics,
        results_dir,
        eval_every=-1,
        do_initial=False,
        num_actors=1,
        **actor_args
    ):
        """Init.

        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        :param do_initial: whether to evaluate before each `train` call.
            Occasionally needed becuase some metrics need to know the
            accuracy before training.
        """
        super().__init__()
        # you can create multiple actors in parallel here.
        # keep in mind that you need to have enough resources for all of them.
        # self.eval_actor = EvaluationActor.remote(**actor_args)
        self.metrics = metrics
        self.results_dir = results_dir
        self.num_actors = num_actors
        self.eval_actors, self.eval_plugins = self.create_actors(
            plugins, actor_args, num_actors
        )
        self.eval_every = eval_every
        self.do_initial = do_initial and eval_every > -1
        self.task_queue = []

    def before_training(self, strategy, **kwargs):
        # Here it's causing a huge slowdown if we use 
        # before_training_exp, so we use before_training instead
        # Update stream refs
        #if self.do_initial:
        self.stream_refs = []
        for el in strategy._eval_streams:
            self.stream_refs.append(ray.put(el))
        self._peval(strategy, **kwargs)

    def after_training(self, strategy, **kwargs):
        """
        Clear task_queue
        """
        ray.get(self.task_queue)  # wait for all the evals to finish.
        self.task_queue.clear()
        tasks = []
        for key, actors in self.eval_actors.items():
            for a in actors:
                t = a.write_files.remote()
                tasks.append(t)
        ray.get(tasks)

    def after_training_exp(self, strategy, **kwargs):
        """Final eval after a learning experience."""
        self._peval(strategy, **kwargs)
        #self._maybe_peval(strategy, strategy.clock.train_exp_iterations, **kwargs)

    def _peval(self, strategy, **kwargs):
        # you can run multiple actors in parallel here.
        clock_ref = ray.put(strategy.clock)
        for stream_ref in self.stream_refs:
            # Strategy model
            actor = self.eval_actors["strategy"][strategy.clock.train_iterations%self.num_actors]
            future = actor.eval.remote(
                copy.deepcopy(strategy.model).cpu(),
                clock_ref,
                stream_ref,
                persistent_workers=True if kwargs["num_workers"] > 0 else False,
            )
            self.task_queue.append(future)
            # Potential plugin models
            for key, plugin in self.eval_plugins.items():
                actor = self.eval_actors[key][strategy.clock.train_iterations%self.num_actors]
                future = actor.eval.remote(
                    copy.deepcopy(plugin.model_to_use_at_eval).cpu(),
                    clock_ref,
                    stream_ref,
                    persistent_workers=True if kwargs["num_workers"] > 0 else False,
                )
                self.task_queue.append(future)

    def _maybe_peval(self, strategy, counter, **kwargs):
        if self.eval_every > 0 and counter % self.eval_every == 0:
            # print("LAUNCHING EVAL ON ITERATION {}".format(counter))
            self._peval(strategy, **kwargs)

    def create_actors(self, plugins, actor_args, num_actors):
        actors = collections.defaultdict(lambda: [])
        # Make sure we use the "0" device for evaluation, actual devices will be managed by ray
        actor_args.update({"device":"cuda:0"})

        # Create actors for the strategy model
        for i in range(num_actors):
            actor = EvaluationActor.remote(self.results_dir, self.metrics, **actor_args)
            actors["strategy"].append(actor)

        # Create actors for the potential plugins
        eval_plugins = {}
        for p in plugins:
            if hasattr(p, "model_to_use_at_eval"):
                dirname = os.path.join(self.results_dir, str(p))
                for i in range(num_actors):
                    actor = EvaluationActor.remote(dirname, self.metrics, **actor_args)
                    actors[str(p)].append(actor)
                eval_plugins[str(p)] = p

        return actors, eval_plugins


if __name__ == "__main__":
    ray.init(num_cpus=12, num_gpus=2)

    scenario = SplitMNIST(5)
    model = SimpleMLP(10)
    optimizer = SGD(model.parameters(), lr=0.01)

    # NOTE: eval_every must be -1 to disable the main strategy's PeriodicEval
    # because we are going to use ParallelEval to evaluate the model.
    strat = OnlineNaive(
        model=model,
        optimizer=optimizer,
        train_mb_size=128,
        train_passes=1,
        eval_every=-1,
        eval_mb_size=512,
        device="cuda",
        plugins=[ParallelEval(eval_every=10)],
    )

    strat.train(scenario.train_stream, num_workers=8)
