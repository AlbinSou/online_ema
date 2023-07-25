#!/usr/bin/env python3
import copy

import ray
import torch
from torch.nn import CrossEntropyLoss

from avalanche.training.plugins import (LRSchedulerPlugin, MIRPlugin)
from src.plugins import (ConcatReplayPlugin, DERPlugin, LambdaScheduler, 
                         MeanEvaluation, RetainBestValModel)
from src.plugins.mean_ensembling import MeanEnsemblingEvaluation
from src.plugins.mean_evaluation_feeding import MeanEvaluationSelfFeeding
from src.plugins.early_stopping import MemoryEarlyStopping
from toolkit.online_parallel_eval import ParallelEval
from toolkit.utils import step_rampup
from avalanche.training import *

dataset_lengths = {
    "split_mnist": 60000,
    "cifar10": 50000,
    "cifar100": 50000,
    "mini_imagenet": 50000,
}


def create_strategy(
    model,
    optimizer,
    plugins,
    evaluator,
    device,
    config,
    training_metrics=None,
    evaluation_metrics=None,
):

    arguments = {
        "criterion": CrossEntropyLoss(),
        "train_mb_size": config.batch_size,
        "eval_mb_size": config.batch_size,
        "device": device,
        "train_passes": config.nepochs,
    }

    naive_strategy_arguments = copy.deepcopy(arguments)

    arguments.update(
        {
            "model": model,
            "optimizer": optimizer,
            "plugins": plugins,
            "evaluator": evaluator,
            "eval_every": config.every,
        }
    )

    strategy_dict = {}
    strategy_name = None

    num_iterations = (
        (
            (
                dataset_lengths[config.dataset]
                - config.val_size * dataset_lengths[config.dataset]
            )
            / (config.batch_size * config.num_tasks)
        )
        * config.nepochs
        // 1
    )
    print(num_iterations)

    if config.strategy == "naive":
        strategy_name = "OnlineNaive"

    elif config.strategy == "er_ace":
        strategy_name = "OnlineER_ACE"
        strategy_dict = {
            "mem_size": config.memory_size,
            "batch_size_mem": config.batch_size,
        }

    elif config.strategy == "der":
        strategy_name = "OnlineNaive"
        der = DERPlugin(
            mem_size=config.memory_size,
            mode="++",
            batch_size_mem=config.batch_size,
            alpha=0.1,
            beta=0.5,
        )
        plugins.append(der)

    elif config.strategy == "er_early_stopping":
        strategy_name = "OnlineNaive"
        replay = ConcatReplayPlugin(
            int(config.memory_size - config.lmb2), batch_size_mem=config.batch_size
        )
        mean_evaluation = MeanEvaluation(momentum=0.99, update_at="experience")
        early_stopping = MemoryEarlyStopping(
            int(config.lmb2),
            patience=200,
            peval_mode="iteration",
            optional_plugin=mean_evaluation,
        )
        plugins.extend([replay, early_stopping, mean_evaluation])

    elif config.strategy == "mir":
        strategy_name = "OnlineNaive"
        replay = MIRPlugin(
            config.memory_size, subsample=200, batch_size_mem=config.batch_size
        )
        plugins.append(replay)

    elif config.strategy == "er":
        strategy_name = "OnlineNaive"
        replay = ConcatReplayPlugin(
            config.memory_size, batch_size_mem=config.batch_size
        )
        plugins.append(replay)

    elif config.strategy == "mean_ensembling":
        strategy_name = "OnlineNaive"
        replay = ConcatReplayPlugin(
            config.memory_size, batch_size_mem=config.batch_size
        )
        mean_ensembling_eval = MeanEnsemblingEvaluation(weighting="quadratic")
        plugins.append(mean_ensembling_eval)
        plugins.append(replay)

    elif config.strategy == "test_low_lr":
        strategy_name = "OnlineNaive"
        replay = ConcatReplayPlugin(
            config.memory_size, batch_size_mem=config.batch_size
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [num_iterations], gamma=0.01
        )
        plugins.append(LRSchedulerPlugin(scheduler, step_granularity="iteration"))
        plugins.append(replay)

    elif config.strategy == "test_self_feeding":
        strategy_name = "OnlineNaive"
        replay = ConcatReplayPlugin(
            config.memory_size, batch_size_mem=config.batch_size
        )
        mean_eval = MeanEvaluationSelfFeeding(momentum=config.lmb2)
        scheduler = LambdaScheduler(
            mean_eval,
            "momentum",
            num_warmup_iterations=num_iterations,
            schedule_by="iteration",
            start_value=0.9,
            scheduling_function=step_rampup,
            end_value=config.lmb2,
        )
        plugins.append(scheduler)
        plugins.append(mean_eval)
        plugins.append(replay)

    # Add additional plugins and reorder them

    add_plugins(config, plugins, num_iterations)

    # Insert potential ParallelEval Plugin
    if config.parallel_evaluation:
        ray.init(num_cpus=24, num_gpus=2)
        parallel_eval_plugin = ParallelEval(
            plugins,
            evaluation_metrics,
            config.results_path,
            eval_every=config.every,
            num_actors=1,
            **naive_strategy_arguments,
        )
        plugins.append(parallel_eval_plugin)
        # Deactivate Sequential PEval plugin
        # If we want to evaluate also the normal model,
        # Use a plugin that has it as its use_at_eval attribute
        strategy_dict["eval_every"] = -1

    arguments.update(strategy_dict)
    arguments["plugins"] = plugins

    cl_strategy = globals()[strategy_name](**arguments)

    print(strategy_name)

    return cl_strategy


def add_plugins(config, plugins, num_iterations):
    if config.mean_evaluation:
        mean_evaluation = MeanEvaluation(
            config.lmb2,
            replace=False,
            update_at="iteration",
        )
        scheduler = LambdaScheduler(
            mean_evaluation,
            "momentum",
            num_warmup_iterations=num_iterations,
            schedule_by="iteration",
            start_value=0.9,
            scheduling_function=step_rampup,
            end_value=config.lmb2,
        )
        plugins.append(mean_evaluation)
        plugins.append(scheduler)
    if config.retain_best:
        mean_evaluation = mean_evaluation if config.mean_evaluation else None
        retain_best = RetainBestValModel(
            int(config.lmb),
            replace=False,
            optional_plugin=mean_evaluation,
        )
        plugins.append(retain_best)
