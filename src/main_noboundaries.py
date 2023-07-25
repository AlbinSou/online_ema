#!/usr/bin/env python3
import os
import json

import numpy as np
import torch
from torch.optim import SGD, Adam

from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from src.arguments import parse_arguments
from src.online_factories import create_strategy
from toolkit.dataset import (adapt_model, create_dataset, create_model)
from toolkit.gradient_clipping import GradClippingPlugin
from toolkit.json_logger import ParallelJSONLogger
from toolkit.metrics import (ParameterNormPluginMetric,
                             per_experience_accuracy_metrics,
                             task_aware_accuracy_metrics, CumulativeAccuracyPluginMetric)
from toolkit.metrics.online_continual_eval_metrics import (
    TaskTrackingMINAccuracyPluginMetric, WCACCPluginMetric, AAAPluginMetric)
from toolkit.utils import set_seed
from src.plugins import MeanEvaluation, RetainBestValModel


def main(args):
    # Seed everything
    set_seed(args.seed)

    ##############################
    #  Get data and adapt model  #
    ##############################

    if args.dataset == "cifar10":
        fixed_class_order = np.arange(10)
    elif args.dataset == "split_mnist":
        fixed_class_order = np.arange(10)
    elif args.dataset == "cifar100":
        fixed_class_order = np.arange(100)
    elif args.dataset == "mini_imagenet":
        fixed_class_order = np.arange(100)
    else:
        fixed_class_order = None

    scenario = create_dataset(
        args.scenario,
        args.dataset,
        num_tasks=args.num_tasks,
        seed=args.seed,
        val_size=args.val_size,
        restricted=args.restricted,
        fixed_class_order=fixed_class_order,
        use_transforms=args.use_transforms,
    )

    if "cifar" in args.dataset:
        input_size = (3, 32, 32)
    elif args.dataset == "split_mnist":
        input_size = (1, 28, 28)
    elif args.dataset == "mini_imagenet":
        input_size = (3, 84, 84)

    model = create_model(args.model, input_size)
    model = adapt_model(model, args.scenario)

    ######################
    #  Create Optimizer  #
    ######################

    if args.optimizer == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)

    out_filename = os.path.join(args.results_path, "log.txt")
    text_logger = TextLogger(open(out_filename, "w"))
    file_json = open(os.path.join(args.results_path, "training_logs.json"), "w")
    file_json.close()
    json_logger = ParallelJSONLogger(
        os.path.join(args.results_path, "training_logs.json"), autoupdate=True
    )
    interactive_logger = InteractiveLogger()

    ################################
    #  Create Evaluation Plugin  #
    ################################

    loggers = [text_logger, interactive_logger, json_logger]

    if args.tensorboard:
        tensorboard_logger = TensorboardLogger(
            tb_log_dir=os.path.join(args.results_path, "tb_log_dir"),
        )
        tensorboard_logger.writer.add_text("parameters", str(args))
        loggers.append(tensorboard_logger)

    min_acc_plugin = TaskTrackingMINAccuracyPluginMetric()

    training_metrics = [
        ParameterNormPluginMetric(),
    ]

    accuracy_epoch, accuracy_stream = accuracy_metrics(epoch=True, stream=True)

    evaluation_metrics = [
        accuracy_epoch,
        accuracy_stream,
        per_experience_accuracy_metrics(stream=True),
        loss_metrics(epoch=True, stream=True),
        min_acc_plugin,
        WCACCPluginMetric(min_acc_plugin),
        AAAPluginMetric(accuracy_stream),
    ]

    if args.scenario == "new_classes_incremental":
        evaluation_metrics.append(task_aware_accuracy_metrics(stream=True))
        evaluation_metrics.append(CumulativeAccuracyPluginMetric())

    # Create main evaluator that will be used by the training actor
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )

    ####################
    #  Plugin addition #
    ####################

    plugins = [GradClippingPlugin(clipgrad=args.clipping)]

    if args.device == "cuda":
        args.device += f":{args.gpu}"

    device = torch.device(args.device)

    num_iterations = (
        (len(scenario.train_stream[0].dataset) / args.batch_size) * args.nepochs // 1
    )
    print(num_iterations)
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations, eta_min=1e-7)
        plugins.append(LRSchedulerPlugin(scheduler, step_granularity="iteration"))

    #######################
    #  Strategy Creation  #
    #######################

    cl_strategy = create_strategy(
        model,
        optimizer,
        plugins,
        evaluator,
        device,
        args,
        training_metrics,
        evaluation_metrics,
    )

    ###################
    #  TRAINING LOOP  #
    ###################

    print("Starting experiment...")
    results = []
    results_ema = []

    print([p.__class__.__name__ for p in cl_strategy.plugins])

    # For online scenario
    batch_streams = scenario.streams.values()

    for t, (experience, val_stream) in enumerate(
        zip(scenario.train_stream, scenario.valid_stream)
    ):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=args.batch_size,
            access_task_boundaries=True,
        )

        if args.eval_on_previous:
            validation_stream = [scenario.valid_stream[: t + 1]]
        else:
            validation_stream = [val_stream]

        # Attribute to use at evaluation time
        # for WCACC and MinAcc computation
        # into toolkit/metrics/online_continual_eval_metrics.py
        # ! Not used during training !
        cl_strategy.clock.current_training_meta_task = experience.current_experience

        cl_strategy.train(
            ocl_scenario.train_stream,
            eval_streams=validation_stream,
            num_workers=args.num_workers,
            drop_last=True,
        )

        cl_strategy.eval(scenario.test_stream[: t + 1])

        if args.save_models:
            torch.save(
                cl_strategy.model.state_dict(),
                os.path.join(args.results_path, f"model_{t}.ckpt"),
            )
            for p in cl_strategy.plugins:
                if isinstance(p, MeanEvaluation):
                    torch.save(
                        p.running_model.state_dict(),
                        os.path.join(args.results_path, f"running_model_{t}.ckpt"),
                    )
                    training_model = cl_strategy.model
                    cl_strategy.model = p.running_model.eval()
                    cl_strategy.eval(scenario.test_stream[:t+1])
                    cl_strategy.model = training_model
                if isinstance(p, RetainBestValModel):
                    torch.save(
                        p.best_state.state_dict(),
                        os.path.join(args.results_path, f"best_ema_{t}.ckpt"),
                    )


    # Only evaluate at the end on the test stream
    results = cl_strategy.eval(scenario.test_stream)

    for p in cl_strategy.plugins:
        if isinstance(p, MeanEvaluation):
            training_model = cl_strategy.model
            cl_strategy.model = p.running_model.eval()
            results_ema = cl_strategy.eval(scenario.test_stream)
            cl_strategy.model = training_model
            with open(
                os.path.join(args.results_path, "final_test_results_ema.json"),
                "w",
            ) as f:
                json.dump(results_ema, f)


    with open(
        os.path.join(args.results_path, "final_test_results.json"),
        "w",
    ) as f:
        json.dump(results, f)



if __name__ == "__main__":
    args = parse_arguments(save_config=True)
    main(args)
