#!/usr/bin/env python3
import numpy as np
import torch
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins import (EvaluationPlugin, MIRPlugin,
                                        ReplayLikeMIR)
from avalanche.training.supervised import Naive, OnlineNaive
import torchvision.transforms as transforms


def main():
    # Fix seed
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fixed_class_order = np.arange(10)

    scenario = SplitCIFAR10(
        5,
        return_task_id=False,
        seed=0,
        fixed_class_order=fixed_class_order,
        train_transform=transforms.ToTensor(),
        eval_transform=transforms.ToTensor(),
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
        dataset_root="/DATA/data/",
    )

    scenario = benchmark_with_validation_stream(scenario, 0.05)

    input_size = (3, 32, 32)

    model = SlimResNet18(10)
    # model.linear = IncrementalClassifier(model.linear.in_features, 1)

    optimizer = SGD(model.parameters(), lr=0.05)

    interactive_logger = InteractiveLogger()

    loggers = [interactive_logger]

    training_metrics = []

    evaluation_metrics = [
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
    ]

    # Create main evaluator that will be used by the training actor
    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )

    plugins = [MIRPlugin(1000, subsample=50, batch_size_mem=10)]

    device = torch.device("cuda")

    #######################
    #  Strategy Creation  #
    #######################

    cl_strategy = OnlineNaive(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        evaluator=evaluator,
        device=device,
        train_mb_size=10,
        eval_mb_size=64,
    )

    ###################
    #  TRAINING LOOP  #
    ###################

    print("Starting experiment...")
    results = []

    print([p.__class__.__name__ for p in cl_strategy.plugins])

    # For online scenario
    batch_streams = scenario.streams.values()

    for t, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        ocl_scenario = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=experience,
            experience_size=10,
            access_task_boundaries=False,
        )

        cl_strategy.train(
            ocl_scenario.train_stream,
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )

        cl_strategy.eval(scenario.test_stream[: t + 1])

    # Only evaluate at the end on the test stream
    results.append(cl_strategy.eval(scenario.test_stream))

    return results


if __name__ == "__main__":
    main()
