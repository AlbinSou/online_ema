#!/usr/bin/env python3
from typing import List

from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.models.dynamic_modules import (MultiHeadClassifier,
                                              MultiTaskModule)


def acc_datasets(scenario) -> AvalancheDataset:
    """
    Accumulate the datasets present in a scenario
    param: scenario
    param: growing: Boolean, whether to return a list of
                    growing datasets or just one big dataset
    """
    train_dataset = None
    test_dataset = None

    for train_experience, test_experience in zip(
        scenario.train_stream, scenario.test_stream
    ):
        if train_dataset is None:
            train_dataset = train_experience.dataset
            test_dataset = test_experience.dataset
        else:
            train_dataset = AvalancheConcatDataset(
                [train_dataset, train_experience.dataset]
            )
            test_dataset = AvalancheConcatDataset(
                [test_dataset, test_experience.dataset]
            )

    return train_dataset, test_dataset


def acc_growing_datasets(scenario) -> List[AvalancheDataset]:
    """
    Accumulate the datasets present in a scenario
    param: scenario
    param: growing: Boolean, whether to return a list of
                    growing datasets or just one big dataset
    """
    train_dataset = None
    test_dataset = None

    train_datasets = []
    test_datasets = []

    for train_experience, test_experience in zip(
        scenario.train_stream, scenario.test_stream
    ):
        if train_dataset is None:
            train_dataset = train_experience.dataset
            test_dataset = test_experience.dataset
        else:
            train_dataset = AvalancheConcatDataset(
                [train_dataset, train_experience.dataset]
            )
            test_dataset = AvalancheConcatDataset(
                [test_dataset, test_experience.dataset]
            )

        train_datasets.append(train_dataset)
        test_datasets.append(train_dataset)

    return train_datasets, test_datasets


def extract_known_tasks(model):
    tasks = set([0])
    if isinstance(model, MultiTaskModule):
        for module in model.modules():
            if isinstance(module, MultiHeadClassifier):
                for k in module.classifiers.keys():
                    tasks.add(int(k))
    return tasks

def has_plugin(strategy, plugin_type):
    for p in strategy.plugins:
        if isinstance(p, plugin_type):
            return True, p
    return False, None
