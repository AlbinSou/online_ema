#!/usr/bin/env python3
from typing import Dict

import torch
from pytorchcv.model_provider import get_model
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor

from avalanche.benchmarks import (OnlineCLExperience,
                                  benchmark_with_validation_stream,
                                  dataset_benchmark, nc_benchmark,
                                  ni_benchmark)
from avalanche.benchmarks.classic import PermutedMNIST, SplitImageNet
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.models import DynamicModule, SimpleMLP, as_multitask
from avalanche.models.dynamic_modules import IncrementalClassifier
from toolkit.miniimagenet_benchmark import SplitMiniImageNet
from toolkit.models import resnet32, simple_cnn
from toolkit.resnet18 import resnet18

DATADIR = "/DATA/data/"


def model_adaptation(model, experience):
    if isinstance(experience, OnlineCLExperience):
        if experience.access_task_boundaries:
            experience = experience.origin_experience
    for module in model.modules():
        if isinstance(module, DynamicModule):
            module.adaptation(experience)


def create_model(model_type: str, input_size=(3, 32, 32), dropout=0.25):
    if model_type == "resnet18":
        model = resnet18(1, input_size=input_size)
    elif model_type == "resnet32":
        model = resnet32(1)
    elif model_type == "resnet50":
        model = get_model("resnet50", pretrained=False)
    elif model_type == "cnn":
        model = simple_cnn(1, dropout=dropout)
    elif model_type == "mlp":
        model = SimpleMLP(1, hidden_size=400, input_size=input_size[1] * input_size[2])
    return model


def create_dataset(
    scenario_type: str,
    dataset: str,
    num_tasks: int,
    seed: int,
    val_size: float,
    datadir: str = DATADIR,
    use_transforms: bool = False,
    **kwargs,
):
    """Provides the right scenario type for the given dataset"""

    if dataset == "cifar100":
        # --- TRANSFORMATIONS
        train_transforms_list = []

        if use_transforms:
            train_transforms_list.extend(
                [RandomCrop(32, padding=4), RandomHorizontalFlip()]
            )

        train_transforms_list.extend(
            [
                ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
                ),
            ]
        )

        train_transform = transforms.Compose(train_transforms_list)

        test_transform = transforms.Compose(
            [
                ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
                ),
            ]
        )

        ds_train = CIFAR100(root=datadir, train=True, download=True)
        ds_test = CIFAR100(root=datadir, train=False, download=True)

    elif dataset == "cifar10":
        # --- TRANSFORMATIONS
        train_transforms_list = []

        if use_transforms:
            train_transforms_list.extend(
                [RandomCrop(32, padding=4), RandomHorizontalFlip()]
            )

        train_transforms_list.extend(
            [
                ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_transform = transforms.Compose(train_transforms_list)

        test_transform = transforms.Compose(
            [
                ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        ds_train = CIFAR10(root=datadir, train=True, download=True)
        ds_test = CIFAR10(root=datadir, train=False, download=True)

    elif dataset == "cifar10_special":
        # --- TRANSFORMATIONS
        train_transform = transforms.Compose(
            [
                ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        ds_train = CIFAR10(root=datadir, train=True, download=True)
        ds_test = CIFAR10(root=datadir, train=False, download=True)

        per_exp_classes = {0: 2, 1: 3, 2: 2, 3: 3}
        kwargs.update({"per_exp_classes": per_exp_classes})

    elif dataset == "permuted_mnist":
        assert scenario_type == "new_instances"
        scenario = PermutedMNIST(
            num_tasks,
            dataset_root=DATADIR,
            seed=seed,
        )
        scenario = benchmark_with_validation_stream(scenario, val_size, shuffle=True)
        return scenario

    elif dataset == "split_mnist":
        train_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )

        ds_train = MNIST(root=datadir, train=True, download=True)
        ds_test = MNIST(root=datadir, train=False, download=True)

    elif dataset == "mini_imagenet":
        assert scenario_type in ["new_classes_multitask", "new_classes_incremental"]
        if scenario_type == "new_classes_multitask":
            ret_tid = True
        else:
            ret_tid = False

        train_transform_list = [transforms.ToPILImage()]

        if use_transforms:
            train_transform_list.extend(
                [RandomCrop(84, padding=10), RandomHorizontalFlip()]
            )

        train_transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_transform = transforms.Compose(train_transform_list)

        test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        scenario = SplitMiniImageNet(
            n_experiences=num_tasks,
            root_path=DATADIR,
            seed=seed,
            train_transform=train_transform,
            test_transform=test_transform,
            return_task_id=ret_tid,
            **kwargs,
        )
        scenario = benchmark_with_validation_stream(scenario, val_size, shuffle=True)
        return scenario

    elif dataset == "split_imagenet":
        assert scenario_type in ["new_classes_multitask", "new_classes_incremental"]
        if scenario_type == "new_classes_multitask":
            ret_tid = True
        else:
            ret_tid = False

        scenario = SplitImageNet(
            n_experiences=num_tasks,
            dataset_root=DATADIR,
            seed=seed,
            return_task_id=ret_tid,
            **kwargs,
        )
        scenario = benchmark_with_validation_stream(scenario, val_size, shuffle=True)
        return scenario

    try:
        scenario = globals()[scenario_type](
            ds_train,
            ds_test,
            train_transform,
            test_transform,
            num_tasks=num_tasks,
            seed=seed,
            val_size=val_size,
            **kwargs,
        )
    except KeyError:
        raise AttributeError(f"The following scenario is not available {scenario_type}")

    return scenario


def adapt_model(model, scenario_type, scenario=None, t=None):
    """
    Adapts a model to a scenario_type

        scenario_type: str
        scenario: Optional Scenario, if given, will do the full model
                  adaptation (provide if testing is required
                              but not for training)
    """
    last_layer_name = list(model.named_parameters())[-1][0].split(".")[0]
    if (
        scenario_type == "new_classes_multitask"
        or scenario_type == "new_classes_multitask_unbalanced"
    ):
        model = as_multitask(model, last_layer_name)
    else:
        classifier = IncrementalClassifier(
            getattr(model, last_layer_name).in_features, 1
        )
        setattr(model, last_layer_name, classifier)

    if scenario:
        if t is None:
            t = len(scenario.train_stream)
        for experience in scenario.train_stream[: t + 1]:
            model_adaptation(model, experience)

    return model


def new_instances(
    ds_train,
    ds_test,
    train_transform,
    test_transform,
    num_tasks=10,
    seed=0,
    val_size=0.05,
    **kwargs,
):
    # Create benchmark

    scenario_ni = ni_benchmark(
        ds_train,
        ds_test,
        num_tasks,
        task_labels=False,
        train_transform=train_transform,
        eval_transform=test_transform,
        seed=seed,
        balance_experiences=True,
    )

    scenario = benchmark_with_validation_stream(
        scenario_ni, validation_size=val_size, shuffle=True
    )

    return scenario


def new_classes_multitask(
    ds_train,
    ds_test,
    train_transform,
    test_transform,
    num_tasks=10,
    seed=0,
    val_size=0.05,
    fixed_class_order=None,
    per_exp_classes=None,
    **kwargs,
):
    """
    Creates a benchmark of num_tasks tasks that
    are each composed of hard class splits of the
    provided training dataset.

    Task id should be provided at test time so
    that the network knows which task to perform.
    Initial labels are turned into task-specific labels,
    all starting from 0.

    i.e Initial classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Task 0. : [0, 1] -> [0, 1]

    Task 1. : [2, 3] -> [0, 1]

    [...]

    Task 4. : [8, 9] -> [0, 1]

    """
    # Create benchmark

    scenario_nc = nc_benchmark(
        ds_train,
        ds_test,
        num_tasks,
        task_labels=True,
        train_transform=train_transform,
        eval_transform=test_transform,
        seed=seed,
        class_ids_from_zero_in_each_exp=True,
        fixed_class_order=fixed_class_order,
        per_exp_classes=per_exp_classes,
    )
    print(scenario_nc.classes_order_original_ids)
    scenario = benchmark_with_validation_stream(
        scenario_nc, validation_size=val_size, shuffle=True
    )

    return scenario


def new_classes_multitask_unbalanced(
    ds_train,
    ds_test,
    train_transform,
    test_transform,
    num_tasks=10,
    seed=0,
    val_size=0.05,
    restricted: Dict[int, int] = None,
    fixed_class_order=None,
    **kwargs,
):
    """
    Creates a benchmark of num_tasks tasks that
    are each composed of hard class splits of the
    provided training dataset.

    Task id should be provided at test time so
    that the network knows which task to perform.
    Initial labels are turned into task-specific labels,
    all starting from 0

    Each task contains a

    i.e Initial classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Task 0. : [0, 1] -> [0, 1]

    Task 1. : [2, 3] -> [0, 1]

    [...]

    Task 4. : [8, 9] -> [0, 1]

    """

    if restricted is not None:
        for k in restricted:
            assert int(k) < num_tasks

    scenario_nc = nc_benchmark(
        ds_train,
        ds_test,
        num_tasks,
        task_labels=True,
        train_transform=None,
        eval_transform=None,
        seed=seed,
        class_ids_from_zero_in_each_exp=True,
        fixed_class_order=fixed_class_order,
    )

    print(scenario_nc.classes_order_original_ids)

    scenario_nc = benchmark_with_validation_stream(
        scenario_nc, validation_size=val_size, shuffle=True
    )

    modified_train_ds = []
    modified_test_ds = []
    modified_valid_ds = []

    for i, (train_ds, test_ds, valid_ds) in enumerate(
        zip(scenario_nc.train_stream, scenario_nc.test_stream, scenario_nc.valid_stream)
    ):
        if restricted is not None and str(i) in restricted:
            print(f"Restricting Task {i} data")
            train_ds_idx, _ = torch.utils.data.random_split(
                torch.arange(len(train_ds.dataset)),
                (
                    int(len(train_ds.dataset) * restricted[str(i)]),
                    len(train_ds.dataset)
                    - int(len(train_ds.dataset) * restricted[str(i)]),
                ),
            )
            dataset = AvalancheSubset(train_ds.dataset, train_ds_idx)
        else:
            dataset = train_ds.dataset

        modified_train_ds.append(dataset)
        modified_test_ds.append(test_ds.dataset)
        modified_valid_ds.append(valid_ds.dataset)

    scenario_nc = dataset_benchmark(
        modified_train_ds,
        modified_test_ds,
        other_streams_datasets={"valid": modified_valid_ds},
        train_transform=train_transform,
        eval_transform=test_transform,
    )

    return scenario_nc


def new_classes_incremental(
    ds_train,
    ds_test,
    train_transform,
    test_transform,
    num_tasks=10,
    seed=0,
    val_size=0.05,
    fixed_class_order=None,
    per_exp_classes=None,
    **kwargs,
):
    # Create benchmark

    scenario_nc = nc_benchmark(
        ds_train,
        ds_test,
        num_tasks,
        task_labels=False,
        train_transform=train_transform,
        eval_transform=test_transform,
        seed=seed,
        class_ids_from_zero_from_first_exp=True,
        fixed_class_order=fixed_class_order,
        per_exp_classes=per_exp_classes,
    )
    print(scenario_nc.classes_order_original_ids)
    scenario = benchmark_with_validation_stream(
        scenario_nc, validation_size=val_size, shuffle=True
    )

    return scenario


def new_classes_incremental_with_labels(
    ds_train,
    ds_test,
    train_transform,
    test_transform,
    num_tasks=10,
    seed=0,
    val_size=0.05,
    fixed_class_order=None,
    per_exp_classes=None,
    **kwargs,
):
    # Create benchmark

    scenario_nc = nc_benchmark(
        ds_train,
        ds_test,
        num_tasks,
        task_labels=True,
        train_transform=train_transform,
        eval_transform=test_transform,
        seed=seed,
        class_ids_from_zero_in_each_exp=False,
        fixed_class_order=fixed_class_order,
        per_exp_classes=per_exp_classes,
    )
    print(scenario_nc.classes_order_original_ids)
    scenario = benchmark_with_validation_stream(
        scenario_nc, validation_size=val_size, shuffle=True
    )

    return scenario
