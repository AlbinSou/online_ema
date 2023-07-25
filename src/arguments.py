#!/usr/bin/env python3
import argparse
import os
import json

import yaml

from toolkit.utils import set_config_args, set_seed


def parse_arguments(save_config=False):
    parser = argparse.ArgumentParser()
    # miscellaneous args
    parser.add_argument(
        "--results",
        type=str,
        default="./results",
        help="Results path (base dir) (default=%(default)s)",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="Experiment name (default=%(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default=%(default)s)"
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Save trained models (default=%(default)s)",
    )

    # benchmark args
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--scenario",
        default="new_instances",
        choices=[
            "new_instances",
            "new_classes_multitask",
            "new_classes_multitask_unbalanced",
            "new_classes_incremental",
            "new_classes_incremental_with_labels",
            "new_classes_incremental_with_labels",
        ],
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        required=False,
        help="Number of subprocesses to use for dataloader (default=%(default)s)",
    )
    parser.add_argument(
        "--pin_memory",
        default=False,
        type=bool,
        required=False,
        help="Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        required=False,
        help="Number of samples per batch to load (default=%(default)s)",
    )
    parser.add_argument(
        "--num_tasks",
        default=10,
        type=int,
        required=False,
        help="Number of tasks per dataset (default=%(default)s)",
    )
    # training args
    parser.add_argument(
        "--strategy",
        type=str,
        default="naive",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer",
        choices=["Adam", "SGD"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model type",
    )
    parser.add_argument(
        "--val_size",
        default=0.05,
        type=float,
        required=False,
        help="Validation size (default=%(default)s)",
    )
    parser.add_argument(
        "--nepochs",
        default=100,
        type=int,
        required=False,
        help="Number of epochs per training session (default=%(default)s)",
    )
    parser.add_argument(
        "--lr",
        default=0.05,
        type=float,
        required=False,
        help="Starting learning rate (default=%(default)s)",
    )
    parser.add_argument(
        "--clipping",
        default=10000,
        type=float,
        required=False,
        help="Clip gradient norm (default=%(default)s)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        required=False,
        help="Momentum factor (default=%(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0002,
        type=float,
        required=False,
        help="Weight decay (L2 penalty) (default=%(default)s)",
    )
    parser.add_argument(
        "--source_results_path",
        default=None,
        type=str,
        required=False,
        help="Sometimes used to load model from another training run",
    )

    parser.add_argument("--stop_after", type=int, default=-1, help="Stop training at task")

    # Dataset
    parser.add_argument("--restricted", type=json.loads, default={}, required=False)

    # Logging and evaluation
    parser.add_argument("--tensorboard", action="store_true", required=False)
    parser.add_argument("--eval_on_previous", action="store_true", required=False)

    # Plugins
    parser.add_argument("--schedule", action="store_true", required=False)
    parser.add_argument("--mean_evaluation", action="store_true", required=False)
    parser.add_argument("--retain_best", action="store_true", required=False)
    parser.add_argument("--parallel_evaluation", action="store_true", required=False)
    parser.add_argument("--eval_mode", type=str, default="iteration", required=False)
    parser.add_argument("--use_transforms", action="store_true", required=False, help="activate input transforms")


    # Hyperparameters for plugins
    parser.add_argument("--lmb", type=float, default=0.4)
    parser.add_argument("--lmb2", type=float, default=0.4)
    parser.add_argument("--every", type=int, default=5, help="Evaluate every k training iterations")
    parser.add_argument("--memory_size", type=int, default=500, required=False)
    parser.add_argument("--start_from_pretrained", type=str, default=None, required=False)

    # Set config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", required=False, default=None)

    # Parses config file to set default arguments
    args, remaining_argv = config_parser.parse_known_args()

    if args.config:
        set_config_args(parser, args.config)

    args = parser.parse_args(remaining_argv)

    # Create Logger
    exp_name = args.exp_name if args.exp_name is not None else "default"
    exp_path = os.path.join(args.results, exp_name)
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    args.results_path = os.path.join(exp_path, str(args.seed))
    if not os.path.isdir(args.results_path):
        os.mkdir(args.results_path)

    if save_config:
        # Save config under results dir
        with open(os.path.join(args.results_path, "config.yml"), "w") as f:
            f.write("!!python/object:argparse.Namespace\n")
            yaml.dump(vars(args), f)

    return args
