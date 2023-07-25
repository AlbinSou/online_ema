#!/usr/bin/env python3
import random
from contextlib import contextmanager
import os
from collections import defaultdict
from typing import Union, List

import numpy as np
from avalanche.models.utils import avalanche_forward
import torch
import yaml
import torch.nn as nn

def _compute_l2_norm(importances):
    norm = 0.
    for (n, imp) in importances:
        norm += torch.norm(imp.view(-1))
    return norm/len(importances)

# Following snippet is licensed under MIT license
# https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/2
@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

@contextmanager
def training(net):
    """Temporarily switch to training mode."""
    istrain = net.training
    try:
        net.train()
        yield net
    finally:
        if not istrain:
            net.eval()

@torch.no_grad()
def adapt_batch_norm(model, dataset, batch_size, max_batch_adapt=70, device="cuda"):
    # Reset bn stats
    bn_modules = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean = torch.zeros(len(m.running_mean))
            m.running_var = torch.ones(len(m.running_var))
            bn_modules.append(m)

    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # Adapt them
    n = 0
    with training(model) as net:
        for i, (batch_x, batch_y, t) in enumerate(dataloader):
            batch_x, batch_y, t = batch_x.to(device), batch_y.to(device), t.to(device)
            b = len(batch_x)
            if i > max_batch_adapt:
                break

            # Used in SWA
            momentum = b / (n + b)
            for module in bn_modules:
                module.momentum = momentum

            avalanche_forward(net, batch_x, t)
            n += b
    return

def cross_entropy_with_soft_targets(
    outputs, targets, exp=1.0, size_average=True, eps=1e-5, apply_softmax=True,
):
    """Calculates cross-entropy with temperature scaling
    outputs and targets should already sum to 1"""

    if apply_softmax:
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
    else:
        out = outputs
        tar = targets

    if exp != 1:
        out = out.pow(exp)
        out = out / (out.sum(1).view(-1, 1).expand_as(out))
        tar = tar.pow(exp)
        tar = tar / (tar.sum(1).view(-1, 1).expand_as(tar))
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


def cycle(loader):
    while True:
        for elem in loader:
            yield elem


def set_seed(seed):
    # REPRODUCTIBILITY
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(name, parameters, **kwargs):
    if name == "SGD":
        optimizer = torch.optim.SGD(parameters, **kwargs)
    elif name == "Adam":
        optimizer = torch.optim.Adam(parameters, **kwargs)
    return optimizer


def set_config_args(parser, config_file: str):
    defaults = {}
    if config_file:
        with open(config_file) as f:
            file_args = yaml.load(f, Loader=yaml.UnsafeLoader)
        defaults.update(vars(file_args))
        _check_defaults(parser, defaults)
        parser.set_defaults(**defaults)


def get_config_args(config_file: str):
    with open(config_file) as f:
        file_args = yaml.load(f, Loader=yaml.UnsafeLoader)
    return file_args


def save_config(args, filename: str):
    # Save config under results dir
    with open(filename, "w") as f:
        f.write("!!python/object:argparse.Namespace\n")
        yaml.dump(vars(args), f)


def _check_defaults(parser, defaults):
    action_list = parser._actions
    accepted_strings = []
    for action in action_list:
        strings = action.option_strings
        for s in strings:
            accepted_strings.append(s.lstrip("-"))
    for key in defaults:
        if key not in accepted_strings and key != "results_path":
            raise ValueError(
                f"Key {key} is present in config file but not handled by parser"
            )


def linear_schedule(num_warmup_iterations, start_value, end_value):
    """
    Returns a scheduling function that goes from start value to 
    end value before stagnating to end_value after num_warmup_iterations
    """
    def _lambda(iter_count):
        if iter_count <= num_warmup_iterations:
            return start_value + (iter_count / num_warmup_iterations) * (
                end_value - start_value
            )
        else:
            return end_value
    return _lambda


def sigmoid_rampup(num_warmup_iterations, start_value, end_value):
    """
    rampup scheduling from Mean Teacher paper
    """
    def _lambda(iter_count):
        if iter_count <= num_warmup_iterations:
            x = iter_count / num_warmup_iterations
            return start_value + np.exp(-5*(1-x)**2) * (
                end_value - start_value
            )
        else:
            return end_value
    return _lambda

def step_rampup(num_warmup_iterations: Union[List, int], start_value, end_value):
    """
    Step rampup
    """

    if isinstance(num_warmup_iterations, List):
        # Treat it as range, taking value end_value in the
        # interval num_warmup_iterations[0], num_warmup_iterations[1]
        def _lambda(iter_count):
            if iter_count < num_warmup_iterations[0] or iter_count > num_warmup_iterations[1]:
                return start_value
            else:
                return end_value

    elif isinstance(num_warmup_iterations, float):
        # Value of start value in interval [0, num_warmup_iterations], end value otherwise
        def _lambda(iter_count):
            if iter_count <= num_warmup_iterations:
                return start_value
            else:
                return end_value

    return _lambda

def clear_tensorboard_files(directory):
    for root, name, files in os.walk(directory):
        for f in files:
            if "events" in f:
                os.system(f" rm {os.path.join(root, f)}")

def reset_optimizer(optimizer, parameters):
    assert len(optimizer.param_groups) == 1
    optimizer.state = defaultdict(dict)
    optimizer.param_groups[0]["params"] = parameters
