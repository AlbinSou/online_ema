#!/usr/bin/env python3

def warmup_lr(num_iterations, num_warmup_iterations):
    def _lambda_lr_func(iter_count):
        if iter_count <= num_warmup_iterations:
            return iter_count/num_warmup_iterations
        else:
            return 1
    return _lambda_lr_func
