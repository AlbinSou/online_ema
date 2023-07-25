#!/usr/bin/env python3
import argparse
import collections
import csv
import os
import warnings

import numpy as np
from toolkit.utils import get_config_args

""" Returns average accuracy after each training task """

streamname = "Top1_Acc_Stream/eval_phase/test_stream/Task000"

def process_directory(root, results_dict, seeds):
    config = get_config_args(os.path.join(root, "config.yml"))
    seed = config.seed
    assert seed not in seeds
    seeds.append(seed)

    results_path = os.path.join(root, "results.csv")
    with open(results_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            # Get the index of test acc
            if i == 0:
                indexes = []
                keys = row
                for i, k in enumerate(keys):
                    if streamname in k:
                        index = i
            else:
                last_row = row

    for i, elem in enumerate(last_row):
        if i == index and len(elem) > 0:
            results_dict.append(float(elem))



def main(args, _print=False):
    results_dict = []
    seeds = []

    ### Results dict format:

    # results_dict = [seed0, seed1, ...]

    for root, dir, files in os.walk(args.pathname):
        if "config.yml" in files:
            if "results.csv" not in files:
                warnings.warn(f"No results found for dir {root}")
                continue
            process_directory(root, results_dict, seeds)


    if _print:
        #print(
        #    f"Per Task Averages: {mean_results} \n\n\
#Std    : {std_results}"
        #)
        #print()
        #print(per_seed_task_average)
        print(f"Across Tasks Average: {np.mean(results_dict)}, Std: {np.std(results_dict)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pathname",
        default="./results/default",
        help="Directory name without seed indicator",
    )
    args = parser.parse_args()
    main(args, _print=True)
