#!/usr/bin/env python3
import pandas as pd
import collections
import os
import jsonlines
from typing import List, Dict 
import numpy as np
import json

filenames = ["logs.json", "training_logs.json"]

def extract_final_test(paths):
    jsonfiles = collections.defaultdict(lambda: [])
    if not isinstance(paths, List):
        paths = [paths]
    for path in paths:
        for dirpath, dirnames, files in os.walk(path):
            for f in files:
                if f in ["final_test_results.json", "final_test_results_ema.json"]:
                    name, seed = dirpath.split("/")[-2:]
                    jsonfiles[name].append(os.path.join(dirpath, f))
    return jsonfiles

def extract_json_files(paths, add_ema: bool = True, only_ema: bool = False, add_probing: bool = False):
    jsonfiles = collections.defaultdict(lambda: [])
    if not isinstance(paths, List):
        paths = [paths]
    for path in paths:
        for dirpath, dirnames, files in os.walk(path):
            for f in files:
                if f in filenames:
                    maybe_ema = dirpath.split("/")[-1]
                    if "ema" in maybe_ema:
                        if add_ema:
                            name, seed, ema = dirpath.split("/")[-3:]
                            jsonfiles[".".join([name, ema])].append(
                                (os.path.join(dirpath, f), int(seed))
                            )
                    elif "probing" in maybe_ema:
                        if add_probing:
                            seed, name = dirpath.split("/")[-2:]
                            jsonfiles[name].append(
                                (os.path.join(dirpath, f), int(seed))
                            )
                        
                    else:
                        if not only_ema:
                            name, seed = dirpath.split("/")[-2:]
                            jsonfiles[name].append((os.path.join(dirpath, f), int(seed)))
    return jsonfiles

def extract_records_from_files(jsonfiles) -> Dict[str, pd.DataFrame]:
    records = {}
    print("Scanning JSON Files ... ")
    for name, files in jsonfiles.items():
        records[name] = []

        # This line could cause problems
        num_seeds = len(np.unique(np.array([f[1] for f in files])))
        print(f"Found {num_seeds} seeds for {name}")

        for f, s in files:
            jsonfile = open(f, "r")
            reader = jsonlines.Reader(jsonfile)
            previous_step = None
            for line in reader:
                step = line.pop("step")

                if previous_step is None:
                    df_line = {"seed": s, "step": step}

                if previous_step is not None and step != previous_step:
                    records[name].append(df_line)
                    df_line = {"seed": s, "step": step}

                # We will add one line per step
                df_line.update(line)

                previous_step = step
            records[name].append(df_line)
            jsonfile.close()

    for name in records:
        df = pd.DataFrame(records[name]).groupby(["step", "seed"], as_index=False).sum()
        records[name] = df

    return records


if __name__ == "__main__":
    jsonfiles = extract_json_files("/home/albin/Documents/Projects/continual_evaluation/results/cifar10/cifar10_er/")
    records = extract_records_from_files(jsonfiles)
    df = records["cifar10_er"]
    print(records)
