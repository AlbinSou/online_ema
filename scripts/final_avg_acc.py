#!/usr/bin/env python3
import argparse
import json
from typing import List, Dict
import collections

import numpy as np

from scripts.utils import extract_final_test

def extract_key_from_json(jsonfiles, key) -> Dict[str, List]:
    records = collections.defaultdict(lambda: [])
    for name in jsonfiles:
        for filename in jsonfiles[name]:
            with open(filename, "r") as f:
                dict_f = json.load(f)
                if "ema" in filename:
                    records[name+"_ema"].append(dict_f[key])
                else:
                    records[name].append(dict_f[key])
    return records

def main(args):
    jsonfiles = extract_final_test(args.paths)
    records = extract_key_from_json(jsonfiles, args.key)
    for name in records:
        values = records[name]
        print(f"Found {len(values)} seeds for {name}")
        print(values)
        mean = np.mean(values)
        std = np.std(values)
        print(f"Final Avg acc for {name} is {mean} +/- {std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+")
    parser.add_argument(
        "--seeds",
        type=int,
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Top1_Acc_Stream/eval_phase/test_stream/Task000",
    )
    args = parser.parse_args()
    main(args)
