#!/usr/bin/env python3
import argparse

import numpy as np

from scripts.utils import extract_json_files, extract_records_from_files

def main(args):
    jsonfiles = extract_json_files(
        args.paths, args.add_ema, args.only_ema, False
    )
    records = extract_records_from_files(jsonfiles)
    for name in records:
        df = records[name]
        seeds_list = np.unique(df["seed"].to_numpy())
        accuracies = []
        notnull = df[args.key].notnull()
        for s in seeds_list:
            # Select df that contains only seed
            df_seed = df[df.seed == s]
            max_step = df_seed.step[df_seed["step"].idxmax()]
            
            acc = df_seed[args.key][df_seed.step == max_step]
            accuracies.append(acc)

        print(accuracies)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"Final {args.key} for {name} is {mean} +/- {std}")


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
        default="TRACK_MB_WCACC/eval_phase/valid_stream/Task000",
    )
    parser.add_argument(
        "--add_ema",
        action="store_true",
    )
    parser.add_argument(
        "--only_ema",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
