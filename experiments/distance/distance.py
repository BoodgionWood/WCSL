#!/usr/bin/env python3
"""
distance.py
Creates coreset samples and writes distance CSV files.
Run this first, then Figure_4or9_distance.py to make the figures.
"""

import time
import torch
from pathlib import Path

from config import configuration, prepare_args_for_grid_search
from pathcfg import DIST_DIR, OUTPUT_DIR, MODEL_DIR, FIG_DIR
from methods.methods import representative_sampling
from dataset.simulations import sample_sim_dataset
from dataset.images import load_image_dataset
from experiments.distance.util import calculate_distance_and_write
from utils.output import write_to_csv


def main(args, method_name, sampling_round, name,
         data_type, num_distance_rounds=10, device="cuda:0"):

    distances_filename_full = DIST_DIR / f"distances_{name}_full.csv"
    distances_filename_avg  = DIST_DIR / f"distances_{name}_avg.csv"
    time_filename           = DIST_DIR / f"time_{name}.csv"

    # ------------------------------------------------------------------ #
    # 1.  Load data
    # ------------------------------------------------------------------ #
    if data_type == "image":
        train_loader = load_image_dataset(args, args.num_pts)
    elif data_type == "sim":
        train_loader = sample_sim_dataset(args, args.num_pts)
    else:
        raise ValueError("data_type must be 'image' or 'sim'")

    # ------------------------------------------------------------------ #
    # 2.  Representative sampling
    # ------------------------------------------------------------------ #
    print(f"{method_name} sampling started!")
    start_time = time.time()
    X_rep = representative_sampling(args, method_name, train_loader, device=device)
    elapsed_time = time.time() - start_time

    header_time = ["dataset", "dim", "num_samples",
                   "method", "sampling_round", "time_value"]
    write_to_csv(
        time_filename,
        [args.dataset, args.data_dim, args.num_samples,
         method_name, sampling_round, elapsed_time],
        header_time,
    )

    # ------------------------------------------------------------------ #
    # 3.  Distance computation
    # ------------------------------------------------------------------ #
    calculate_distance_and_write(
        args, X_rep, method_name, sampling_round,
        num_distance_rounds,
        distances_filename_full, distances_filename_avg,
        device,
    )


# ---------------------------------------------------------------------- #
#                       Script entry point
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    args = configuration()
    args = prepare_args_for_grid_search(args)

    # —— derive repo-relative paths so nothing is hard-coded ——
    args.output_path = str(OUTPUT_DIR)      # e.g. repo/output
    args.plot_path   = str(FIG_DIR)
    args.model_path  = str(MODEL_DIR)

    # The util helper will mkdir these again if needed; safe to ignore errors
    Path(args.plot_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_type == "sim":
        for i in range(10):
            for dist_type in ["normal","t"]:
                args.dataset = dist_type
                for dim in [100]:
                    for num_samples in [25, 50, 100, 200, 300, 400]:
                        args.data_dim   = dim
                        args.num_pts    = 10000
                        args.num_samples = num_samples
                        args.num_epochs = 500
                        dist = f"{args.dataset}_{args.data_dim}"
                        print(f"Dataset: {dist}")
                        print(f"args: {args}")
                        for method in ["Random", "K-means", "SCCP",
                                       "Kernel_Thinning", "WCSL"]:
                            main(args, method, i, dist_type, "sim")

    elif args.data_type == "image":
        for i in range(10):
            for dist_type in ["MNIST", "FashionMNIST"]:
                args.dataset = dist_type
                for num_samples in [25, 50, 100, 200, 300, 400]:
                    args.num_samples = num_samples
                    args.data_dim    = 28 * 28
                    args.num_epochs  = 50
                    print(f"args: {args}")
                    for method in ["Random", "K-means", "SCCP", "WCSL"]:
                        main(args, method, i, dist_type, "image")
