# experiments/distance/util.py
# ---------------------------------------------------------------------------
# Utilities for distance-based evaluation of representative (coreset) samples
# ---------------------------------------------------------------------------
#
# Supported distances
#   • "W"   : 2-Wasserstein via entropic Sinkhorn regularisation
#   • "MMD" : Maximum Mean Discrepancy
#
# All file writing is delegated to utils.output.write_to_csv, which appends if
# the file already exists.  Header rows are written automatically on the first
# call.

from __future__ import annotations

import numpy as np
import torch

from dataset.simulations import sample_sim_data
from dataset.images import load_image_dataset
from dataset.utils import sample_from_loader
from loss.loss import mmd, wasserstein_distance
from utils.output import write_to_csv


# --------------------------------------------------------------------------- #
#  Data helpers                                                               #
# --------------------------------------------------------------------------- #
def sample_data(args, num_samples: int, device: str = "cuda:0") -> torch.Tensor:
    """
    Draw `num_samples` fresh points from the dataset specified by *args*.

    Returns
    -------
    torch.Tensor  on *device*
    """
    if args.data_type == "image":
        img_loader = load_image_dataset(args, args.num_pts)
        batch = sample_from_loader(img_loader, num_samples)
    else:
        batch = sample_sim_data(args, num_samples)

    return batch.to(device)


# --------------------------------------------------------------------------- #
#  Distance computation & logging                                             #
# --------------------------------------------------------------------------- #
def calculate_distance_and_write(
    args,
    X_rep: torch.Tensor,
    method_name: str,
    sampling_round: int,
    num_distance_rounds: int,
    distances_filename_full: str | "os.PathLike[str]",
    distances_filename_avg: str | "os.PathLike[str]",
    device: str = "cuda:0",
) -> None:
    """
    Compute W and MMD between X_rep and random batches Z, repeating
    *num_distance_rounds* times.  Write both per-round distances (full CSV)
    and their averages (avg CSV).

    Parameters
    ----------
    args : Namespace
        Must carry .dataset, .data_dim, .num_samples, .loss_reg, .m_distance
    X_rep : torch.Tensor
        Representative / coreset sample (already on *device*)
    method_name : str
        The sampling algorithm that produced X_rep (e.g. "WCSL")
    sampling_round : int
        Repetition index of the sampling stage (for robustness)
    num_distance_rounds : int
        How many independent Z batches to average over
    distances_filename_full, distances_filename_avg : str or Path
        Target CSV files.  They are created (with header) if absent.
    device : str
        "cuda:0" or "cpu"

    Notes
    -----
    • This helper never overwrites rows; write_to_csv appends idempotently.
    • Normalisation against the Random baseline happens later in plotting.
    """

    # CSV headers
    header_full = [
        "dataset",
        "dim",
        "num_samples",
        "distance_type",
        "method",
        "sampling_round",
        "distance_round",
        "distance_value",
    ]
    header_avg = [
        "dataset",
        "dim",
        "num_samples",
        "distance_type",
        "method",
        "sampling_round",
        "distance_value",
    ]

    # Containers for averages
    distances = {"W": {method_name: []}, "MMD": {method_name: []}}

    for dist_round in range(num_distance_rounds):
        # Fresh comparison batch
        Z = sample_data(args, args.num_samples, device=device)

        # ➤ Wasserstein
        w_val = wasserstein_distance(
            X_rep, Z, reg=args.loss_reg, m_distance=args.m_distance
        ).item()
        write_to_csv(
            distances_filename_full,
            [
                args.dataset,
                args.data_dim,
                args.num_samples,
                "W",
                method_name,
                sampling_round,
                dist_round,
                w_val,
            ],
            header_full,
        )
        distances["W"][method_name].append(w_val)

        # ➤ MMD
        mmd_val = mmd(X_rep, Z).item()
        write_to_csv(
            distances_filename_full,
            [
                args.dataset,
                args.data_dim,
                args.num_samples,
                "MMD",
                method_name,
                sampling_round,
                dist_round,
                mmd_val,
            ],
            header_full,
        )
        distances["MMD"][method_name].append(mmd_val)

    # ------------------------------------------------------------- #
    #  Averaged values                                              #
    # ------------------------------------------------------------- #
    for dist_type, method_dict in distances.items():
        avg_val = float(np.mean(method_dict[method_name]))
        write_to_csv(
            distances_filename_avg,
            [
                args.dataset,
                args.data_dim,
                args.num_samples,
                dist_type,
                method_name,
                sampling_round,
                avg_val,
            ],
            header_avg,
        )
