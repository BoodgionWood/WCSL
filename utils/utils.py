import torch
import json
import argparse


def args_to_string(args):
    # Convert args (Namespace) to a dictionary
    args_dict = vars(args)

    # Convert dictionary to a JSON string
    return json.dumps(args_dict)


def string_to_args(args_str):
    # Convert JSON string back to a dictionary
    args_dict = json.loads(args_str)

    # Update the args Namespace with values from the dictionary
    # for key, value in args_dict.items():
        # setattr(args, key, value)
    args = argparse.Namespace(**args_dict)

    return args


def find_closest_points(data, samples, labels=None):
    """
    For each sample, find the closest data point and return them along with corresponding labels.

    Parameters:
    - data (N, p): Numpy array with N data points of dimension p
    - labels (N,): Numpy array with labels corresponding to the data points
    - samples (n, p): Numpy array with n samples of dimension p

    Returns:
    - closest_samples (n, p): Numpy array with the closest data points to the samples
    - closest_labels (n,): Numpy array with labels corresponding to the closest_samples
    """
    distances = torch.cdist(samples, data)  # compute pairwise distances between samples and data
    min_distances, min_indices = torch.min(distances, dim=1)  # find the closest point in data for each point in samples
    closest_points = torch.index_select(data, dim=0, index=min_indices)  # select the closest points in data
    if labels is not None:
        closest_labels = torch.index_select(labels, dim=0, index=min_indices)
        return closest_points, closest_labels
    return closest_points


def nan_check(x):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("nan or inf")
        # torch.set_printoptions(profile="full")
        print(x)  # prints the whole tensor
        # torch.set_printoptions(profile="default")
        return True
    else:
        print("no nan or inf")
        return False


def calculate_point_rho(points):
    return torch.norm(points, dim=1)
