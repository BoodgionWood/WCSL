import torch
from tqdm import tqdm

from dataset.utils import sample_from_loader



def q_xi_Y(X, Y):
    X_flat = X.view(X.shape[0], -1)
    Y_flat = Y.view(Y.shape[0], -1)
    dists = torch.cdist(X_flat, Y_flat, p=2)
    dists[dists<1e-6]=1e-6
    return torch.sum(dists.reciprocal(), dim=-1)


def first_term(X, n, N):
    X_flat = X.view(X.shape[0], -1)
    dists = torch.cdist(X_flat, X_flat, p=2) + torch.eye(n).to(X.device) * 1e9
    dists[dists < 1e-6] = 1e-6
    result_flat = ((N / n) * torch.sum((X_flat.unsqueeze(1) - X_flat.unsqueeze(0)) / dists.unsqueeze(-1), dim=1))
    return result_flat.view(X.shape)


def second_term(X, Y):
    X_flat = X.view(X.shape[0], -1)
    Y_flat = Y.view(Y.shape[0], -1)
    dists = torch.cdist(X_flat, Y_flat, p=2)
    dists[dists < 1e-6] = 1e-6
    return torch.sum(Y_flat / dists.unsqueeze(-1), dim=1).view_as(X_flat)  # Modified to return flat


def M(X, Y):
    n = X.shape[0]
    N = Y.shape[0]
    q_values = q_xi_Y(X, Y)

    # Modified to work with flat structure and then reshape accordingly
    Z = q_values.reciprocal().view(-1, 1) * (first_term(X, n, N).view(X.shape[0], -1) + second_term(X, Y))
    return Z.view_as(X)


def update_k(X, Y, d, w_l):
    k_l = w_l * q_xi_Y(X, Y) / (w_l * q_xi_Y(X, Y) + (1 - w_l) * d)
    return k_l


def update_X(X, Y, k_l):
    X_new = (1 - k_l).view(-1,1) * X + k_l.view(-1,1) * M(X, Y)
    return X_new


def update_d(X, Y, d, w_l):
    d_new = (1 - w_l) * d + w_l * q_xi_Y(X, Y)
    return d_new


def sccp(args, train_loader, device="cuda:0"):
    args.method = "sccp"

    samples = sample_from_loader(train_loader, args.num_samples).to(device)
    samples_shape = samples.shape
    samples = samples.view(samples.shape[0], -1)
    p = samples.numel() // args.num_samples
    d = torch.zeros(args.num_samples).to(device)

    l = 0
    for epoch in tqdm(range(args.num_epochs)):
        for batch_idx, (data, _) in enumerate(train_loader):
            data_ = data.to(device).view(data.shape[0], -1)
            w_l = args.num_samples * p / (args.num_samples * p + l)

            k_l = update_k(samples, data_, d, w_l)
            samples_new = update_X(samples, data_, k_l)
            d_new = update_d(samples, data_, d, w_l)

            samples = samples_new
            d = d_new
            l += 1

    return samples.view(samples_shape)


