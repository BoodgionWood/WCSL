# loss.py

import torch
from loss.sinkhorn import SinkhornLoss                 # external helper


# --------------------------------------------------------------------- #
#   MMD                                #
# --------------------------------------------------------------------- #
def _gaussian_kernel_depth_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Isotropic Gaussian kernel with automatic bandwidth scaling by feature depth
        k(x,y) = exp(-‖x-y‖² / d)
    Works on arbitrary tensor shapes by flattening the last dimensions.
    """
    a, b = a.detach().cpu(), b.detach().cpu()
    n, m     = a.shape[0], b.shape[0]
    d        = a.view(n, -1).shape[1]                 # feature dimension
    a        = a.view(n,   1, d)
    b        = b.view(1,   m, d)
    sq_dist  = ((a - b) ** 2).mean(-1) / d
    return torch.exp(-sq_dist)


def mmd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (depth-normalised RBF kernel).

    Parameters
    ----------
    P, Q : torch.Tensor
        Batches of samples with the same shape [batch, ...].

    Returns
    -------
    torch.Tensor (scalar)
    """
    k_pp = _gaussian_kernel_depth_norm(P, P).mean()
    k_qq = _gaussian_kernel_depth_norm(Q, Q).mean()
    k_pq = _gaussian_kernel_depth_norm(P, Q).mean()
    return k_pp + k_qq - 2.0 * k_pq


# --------------------------------------------------------------------- #
#   Wasserstein distance via Sinkhorn                                   #
# --------------------------------------------------------------------- #

def compute_M(P_samples, Q_samples,m_distance="L2square",lambda_val=1e-3):
    if m_distance=="L2":
        M = torch.cdist(P_samples.view(P_samples.shape[0], -1), Q_samples.view(Q_samples.shape[0], -1), p=2)
    elif m_distance=="L2square":
        M = torch.square(torch.cdist(P_samples.view(P_samples.shape[0], -1), Q_samples.view(Q_samples.shape[0], -1), p=2))
    elif m_distance=="spin":
        # Ensure P_samples and Q_samples are broadcastable to (n, m, p)
        P_extended = P_samples.unsqueeze(1)  # Shape becomes (n, 1, p)
        Q_extended = Q_samples.unsqueeze(0)  # Shape becomes (1, m, p)

        # Compute (x_i - y_i)^2 for each dimension i and for each pair (x, y)
        diff_squared = torch.square(P_extended - Q_extended)

        M = -torch.prod(1 / (diff_squared + lambda_val), dim=2)
    else:
        raise ValueError('Unknown M_distance type: ' + m_distance)
    M = M.to(dtype=torch.float64)
    return M


def compute_M(P_samples, Q_samples,m_distance="L2square",lambda_val=1e-3):
    if m_distance=="L2":
        M = torch.cdist(P_samples.view(P_samples.shape[0], -1), Q_samples.view(Q_samples.shape[0], -1), p=2)
    elif m_distance=="L2square":
        M = torch.square(torch.cdist(P_samples.view(P_samples.shape[0], -1), Q_samples.view(Q_samples.shape[0], -1), p=2))
    elif m_distance=="spin":
        # Ensure P_samples and Q_samples are broadcastable to (n, m, p)
        P_extended = P_samples.unsqueeze(1)  # Shape becomes (n, 1, p)
        Q_extended = Q_samples.unsqueeze(0)  # Shape becomes (1, m, p)

        # Compute (x_i - y_i)^2 for each dimension i and for each pair (x, y)
        diff_squared = torch.square(P_extended - Q_extended)

        M = -torch.prod(1 / (diff_squared + lambda_val), dim=2)
    else:
        raise ValueError('Unknown M_distance type: ' + m_distance)
    M = M.to(dtype=torch.float64)
    return M


def wasserstein_distance(P_samples, Q_samples, reg=0.01, m_distance="L2square"):
    M= compute_M(P_samples, Q_samples,m_distance)
    lossfn = SinkhornLoss(reg=reg, max_iter=10000)
    return lossfn(M)
