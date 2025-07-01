import numpy as np
import torch
from scipy.stats import multivariate_normal, multivariate_t
from torch.utils.data import DataLoader, TensorDataset


def sample_sim_data(args, num_samples):
    dimension = args.data_dim
    distribution_type = args.dataset
    q = 1.25
    batch_size = 1000
    if args.m_distance == "L2":
        q = args.data_dim / (args.data_dim - 1)
    if args.m_distance == "L2square":
        q = 2 * args.data_dim / (args.data_dim - 2)
    if distribution_type == "normal":
        # Generate random orthogonal matrix
        H = np.random.rand(dimension, dimension)
        Q, _ = np.linalg.qr(H)

        # Create a diagonal matrix D of dimension
        D = np.zeros((dimension, dimension))
        non_zero_eigenvalues = min(dimension // 2, 20)

        for i in range(non_zero_eigenvalues):
            D[i, i] = np.random.rand()

        # Construct covariance matrix
        cov = np.dot(Q, np.dot(D, Q.T))

        mean = np.zeros(dimension)
        samples = multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)
    elif args.dataset == 't':
        # Generate random orthogonal matrix
        H = np.random.rand(dimension, dimension)
        Q, _ = np.linalg.qr(H)

        # Create a diagonal matrix D of dimension
        D = np.zeros((dimension, dimension))
        non_zero_eigenvalues = min(dimension // 2, 20)

        for i in range(non_zero_eigenvalues):
            D[i, i] = np.random.rand()

        # Construct covariance matrix
        cov = np.dot(Q, np.dot(D, Q.T))

        mean = np.zeros(dimension)
        samples = multivariate_t.rvs(loc=mean, shape=cov, df=q, size=num_samples)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    return torch.from_numpy(samples).float()


def sample_sim_dataset(args, n_samples, shuffle=True, drop_last=True):
    data=sample_sim_data(args, n_samples)
    Y=torch.zeros(n_samples, dtype=torch.long)
    dataset = TensorDataset(data,Y)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)






