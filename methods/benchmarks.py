import torch
import numpy as np
from math import log
from sklearn.cluster import MiniBatchKMeans
from goodpoints import kt


def get_kmeans_centroids(dataloader, sample_size, batch_size):
    original_shape = None
    # Use MiniBatchKMeans to handle large datasets in chunks
    kmeans = MiniBatchKMeans(n_clusters=sample_size, batch_size=batch_size, compute_labels=False)

    # Iterate through the dataloader and partial fit the data
    for batch in dataloader:
        # Assuming the data is the first item in the batch
        data = batch[0]
        if original_shape is None:
            # Store the original shape (excluding batch size)
            original_shape = data.shape[1:]  # (a,b,c) if 4D or (p) if 2D
        data_reshaped = data.view(data.size(0), -1)  # (batch_size, d)
        data_numpy = data_reshaped.numpy()
        kmeans.partial_fit(data_numpy)

    # Get centroids and convert them back to a PyTorch tensor
    centroids = kmeans.cluster_centers_
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    # Reshape the centroids back to the original shape (sample_size, a, b, c) or (sample_size, p)
    centroids_tensor = centroids_tensor.view((sample_size,) + original_shape)

    return centroids_tensor


def kernel_thinning(dataloader, sample_size, split_var=1.0, swap_var=1.0, delta=0.5, seed=None):
    """
    Produce a kernel thinning coreset of the given sample_size from the data
    provided by the dataloader. The coreset is returned as a torch.Tensor.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader that yields batches of data,
            each batch being a torch.Tensor of shape (batch_size, d).
        sample_size (int): Desired coreset size, must be 2^m for some integer m.
        split_var (float): Variance parameter for the split kernel (often smaller than swap_var).
        swap_var (float): Variance parameter for the swap kernel.
        delta (float): Failure probability parameter for KT-SPLIT.
        seed (int or None): Random seed for reproducibility. If None, no seed is set.

    Returns:
        torch.Tensor: A tensor of shape (sample_size, d) representing the coreset.
    """
    original_shape = None
    # Gather all data from the dataloader
    X_list = []
    for batch in dataloader:
        batch_X, batch_y = batch
        if original_shape is None:
            original_shape = batch_X.shape[1:]
        batch_X_reshaped = batch_X.view(batch_X.size(0), -1)  # (batch_size, d)
        X_list.append(batch_X_reshaped)
        # If it only returns X:
        # X_list.append(batch)
    X_full = torch.cat(X_list, dim=0)  # shape: (N, d)

    # Convert to NumPy for use with goodpoints.kt
    X_np = X_full.detach().cpu().numpy()
    N, d = X_np.shape

    # Set the m by the largest integer m such that 2^m * sample_size<=N
    m = int(log(N/sample_size)/log(2))
    print(m)

    # The kernel thinning procedure used here (thin) expects an input sample of size n = (2^(2*m)).
    # After m rounds, it returns a coreset of size 2^m = sample_size.
    # So we need at least sample_size^2 samples in X.
    required_n = sample_size * (2 ** m)
    if N < required_n:
        raise ValueError(f"Not enough samples. Need at least {required_n}, got {N}.")

    # Truncate or use the first required_n samples
    X_np = X_np[:required_n]

    # Define the kernels
    def gaussian_kernel(y, X, var):
        # y: (d,), X: (N, d)
        diff = X - y
        sqdist = np.sum(diff ** 2, axis=1)
        return np.exp(-sqdist / (2 * var))
    split_kernel = lambda y, X: gaussian_kernel(y, X, split_var)
    swap_kernel = lambda y, X: gaussian_kernel(y, X, swap_var)

    ''' This part not working.
    def gaussian_kernel_depth_normalized(a, b):
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        # Reshape to flatten all dimensions except the first
        depth = a.reshape(dim1_1, -1).shape[1]
        a = a.reshape(dim1_1, 1, depth)
        b = b.reshape(1, dim1_2, depth)

        # Calculate the mean squared difference along the last dimension
        numerator = ((a - b) ** 2).mean(axis=-1) / depth

        return np.exp(-numerator)

    split_kernel = lambda y, X: gaussian_kernel_depth_normalized(y, X)
    swap_kernel = lambda y, X: gaussian_kernel_depth_normalized(y, X)
    '''

    # Optionally set seed
    if seed is not None:
        np.random.seed(seed)

    # Run kernel thinning
    coreset_indices = kt.thin(X_np, m, split_kernel, swap_kernel, delta=delta, seed=seed, store_K=False)

    # Extract the coreset and convert back to torch.Tensor
    coreset_np = X_np[coreset_indices]
    coreset_torch = torch.from_numpy(coreset_np)
    coreset_torch = coreset_torch.view((sample_size,) + original_shape)
    print(coreset_torch.size())

    return coreset_torch


if __name__ == "__main__":
    from dataset.utils import pack_data_to_loader
    # Create a dummy dataloader
    data = torch.randn(1000, 2)
    labels = torch.randint(0, 10, (1000,))
    dataloader = pack_data_to_loader(data, labels, batch_size=64, shuffle=True, drop_last=False, print_count=True)

    # Get centroids
    centroids = get_kmeans_centroids(dataloader, 5, 64)
    print(centroids)