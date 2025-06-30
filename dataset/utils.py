import torch
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader



def sample_from_loader(train_loader, n=50):
    selected_samples = []
    total_samples = 0

    for data, _ in tqdm(train_loader):
        for sample in data:
            total_samples += 1
            if len(selected_samples) < n:
                # Initially fill the reservoir
                selected_samples.append(sample)
            else:
                # Randomly replace elements in the reservoir
                # with a decreasing probability.
                # Probability of replacing is n/total_samples
                s = random.randint(0, total_samples - 1)
                if s < n:
                    selected_samples[s] = sample

    return torch.stack(selected_samples)


def pack_data_to_loader(data_tensor, labels=None, batch_size=64, shuffle=True, drop_last=False, print_count=False):
    """
    Pack the simulated data and optionally its labels into a DataLoader.

    Parameters:
    - data_tensor (torch.Tensor): The simulated data.
    - labels (torch.Tensor or None): The labels corresponding to the simulated data.
                                   If None, it will default all labels to zeros.
    - batch_size (int): The size of each batch.
    - shuffle (bool): Whether to shuffle the data.
    - drop_last (bool): Whether to drop the last batch if its size is not equal to batch_size.

    Returns:
    - DataLoader object containing the simulated data and labels.
    """

    # If labels are not provided, create dummy labels of zeros
    if labels is None:
        labels = torch.zeros(data_tensor.size(0), dtype=torch.long)

    # Print number of samples in each class for new labels
    if print_count:
        unique_label, label_counts = torch.unique(labels, return_counts=True)
        print("\nPacked to dataloader label counts:")
        for label, count in zip(unique_label, label_counts):
            print(f"Class {label.item()}: {count.item()} samples")

    # Create a TensorDataset from the simulated data and labels
    dataset = TensorDataset(data_tensor, labels)

    # Create and return a DataLoader from the TensorDataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader
