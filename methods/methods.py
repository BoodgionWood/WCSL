from methods.mvpts_core import mvpts
from methods.sccp_core import sccp
from methods.benchmarks import get_kmeans_centroids, kernel_thinning
from dataset.utils import sample_from_loader


def representative_sampling(args, method, train_loader, start_time=None, device="cuda:0"):
    if method == "WCSL":
        if args.phi_mode == "none":
            return mvpts(args, train_loader, start_time=start_time, device=device).to(device)
        else:
            raise NotImplementedError("Phi mode not implemented!")
    elif method == "SCCP":
        return sccp(args, train_loader, device).to(device)
    elif method == "Random":
        return sample_from_loader(train_loader, args.num_samples).to(device)
    elif method == "K-means":
        return get_kmeans_centroids(train_loader, args.num_samples, args.batch_size).to(device)
    elif method == "Kernel_Thinning":
        return kernel_thinning(train_loader, args.num_samples).to(device)
    else:
        raise NotImplementedError("Method not implemented!")