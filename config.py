import argparse
from pathcfg import PROJECT_ROOT, OUTPUT_DIR, DATA_DIR        # NEW

def configuration():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="pareto", help="type of distribution we use")

    parser.add_argument("--data_type", type=str, default="sim", help="sim or image")

    parser.add_argument("--method", type=str, default="mvpts", help="moving_pts or sccp")

    parser.add_argument("--model", type=str, default="svm", help="svm or logit")

    parser.add_argument("--path", type=str, default=str(PROJECT_ROOT), help="repo root")
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_DIR), help="output dir")
    parser.add_argument("--plot_path", type=str, default=str(OUTPUT_DIR), help="plot dir")
    parser.add_argument("--model_path", type=str, default=str(OUTPUT_DIR / 'models'), help="model dir")
    parser.add_argument("--data_path", type=str, default=str(DATA_DIR), help="dataset root")

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument("--assign_label", type=str, default="soft", help="hard or soft")

    parser.add_argument("--k", type=int, default=2, help="number of classes in dataset.")

    parser.add_argument(
        "--num_pts", type=int, default=2000, help="number of point in sets"
    )

    parser.add_argument(
        "--num_samples", type=int, default=100, help="number of support points to train"
    )

    parser.add_argument(
        "--num_test", type=int, default=100, help="batch size used during training"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=300, help="number of epochs to train for"
    )

    parser.add_argument(
        "--data_dim", type=int, default=10, help="dimension of data"
    )

    parser.add_argument(
        "--in_channels", type=int, default=1, help="channels of input image, MNIST/FashionMNIST is 1, CIFAR10 is 3"
    )

    parser.add_argument(
        "--image_width", type=int, default=28, help="width of input image, MNIST/FashionMNIST is 28, CIFAR10 is 32"
    )

    parser.add_argument(
        "--scale", type=int, default=224, help="scale of medical images"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1000, help="batch size used during training"
    )

    parser.add_argument(
        "--sample_batch_size", type=int, default=1000, help="batch size used during training"
    )

    parser.add_argument(
        "--reg", type=float, default=1e-2, help="regularization parameter for Sinkhorn method"
    )

    parser.add_argument(
        "--loss_reg", type=float, default=1e-2, help="regularization parameter for Sinkhorn method"
    )

    parser.add_argument(
        "--shift", type=float, default=1.00, help="distribution shift for generating simulated data"
    )

    parser.add_argument(
        "--dataset_noise", type=float, default=0.00, help="noise for generating simulated 2d data"
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for mvpts")

    parser.add_argument("--lr_prediction", type=float, default=1e-4, help="Learning rate to train prediction model")

    parser.add_argument("--num_epochs_prediction", type=int, default=2000, help="Number of epochs to train prediction model")

    parser.add_argument("--m_distance", type=str, default="L2square", help="distance used for compute_M")

    parser.add_argument("--use_clustering", action="store_true", help="use clustering method, instead of using which distribution the sample comes from, to assign labels")

    parser.add_argument("--svm_c", type=float, default=1.0, help="C parameter for SVM")

    parser.add_argument("--svm_kernel", type=str, default="rbf", help="kernel parameter for SVM")

    parser.add_argument(
        "--train_round", type=int, default=0, help="number of training round has been done"
    )

    parser.add_argument(
        "--unique_name", type=str, default="", help="unique name for folder to save outputs"
    )

    # Parameters for phi transformation

    parser.add_argument("--phi_mode", type=str, default="none", help="none, kernel or transform")

    parser.add_argument(
        "--phi_epochs", type=int, default=0, help="number of epochs to update phi for"
    )

    parser.add_argument(
        "--output_dim", type=int, default=1, help="output dimension of phi transform"
    )

    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="regularization parameter for gradient penalty"
    )

    parser.add_argument(
        "--phi_clip", type=float, default=0.1, help="clip parameter for phi"
    )

    parser.add_argument(
        "--phi_k", type=int, default=1, help="number of iterations to update phi for"
    )

    parser.add_argument(
        "--mvpts_thred", type=float, default=-1.0, help="convergence threshold for mvpts, negative means no threshold"
    )

    args = parser.parse_args()

    return args


def prepare_args_for_grid_search(args):
    if args.data_type=="image":
        if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
            args.data_dim = 28 * 28
            args.in_channels = 1
            args.image_width = 28
        elif args.dataset == "CIFAR10":
            args.data_dim = 32 * 32 * 3
            args.in_channels = 3
            args.image_width = 32
        args.k = 10
        args.assign_label = "hard"
        args.batch_size = 1000
        args.sample_batch_size = 100
        args.num_samples = 30
        args.num_epochs = 10
        args.num_epochs_prediction = 30
        # These two parameters will not work, it is just for saving at csv file
        args.num_pts=50000
        args.num_test=10000

    elif args.data_type=="sim":
        if args.model=="svm":
            args.assign_label = "hard"
        elif args.model=="logit":
            args.assign_label = "soft"
        else:
            raise NotImplementedError
        args.num_epochs_prediction = 10000
        args.batch_size = 1000
        args.num_epochs = 2000
        args.num_pts = 20000
        args.num_test = 10000
        if args.sample_batch_size > args.num_samples:
            args.sample_batch_size = args.num_samples

    else:
        raise NotImplementedError
    return args
