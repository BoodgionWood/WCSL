import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_rescaled_mnist_dataset(n=28, data_path='./data', download=True, batch_size=64, train=True, drop_last=True):
    transform = transforms.Compose([
        transforms.Resize((n, n)),  # resize images to n*n
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_data = datasets.MNIST(root=data_path, train=train, download=download, transform=transform)
    mnist_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    return mnist_loader


def save_generated_MNIST_samples(sample_images, filename='./output/MNIST/generated_samples.npy', num_samples=50):
    generated_samples_np = sample_images.detach().cpu().numpy().reshape(num_samples,-1)
    np.save(filename, generated_samples_np)


def load_fashion_mnist_dataset(data_path='./data', download=False, batch_size=64, train=True, drop_last=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root=data_path, train=train, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return dataloader


def load_cifar10_dataset(data_path='./data', download=False, batch_size=64, train=True, drop_last=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_path, train=train, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return dataloader


def load_image_dataset(args, train=True):
    if args.dataset == "MNIST":
        train_loader = load_rescaled_mnist_dataset(n=28, download=args.download, data_path=args.data_path, batch_size=args.batch_size, train=train,drop_last=True)
    elif args.dataset == "FashionMNIST":
        train_loader = load_fashion_mnist_dataset(download= args.download, data_path=args.data_path, batch_size=args.batch_size, train=train, drop_last=True)
    elif args.dataset == "CIFAR10":
        train_loader = load_cifar10_dataset(download=args.download, data_path=args.data_path, batch_size=args.batch_size, train=train, drop_last=True)
    else:
        raise NotImplementedError
    return train_loader
