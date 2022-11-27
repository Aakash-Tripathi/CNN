import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist_data():
    """ Load MNIST data from torchvision.datasets

    Returns:
        train_data: training data
        valid_data: validation data
        test_data: test data
        train_loader: training data loader
        valid_loader: validation data loader
        test_loader: test data loader
    """

    # load MNIST dataset
    data_folder = '../data'

    # define the transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_folder, train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_folder, train=False, download=True, transform=transform),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_folder, train=False, download=True, transform=transform),
        batch_size=64, shuffle=True)

    # Train test validation split
    train_data = torchvision.datasets.MNIST(data_folder, train=True, download=True, transform=transform)
    valid_data = torchvision.datasets.MNIST(data_folder, train=False, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(data_folder, train=False, download=True, transform=transform)

    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader