from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Tuple
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST



def get_data_loader(batch_size, dataset='mnist') -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset == 'mnist':
        download_root = './MNIST_DATASET'

        train_dataset = MNIST(download_root, transform=transform, train=True, download=True)
        test_dataset = MNIST(download_root, transform=transform, train=False, download=True)
    elif dataset == 'fashion_mnist':
        download_root = './FashionMNIST_DATASET'

        train_dataset = FashionMNIST(download_root, transform=transform, train=True, download=True)
        test_dataset = FashionMNIST(download_root, transform=transform, train=False, download=True)
    
    elif dataset == "cifar10":
        download_root = './CIFAR10_DATASET'
        train_dataset = CIFAR10(download_root, transform=transform, train=True, download=True)
        test_dataset = CIFAR10(download_root, transform=transform, train=True, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size) # remove shuffle=True for overfittings
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return (train_loader, test_loader)

if __name__ == '__main__':
    train_loader, _ = get_data_loader(64, 'fashion_mnist')
    print(len(train_loader.dataset))