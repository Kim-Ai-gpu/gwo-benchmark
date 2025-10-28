import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .base import BaseDataset

class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset."""

    def get_loaders(self, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
        """Returns CIFAR-10 train and test dataloaders."""
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return trainloader, testloader
