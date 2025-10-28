from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def get_loaders(self, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
        """Returns train and test dataloaders."""
        pass
