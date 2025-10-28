import pkgutil
import inspect
import importlib
from .base import BaseDataset

DATASET_REGISTRY = {}

def register_dataset(name: str):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def discover_datasets():
    """Automatically discovers and imports datasets in this package."""
    for _, name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f".{name}", __name__)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseDataset) and cls is not BaseDataset:
                dataset_name = cls.__name__.lower().replace("dataset", "")
                DATASET_REGISTRY[dataset_name] = cls

discover_datasets()
