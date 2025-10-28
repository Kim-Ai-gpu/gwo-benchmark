import time
import torch
from .base import GWOModule
from .datasets import DATASET_REGISTRY
from .hardware import measure_hardware_performance
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    @abstractmethod
    def evaluate(self, model: GWOModule) -> dict:
        """Runs the evaluation pipeline and returns a dictionary with performance metrics."""
        pass

class Evaluator(BaseEvaluator):
    """Provides a standardized training and evaluation pipeline."""

    def __init__(self, dataset_name: str, train_config: dict):
        self.dataset_name = dataset_name
        self.train_config = train_config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: GWOModule) -> dict:
        """Runs the evaluation pipeline."""
        dataset_cls = DATASET_REGISTRY.get(self.dataset_name.lower())
        if not dataset_cls:
            raise NotImplementedError(f"Dataset '{self.dataset_name}' is not supported.")
        
        dataset = dataset_cls()
        batch_size = self.train_config.get('batch_size', 4)
        num_workers = self.train_config.get('num_workers', 2)
        trainloader, testloader = dataset.get_loaders(batch_size, num_workers)

        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.train_config.get('lr', 0.001), momentum=self.train_config.get('momentum', 0.9))

        for epoch in range(self.train_config.get('epochs', 1)):
            with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
                for data in tepoch:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

        train_accuracy = self._calculate_accuracy(model, trainloader, desc="Calculating Train Accuracy")
        test_accuracy = self._calculate_accuracy(model, testloader, desc="Calculating Test Accuracy")

        perf_metrics = measure_hardware_performance(model, testloader, self.device, batch_size)

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "latency_ms": perf_metrics["latency_ms"],
            "throughput_imgs_sec": perf_metrics["throughput_imgs_sec"],
        }

    def _calculate_accuracy(self, model, dataloader, desc: str) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(dataloader, desc=desc):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
