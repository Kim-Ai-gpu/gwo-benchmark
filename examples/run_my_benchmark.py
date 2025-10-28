import torch
from torch.utils.data import DataLoader
from gwo_benchmark import run, Evaluator, BaseEvaluator
from gwo_benchmark.base import GWOModule
from examples.models.deformable_conv import DeformableConvGWO
from examples.models.standard_conv import StandardConvGWO
from gwo_benchmark.datasets import DATASET_REGISTRY
from tqdm import tqdm

class SimpleNet(torch.nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.conv1 = conv_layer(3, 16, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = StandardConvGWO(16, 32, 3)
        # Calculate the flattened size dynamically
        self._calculate_flattened_size()
        self.fc1 = torch.nn.Linear(self.flattened_size, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def _calculate_flattened_size(self):
        # Create a dummy input tensor to determine the size after conv and pooling
        dummy_input = torch.randn(1, 3, 32, 32) # Assuming input images are 32x32 (CIFAR10)
        x = self.pool(self.conv1(dummy_input))
        x = self.pool(self.conv2(x))
        self.flattened_size = torch.flatten(x, 1).size(1)


    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StandardConvNetGWO(GWOModule):
    def __init__(self):
        super().__init__()
        self.net = SimpleNet(StandardConvGWO)

    @property
    def C_D(self) -> int:
        return self.net.conv1.C_D + self.net.conv2.C_D

    def get_parametric_complexity_modules(self) -> list[torch.nn.Module]:
        return self.net.conv1.get_parametric_complexity_modules() + self.net.conv2.get_parametric_complexity_modules()

    def forward(self, x):
        return self.net(x)

class DeformableConvNetGWO(GWOModule):
    def __init__(self):
        super().__init__()
        self.net = SimpleNet(DeformableConvGWO)

    @property
    def C_D(self) -> int:
        return self.net.conv1.C_D + self.net.conv2.C_D

    def get_parametric_complexity_modules(self) -> list[torch.nn.Module]:
        return self.net.conv1.get_parametric_complexity_modules() + self.net.conv2.get_parametric_complexity_modules()

    def forward(self, x):
        return self.net(x)

class CustomEvaluator(BaseEvaluator):
    def __init__(self, test_loader: DataLoader, device):
        self.test_loader = test_loader
        self.device = device

    def evaluate(self, model: GWOModule) -> dict:
        print(f"Running custom evaluation for {model.__class__.__name__}...")
        model.to(self.device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="Custom Test Accuracy"):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total

        return {
            "train_accuracy": 0.0,
            "test_accuracy": test_accuracy,
            "latency_ms": -1.0,
            "throughput_imgs_sec": -1.0,
        }


def main():
    """Main function to run the benchmarks."""
    train_config = {
        "epochs": 1,
        "lr": 0.001,
        "momentum": 0.9,
        "batch_size": 4,
        "num_workers": 2,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Option 1: Single run benchmark ---
    print("--- Running a single benchmark for Deformable Convolution ---")
    deform_conv_model = DeformableConvNetGWO()
    standard_evaluator = Evaluator(dataset_name="cifar10", train_config=train_config)
    single_result = run(deform_conv_model, standard_evaluator, result_dir="results")
    print(single_result)

    # --- Option 2: Multi-seed aggregated benchmark ---
    print("\n--- Running a multi-seed benchmark for Standard Convolution ---")
    seeds_to_run = [42, 123, 1024] # Define the seeds for multiple runs
    std_conv_model = StandardConvNetGWO()

    # The same evaluator can be used for multiple runs
    aggregated_result = run(
        std_conv_model,
        standard_evaluator,
        result_dir="results",
        seeds=seeds_to_run
    )
    print(aggregated_result)


if __name__ == "__main__":
    main()