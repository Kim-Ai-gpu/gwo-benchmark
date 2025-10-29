import torch
import torch.nn as nn
import pandas as pd
from gwo_benchmark import run, Evaluator, GWOModule

from examples.models.resnet import ResNet18GWO

class BaselineNet(nn.Module):
    def __init__(self, gwo_layer_class, in_channels, out_channels, num_classes=10):
        super().__init__()
        self.gwo_layer1 = gwo_layer_class(in_channels, out_channels, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.gwo_layer2 = gwo_layer_class(out_channels, out_channels*2, kernel_size=3)
        
        dummy_input = torch.randn(1, in_channels, 32, 32)
        x = self.pool(self.gwo_layer1(dummy_input))
        x = self.pool(self.gwo_layer2(x))
        flattened_size = torch.flatten(x, 1).size(1)
        
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.gwo_layer1(x)))
        x = self.pool(torch.relu(self.gwo_layer2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BaselineNetGWO(GWOModule):
    def __init__(self, gwo_layer_class):
        super().__init__()
        self.net = BaselineNet(gwo_layer_class, in_channels=3, out_channels=16)

    @property
    def C_D(self) -> int:
        return self.net.gwo_layer1.C_D + self.net.gwo_layer2.C_D

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        modules = self.net.gwo_layer1.get_parametric_complexity_modules()
        modules += self.net.gwo_layer2.get_parametric_complexity_modules()
        return modules

    def forward(self, x):
        return self.net(x)

def main():
    models_to_benchmark = {
        "ResNet18": ResNet18GWO
    }
    
    train_config = {
        "epochs": 30,
        "lr": 0.001,
        "momentum": 0.9,
        "batch_size": 64,
        "num_workers": 2,
    }

    evaluator = Evaluator(dataset_name="cifar10", train_config=train_config)
    
    results_list = []
    
    for name, gwo_class in models_to_benchmark.items():
        print(f"\n{'='*20} BENCHMARKING: {name} {'='*20}")
        model = gwo_class()
        
        result = run(model, evaluator, result_dir="baseline_results")
        
        result_dict = {
            "Model": result.model_name.replace("BaselineNetGWO", f"({name})"),
            "Score": result.score,
            "Test Acc (%)": result.test_accuracy,
            "Ω_proxy": result.omega_proxy,
            "C_D": result.c_d,
            "C_P (M)": result.c_p_M,
            "Latency (ms)": result.latency_ms
        }
        results_list.append(result_dict)
        
    df = pd.DataFrame(results_list)
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    print("\n\n" + "="*60)
    print("                 GWO BENCHMARK BASELINE RESULTS")
    print("="*60)
    print(df.to_string())
    print("="*60)
    print("\n* Score = (100 * Test Acc) / (1 + Ω_proxy). Higher is better.")

if __name__ == "__main__":
    main()