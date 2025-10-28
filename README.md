# GWO Benchmark: The Architect's Arena

[![PyPI version](https://badge.fury.io/py/gwo-benchmark.svg)](https://badge.fury.io/py/gwo-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Is your neural network 'smart' or just big? This benchmark tells you the difference.**

This Python package provides a framework for benchmarking neural network operations, inspired by the GWO (Generalized Windowed Operation) theory from the paper ["Window is Everything: A Grammar for Neural Operations"](https://zenodo.org/records/17103133).

Instead of just measuring accuracy, this benchmark scores operations on their **architectural efficiency**. It quantifies the relationship between an operation's theoretical **Operational Complexity (`Ω_proxy`)** and its real-world performance, helping you design smarter, more efficient models.

---

## Key Concepts in 1 Minute

The core idea is to break down any neural network operation (like Convolution or Self-Attention) into its fundamental building blocks and score its complexity.

- **GWO (Generalized Windowed Operation):** A "grammar" that describes any operation using three components:
    - **Path (P):** *Where* to look for information (e.g., a local sliding window).
    - **Shape (S):** *What form* of information to look for (e.g., a square patch).
    - **Weight (W):** *What* to value in that information (e.g., a learnable kernel).

- **Operational Complexity (`Ω_proxy`):** The "intelligence score" of your operation. A lower score for the same performance means a more efficient design. It's calculated as:
    `Ω_proxy = C_D (Structural Complexity) + α * C_P (Parametric Complexity)`

    - **`C_D` (Descriptive Complexity):** How many basic "primitives" does it take to describe your operation's structure? (You define this based on our guide).
    - **`C_P` (Parametric Complexity):** How many extra parameters are needed to *generate* the operation's behavior dynamically? (e.g., the offset prediction network in Deformable Convolution). This is calculated automatically.

## Installation

```bash
pip install gwo-benchmark
```
Or for development from this repository:
```bash
git clone https://github.com/Kim-Ai-gpu/gwo-benchmark.git
cd gwo-benchmark
pip install -e .
```

## Quick Start in 3 Steps

Let's benchmark a simple custom CNN on CIFAR-10.

**Step 1: Define your model inheriting from `GWOModule`**

Create your model file `my_models.py`:
```python
# my_models.py
import torch.nn as nn
from gwo_benchmark import GWOModule

class MySimpleConv(GWOModule):
    # PRIMITIVES: STATIC_SLIDING(1) + DENSE_SQUARE(1) + SHARED_KERNEL(1)
    # Based on the official primitive guide, the complexity is 3.
    C_D = 3

    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

    # This model has no dynamic components, so C_P is zero.
    def get_parametric_complexity_modules(self):
        return []
```

**Step 2: Create your benchmark script**

Create your main script `run_benchmark.py`:
```python
# run_benchmark.py
from gwo_benchmark import run, Evaluator
from my_models import MySimpleConv

# 1. Instantiate your model
model = MySimpleConv()

# 2. Configure the evaluation environment
#    The standard Evaluator handles training and testing for you.
evaluator = Evaluator(
    dataset_name="cifar10",
    train_config={ "epochs": 2, "batch_size": 64 }
)

# 3. Run the benchmark!
if __name__ == "__main__":
    result = run(model, evaluator, result_dir="benchmark_results")
    print(result)
```

**Step 3: Run from your terminal**

```bash
python run_benchmark.py
```

You'll see a detailed analysis of your model's complexity and performance, saved in the `benchmark_results` directory.

---

## Understanding Your Score: The Tier System

A high score is good, but how high is high enough? To give your results context, we've established a tier system based on the performance of well-known baseline operations.

Your goal is to design an operation that reaches A-Tier or pushes the boundaries into S-Tier.

### The Leaderboard: Official Baselines (CIFAR-10, Official Track)

This table serves as your primary reference point. The `Score` is a measure of efficiency (higher is better).

| Model              | Score      | Test Acc (%) | Ω_proxy  | C_D  | C_P (M) | Latency (ms) | Tier |
| ------------------ | ---------- | ------------ | -------- | ---- | ------- | ------------ | ---- |
| **(StandardConv)** | **990.14** | **69.31**    | **6.00** | **6**| **0.0** | **0.50**     | **B**|
| (DeformableConv)   | 771.40     | 69.45        | 8.00     | 8    | 0.003   | 1.63         | **C**|
| (DepthwiseConv)    | 681.67     | 61.35        | 8.00     | 8    | 0.0     | 0.53         | **C**|

### Tier Definitions

-   **🏆 S-Tier (State-of-the-Art):** Your operation's efficiency score significantly surpasses all established baselines. It sets a new standard for what is possible and pushes the Pareto frontier.

-   **🚀 A-Tier (Excellent):** Clearly outperforms the **StandardConv** baseline efficiency score. This indicates a highly competitive and well-designed operation that is production-ready.

-   **✅ B-Tier (Solid Baseline):** Achieves an efficiency score comparable to **StandardConv**. Your operation is a robust and viable alternative to a classic, strong performer. This is the minimum target for a competitive design.

-   **💡 C-Tier (Promising / Situational):** Demonstrates functionality but does not yet match the efficiency of StandardConv, like the Deformable and Depthwise baselines in our test. It might be valuable for specific niches that require its unique properties.

-   **🔬 D-Tier (Experimental):** A novel idea that requires more tuning or fundamental refinement to become competitive against the established baselines.

---

## Calculating Descriptive Complexity (`C_D`) with an LLM

Calculating `C_D` requires mapping your operation's logic to our official "primitive" vocabulary. For complex operations, a Large Language Model (LLM) like GPT-4, Claude, or Gemini can help you with this analysis.

Here is a ready-to-use prompt template. Simply replace the placeholder with your `GWOModule` code.

```prompt
You are an expert in the GWO (Generalized Windowed Operation) framework for neural networks. Your task is to analyze a given GWOModule PyTorch code and calculate its Descriptive Complexity (CD).

1. Official Primitive Dictionary (v0.1):
You MUST use the following primitives and their corresponding complexity scores.

   Path (P) Primitives:
       STATIC_SLIDING: 1 (Fixed, local sliding window, e.g., standard convolution)
       GLOBAL_INDEXED: 1 (Fixed, global connectivity, e.g., matrix multiplication)
       CONTENT_AWARE: 2 (Data-dependent connectivity, requires a sub-network, e.g., deformable convolution)

   Shape (S) Primitives:
       DENSE_SQUARE(k): 1 (A dense kxk square, e.g., standard convolution)
       FULL_ROW: 1 (An entire row, e.g., matrix multiplication)
       CAUSAL_1D: 1 (1D causal mask, e.g., autoregressive models)

   Weight (W) Primitives:
       IDENTITY: 1 (Weights are the input values themselves, unparameterized)
       SHARED_KERNEL: 1 (A single, learnable kernel shared across all positions, e.g., convolution)
       DYNAMIC_ATTENTION: 2 (Weights are computed dynamically based on input, requires a sub-network, e.g., self-attention)

2. Your Task:
Analyze the PyTorch code for the GWOModule provided below. Break down its core operation into the GWO (P, S, W) components. For each component, identify the most appropriate primitive from the dictionary. Finally, sum the scores of the chosen primitives to determine the final CD. Provide a step-by-step reasoning for your choices.

3. PyTorch Code to Analyze:

{{ PASTE YOUR GWOMODULE CODE HERE }}

4. Expected Output Format:

Path (P) Analysis:
  [Your reasoning for choosing the Path primitive]
  Chosen Primitive: [PRIMITIVE_NAME] (Score: X)

Shape (S) Analysis:
  [Your reasoning for choosing the Shape primitive]
  Chosen Primitive: [PRIMITIVE_NAME] (Score: Y)

Weight (W) Analysis:
  [Your reasoning for choosing the Weight primitive]
  Chosen Primitive: [PRIMITIVE_NAME] (Score: Z)

Final Calculation:
  Total CD = X + Y + Z
```

## How It Works

The framework is designed for flexibility and extension.

1.  **`GWOModule` (`gwo_benchmark.base.GWOModule`):** The heart of your submission. You must inherit from this abstract class and implement:
    - `C_D` (property): Your calculation of the Descriptive Complexity.
    - `get_parametric_complexity_modules()` (method): A list of `nn.Module`s that contribute to `C_P`.

2.  **`Evaluator` (`gwo_benchmark.evaluator.BaseEvaluator`):** This class encapsulates all evaluation logic (training, testing, performance measurement).
    - Use the built-in `Evaluator` for standard datasets like CIFAR-10.
    - Create your own custom evaluation loop by inheriting from `BaseEvaluator` for specialized tasks.

3.  **Datasets (`gwo_benchmark.datasets`):** Easily add support for new datasets by inheriting from `BaseDataset` and registering your class. See the `datasets` directory for examples.

## Contributing

We welcome contributions! This project is in its early stages, and we believe it can grow into a standard tool for the deep learning community.

-   **Add New GWO Models:** Implement novel or existing operations (like Transformers, Attention variants, MLPs) as `GWOModule`s in the `examples` directory.
-   **Support More Datasets:** Help us expand the benchmark to new domains like NLP, Graphs, etc.
-   **Improve the Core Engine:** Enhance the `Evaluator`, `ComplexityCalculator`, or add new analysis tools.

Please see our `CONTRIBUTING.md` for more details.

## Running Tests

To ensure the integrity of the framework, please run tests before submitting a pull request.

```bash
python -m unittest discover tests```

## Citation

If you use this framework in your research, please consider citing the original paper:
*@article{https://doi.org/10.5281/zenodo.17103133, doi = {10.5281/ZENODO.17103133}, url = {https://zenodo.org/doi/10.5281/zenodo.17103133}, author = {Kim, Youngseong}, keywords = {Machine learning, Machine Learning, Supervised Machine Learning, Machine Learning/classification, Machine Learning/ethics, Machine Learning/standards, Unsupervised Machine Learning, Machine Learning/history, Machine Learning/trends, Machine Learning/economics, Supervised Machine Learning/standards, Unsupervised Machine Learning/classification}, language = {en}, title = {Window is Everything: A Grammar for Neural Operations}, publisher = {Zenodo}, year = {2025}, copyright = {Creative Commons Attribution 4.0 International}}