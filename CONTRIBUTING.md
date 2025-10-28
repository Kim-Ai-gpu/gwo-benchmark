# Contributing to GWO Benchmark

First off, thank you for considering contributing! We're thrilled you're interested in helping us build a better tool for the deep learning community. Every contribution, from a small typo fix to a new core feature, is valuable.

This document provides guidelines for contributing to the GWO Benchmark project. Please take a moment to review it before you get started.

## Project Philosophy

Our goal is to create a benchmark that is not just a leaderboard, but a **scientific instrument** for understanding neural network architectures. When contributing, please keep these principles in mind:

1.  **Clarity over Complexity:** Code should be clear, well-documented, and easy to understand.
2.  **Rigor over Speed:** Accuracy in our measurements and calculations is paramount.
3.  **Modularity and Extensibility:** Contributions should be designed to be easily extended by others.

## How Can I Contribute?

There are many ways to contribute to the project. Here are a few ideas:

#### 🐛 Reporting Bugs
If you find a bug, please open an issue and provide as much detail as possible, including:
- A clear and descriptive title.
- Steps to reproduce the bug.
- The expected behavior and what actually happened.
- Your operating system, Python version, and relevant package versions.

#### 💡 Suggesting Enhancements
Have an idea for a new feature or an improvement to an existing one? Open an issue with a clear description of your suggestion and why it would be valuable.

#### 📝 Improving Documentation
Clear documentation is crucial. If you find parts of the documentation that are unclear, confusing, or incomplete, please submit a pull request with your improvements.

#### 💻 Writing Code
This is the most direct way to contribute. We have a special need for contributions in the following areas:

-   **Implement New GWO Models:** Implement a popular or novel neural network operation as a `GWOModule`. This is a great way to start! Look for models we need in the "Good First Issues" label on GitHub.
    -   Examples: Transformer blocks, MLP-Mixer layers, various Attention mechanisms, Graph Convolutions.
-   **Add Support for New Datasets:** Expand the benchmark's reach by adding support for new datasets in NLP, Graph ML, or other domains. You'll need to create a new `BaseDataset` subclass.
-   **Enhance the Core Framework:** Improve the `Evaluator`, add new metrics to `BenchmarkResult`, or refine the `ComplexityCalculator`.

## Your First Code Contribution

Ready to write some code? Here’s how to set up your environment and submit your first pull request.

#### 1. Fork & Clone the Repository
- **Fork** this repository on GitHub.
- **Clone** your forked repository to your local machine:
  ```bash
  git clone https://github.com/Kim-Ai-gpu/gwo-benchmark.git
  cd gwo-benchmark
  ```

#### 2. Set Up Your Development Environment
We recommend using a virtual environment.
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies in editable mode
pip install -e .
pip install -r requirements-dev.txt # Install testing dependencies
```

#### 3. Create a New Branch
Create a branch for your changes. Please use a descriptive name.
```bash
git checkout -b feature/implement-transformer-block
```

#### 4. Write Your Code & Add Tests
- Make your changes to the code.
- **Add tests!** For any new feature, please add corresponding unit tests in the `tests/` directory. For bug fixes, add a test that would have failed before your fix.

#### 5. Run Tests
Before submitting, ensure all tests pass.
```bash
python -m unittest discover tests
```

#### 6. Format Your Code
We use `black` for code formatting. Please format your code before committing.
```bash
black .
```

#### 7. Commit and Push
- Commit your changes with a clear and descriptive commit message.
- Push your branch to your forked repository on GitHub.
  ```bash
  git add .
  git commit -m "feat: Implement Transformer block as a GWOModule"
  git push origin feature/implement-transformer-block
  ```

#### 8. Open a Pull Request
- Go to the original GWO Benchmark repository on GitHub and open a pull request.
- Provide a clear description of your changes and link to any relevant issues.
- We will review your pull request as soon as possible.

Thank you again for your interest in contributing. We look forward to your pull requests!
