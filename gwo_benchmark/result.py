import dataclasses
from typing import Optional

@dataclasses.dataclass
class BenchmarkResult:
    model_name: str
    score: float
    test_accuracy: float
    train_accuracy: float
    generalization_gap: float
    omega_proxy: float
    c_d: int
    c_p_M: float
    latency_ms: float
    throughput_imgs_sec: float
    seed: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"--- Benchmark Result for: {self.model_name} (Seed: {self.seed}) ---\n"
            f"  >> Final Score:         {self.score:.2f} <<\n"
            f"--------------------------------------------------\n"
            f"  Accuracy (Test/Train): {self.test_accuracy:.2f}% / {self.train_accuracy:.2f}%\n"
            f"  Generalization Gap:    {self.generalization_gap:.2f}%\n"
            f"  Complexity (Ω_proxy):  {self.omega_proxy:.4f}\n"
            f"    - C_D (Descriptive): {self.c_d}\n"
            f"    - C_P (Parametric):  {self.c_p_M:.4f} M\n"
            f"  Performance:\n"
            f"    - Latency:           {self.latency_ms:.2f} ms/batch\n"
            f"    - Throughput:        {self.throughput_imgs_sec:.2f} images/sec\n"
            f"--------------------------------------------------"
        )

@dataclasses.dataclass
class AggregatedBenchmarkResult:
    """Dataclass to hold aggregated (mean/std) benchmark results."""
    model_name: str
    num_runs: int
    score_mean: float
    score_std: float
    test_accuracy_mean: float
    test_accuracy_std: float
    latency_ms_mean: float
    latency_ms_std: float
    throughput_imgs_sec_mean: float
    throughput_imgs_sec_std: float

    def __str__(self) -> str:
        return (
            f"--- Aggregated Benchmark Result for: {self.model_name} ({self.num_runs} runs) ---\n"
            f"  >> Final Score:         {self.score_mean:.2f} \u00b1 {self.score_std:.2f} <<\n"
            f"--------------------------------------------------\n"
            f"  Accuracy (Test): {self.test_accuracy_mean:.2f}% \u00b1 {self.test_accuracy_std:.2f}%\n"
            f"  Performance:\n"
            f"    - Latency:           {self.latency_ms_mean:.2f} \u00b1 {self.latency_ms_std:.2f} ms/batch\n"
            f"    - Throughput:        {self.throughput_imgs_sec_mean:.2f} \u00b1 {self.throughput_imgs_sec_std:.2f} images/sec\n"
            f"--------------------------------------------------"
        )
