from .base import GWOModule
from .complexity import ComplexityCalculator
from .evaluator import BaseEvaluator, Evaluator
from .result import BenchmarkResult, AggregatedBenchmarkResult
import csv
import os
import dataclasses
from datetime import datetime
from typing import List, Optional, Union
import random
import numpy as np
import torch
import pandas as pd

def seed_everything(seed: int):
    """Sets the seed for all relevant random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _single_run(model: GWOModule, evaluator: BaseEvaluator, result_dir: str, seed: Optional[int]) -> BenchmarkResult:
    """Performs a single benchmark run."""
    if seed is not None:
        seed_everything(seed)

    complexity_calculator = ComplexityCalculator()
    complexity_results = complexity_calculator.calculate(model)

    eval_results = evaluator.evaluate(model)

    generalization_gap = eval_results["train_accuracy"] - eval_results["test_accuracy"]

    score = (100 * eval_results["test_accuracy"]) / (1 + complexity_results["omega_proxy"])

    result = BenchmarkResult(
        model_name=model.__class__.__name__,
        score=score,
        test_accuracy=eval_results["test_accuracy"],
        train_accuracy=eval_results["train_accuracy"],
        generalization_gap=generalization_gap,
        omega_proxy=complexity_results["omega_proxy"],
        c_d=complexity_results["c_d"],
        c_p_M=complexity_results["c_p_M"],
        latency_ms=eval_results["latency_ms"],
        throughput_imgs_sec=eval_results["throughput_imgs_sec"],
        seed=seed,
    )

    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "benchmark_results.csv")
    write_header = not os.path.exists(result_file)

    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["timestamp"] + [f.name for f in dataclasses.fields(result)]
            writer.writerow(header)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp] + [getattr(result, f.name) for f in dataclasses.fields(result)]
        writer.writerow(row)

    return result

def run(
    model: GWOModule, 
    evaluator: BaseEvaluator, 
    result_dir: str = ".", 
    seeds: Optional[List[int]] = None
) -> Union[BenchmarkResult, AggregatedBenchmarkResult]:
    """
    Main entry point for running the benchmark.

    If seeds are provided, runs the benchmark for each seed and returns aggregated results.
    If seeds is None, performs a single run.
    """
    if seeds:
        results = [_single_run(model, evaluator, result_dir, seed) for seed in seeds]
        
        df = pd.DataFrame(results)
        
        agg_result = AggregatedBenchmarkResult(
            model_name=model.__class__.__name__,
            num_runs=len(seeds),
            score_mean=df["score"].mean(),
            score_std=df["score"].std(),
            test_accuracy_mean=df["test_accuracy"].mean(),
            test_accuracy_std=df["test_accuracy"].std(),
            latency_ms_mean=df["latency_ms"].mean(),
            latency_ms_std=df["latency_ms"].std(),
            throughput_imgs_sec_mean=df["throughput_imgs_sec"].mean(),
            throughput_imgs_sec_std=df["throughput_imgs_sec"].std(),
        )
        return agg_result
    else:
        return _single_run(model, evaluator, result_dir, seed=None)
