import time
import torch
from .base import GWOModule

def measure_hardware_performance(model: GWOModule, testloader: torch.utils.data.DataLoader, device: torch.device, batch_size: int) -> dict:
    """Measures latency and throughput with warm-up and synchronization."""
    model.eval()

    try:
        dummy_input, _ = next(iter(testloader))
        dummy_input = dummy_input.to(device)
    except StopIteration:
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    warmup_iters = 10
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    repetitions = 100
    timings = []
    for _ in range(repetitions):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000)
    
    latency_ms = sum(timings) / repetitions

    duration = 1
    num_images = 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < duration:
        with torch.no_grad():
            _ = model(dummy_input)
        num_images += batch_size
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    throughput_imgs_sec = num_images / total_duration

    return {
        "latency_ms": latency_ms,
        "throughput_imgs_sec": throughput_imgs_sec,
    }
