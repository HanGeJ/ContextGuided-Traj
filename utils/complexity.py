from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, Iterable

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module, bytes_per_parameter: int = 4) -> int:
    return int(sum(p.numel() for p in model.parameters()) * bytes_per_parameter)


def estimate_flops(model: nn.Module, input_batch: dict) -> int:
    """Estimate major Linear/GRU/MHA FLOPs for one network forward pass.

    This remains a deterministic analytical estimate rather than a profiler.
    It is intended for matched controlled comparisons inside this repository.
    """
    flops = 0
    hooks = []

    def hook(module: nn.Linear, inputs, outputs):
        nonlocal flops
        x = inputs[0]
        num_vectors = 1
        for dim in x.shape[:-1]:
            num_vectors *= int(dim)
        macs = num_vectors * module.in_features * module.out_features
        bias_ops = num_vectors * module.out_features if module.bias is not None else 0
        flops += 2 * macs + bias_ops

    def gru_hook(module: nn.GRU, inputs, outputs):
        nonlocal flops
        x = inputs[0]
        batch = int(x.shape[0]) if module.batch_first else int(x.shape[1])
        steps = int(x.shape[1]) if module.batch_first else int(x.shape[0])
        input_dim = int(module.input_size)
        hidden = int(module.hidden_size)
        layers = int(module.num_layers)
        # Three GRU gates. Count multiply-adds as two FLOPs.
        for layer in range(layers):
            in_dim = input_dim if layer == 0 else hidden
            macs = batch * steps * 3 * (in_dim * hidden + hidden * hidden)
            bias_ops = batch * steps * 6 * hidden if module.bias else 0
            flops += 2 * macs + bias_ops

    def mha_hook(module: nn.MultiheadAttention, inputs, outputs):
        nonlocal flops
        q = inputs[0]
        if module.batch_first:
            batch, steps, dim = map(int, q.shape[:3])
        else:
            steps, batch, dim = map(int, q.shape[:3])
        # Q/K/V projections + output projection + attention score/value products.
        proj_macs = batch * steps * 4 * dim * dim
        attn_macs = batch * 2 * steps * steps * dim
        flops += 2 * (proj_macs + attn_macs)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook))
        elif isinstance(module, nn.GRU):
            hooks.append(module.register_forward_hook(gru_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))

    training = model.training
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prepared = {
        key: (value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) else value)
        for key, value in input_batch.items()
    }
    with torch.no_grad():
        model(
            prepared["obs"],
            ego_obs_rel=prepared.get("ego_obs_rel"),
            ego_valid=prepared.get("ego_valid"),
            neighbor_rel=prepared.get("neighbor_rel"),
            neighbor_mask=prepared.get("neighbor_mask"),
        )
    for handle in hooks:
        handle.remove()
    model.train(training)
    return int(flops)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(fn: Callable[[], object], device: torch.device,
                       warmup: int = 100, repeats: int = 1000,
                       deadlines_ms: Iterable[float] = (20.0, 50.0, 100.0)) -> Dict[str, float]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            fn()
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            _sync(device)
            start = time.perf_counter_ns()
            fn()
            _sync(device)
            timings.append((time.perf_counter_ns() - start) / 1e6)

    ordered = sorted(timings)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        alpha = position - lo
        return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha

    result: Dict[str, float] = {
        "mean_ms": float(statistics.fmean(timings)),
        "median_ms": float(percentile(0.50)),
        "p95_ms": float(percentile(0.95)),
        "p99_ms": float(percentile(0.99)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "repeats": int(repeats),
    }
    for deadline in deadlines_ms:
        result[f"deadline_violation_rate_{deadline:g}ms"] = float(
            sum(t > deadline for t in timings) / len(timings)
        )
    if device.type == "cuda":
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return result


def vehicle_travel_distance_m(latency_ms: float, speed_kmh: float) -> float:
    return float(speed_kmh / 3.6 * latency_ms / 1000.0)


def measure_inference_time(model: nn.Module, loader, device: torch.device,
                           warmup_batches: int = 2,
                           measure_batches: int = 10) -> Dict[str, float]:
    """Compatibility helper for average loader-batch inference timing."""
    model.eval()
    measured = 0
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            obs = batch["obs"].to(device)
            ego_obs_rel = batch.get("ego_obs_rel")
            ego_valid = batch.get("ego_valid")
            neighbor_rel = batch.get("neighbor_rel")
            neighbor_mask = batch.get("neighbor_mask")
            if ego_obs_rel is not None:
                ego_obs_rel = ego_obs_rel.to(device)
            if ego_valid is not None:
                ego_valid = ego_valid.to(device)
            if neighbor_rel is not None:
                neighbor_rel = neighbor_rel.to(device)
            if neighbor_mask is not None:
                neighbor_mask = neighbor_mask.to(device)
            model_kwargs = dict(
                ego_obs_rel=ego_obs_rel, ego_valid=ego_valid,
                neighbor_rel=neighbor_rel, neighbor_mask=neighbor_mask,
            )
            if batch_idx < warmup_batches:
                model(obs, **model_kwargs)
                _sync(device)
                continue
            if measured >= measure_batches:
                break
            _sync(device)
            start = time.perf_counter()
            model(obs, **model_kwargs)
            _sync(device)
            total_time += time.perf_counter() - start
            total_samples += obs.size(0)
            measured += 1
    if measured == 0 or total_samples == 0:
        return {
            "inference_time_ms_per_batch": float("nan"),
            "inference_time_ms_per_sample": float("nan"),
            "timed_batches": 0,
            "timed_samples": 0,
        }
    return {
        "inference_time_ms_per_batch": 1000.0 * total_time / measured,
        "inference_time_ms_per_sample": 1000.0 * total_time / total_samples,
        "timed_batches": measured,
        "timed_samples": total_samples,
    }
