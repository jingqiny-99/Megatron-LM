# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact per-update sample depths shared by pipeline and RSBF schedules."""

import random
from functools import lru_cache


def validate_stratified_depth(config):
    """Validate exact sample counts and their grouping before building an execution plan."""
    counts = config.loop_depth_counts
    group = config.loop_depth_group_size
    if counts is None or len(counts) != config.loop_iterations:
        raise ValueError("loop_depth_counts needs one sample count per possible depth")
    if not isinstance(group, int) or group < 1:
        raise ValueError("loop_depth_group_size must be a positive integer")
    if not all(isinstance(n, int) and n >= 0 and n % group == 0 for n in counts):
        raise ValueError("loop_depth_counts must be nonnegative multiples of loop_depth_group_size")
    if not any(counts) or any(counts[: config.loop_min_rounds - 1]):
        raise ValueError("loop_depth_counts must assign samples only to allowed exit depths")


@lru_cache(maxsize=32)
def _group_depths(counts, group_size, seed, iteration):
    depths = [d for d, n in enumerate(counts, 1) for _ in range(n // group_size)]
    # A private host RNG keeps the ordering independent of device, rank, waves, and model RNG.
    random.Random(f"loop-depth/v1/{seed}/{iteration}").shuffle(depths)
    return tuple(depths)


def stratified_depth(config, iteration: int, global_sample_index: int) -> int:
    """Depth of the zero-based sample inside this optimizer update, before any placement."""
    group_size = config.loop_depth_group_size
    groups = _group_depths(
        tuple(config.loop_depth_counts), group_size, config.loop_depth_seed, iteration
    )
    if not 0 <= global_sample_index < len(groups) * group_size:
        raise ValueError(f"sample index {global_sample_index} outside this update's depth schedule")
    return groups[global_sample_index // group_size]
