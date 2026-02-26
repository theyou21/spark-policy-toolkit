"""Minimal benchmark entrypoint for VectorizedArrayTree batch inference."""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.vectorized_tree import VectorizedArrayTree


def build_balanced_continuous_tree(depth: int, n_features: int) -> dict[str, np.ndarray]:
    """Construct a synthetic balanced tree with continuous splits only."""
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if n_features <= 0:
        raise ValueError("n_features must be > 0")

    n_nodes = 2 ** (depth + 1) - 1
    first_leaf = 2**depth - 1

    node_type = np.zeros(n_nodes, dtype=np.int8)
    node_type[:first_leaf] = 1

    rng = np.random.default_rng(7)
    feature_idx = np.zeros(n_nodes, dtype=np.int64)
    feature_idx[:first_leaf] = rng.integers(0, n_features, size=first_leaf, dtype=np.int64)

    thresholds = np.zeros(n_nodes, dtype=np.float64)
    thresholds[:first_leaf] = rng.uniform(-1.5, 1.5, size=first_leaf)

    left_child = np.arange(n_nodes, dtype=np.int64)
    right_child = np.arange(n_nodes, dtype=np.int64)
    internal = np.arange(first_leaf, dtype=np.int64)
    left_child[internal] = internal * 2 + 1
    right_child[internal] = internal * 2 + 2

    tree = {
        "node_type": node_type,
        "feature_idx": feature_idx,
        "thresholds": thresholds,
        "cat_value": np.zeros(n_nodes, dtype=np.float64),
        "left_child": left_child,
        "right_child": right_child,
        "nan_goes_left": rng.random(n_nodes) < 0.5,
        "leaf_response_rates": rng.uniform(0.0, 1.0, size=(n_nodes, 4)).astype(np.float64),
    }
    return tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    X = rng.normal(size=(args.rows, args.features)).astype(np.float64)
    if args.missing_rate > 0:
        missing_mask = rng.random(size=X.shape) < args.missing_rate
        X[missing_mask] = np.nan

    tree = VectorizedArrayTree(build_balanced_continuous_tree(args.depth, args.features))

    tree.predict_numpy(X[:1000])

    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        _ = tree.predict_numpy(X)
        timings.append(time.perf_counter() - start)

    best = min(timings)
    throughput = args.rows / best
    print(f"rows={args.rows} depth={args.depth} features={args.features} runs={args.runs}")
    print(f"best_seconds={best:.6f}")
    print(f"rows_per_second={throughput:.2f}")


if __name__ == "__main__":
    main()
