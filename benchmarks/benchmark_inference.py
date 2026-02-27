"""Spark benchmark entrypoint for policy scorer inference paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession

# Support direct script execution: `python benchmarks/benchmark_inference.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import run_policy_scorer_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--treatments", type=int, default=4)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--no-antipattern", action="store_true")
    parser.add_argument("--no-jvm-reference", action="store_true")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("spark-policy-toolkit-benchmark")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .getOrCreate()
    )
    try:
        run_policy_scorer_benchmark(
            spark=spark,
            n_rows=args.rows,
            n_features=args.features,
            depth=args.depth,
            n_treatments=args.treatments,
            n_trees=args.trees,
            missing_rate=args.missing_rate,
            partitions=args.partitions,
            include_antipattern=not args.no_antipattern,
            include_jvm_reference=not args.no_jvm_reference,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
