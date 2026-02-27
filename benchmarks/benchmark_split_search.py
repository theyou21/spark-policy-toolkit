"""Minimal benchmark entrypoint for collect-less split search."""

from __future__ import annotations

import time

from pyspark.sql import DataFrame, SparkSession, functions as F

from src.split_search import MetricMode, build_prefix_sums, score_candidates_collectless


def _build_synthetic_split_df(
    spark: SparkSession,
    *,
    n_rows: int = 1_000_000,
    missing_rate: float = 0.1,
    seed: int = 23,
) -> DataFrame:
    base = spark.range(n_rows)
    return base.select(
        F.when(
            F.rand(seed) < F.lit(missing_rate),
            F.lit(None).cast("double"),
        )
        .otherwise(F.randn(seed + 1))
        .alias("feature"),
        F.when(F.rand(seed + 2) < F.lit(0.5), F.lit("control"))
        .when(F.rand(seed + 3) < F.lit(0.5), F.lit("t1"))
        .otherwise(F.lit("t2"))
        .alias("treatment"),
        F.when(F.rand(seed + 4) < F.lit(0.3), F.lit(1.0))
        .otherwise(F.lit(0.0))
        .alias("outcome"),
    )


def run_split_search_benchmark(
    spark: SparkSession,
    *,
    n_rows: int = 1_000_000,
    n_quantile_splits: int = 32,
    evaluation_mode: MetricMode = "sql",
) -> None:
    """Time D2.1 + D2.2 on synthetic data and print the winning split."""
    df = _build_synthetic_split_df(
        spark=spark,
        n_rows=n_rows,
    ).cache()
    _ = df.count()

    t0 = time.perf_counter()
    prefix_sums_df = build_prefix_sums(
        df,
        feature_col="feature",
        treatment_col="treatment",
        outcome_col="outcome",
        num_quantile_splits=n_quantile_splits,
    )
    best_split = score_candidates_collectless(
        prefix_sums_df,
        evaluation_mode=evaluation_mode,
    )
    elapsed = time.perf_counter() - t0

    print(f"n_rows={n_rows} n_quantile_splits={n_quantile_splits}")
    print(f"evaluation_mode={evaluation_mode}")
    print(f"best_split={best_split}")
    print(f"elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    spark_session = (
        SparkSession.builder.master("local[*]")
        .appName("benchmark_split_search")
        .getOrCreate()
    )
    try:
        run_split_search_benchmark(spark_session)
    finally:
        spark_session.stop()
