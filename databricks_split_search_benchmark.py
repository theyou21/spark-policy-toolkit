# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Policy Toolkit — Direction 2 Benchmark
# MAGIC
# MAGIC Self-contained notebook to validate and benchmark Direction 2 (`build_prefix_sums` + `score_candidates_collectless`).
# MAGIC
# MAGIC It covers:
# MAGIC - Tiny-data correctness: collect-less vs driver-collect parity
# MAGIC - Determinism checks
# MAGIC - Physical-plan sanity for D2.1/D2.2
# MAGIC - Runtime benchmarking for `evaluation_mode="sql"` and `evaluation_mode="mapInPandas"`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Parameters

# COMMAND ----------

# ---- correctness knobs ----
TINY_SPLITS = [-0.2, 0.2]

# ---- benchmark knobs ----
N_ROWS_BENCH = 2_000_000
N_ROWS_DRIVER_BASELINE = 200_000
N_QUANTILE_SPLITS = 32
MISSING_RATE = 0.10
BENCH_REPEATS = 3
SEED = 41

# ---- Spark knobs ----
SHUFFLE_PARTITIONS = 200

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Spark Configuration

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
spark.conf.set("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))

print("spark.sql.shuffle.partitions =", spark.conf.get("spark.sql.shuffle.partitions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Imports

# COMMAND ----------

from __future__ import annotations

import time

from pyspark.sql import functions as F

from src.split_search import (
    _build_control_treatment_df,
    _expand_nan_directions,
    _score_candidates_sql,
    best_split_driver_collect,
    build_prefix_sums,
    score_candidates_collectless,
)

print("Imports loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Test Data Builders

# COMMAND ----------

def make_tiny_df():
    return spark.createDataFrame(
        [
            (-1.00, "control", 0.0),
            (-0.10, "t1", 1.0),
            (0.10, "control", 1.0),
            (0.60, "t1", 0.0),
            (None, "control", 1.0),
            (float("nan"), "t1", 1.0),
        ],
        schema="feature double, treatment string, outcome double",
    )


def make_benchmark_df(n_rows: int, missing_rate: float, seed: int):
    base = spark.range(n_rows)
    return base.select(
        F.when(F.rand(seed) < F.lit(missing_rate / 2.0), F.lit(None).cast("double"))
        .when(
            F.rand(seed + 1) < F.lit(missing_rate),
            F.lit(float("nan")),
        )
        .otherwise(F.randn(seed + 2))
        .alias("feature"),
        F.when(F.rand(seed + 3) < F.lit(0.50), F.lit("control"))
        .when(F.rand(seed + 4) < F.lit(0.50), F.lit("t1"))
        .otherwise(F.lit("t2"))
        .alias("treatment"),
        F.when(F.rand(seed + 5) < F.lit(0.30), F.lit(1.0))
        .otherwise(F.lit(0.0))
        .alias("outcome"),
    )


def assert_prefix_schema(prefix_df):
    expected_schema = [
        ("feature", "string"),
        ("candidate_bin", "int"),
        ("bin_boundary", "double"),
        ("treatment", "string"),
        ("left_count_base", "bigint"),
        ("left_sum_base", "double"),
        ("right_count_base", "bigint"),
        ("right_sum_base", "double"),
        ("missing_count", "bigint"),
        ("missing_sum", "double"),
    ]
    actual_schema = [(field.name, field.dataType.simpleString()) for field in prefix_df.schema]
    assert actual_schema == expected_schema, (
        f"prefix schema mismatch. expected={expected_schema} actual={actual_schema}"
    )


print("Builders ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Tiny Correctness Test (Parity with Driver Baseline)

# COMMAND ----------

tiny_df = make_tiny_df()
prefix_tiny = build_prefix_sums(
    tiny_df,
    feature_col="feature",
    treatment_col="treatment",
    outcome_col="outcome",
    splits=TINY_SPLITS,
)
assert_prefix_schema(prefix_tiny)

best_sql = score_candidates_collectless(
    prefix_tiny,
    evaluation_mode="sql",
)
best_map = score_candidates_collectless(
    prefix_tiny,
    evaluation_mode="mapInPandas",
)
best_driver = best_split_driver_collect(
    tiny_df,
    feature_col="feature",
    treatment_col="treatment",
    outcome_col="outcome",
    splits=TINY_SPLITS,
)

print("best_sql   :", best_sql)
print("best_map   :", best_map)
print("best_driver:", best_driver)

assert best_sql.feature == best_driver.feature
assert best_sql.candidate_bin == best_driver.candidate_bin
assert best_sql.nan_direction == best_driver.nan_direction
assert abs(best_sql.score - best_driver.score) < 1e-12

assert best_map.feature == best_driver.feature
assert best_map.candidate_bin == best_driver.candidate_bin
assert best_map.nan_direction == best_driver.nan_direction
assert abs(best_map.score - best_driver.score) < 1e-12

display(prefix_tiny.orderBy("candidate_bin", "treatment"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 — Determinism Check

# COMMAND ----------

det_runs = []
for i in range(5):
    result = score_candidates_collectless(prefix_tiny, evaluation_mode="sql")
    det_runs.append((result.feature, result.candidate_bin, result.nan_direction, result.score))

print("determinism runs:")
for idx, item in enumerate(det_runs):
    print(idx, item)

first = det_runs[0]
for item in det_runs[1:]:
    assert item == first, f"non-deterministic result: first={first} other={item}"

print("Determinism check passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 — Physical Plan Sanity
# MAGIC
# MAGIC - D2.1 should show `Bucketizer` + groupBy + window operations.
# MAGIC - D2.2 SQL scoring should show Spark aggregations/windows without Python UDF.

# COMMAND ----------

print("D2.1 prefix plan:")
prefix_tiny.explain(True)

print("\nD2.2 SQL scoring plan (internal DF before final best-row selection):")
expanded_tiny = _expand_nan_directions(prefix_tiny)
control_tiny = _build_control_treatment_df(prefix_tiny)
scored_sql_tiny = _score_candidates_sql(
    expanded_df=expanded_tiny,
    control_treatment_df=control_tiny,
    min_leaf_size=1,
    min_uplift=0.0,
    prefer_nan_direction_on_tie="right",
)
scored_sql_tiny.explain(True)
display(scored_sql_tiny.orderBy(F.col("score").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 — Benchmark Dataset

# COMMAND ----------

bench_df = make_benchmark_df(
    n_rows=N_ROWS_BENCH,
    missing_rate=MISSING_RATE,
    seed=SEED,
).cache()
bench_rows = bench_df.count()
print(f"bench_df rows={bench_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 — D2.1 Timing

# COMMAND ----------

t0 = time.perf_counter()
prefix_bench = build_prefix_sums(
    bench_df,
    feature_col="feature",
    treatment_col="treatment",
    outcome_col="outcome",
    num_quantile_splits=N_QUANTILE_SPLITS,
).cache()
prefix_rows = prefix_bench.count()
t_d21 = time.perf_counter() - t0
assert_prefix_schema(prefix_bench)

print(f"D2.1 done: rows={prefix_rows:,} elapsed_seconds={t_d21:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 — D2.2 Timing (`sql` vs `mapInPandas`)

# COMMAND ----------

def timed_collectless(prefix_df, mode: str, repeats: int):
    records = []
    for i in range(repeats):
        start = time.perf_counter()
        best = score_candidates_collectless(
            prefix_df,
            evaluation_mode=mode,
        )
        elapsed = time.perf_counter() - start
        records.append((mode, i, elapsed, best.feature, best.candidate_bin, best.nan_direction, best.score))
    return records


rows = []
rows.extend(timed_collectless(prefix_bench, "sql", BENCH_REPEATS))
rows.extend(timed_collectless(prefix_bench, "mapInPandas", BENCH_REPEATS))

bench_result_df = spark.createDataFrame(
    rows,
    schema=(
        "mode string, run_idx int, elapsed_seconds double, "
        "feature string, candidate_bin int, nan_direction string, score double"
    ),
)

display(bench_result_df.orderBy("mode", "run_idx"))
display(
    bench_result_df.groupBy("mode")
    .agg(
        F.avg("elapsed_seconds").alias("avg_seconds"),
        F.min("elapsed_seconds").alias("min_seconds"),
        F.max("elapsed_seconds").alias("max_seconds"),
    )
    .orderBy("mode")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 — Optional Driver-Collect Baseline Timing (smaller sample)
# MAGIC
# MAGIC This is intentionally driver-heavy, so run on fewer rows.

# COMMAND ----------

driver_df = bench_df.limit(N_ROWS_DRIVER_BASELINE).cache()
_ = driver_df.count()

t0 = time.perf_counter()
best_driver_bench = best_split_driver_collect(
    driver_df,
    feature_col="feature",
    treatment_col="treatment",
    outcome_col="outcome",
    splits=None,
)
t_driver = time.perf_counter() - t0

print(f"driver baseline rows={N_ROWS_DRIVER_BASELINE:,}")
print("best_driver_bench:", best_driver_bench)
print(f"driver baseline elapsed_seconds={t_driver:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12 — Cleanup

# COMMAND ----------

bench_df.unpersist()
prefix_bench.unpersist()
driver_df.unpersist()

print("Done.")
