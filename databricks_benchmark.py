# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Policy Toolkit — Databricks Benchmark
# MAGIC
# MAGIC Self-contained notebook that benchmarks the three policy-scorer inference paths:
# MAGIC - **mapInPandas** — broadcast forest, vectorized NumPy scoring
# MAGIC - **mapInArrow** — broadcast forest, PyArrow RecordBatch scoring
# MAGIC - **anti-pattern** — per-row JSON parse + re-build (intentionally slow baseline)
# MAGIC
# MAGIC Upload this notebook to a Databricks workspace and attach to a cluster. No extra packages are required beyond what ships with Databricks Runtime.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Benchmark Parameters
# MAGIC
# MAGIC Adjust these before running.

# COMMAND ----------

# ---- Benchmark knobs ----
N_ROWS        = 500_000   # number of synthetic rows
N_FEATURES    = 32        # feature columns
DEPTH         = 7         # tree depth (nodes = 2^(depth+1)-1)
N_TREATMENTS  = 4         # treatment arms per leaf
N_TREES       = 50        # trees in the forest
MISSING_RATE  = 0.1       # fraction of NaN in feature columns
PARTITIONS    = 64        # Spark repartition count
INCLUDE_ANTIPATTERN = True  # set False to skip the slow baseline

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Spark Configuration

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — VectorizedArrayTree (inline source)

# COMMAND ----------

from __future__ import annotations

import json
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


class VectorizedArrayTree:
    """Vectorized inference for array-native uplift trees."""

    LEAF = 0
    CONTINUOUS = 1
    CATEGORICAL = 2

    REQUIRED_KEYS = (
        "node_type", "feature_idx", "thresholds", "cat_value",
        "left_child", "right_child", "nan_goes_left", "leaf_response_rates",
    )

    def __init__(self, tree_arrays: Mapping[str, np.ndarray]) -> None:
        if not isinstance(tree_arrays, Mapping):
            raise TypeError("tree_arrays must be a mapping of array fields.")
        missing = [k for k in self.REQUIRED_KEYS if k not in tree_arrays]
        if missing:
            raise ValueError(f"Missing required tree arrays: {missing}")

        self.node_type = np.asarray(tree_arrays["node_type"], dtype=np.int8)
        self.feature_idx = np.asarray(tree_arrays["feature_idx"], dtype=np.int64)
        self.thresholds = np.asarray(tree_arrays["thresholds"], dtype=np.float64)
        self.cat_value = np.asarray(tree_arrays["cat_value"])
        self.left_child = np.asarray(tree_arrays["left_child"], dtype=np.int64)
        self.right_child = np.asarray(tree_arrays["right_child"], dtype=np.int64)
        self.nan_goes_left = np.asarray(tree_arrays["nan_goes_left"], dtype=bool)
        self.leaf_response_rates = np.asarray(tree_arrays["leaf_response_rates"], dtype=np.float64)
        self._validate_schema()

    @property
    def n_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def n_treatments(self) -> int:
        return int(self.leaf_response_rates.shape[1])

    def predict_numpy(self, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D NumPy array.")
        n_rows, n_features = X_arr.shape
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError("feature_names length must match X.shape[1].")
        self._validate_feature_indices(n_features=n_features)
        if n_rows == 0:
            return np.empty((0, self.n_treatments), dtype=np.float64)

        current_nodes = np.zeros(n_rows, dtype=np.int64)
        for _ in range(self.n_nodes + 1):
            node_types = self.node_type[current_nodes]
            active_mask = node_types != self.LEAF
            if not np.any(active_mask):
                break
            active_rows = np.flatnonzero(active_mask)
            active_nodes = current_nodes[active_rows]
            active_types = node_types[active_rows]

            # Continuous splits
            continuous_mask = active_types == self.CONTINUOUS
            if np.any(continuous_mask):
                rows = active_rows[continuous_mask]
                nodes = active_nodes[continuous_mask]
                values = X_arr[rows, self.feature_idx[nodes]]
                missing = self._is_missing(values)
                go_left = np.zeros(rows.shape[0], dtype=bool)
                non_missing = ~missing
                if np.any(non_missing):
                    go_left[non_missing] = np.asarray(
                        values[non_missing] <= self.thresholds[nodes[non_missing]], dtype=bool
                    )
                go_left[missing] = self.nan_goes_left[nodes[missing]]
                current_nodes[rows] = np.where(go_left, self.left_child[nodes], self.right_child[nodes])

            # Categorical splits
            categorical_mask = active_types == self.CATEGORICAL
            if np.any(categorical_mask):
                rows = active_rows[categorical_mask]
                nodes = active_nodes[categorical_mask]
                values = X_arr[rows, self.feature_idx[nodes]]
                missing = self._is_missing(values)
                go_left = np.zeros(rows.shape[0], dtype=bool)
                non_missing = ~missing
                if np.any(non_missing):
                    go_left[non_missing] = np.asarray(
                        values[non_missing] == self.cat_value[nodes[non_missing]], dtype=bool
                    )
                go_left[missing] = self.nan_goes_left[nodes[missing]]
                current_nodes[rows] = np.where(go_left, self.left_child[nodes], self.right_child[nodes])
        else:
            raise ValueError("Tree traversal exceeded node budget.")

        if np.any(self.node_type[current_nodes] != self.LEAF):
            raise ValueError("Traversal ended on non-leaf nodes.")
        return self.leaf_response_rates[current_nodes].astype(np.float64, copy=False)

    @staticmethod
    def _is_missing(values: np.ndarray) -> np.ndarray:
        if values.dtype.kind in {"f", "c"}:
            return np.isnan(values)
        return np.asarray(pd.isna(values), dtype=bool)

    def _validate_schema(self) -> None:
        array_fields = (
            ("node_type", self.node_type), ("feature_idx", self.feature_idx),
            ("thresholds", self.thresholds), ("cat_value", self.cat_value),
            ("left_child", self.left_child), ("right_child", self.right_child),
            ("nan_goes_left", self.nan_goes_left),
        )
        if self.node_type.ndim != 1:
            raise ValueError("node_type must be a 1D array.")
        n_nodes = self.node_type.shape[0]
        if n_nodes == 0:
            raise ValueError("Tree must contain at least one node.")
        for name, arr in array_fields[1:]:
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array.")
            if arr.shape[0] != n_nodes:
                raise ValueError(f"{name} length must equal node_type length ({n_nodes}).")
        if self.leaf_response_rates.ndim != 2:
            raise ValueError("leaf_response_rates must be a 2D array.")
        if self.leaf_response_rates.shape[0] != n_nodes:
            raise ValueError("leaf_response_rates first dim must equal number of nodes.")
        if self.leaf_response_rates.shape[1] == 0:
            raise ValueError("leaf_response_rates must include at least one treatment column.")
        valid_types = {self.LEAF, self.CONTINUOUS, self.CATEGORICAL}
        invalid = sorted(set(np.unique(self.node_type).tolist()) - valid_types)
        if invalid:
            raise ValueError(f"node_type contains invalid values {invalid}.")
        split_mask = self.node_type != self.LEAF
        if np.any(split_mask):
            for name, child_arr in (("left_child", self.left_child), ("right_child", self.right_child)):
                bad = (child_arr[split_mask] < 0) | (child_arr[split_mask] >= n_nodes)
                if np.any(bad):
                    raise ValueError(f"{name} contains out-of-range child indices.")

    def _validate_feature_indices(self, n_features: int) -> None:
        split_mask = self.node_type != self.LEAF
        if not np.any(split_mask):
            return
        split_features = self.feature_idx[split_mask]
        if np.any(split_features < 0):
            raise ValueError("feature_idx must be non-negative for non-leaf nodes.")
        if np.any(split_features >= n_features):
            raise ValueError(f"feature_idx references feature >= {n_features}.")


print("VectorizedArrayTree defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Inference helpers (inline source)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, LongType, StructField, StructType


def _build_vectorized_forest(forest):
    models = [VectorizedArrayTree(tree_arrays=tree) for tree in forest]
    if not models:
        raise ValueError("forest must contain at least one tree.")
    n_treatments = models[0].n_treatments
    for idx, model in enumerate(models[1:], start=1):
        if model.n_treatments != n_treatments:
            raise ValueError(f"Tree {idx} has inconsistent n_treatments.")
    return models, n_treatments


def _predict_forest_mean_numpy(X, models):
    acc = None
    for model in models:
        pred = model.predict_numpy(X)
        acc = np.array(pred, dtype=np.float64, copy=True) if acc is None else acc + pred
    acc /= float(len(models))
    return acc


def _validate_apply_inputs(df, forest, feature_cols, out_col):
    if not forest:
        raise ValueError("forest must contain at least one tree.")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature name.")
    if not out_col:
        raise ValueError("out_col must be non-empty.")
    if out_col in df.columns:
        raise ValueError(f"out_col '{out_col}' already exists in DataFrame.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature_cols not found in DataFrame: {missing}")


def _with_array_output_schema(df, out_col):
    return StructType(
        list(df.schema.fields)
        + [StructField(out_col, ArrayType(DoubleType(), containsNull=False), nullable=False)]
    )


def _resolve_feature_positions(df, feature_cols):
    positions = {name: idx for idx, name in enumerate(df.schema.names)}
    return [positions[name] for name in feature_cols]


def _record_batch_features_to_numpy(batch, feature_positions):
    if len(feature_positions) == 0:
        return np.empty((batch.num_rows, 0), dtype=np.float64)
    if batch.num_rows == 0:
        return np.empty((0, len(feature_positions)), dtype=np.float64)
    cols = [np.asarray(batch.column(pos).to_numpy(zero_copy_only=False)) for pos in feature_positions]
    return np.column_stack(cols)


def _scores_to_arrow_list_array(scores, n_treatments, pa):
    flat_values = pa.array(np.asarray(scores, dtype=np.float64).reshape(-1), type=pa.float64())
    offsets = pa.array(
        np.arange(0, (scores.shape[0] + 1) * n_treatments, n_treatments, dtype=np.int32),
        type=pa.int32(),
    )
    return pa.ListArray.from_arrays(offsets, flat_values)


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _forest_json_schema():
    tree_schema = StructType([
        StructField("node_type", ArrayType(LongType(), containsNull=False), nullable=False),
        StructField("feature_idx", ArrayType(LongType(), containsNull=False), nullable=False),
        StructField("thresholds", ArrayType(DoubleType(), containsNull=False), nullable=False),
        StructField("cat_value", ArrayType(DoubleType(), containsNull=False), nullable=False),
        StructField("left_child", ArrayType(LongType(), containsNull=False), nullable=False),
        StructField("right_child", ArrayType(LongType(), containsNull=False), nullable=False),
        StructField("nan_goes_left", ArrayType(BooleanType(), containsNull=False), nullable=False),
        StructField("leaf_response_rates", ArrayType(ArrayType(DoubleType(), containsNull=False), containsNull=False), nullable=False),
    ])
    return ArrayType(tree_schema, containsNull=False)


print("Inference helpers defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Scorer entry-points

# COMMAND ----------

def apply_policy_scorer_mapinpandas(df, forest, feature_cols, out_col):
    _validate_apply_inputs(df=df, forest=forest, feature_cols=feature_cols, out_col=out_col)
    spark_session = df.sparkSession
    forest_bc = spark_session.sparkContext.broadcast(list(forest))
    output_schema = _with_array_output_schema(df=df, out_col=out_col)

    def score_batches(pdf_iter):
        models = None
        n_treatments = None
        for pdf in pdf_iter:
            if models is None:
                models, n_treatments = _build_vectorized_forest(forest_bc.value)
            X = pdf.loc[:, list(feature_cols)].to_numpy(copy=False)
            mean_scores = _predict_forest_mean_numpy(X, models)
            out_pdf = pdf.copy(deep=False)
            out_pdf[out_col] = mean_scores.tolist()
            yield out_pdf

    return df.mapInPandas(score_batches, schema=output_schema)


def apply_policy_scorer_mapinarrow(df, forest, feature_cols, out_col):
    import pyarrow as pa
    _validate_apply_inputs(df=df, forest=forest, feature_cols=feature_cols, out_col=out_col)
    spark_session = df.sparkSession
    forest_bc = spark_session.sparkContext.broadcast(list(forest))
    output_schema = _with_array_output_schema(df=df, out_col=out_col)
    feature_positions = _resolve_feature_positions(df=df, feature_cols=feature_cols)

    def score_batches(batch_iter):
        models = None
        n_treatments = None
        for batch in batch_iter:
            if models is None:
                models, n_treatments = _build_vectorized_forest(forest_bc.value)
            X = _record_batch_features_to_numpy(batch=batch, feature_positions=feature_positions)
            mean_scores = _predict_forest_mean_numpy(X, models)
            output_array = _scores_to_arrow_list_array(scores=mean_scores, n_treatments=n_treatments, pa=pa)
            out_columns = [batch.column(i) for i in range(batch.num_columns)] + [output_array]
            out_names = list(batch.schema.names) + [out_col]
            yield pa.RecordBatch.from_arrays(out_columns, names=out_names)

    return df.mapInArrow(score_batches, schema=output_schema)


def apply_policy_scorer_antipattern(df, forest, feature_cols, out_col, *, simulate_spark_from_json=False):
    _validate_apply_inputs(df=df, forest=forest, feature_cols=feature_cols, out_col=out_col)
    forest_json = json.dumps(_to_jsonable(list(forest)))
    temp_json_col = "__forest_json"
    temp_parsed_col = "__forest_json_parsed"
    working_df = df.withColumn(temp_json_col, F.lit(forest_json))
    if simulate_spark_from_json:
        working_df = working_df.withColumn(
            temp_parsed_col, F.from_json(F.col(temp_json_col), _forest_json_schema()),
        )
    output_schema = _with_array_output_schema(df=df, out_col=out_col)
    keep_cols = list(df.columns)

    def score_rows(pdf_iter):
        def score_one_row(row):
            per_row_forest = json.loads(row[temp_json_col])
            x_row = row.loc[list(feature_cols)].to_numpy(copy=False).reshape(1, -1)
            models, _ = _build_vectorized_forest(per_row_forest)
            return _predict_forest_mean_numpy(x_row, models)[0].tolist()
        for pdf in pdf_iter:
            out_pdf = pdf.copy(deep=False)
            out_pdf[out_col] = out_pdf.apply(score_one_row, axis=1)
            out_pdf = out_pdf.drop(columns=[temp_json_col, temp_parsed_col], errors="ignore")
            yield out_pdf.loc[:, keep_cols + [out_col]]

    return working_df.mapInPandas(score_rows, schema=output_schema)


print("Scorer entry-points defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 — Synthetic data & forest generators

# COMMAND ----------

def generate_synthetic_dataframe(spark_session, n_rows, n_features, missing_rate=0.0, seed=7, partitions=None):
    base = spark_session.range(n_rows)
    if partitions is not None and partitions > 0:
        base = base.repartition(partitions)
    cols = []
    for idx in range(n_features):
        feature_col = F.randn(seed + idx).cast("double")
        if missing_rate > 0.0:
            mask = F.rand(seed + 10_000 + idx) < F.lit(missing_rate)
            feature_col = F.when(mask, F.lit(None).cast("double")).otherwise(feature_col)
        cols.append(feature_col.alias(f"f{idx}"))
    return base.select(*cols)


def _generate_one_synthetic_tree(depth, n_treatments, n_features, missing_routing, seed):
    rng = np.random.default_rng(seed)
    n_nodes = 2 ** (depth + 1) - 1
    first_leaf = 2**depth - 1
    internal_nodes = np.arange(first_leaf, dtype=np.int64)

    node_type = np.zeros(n_nodes, dtype=np.int8)
    node_type[internal_nodes] = np.where((internal_nodes % 2) == 0, 1, 2).astype(np.int8)

    feature_idx = np.zeros(n_nodes, dtype=np.int64)
    feature_idx[internal_nodes] = rng.integers(0, n_features, size=internal_nodes.shape[0], dtype=np.int64)

    thresholds = np.zeros(n_nodes, dtype=np.float64)
    thresholds[internal_nodes] = rng.uniform(-1.0, 1.0, size=internal_nodes.shape[0])

    cat_value = np.zeros(n_nodes, dtype=np.float64)
    cat_value[internal_nodes] = rng.integers(0, 4, size=internal_nodes.shape[0]).astype(np.float64)

    left_child = np.arange(n_nodes, dtype=np.int64)
    right_child = np.arange(n_nodes, dtype=np.int64)
    left_child[internal_nodes] = internal_nodes * 2 + 1
    right_child[internal_nodes] = internal_nodes * 2 + 2

    if missing_routing == "left":
        nan_goes_left = np.ones(n_nodes, dtype=bool)
    elif missing_routing == "right":
        nan_goes_left = np.zeros(n_nodes, dtype=bool)
    else:
        nan_goes_left = rng.random(n_nodes) < 0.5

    leaf_response_rates = np.zeros((n_nodes, n_treatments), dtype=np.float64)
    leaf_response_rates[first_leaf:] = rng.uniform(0.0, 1.0, size=(n_nodes - first_leaf, n_treatments))

    return {
        "node_type": node_type, "feature_idx": feature_idx, "thresholds": thresholds,
        "cat_value": cat_value, "left_child": left_child, "right_child": right_child,
        "nan_goes_left": nan_goes_left, "leaf_response_rates": leaf_response_rates,
    }


def generate_synthetic_forest(depth, n_treatments, n_trees, n_features=16, missing_routing="random", seed=13):
    return [
        _generate_one_synthetic_tree(depth, n_treatments, n_features, missing_routing, seed + i * 97)
        for i in range(n_trees)
    ]


print("Generators defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 — Run Benchmark

# COMMAND ----------

# --- Print Spark Arrow confs ---
arrow_keys = [
    "spark.sql.execution.arrow.pyspark.enabled",
    "spark.sql.execution.arrow.maxRecordsPerBatch",
    "spark.sql.execution.arrow.pyspark.fallback.enabled",
    "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
    "spark.sql.execution.pythonUDF.arrow.enabled",
]
print("=== Arrow-relevant Spark confs ===")
for key in arrow_keys:
    try:
        print(f"{key}={spark.conf.get(key)}")
    except Exception:
        print(f"{key}=<unset>")

# --- Build synthetic data ---
feature_cols = [f"f{i}" for i in range(N_FEATURES)]
forest = generate_synthetic_forest(
    depth=DEPTH, n_treatments=N_TREATMENTS, n_trees=N_TREES,
    n_features=N_FEATURES, missing_routing="random",
)
df = generate_synthetic_dataframe(
    spark, n_rows=N_ROWS, n_features=N_FEATURES,
    missing_rate=MISSING_RATE, partitions=PARTITIONS,
).cache()
_ = df.count()  # materialize cache
print(f"\nSynthetic DataFrame cached: {N_ROWS:,} rows x {N_FEATURES} features")

# --- Benchmark runner ---
results = []

def run_one(label, out_df):
    start = time.perf_counter()
    rows = out_df.count()
    elapsed = time.perf_counter() - start
    rps = 0.0 if elapsed == 0 else rows / elapsed
    print(f"{label}: rows={rows:,}  elapsed={elapsed:.4f}s  rows/sec={rps:,.0f}")
    results.append({"label": label, "rows": rows, "elapsed_sec": round(elapsed, 4), "rows_per_sec": round(rps, 2)})

# --- mapInPandas ---
run_one("mapInPandas", apply_policy_scorer_mapinpandas(
    df=df, forest=forest, feature_cols=feature_cols, out_col="score_mapinpandas",
))

# --- mapInArrow ---
try:
    run_one("mapInArrow", apply_policy_scorer_mapinarrow(
        df=df, forest=forest, feature_cols=feature_cols, out_col="score_mapinarrow",
    ))
except Exception as exc:
    print(f"mapInArrow: skipped ({exc})")

# --- anti-pattern ---
if INCLUDE_ANTIPATTERN:
    run_one("anti_pattern", apply_policy_scorer_antipattern(
        df=df, forest=forest, feature_cols=feature_cols,
        out_col="score_antipattern", simulate_spark_from_json=True,
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 — Results Summary

# COMMAND ----------

results_df = pd.DataFrame(results)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 — Correctness Checks (single cell)
# MAGIC
# MAGIC This cell validates correctness beyond throughput:
# MAGIC - deterministic tree traversal outputs
# MAGIC - parity across mapInPandas / mapInArrow / anti-pattern
# MAGIC - output schema and vector length contract

# COMMAND ----------

# 1) Deterministic VectorizedArrayTree correctness check
tree_test = {
    "node_type": np.array([1, 0, 2, 0, 0], dtype=np.int8),
    "feature_idx": np.array([0, 0, 1, 0, 0], dtype=np.int64),
    "thresholds": np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "cat_value": np.array([0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float64),
    "left_child": np.array([1, 0, 3, 0, 0], dtype=np.int64),
    "right_child": np.array([2, 0, 4, 0, 0], dtype=np.int64),
    "nan_goes_left": np.array([True, True, False, True, True], dtype=bool),
    "leaf_response_rates": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    ),
}

model_test = VectorizedArrayTree(tree_test)
X_test = np.array(
    [
        [0.1, 1.0],      # root left
        [0.9, 3.0],      # root right, categorical left
        [0.9, 2.0],      # root right, categorical right
        [np.nan, 3.0],   # root NaN -> left
        [0.9, np.nan],   # categorical NaN -> right
    ],
    dtype=np.float64,
)
expected_test = np.array(
    [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.1, 0.2, 0.3],
        [0.7, 0.8, 0.9],
    ],
    dtype=np.float64,
)
pred_test = model_test.predict_numpy(X_test)
assert pred_test.dtype == np.float64
assert pred_test.shape == expected_test.shape
assert np.allclose(pred_test, expected_test), "Deterministic traversal check failed."

print("Deterministic VectorizedArrayTree check passed.")


# 2) Small Spark parity check across all three scorer paths
N_ROWS_TEST = 2_000
N_FEATURES_TEST = 8
DEPTH_TEST = 4
N_TREATMENTS_TEST = 4
N_TREES_TEST = 5
MISSING_RATE_TEST = 0.2

feature_cols_test = [f"f{i}" for i in range(N_FEATURES_TEST)]
forest_test = generate_synthetic_forest(
    depth=DEPTH_TEST,
    n_treatments=N_TREATMENTS_TEST,
    n_trees=N_TREES_TEST,
    n_features=N_FEATURES_TEST,
    missing_routing="random",
    seed=123,
)

df_test = (
    generate_synthetic_dataframe(
        spark,
        n_rows=N_ROWS_TEST,
        n_features=N_FEATURES_TEST,
        missing_rate=MISSING_RATE_TEST,
        partitions=1,
    )
    .withColumn("__id", F.monotonically_increasing_id())
    .cache()
)
_ = df_test.count()

out_pd = apply_policy_scorer_mapinpandas(
    df=df_test,
    forest=forest_test,
    feature_cols=feature_cols_test,
    out_col="score_pd",
)
out_ar = apply_policy_scorer_mapinarrow(
    df=df_test,
    forest=forest_test,
    feature_cols=feature_cols_test,
    out_col="score_ar",
)
out_ap = apply_policy_scorer_antipattern(
    df=df_test,
    forest=forest_test,
    feature_cols=feature_cols_test,
    out_col="score_ap",
    simulate_spark_from_json=True,
)


# 3) Output contract checks
for scored_df, score_col in [(out_pd, "score_pd"), (out_ar, "score_ar"), (out_ap, "score_ap")]:
    dtype = scored_df.schema[score_col].dataType
    assert isinstance(dtype, ArrayType), f"{score_col} is not ArrayType."
    assert isinstance(dtype.elementType, DoubleType), f"{score_col} element type is not DoubleType."
    assert scored_df.where(F.col(score_col).isNull()).count() == 0, f"{score_col} has null arrays."
    assert (
        scored_df.where(F.size(F.col(score_col)) != N_TREATMENTS_TEST).count() == 0
    ), f"{score_col} has invalid vector length."

print("Output contract checks passed.")


# 4) Numeric parity checks
joined = (
    out_pd.select("__id", "score_pd")
    .join(out_ar.select("__id", "score_ar"), "__id", "inner")
    .join(out_ap.select("__id", "score_ap"), "__id", "inner")
)

rows = joined.select("score_pd", "score_ar", "score_ap").collect()
assert len(rows) == N_ROWS_TEST, "Joined parity set does not match expected row count."

for row in rows:
    pd_arr = np.asarray(row["score_pd"], dtype=np.float64)
    ar_arr = np.asarray(row["score_ar"], dtype=np.float64)
    ap_arr = np.asarray(row["score_ap"], dtype=np.float64)
    assert np.allclose(pd_arr, ar_arr, atol=1e-12), "mapInPandas != mapInArrow"
    assert np.allclose(pd_arr, ap_arr, atol=1e-12), "mapInPandas != anti-pattern"

print("Cross-path parity checks passed.")


# 5) Validation-path negative check
try:
    apply_policy_scorer_mapinpandas(
        df=df_test,
        forest=forest_test,
        feature_cols=["missing_feature"],
        out_col="bad_out",
    )
    raise AssertionError("Expected ValueError for missing feature column.")
except ValueError:
    pass

print("Validation-path check passed.")
print("All correctness checks passed.")
