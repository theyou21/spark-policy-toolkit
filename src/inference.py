"""Spark inference entrypoints for array-native policy scoring."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np

from src.vectorized_tree import VectorizedArrayTree

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.types import StructType


def apply_policy_scorer_mapinpandas(
    df: "DataFrame",
    forest: Sequence[Mapping[str, np.ndarray]],
    feature_cols: Sequence[str],
    out_col: str,
) -> "DataFrame":
    """Score a forest with `mapInPandas` and return `df` plus `out_col`.

    This path broadcasts the forest once, then creates `VectorizedArrayTree`
    objects once per partition (lazy initialization inside the iterator).
    """
    _validate_apply_inputs(df=df, forest=forest, feature_cols=feature_cols, out_col=out_col)

    # Local import keeps module importable in non-Spark environments.
    spark = df.sparkSession
    forest_bc = spark.sparkContext.broadcast(list(forest))
    output_schema = _with_array_output_schema(df=df, out_col=out_col)

    def score_batches(pdf_iter: Iterator["pd.DataFrame"]) -> Iterator["pd.DataFrame"]:
        models: list[VectorizedArrayTree] | None = None
        n_treatments: int | None = None

        for pdf in pdf_iter:
            if models is None:
                models, n_treatments = _build_vectorized_forest(forest_bc.value)

            X = pdf.loc[:, list(feature_cols)].to_numpy(copy=False)
            mean_scores = _predict_forest_mean_numpy(X, models)
            if mean_scores.shape[1] != n_treatments:
                raise ValueError("Inconsistent forest prediction width across batches.")

            out_pdf = pdf.copy(deep=False)
            out_pdf[out_col] = mean_scores.tolist()
            yield out_pdf

    return df.mapInPandas(score_batches, schema=output_schema)


def apply_policy_scorer_mapinarrow(
    df: "DataFrame",
    forest: Sequence[Mapping[str, np.ndarray]],
    feature_cols: Sequence[str],
    out_col: str,
) -> "DataFrame":
    """Score a forest with `mapInArrow` over `pyarrow.RecordBatch` iterators."""
    _validate_apply_inputs(df=df, forest=forest, feature_cols=feature_cols, out_col=out_col)

    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError("mapInArrow path requires `pyarrow` to be installed.") from exc

    spark = df.sparkSession
    forest_bc = spark.sparkContext.broadcast(list(forest))
    output_schema = _with_array_output_schema(df=df, out_col=out_col)
    feature_positions = _resolve_feature_positions(df=df, feature_cols=feature_cols)

    def score_batches(
        batch_iter: Iterator["pa.RecordBatch"],
    ) -> Iterator["pa.RecordBatch"]:
        models: list[VectorizedArrayTree] | None = None
        n_treatments: int | None = None

        for batch in batch_iter:
            if models is None:
                models, n_treatments = _build_vectorized_forest(forest_bc.value)

            X = _record_batch_features_to_numpy(batch=batch, feature_positions=feature_positions)
            mean_scores = _predict_forest_mean_numpy(X, models)
            if mean_scores.shape[1] != n_treatments:
                raise ValueError("Inconsistent forest prediction width across batches.")

            output_array = _scores_to_arrow_list_array(
                scores=mean_scores,
                n_treatments=n_treatments,
                pa=pa,
            )
            out_columns = [batch.column(i) for i in range(batch.num_columns)] + [output_array]
            out_names = list(batch.schema.names) + [out_col]
            yield pa.RecordBatch.from_arrays(out_columns, names=out_names)

    return df.mapInArrow(score_batches, schema=output_schema)


def apply_policy_scorer_antipattern(
    df: "DataFrame",
    forest: Sequence[Mapping[str, np.ndarray]],
    feature_cols: Sequence[str],
    out_col: str,
    *,
    simulate_spark_from_json: bool = False,
) -> "DataFrame":
    """Intentionally slow baseline for comparison.

    Simulates anti-pattern behavior:
    - duplicates forest JSON into every row,
    - parses it with `json.loads` for every row,
    - scores rows with `pandas.DataFrame.apply(axis=1)`,
    - optional Spark `from_json` parse overhead.
    """
    _validate_apply_inputs(df=df, forest=forest, feature_cols=feature_cols, out_col=out_col)

    from pyspark.sql import functions as F

    forest_json = json.dumps(_to_jsonable(list(forest)))
    temp_json_col = "__forest_json"
    temp_parsed_col = "__forest_json_parsed"

    working_df = df.withColumn(temp_json_col, F.lit(forest_json))
    if simulate_spark_from_json:
        # Optional extra parse step to emulate an additional Spark-side JSON decode.
        working_df = working_df.withColumn(
            temp_parsed_col,
            F.from_json(F.col(temp_json_col), _forest_json_schema()),
        )

    output_schema = _with_array_output_schema(df=df, out_col=out_col)
    keep_cols = list(df.columns)

    def score_rows(pdf_iter: Iterator["pd.DataFrame"]) -> Iterator["pd.DataFrame"]:
        def score_one_row(row: "pd.Series") -> list[float]:
            per_row_forest = json.loads(row[temp_json_col])
            x_row = row.loc[list(feature_cols)].to_numpy(copy=False).reshape(1, -1)
            models, _ = _build_vectorized_forest(per_row_forest)
            return _predict_forest_mean_numpy(x_row, models)[0].tolist()

        for pdf in pdf_iter:
            out_pdf = pdf.copy(deep=False)
            out_pdf[out_col] = out_pdf.apply(score_one_row, axis=1)
            drop_cols = [temp_json_col, temp_parsed_col]
            out_pdf = out_pdf.drop(columns=drop_cols, errors="ignore")
            yield out_pdf.loc[:, keep_cols + [out_col]]

    return working_df.mapInPandas(score_rows, schema=output_schema)


def generate_synthetic_dataframe(
    spark: "SparkSession",
    n_rows: int,
    n_features: int,
    missing_rate: float = 0.0,
    *,
    seed: int = 7,
    partitions: int | None = None,
) -> "DataFrame":
    """Build a synthetic Spark DataFrame with random feature columns."""
    if n_rows < 0:
        raise ValueError("n_rows must be >= 0")
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0.0, 1.0].")

    from pyspark.sql import functions as F

    base = spark.range(n_rows)
    if partitions is not None and partitions > 0:
        base = base.repartition(partitions)

    cols = []
    for idx in range(n_features):
        feature_col = F.randn(seed + idx).cast("double")
        if missing_rate > 0.0:
            missing = F.rand(seed + 10_000 + idx) < F.lit(missing_rate)
            feature_col = F.when(missing, F.lit(None).cast("double")).otherwise(feature_col)
        cols.append(feature_col.alias(f"f{idx}"))

    return base.select(*cols)


def generate_synthetic_forest(
    depth: int,
    n_treatments: int,
    n_trees: int,
    *,
    n_features: int = 16,
    missing_routing: str = "random",
    seed: int = 13,
) -> list[dict[str, np.ndarray]]:
    """Build a synthetic forest in the array-native schema."""
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if n_treatments <= 0:
        raise ValueError("n_treatments must be > 0")
    if n_trees <= 0:
        raise ValueError("n_trees must be > 0")
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if missing_routing not in {"left", "right", "random"}:
        raise ValueError("missing_routing must be one of {'left', 'right', 'random'}.")

    forest: list[dict[str, np.ndarray]] = []
    for tree_idx in range(n_trees):
        forest.append(
            _generate_one_synthetic_tree(
                depth=depth,
                n_treatments=n_treatments,
                n_features=n_features,
                missing_routing=missing_routing,
                seed=seed + tree_idx * 97,
            )
        )
    return forest


def run_policy_scorer_benchmark(
    spark: "SparkSession",
    *,
    n_rows: int = 500_000,
    n_features: int = 32,
    depth: int = 7,
    n_treatments: int = 4,
    n_trees: int = 50,
    missing_rate: float = 0.1,
    partitions: int = 64,
    include_antipattern: bool = True,
) -> None:
    """Run synthetic throughput benchmark for all available scorer paths."""
    feature_cols = [f"f{i}" for i in range(n_features)]
    forest = generate_synthetic_forest(
        depth=depth,
        n_treatments=n_treatments,
        n_trees=n_trees,
        n_features=n_features,
        missing_routing="random",
    )
    df = generate_synthetic_dataframe(
        spark=spark,
        n_rows=n_rows,
        n_features=n_features,
        missing_rate=missing_rate,
        partitions=partitions,
    ).cache()
    _ = df.count()

    print("=== Arrow-relevant Spark confs ===")
    for key, value in get_arrow_relevant_spark_confs(spark).items():
        print(f"{key}={value}")

    def run_one(label: str, out_df: "DataFrame") -> None:
        start = time.perf_counter()
        rows = out_df.count()
        elapsed = time.perf_counter() - start
        rows_per_sec = 0.0 if elapsed == 0 else rows / elapsed
        print(
            f"{label}: rows={rows} elapsed_sec={elapsed:.4f} rows_per_sec={rows_per_sec:,.2f}"
        )

    run_one(
        "mapInPandas",
        apply_policy_scorer_mapinpandas(
            df=df,
            forest=forest,
            feature_cols=feature_cols,
            out_col="score_mapinpandas",
        ),
    )

    try:
        map_in_arrow_df = apply_policy_scorer_mapinarrow(
            df=df,
            forest=forest,
            feature_cols=feature_cols,
            out_col="score_mapinarrow",
        )
        run_one("mapInArrow", map_in_arrow_df)
    except ImportError as exc:
        print(f"mapInArrow: skipped ({exc})")

    if include_antipattern:
        run_one(
            "anti_pattern",
            apply_policy_scorer_antipattern(
                df=df,
                forest=forest,
                feature_cols=feature_cols,
                out_col="score_antipattern",
                simulate_spark_from_json=True,
            ),
        )


def get_arrow_relevant_spark_confs(spark: "SparkSession") -> dict[str, str]:
    """Return a small snapshot of Spark Arrow-relevant configuration values."""
    keys = [
        "spark.sql.execution.arrow.pyspark.enabled",
        "spark.sql.execution.arrow.maxRecordsPerBatch",
        "spark.sql.execution.arrow.pyspark.fallback.enabled",
        "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
        "spark.sql.execution.pythonUDF.arrow.enabled",
    ]
    confs: dict[str, str] = {}
    for key in keys:
        try:
            confs[key] = spark.conf.get(key)
        except Exception:
            confs[key] = "<unset>"
    return confs


def _build_vectorized_forest(
    forest: Sequence[Mapping[str, Any]],
) -> tuple[list[VectorizedArrayTree], int]:
    models = [VectorizedArrayTree(tree_arrays=tree) for tree in forest]
    if not models:
        raise ValueError("forest must contain at least one tree.")

    n_treatments = models[0].n_treatments
    for idx, model in enumerate(models[1:], start=1):
        if model.n_treatments != n_treatments:
            raise ValueError(
                f"All trees must share n_treatments; tree 0 has {n_treatments}, "
                f"tree {idx} has {model.n_treatments}."
            )
    return models, n_treatments


def _predict_forest_mean_numpy(
    X: np.ndarray,
    models: Sequence[VectorizedArrayTree],
) -> np.ndarray:
    if not models:
        raise ValueError("models must contain at least one tree.")

    acc: np.ndarray | None = None
    for model in models:
        pred = model.predict_numpy(X)
        if acc is None:
            acc = np.array(pred, dtype=np.float64, copy=True)
        else:
            acc += pred
    assert acc is not None
    acc /= float(len(models))
    return acc


def _validate_apply_inputs(
    df: "DataFrame",
    forest: Sequence[Mapping[str, Any]],
    feature_cols: Sequence[str],
    out_col: str,
) -> None:
    if not forest:
        raise ValueError("forest must contain at least one tree.")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature name.")
    if not out_col:
        raise ValueError("out_col must be non-empty.")
    if out_col in df.columns:
        raise ValueError(f"out_col '{out_col}' already exists in DataFrame.")

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"feature_cols not found in DataFrame: {missing}")


def _with_array_output_schema(df: "DataFrame", out_col: str) -> "StructType":
    from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType

    return StructType(
        list(df.schema.fields)
        + [StructField(out_col, ArrayType(DoubleType(), containsNull=False), nullable=False)]
    )


def _resolve_feature_positions(df: "DataFrame", feature_cols: Sequence[str]) -> list[int]:
    positions: dict[str, int] = {name: idx for idx, name in enumerate(df.schema.names)}
    return [positions[name] for name in feature_cols]


def _record_batch_features_to_numpy(
    batch: "pa.RecordBatch",
    feature_positions: Sequence[int],
) -> np.ndarray:
    if len(feature_positions) == 0:
        return np.empty((batch.num_rows, 0), dtype=np.float64)
    if batch.num_rows == 0:
        return np.empty((0, len(feature_positions)), dtype=np.float64)

    cols: list[np.ndarray] = []
    for pos in feature_positions:
        col_arr = np.asarray(batch.column(pos).to_numpy(zero_copy_only=False))
        cols.append(col_arr)
    return np.column_stack(cols)


def _scores_to_arrow_list_array(
    scores: np.ndarray,
    n_treatments: int,
    pa: Any,
) -> "pa.Array":
    flat_values = pa.array(
        np.asarray(scores, dtype=np.float64).reshape(-1),
        type=pa.float64(),
    )
    offsets = pa.array(
        np.arange(0, (scores.shape[0] + 1) * n_treatments, n_treatments, dtype=np.int32),
        type=pa.int32(),
    )
    return pa.ListArray.from_arrays(offsets, flat_values)


def _generate_one_synthetic_tree(
    depth: int,
    n_treatments: int,
    n_features: int,
    missing_routing: str,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_nodes = 2 ** (depth + 1) - 1
    first_leaf = 2**depth - 1

    node_type = np.zeros(n_nodes, dtype=np.int8)
    internal_nodes = np.arange(first_leaf, dtype=np.int64)
    node_type[internal_nodes] = np.where((internal_nodes % 2) == 0, 1, 2).astype(np.int8)

    feature_idx = np.zeros(n_nodes, dtype=np.int64)
    feature_idx[internal_nodes] = rng.integers(
        0, n_features, size=internal_nodes.shape[0], dtype=np.int64
    )

    thresholds = np.zeros(n_nodes, dtype=np.float64)
    thresholds[internal_nodes] = rng.uniform(-1.0, 1.0, size=internal_nodes.shape[0])

    cat_value = np.zeros(n_nodes, dtype=np.float64)
    cat_value[internal_nodes] = rng.integers(0, 4, size=internal_nodes.shape[0]).astype(
        np.float64
    )

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
    leaf_response_rates[first_leaf:] = rng.uniform(
        0.0, 1.0, size=(n_nodes - first_leaf, n_treatments)
    )

    return {
        "node_type": node_type,
        "feature_idx": feature_idx,
        "thresholds": thresholds,
        "cat_value": cat_value,
        "left_child": left_child,
        "right_child": right_child,
        "nan_goes_left": nan_goes_left,
        "leaf_response_rates": leaf_response_rates,
    }


def _forest_json_schema() -> "StructType":
    from pyspark.sql.types import ArrayType, BooleanType, DoubleType, LongType, StructField, StructType

    tree_schema = StructType(
        [
            StructField("node_type", ArrayType(LongType(), containsNull=False), nullable=False),
            StructField("feature_idx", ArrayType(LongType(), containsNull=False), nullable=False),
            StructField("thresholds", ArrayType(DoubleType(), containsNull=False), nullable=False),
            StructField("cat_value", ArrayType(DoubleType(), containsNull=False), nullable=False),
            StructField("left_child", ArrayType(LongType(), containsNull=False), nullable=False),
            StructField("right_child", ArrayType(LongType(), containsNull=False), nullable=False),
            StructField("nan_goes_left", ArrayType(BooleanType(), containsNull=False), nullable=False),
            StructField(
                "leaf_response_rates",
                ArrayType(ArrayType(DoubleType(), containsNull=False), containsNull=False),
                nullable=False,
            ),
        ]
    )
    return ArrayType(tree_schema, containsNull=False)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
