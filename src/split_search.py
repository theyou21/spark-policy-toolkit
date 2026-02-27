"""Public API contracts for collect-less split search (Direction 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


NanDirection = Literal["left", "right"]
MetricMode = Literal["sql", "mapInPandas"]


@dataclass(frozen=True, slots=True)
class BestSplit:
    """Final split choice returned by collect-less search.

    Fields:
    - feature: Feature name evaluated for the split.
    - bin_boundary: Continuous threshold represented by the candidate bin boundary.
      This can be None for non-threshold variants.
    - candidate_bin: Integer bucket index used for the candidate threshold.
    - nan_direction: Which side receives missing-bin rows for the selected split.
    - score: Final metric value (for example DDP max-envelope).
    - diagnostics: Optional auxiliary values (counts, validity flags, tie-break keys).
    """

    feature: str
    bin_boundary: float | None
    candidate_bin: int
    nan_direction: NanDirection
    score: float
    diagnostics: Mapping[str, Any] | None = None


def build_prefix_sums(
    df: "DataFrame",
    *,
    feature_col: str,
    treatment_col: str,
    outcome_col: str,
    splits: Sequence[float] | None = None,
    num_quantile_splits: int | None = None,
    quantile_relative_error: float = 0.01,
) -> "DataFrame":
    """Build D2.1 prefix-sum candidate stats without collecting to the driver.

    Required behavior:
    - Use JVM-side `Bucketizer(handleInvalid="keep")`.
    - Treat the `handleInvalid="keep"` bucket as the explicit missing bin.
    - Compute tallies via `groupBy(bin, treatment)` and window prefix sums.

    Exact intermediate schema before window prefix sums (`bin_tallies_df`):
    - `feature`: STRING
    - `bin_id`: INT
    - `bin_boundary`: DOUBLE
    - `is_missing_bin`: BOOLEAN
    - `treatment`: STRING
    - `opps`: BIGINT
    - `accepts`: DOUBLE

    Exact D2.1 output schema (`prefix_sums_df`) consumed by D2.2:
    - `feature`: STRING
    - `candidate_bin`: INT
    - `bin_boundary`: DOUBLE
    - `treatment`: STRING
    - `left_count_base`: BIGINT
    - `left_sum_base`: DOUBLE
    - `right_count_base`: BIGINT
    - `right_sum_base`: DOUBLE
    - `missing_count`: BIGINT
    - `missing_sum`: DOUBLE

    D2.2 consumes this schema as-is and applies NaN-left / NaN-right routing by:
    - NaN-left: left += missing, right unchanged.
    - NaN-right: right += missing, left unchanged.
    """
    from pyspark.ml.feature import Bucketizer
    from pyspark.sql import Window, functions as F

    missing_cols = [
        col_name
        for col_name in (feature_col, treatment_col, outcome_col)
        if col_name not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required input columns: {missing_cols}")

    bucketizer_splits = _resolve_bucketizer_splits(
        df=df,
        feature_col=feature_col,
        splits=splits,
        num_quantile_splits=num_quantile_splits,
        quantile_relative_error=quantile_relative_error,
    )
    regular_bin_count = len(bucketizer_splits) - 1
    missing_bin_id = regular_bin_count

    prepared_df = (
        df.select(
            F.col(feature_col).cast("double").alias("__feature_value"),
            F.col(treatment_col).cast("string").alias("__treatment"),
            F.col(outcome_col).cast("double").alias("__outcome"),
        ).withColumn(
            "__feature_for_bucketizer",
            F.when(
                F.col("__feature_value").isNull() | F.isnan(F.col("__feature_value")),
                F.lit(float("nan")),
            ).otherwise(F.col("__feature_value")),
        )
    )

    bucketizer = Bucketizer(
        splits=bucketizer_splits,
        inputCol="__feature_for_bucketizer",
        outputCol="__bin_raw",
        handleInvalid="keep",
    )
    bucketed_df = bucketizer.transform(prepared_df).select(
        F.col("__treatment").alias("treatment"),
        F.col("__outcome").alias("outcome"),
        F.col("__bin_raw").cast("int").alias("bin_id"),
    )

    raw_tallies_df = bucketed_df.groupBy("bin_id", "treatment").agg(
        F.count(F.lit(1)).cast("bigint").alias("opps"),
        F.sum("outcome").cast("double").alias("accepts"),
    )

    spark = df.sparkSession
    bin_metadata_rows = [
        (int(bin_id), float(bucketizer_splits[bin_id + 1]), False)
        for bin_id in range(regular_bin_count)
    ]
    bin_metadata_rows.append((int(missing_bin_id), None, True))
    bin_metadata_df = spark.createDataFrame(
        bin_metadata_rows,
        schema="bin_id int, bin_boundary double, is_missing_bin boolean",
    )

    treatment_levels_df = bucketed_df.select("treatment").distinct()
    bin_tallies_df = (
        bin_metadata_df.crossJoin(treatment_levels_df)
        .join(raw_tallies_df, on=["bin_id", "treatment"], how="left")
        .fillna({"opps": 0, "accepts": 0.0})
        .select(
            F.lit(feature_col).alias("feature"),
            F.col("bin_id").cast("int").alias("bin_id"),
            F.col("bin_boundary").cast("double").alias("bin_boundary"),
            F.col("is_missing_bin").cast("boolean").alias("is_missing_bin"),
            F.col("treatment").cast("string").alias("treatment"),
            F.col("opps").cast("bigint").alias("opps"),
            F.col("accepts").cast("double").alias("accepts"),
        )
    )

    regular_tallies_df = bin_tallies_df.filter(~F.col("is_missing_bin"))
    missing_tallies_df = bin_tallies_df.filter(F.col("is_missing_bin")).select(
        F.col("treatment").cast("string").alias("treatment"),
        F.col("opps").cast("bigint").alias("missing_count"),
        F.col("accepts").cast("double").alias("missing_sum"),
    )

    prefix_window = (
        Window.partitionBy("treatment")
        .orderBy("bin_id")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    total_window = Window.partitionBy("treatment")
    max_candidate_bin = regular_bin_count - 2

    prefix_sums_df = (
        regular_tallies_df.withColumn(
            "left_count_base", F.sum("opps").over(prefix_window).cast("bigint")
        )
        .withColumn("left_sum_base", F.sum("accepts").over(prefix_window).cast("double"))
        .withColumn(
            "__total_count_regular", F.sum("opps").over(total_window).cast("bigint")
        )
        .withColumn(
            "__total_sum_regular", F.sum("accepts").over(total_window).cast("double")
        )
        .withColumn(
            "right_count_base",
            (F.col("__total_count_regular") - F.col("left_count_base")).cast("bigint"),
        )
        .withColumn(
            "right_sum_base",
            (F.col("__total_sum_regular") - F.col("left_sum_base")).cast("double"),
        )
        .filter(F.col("bin_id") <= F.lit(max_candidate_bin))
        .join(missing_tallies_df, on="treatment", how="left")
        .fillna({"missing_count": 0, "missing_sum": 0.0})
        .select(
            F.col("feature").cast("string").alias("feature"),
            F.col("bin_id").cast("int").alias("candidate_bin"),
            F.col("bin_boundary").cast("double").alias("bin_boundary"),
            F.col("treatment").cast("string").alias("treatment"),
            F.col("left_count_base").cast("bigint").alias("left_count_base"),
            F.col("left_sum_base").cast("double").alias("left_sum_base"),
            F.col("right_count_base").cast("bigint").alias("right_count_base"),
            F.col("right_sum_base").cast("double").alias("right_sum_base"),
            F.col("missing_count").cast("bigint").alias("missing_count"),
            F.col("missing_sum").cast("double").alias("missing_sum"),
        )
    )

    return prefix_sums_df


def _resolve_bucketizer_splits(
    df: "DataFrame",
    *,
    feature_col: str,
    splits: Sequence[float] | None,
    num_quantile_splits: int | None,
    quantile_relative_error: float,
) -> list[float]:
    """Resolve Bucketizer split points as `[-inf, ..., +inf]`."""
    from pyspark.sql import functions as F

    if not 0.0 <= quantile_relative_error <= 1.0:
        raise ValueError("quantile_relative_error must be in [0.0, 1.0].")

    if splits is not None:
        raw_thresholds: Sequence[float] = splits
    else:
        effective_bins = num_quantile_splits if num_quantile_splits is not None else 16
        if effective_bins < 1:
            raise ValueError("num_quantile_splits must be >= 1 when provided.")
        quantile_probs = [idx / effective_bins for idx in range(1, effective_bins)]
        non_missing_df = (
            df.select(F.col(feature_col).cast("double").alias("__feature_value"))
            .where(
                F.col("__feature_value").isNotNull()
                & ~F.isnan(F.col("__feature_value"))
            )
        )
        raw_thresholds = non_missing_df.stat.approxQuantile(
            "__feature_value",
            quantile_probs,
            quantile_relative_error,
        )

    cleaned_thresholds: list[float] = []
    for raw_value in raw_thresholds:
        if raw_value is None:
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Split values must be numeric, found {raw_value!r}."
            ) from exc
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        cleaned_thresholds.append(numeric)

    unique_thresholds = sorted(set(cleaned_thresholds))
    return [float("-inf"), *unique_thresholds, float("inf")]


def demo_build_prefix_sums_local(spark: "SparkSession") -> "DataFrame":
    """Create a tiny synthetic frame and validate D2.1 output schema."""
    tiny_df = spark.createDataFrame(
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

    result_df = build_prefix_sums(
        tiny_df,
        feature_col="feature",
        treatment_col="treatment",
        outcome_col="outcome",
        splits=[-0.2, 0.2],
    )

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
    actual_schema = [(field.name, field.dataType.simpleString()) for field in result_df.schema]
    if actual_schema != expected_schema:
        raise AssertionError(
            f"Unexpected D2.1 schema. expected={expected_schema} actual={actual_schema}"
        )

    return result_df


def score_candidates_collectless(
    prefix_sums_df: "DataFrame",
    *,
    min_leaf_size: int = 1,
    min_uplift: float = 0.0,
    metric: Literal["ddp_max"] = "ddp_max",
    evaluation_mode: MetricMode = "sql",
    prefer_nan_direction_on_tie: NanDirection = "right",
) -> BestSplit:
    """Score split candidates from D2.1 and return a deterministic best split.

    Step D2.2 input schema (must match D2.1 output exactly):
    - `feature`: STRING
    - `candidate_bin`: INT
    - `bin_boundary`: DOUBLE
    - `treatment`: STRING
    - `left_count_base`: BIGINT
    - `left_sum_base`: DOUBLE
    - `right_count_base`: BIGINT
    - `right_sum_base`: DOUBLE
    - `missing_count`: BIGINT
    - `missing_sum`: DOUBLE

    D2.2 scoring requirements:
    - Evaluate both `nan_direction='left'` and `nan_direction='right'` for every
      candidate bin.
    - Support multi-treatment DDP max-envelope scoring from UPLIFT section 1.
    - Avoid driver `collect()` for candidate scoring; evaluation runs with:
      - `evaluation_mode="sql"`: Spark SQL expressions only.
      - `evaluation_mode="mapInPandas"`: executor-local batch metric evaluation.

    Exact D2.2 internal scored schema (`scored_candidates_df`):
    - `feature`: STRING
    - `candidate_bin`: INT
    - `bin_boundary`: DOUBLE
    - `nan_direction`: STRING  # "left" | "right"
    - `left_count`: BIGINT
    - `left_sum`: DOUBLE
    - `right_count`: BIGINT
    - `right_sum`: DOUBLE
    - `score`: DOUBLE
    - `is_valid`: BOOLEAN
    - `tie_break_rank`: INT

    Deterministic tie-break contract (descending score):
    1) Higher `score`
    2) Lower `bin_boundary`
    3) Lower `candidate_bin`
    4) Preferred NaN direction from `prefer_nan_direction_on_tie`
    """
    # TODO(D2.2): score_candidates_collectless(...)
    # 1) Expand each candidate into two directions (left/right) without collecting.
    # 2) Materialize direction-specific left/right tallies from D2.1 base columns.
    # 3) Compute branch propensities and DDP max-envelope scores.
    # 4) Apply constraints (leaf size, min uplift, validity checks).
    # 5) Select best candidate with explicit deterministic tie-breaking.
    raise NotImplementedError("D2.2 score_candidates_collectless is not implemented yet.")


def best_split_driver_collect(
    df: "DataFrame",
    *,
    feature_col: str,
    treatment_col: str,
    outcome_col: str,
    splits: Sequence[float] | None = None,
) -> BestSplit:
    """Driver-collect baseline for parity and benchmarking.

    Baseline intent:
    - Build the same bin/treatment tallies as D2.1.
    - Collect candidate aggregates to the driver.
    - Score candidates in local Python and return `BestSplit`.

    This function exists only as a reference baseline; collect-less production
    scoring should use `build_prefix_sums(...)` + `score_candidates_collectless(...)`.
    """
    # TODO(baseline): best_split_driver_collect(...)
    # 1) Reuse D2.1 tallies/prefix logic where possible.
    # 2) Collect candidate rows to driver for local scoring.
    # 3) Reuse the same NaN-direction and tie-break semantics as D2.2.
    raise NotImplementedError("Baseline driver-collect split search is not implemented yet.")
