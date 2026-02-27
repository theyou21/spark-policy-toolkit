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
    if metric != "ddp_max":
        raise ValueError("Only metric='ddp_max' is currently supported.")
    if evaluation_mode not in {"sql", "mapInPandas"}:
        raise ValueError("evaluation_mode must be one of {'sql', 'mapInPandas'}.")
    if min_leaf_size < 1:
        raise ValueError("min_leaf_size must be >= 1.")
    if prefer_nan_direction_on_tie not in {"left", "right"}:
        raise ValueError("prefer_nan_direction_on_tie must be 'left' or 'right'.")

    _validate_prefix_sums_schema(prefix_sums_df)
    expanded_df = _expand_nan_directions(prefix_sums_df)
    control_treatment_df = _build_control_treatment_df(prefix_sums_df)

    if evaluation_mode == "sql":
        scored_candidates_df = _score_candidates_sql(
            expanded_df=expanded_df,
            control_treatment_df=control_treatment_df,
            min_leaf_size=min_leaf_size,
            min_uplift=min_uplift,
            prefer_nan_direction_on_tie=prefer_nan_direction_on_tie,
        )
    else:
        scored_candidates_df = _score_candidates_map_in_pandas(
            expanded_df=expanded_df,
            control_treatment_df=control_treatment_df,
            min_leaf_size=min_leaf_size,
            min_uplift=min_uplift,
            prefer_nan_direction_on_tie=prefer_nan_direction_on_tie,
        )

    return _select_best_split(
        scored_candidates_df=scored_candidates_df,
        evaluation_mode=evaluation_mode,
    )


def _validate_prefix_sums_schema(prefix_sums_df: "DataFrame") -> None:
    required_columns = {
        "feature",
        "candidate_bin",
        "bin_boundary",
        "treatment",
        "left_count_base",
        "left_sum_base",
        "right_count_base",
        "right_sum_base",
        "missing_count",
        "missing_sum",
    }
    missing = sorted(required_columns - set(prefix_sums_df.columns))
    if missing:
        raise ValueError(f"prefix_sums_df is missing required columns: {missing}")


def _expand_nan_directions(prefix_sums_df: "DataFrame") -> "DataFrame":
    from pyspark.sql import functions as F

    spark = prefix_sums_df.sparkSession
    direction_df = spark.createDataFrame(
        [("left",), ("right",)],
        schema="nan_direction string",
    )

    return (
        prefix_sums_df.crossJoin(direction_df)
        .withColumn(
            "left_count",
            F.when(
                F.col("nan_direction") == F.lit("left"),
                F.col("left_count_base") + F.col("missing_count"),
            ).otherwise(F.col("left_count_base")),
        )
        .withColumn(
            "left_sum",
            F.when(
                F.col("nan_direction") == F.lit("left"),
                F.col("left_sum_base") + F.col("missing_sum"),
            ).otherwise(F.col("left_sum_base")),
        )
        .withColumn(
            "right_count",
            F.when(
                F.col("nan_direction") == F.lit("right"),
                F.col("right_count_base") + F.col("missing_count"),
            ).otherwise(F.col("right_count_base")),
        )
        .withColumn(
            "right_sum",
            F.when(
                F.col("nan_direction") == F.lit("right"),
                F.col("right_sum_base") + F.col("missing_sum"),
            ).otherwise(F.col("right_sum_base")),
        )
        .select(
            F.col("feature").cast("string").alias("feature"),
            F.col("candidate_bin").cast("int").alias("candidate_bin"),
            F.col("bin_boundary").cast("double").alias("bin_boundary"),
            F.col("treatment").cast("string").alias("treatment"),
            F.col("nan_direction").cast("string").alias("nan_direction"),
            F.col("left_count").cast("bigint").alias("left_count"),
            F.col("left_sum").cast("double").alias("left_sum"),
            F.col("right_count").cast("bigint").alias("right_count"),
            F.col("right_sum").cast("double").alias("right_sum"),
        )
    )


def _build_control_treatment_df(prefix_sums_df: "DataFrame") -> "DataFrame":
    from pyspark.sql import Window, functions as F

    treatment_rank = (
        F.when(F.lower(F.col("treatment")) == F.lit("control"), F.lit(0))
        .when(F.col("treatment") == F.lit("0"), F.lit(1))
        .otherwise(F.lit(2))
    )
    rank_window = Window.orderBy(treatment_rank.asc(), F.col("treatment").asc())

    return (
        prefix_sums_df.select(F.col("treatment").cast("string").alias("treatment"))
        .where(F.col("treatment").isNotNull())
        .distinct()
        .withColumn("__row_num", F.row_number().over(rank_window))
        .where(F.col("__row_num") == F.lit(1))
        .select(F.col("treatment").alias("__control_treatment"))
    )


def _score_candidates_sql(
    *,
    expanded_df: "DataFrame",
    control_treatment_df: "DataFrame",
    min_leaf_size: int,
    min_uplift: float,
    prefer_nan_direction_on_tie: NanDirection,
) -> "DataFrame":
    from pyspark.sql import Window, functions as F

    group_cols = ["feature", "candidate_bin", "bin_boundary", "nan_direction"]
    group_window = Window.partitionBy(*group_cols)

    rates_df = (
        expanded_df.withColumn(
            "left_rate",
            F.when(F.col("left_count") > F.lit(0), F.col("left_sum") / F.col("left_count")),
        )
        .withColumn(
            "right_rate",
            F.when(
                F.col("right_count") > F.lit(0),
                F.col("right_sum") / F.col("right_count"),
            ),
        )
        .crossJoin(control_treatment_df)
        .withColumn(
            "__is_control",
            F.col("treatment") == F.col("__control_treatment"),
        )
        .withColumn(
            "__left_control_rate",
            F.max(F.when(F.col("__is_control"), F.col("left_rate"))).over(group_window),
        )
        .withColumn(
            "__right_control_rate",
            F.max(F.when(F.col("__is_control"), F.col("right_rate"))).over(group_window),
        )
        .withColumn(
            "__left_uplift",
            F.when(
                ~F.col("__is_control"),
                F.col("left_rate") - F.col("__left_control_rate"),
            ),
        )
        .withColumn(
            "__right_uplift",
            F.when(
                ~F.col("__is_control"),
                F.col("right_rate") - F.col("__right_control_rate"),
            ),
        )
    )

    score_expr = F.greatest(
        F.col("__right_max_uplift") - F.col("__left_min_uplift"),
        F.col("__left_max_uplift") - F.col("__right_min_uplift"),
    ).cast("double")

    tie_break_rank_expr = (
        F.when(
            F.col("nan_direction") == F.lit(prefer_nan_direction_on_tie),
            F.lit(0),
        )
        .otherwise(F.lit(1))
        .cast("int")
    )

    return (
        rates_df.groupBy(*group_cols)
        .agg(
            F.sum("left_count").cast("bigint").alias("left_count"),
            F.sum("left_sum").cast("double").alias("left_sum"),
            F.sum("right_count").cast("bigint").alias("right_count"),
            F.sum("right_sum").cast("double").alias("right_sum"),
            F.min("left_count").cast("bigint").alias("__min_left_group_count"),
            F.min("right_count").cast("bigint").alias("__min_right_group_count"),
            F.max(F.when(F.col("__is_control"), F.lit(1)).otherwise(F.lit(0)))
            .cast("int")
            .alias("__has_control"),
            F.sum(F.when(~F.col("__is_control"), F.lit(1)).otherwise(F.lit(0)))
            .cast("int")
            .alias("__num_non_control"),
            F.min("__left_uplift").cast("double").alias("__left_min_uplift"),
            F.max("__left_uplift").cast("double").alias("__left_max_uplift"),
            F.min("__right_uplift").cast("double").alias("__right_min_uplift"),
            F.max("__right_uplift").cast("double").alias("__right_max_uplift"),
        )
        .withColumn("score", score_expr)
        .withColumn(
            "is_valid",
            (F.col("__min_left_group_count") > F.lit(0))
            & (F.col("__min_right_group_count") > F.lit(0))
            & (F.col("left_count") >= F.lit(min_leaf_size))
            & (F.col("right_count") >= F.lit(min_leaf_size))
            & (F.col("__has_control") == F.lit(1))
            & (F.col("__num_non_control") > F.lit(0))
            & F.col("score").isNotNull()
            & (F.col("score") >= F.lit(min_uplift)),
        )
        .withColumn("tie_break_rank", tie_break_rank_expr)
        .select(
            F.col("feature").cast("string").alias("feature"),
            F.col("candidate_bin").cast("int").alias("candidate_bin"),
            F.col("bin_boundary").cast("double").alias("bin_boundary"),
            F.col("nan_direction").cast("string").alias("nan_direction"),
            F.col("left_count").cast("bigint").alias("left_count"),
            F.col("left_sum").cast("double").alias("left_sum"),
            F.col("right_count").cast("bigint").alias("right_count"),
            F.col("right_sum").cast("double").alias("right_sum"),
            F.col("score").cast("double").alias("score"),
            F.col("is_valid").cast("boolean").alias("is_valid"),
            F.col("tie_break_rank").cast("int").alias("tie_break_rank"),
        )
    )


def _score_candidates_map_in_pandas(
    *,
    expanded_df: "DataFrame",
    control_treatment_df: "DataFrame",
    min_leaf_size: int,
    min_uplift: float,
    prefer_nan_direction_on_tie: NanDirection,
) -> "DataFrame":
    with_control_df = expanded_df.crossJoin(control_treatment_df)
    group_cols = ["feature", "candidate_bin", "bin_boundary", "nan_direction"]
    tie_break_preferred = prefer_nan_direction_on_tie

    def _score_group(pdf):  # type: ignore[no-untyped-def]
        import pandas as pd

        feature = str(pdf["feature"].iloc[0])
        candidate_bin = int(pdf["candidate_bin"].iloc[0])
        boundary_value = pdf["bin_boundary"].iloc[0]
        bin_boundary = float(boundary_value) if pd.notna(boundary_value) else None
        nan_direction = str(pdf["nan_direction"].iloc[0])
        control_treatment = str(pdf["__control_treatment"].iloc[0])

        left_counts = pdf["left_count"].astype("float64").to_numpy()
        right_counts = pdf["right_count"].astype("float64").to_numpy()
        left_sums = pdf["left_sum"].astype("float64").to_numpy()
        right_sums = pdf["right_sum"].astype("float64").to_numpy()
        treatments = pdf["treatment"].astype(str).to_numpy()

        left_count_total = int(left_counts.sum())
        right_count_total = int(right_counts.sum())
        left_sum_total = float(left_sums.sum())
        right_sum_total = float(right_sums.sum())
        min_left_group_count = int(left_counts.min()) if left_counts.size else 0
        min_right_group_count = int(right_counts.min()) if right_counts.size else 0

        control_mask = treatments == control_treatment
        non_control_mask = ~control_mask
        has_control = bool(control_mask.any())
        num_non_control = int(non_control_mask.sum())

        score = float("nan")
        if (
            has_control
            and num_non_control > 0
            and min_left_group_count > 0
            and min_right_group_count > 0
        ):
            left_rates = left_sums / left_counts
            right_rates = right_sums / right_counts

            left_control_rate = float(left_rates[control_mask][0])
            right_control_rate = float(right_rates[control_mask][0])

            left_uplifts = left_rates[non_control_mask] - left_control_rate
            right_uplifts = right_rates[non_control_mask] - right_control_rate

            left_min_uplift = float(left_uplifts.min())
            left_max_uplift = float(left_uplifts.max())
            right_min_uplift = float(right_uplifts.min())
            right_max_uplift = float(right_uplifts.max())

            score = max(
                right_max_uplift - left_min_uplift,
                left_max_uplift - right_min_uplift,
            )

        is_valid = (
            min_left_group_count > 0
            and min_right_group_count > 0
            and left_count_total >= min_leaf_size
            and right_count_total >= min_leaf_size
            and has_control
            and num_non_control > 0
            and not math.isnan(score)
            and score >= min_uplift
        )
        tie_break_rank = 0 if nan_direction == tie_break_preferred else 1

        return pd.DataFrame(
            [
                {
                    "feature": feature,
                    "candidate_bin": candidate_bin,
                    "bin_boundary": bin_boundary,
                    "nan_direction": nan_direction,
                    "left_count": left_count_total,
                    "left_sum": left_sum_total,
                    "right_count": right_count_total,
                    "right_sum": right_sum_total,
                    "score": score,
                    "is_valid": is_valid,
                    "tie_break_rank": tie_break_rank,
                }
            ]
        )

    schema = (
        "feature string, candidate_bin int, bin_boundary double, nan_direction string, "
        "left_count bigint, left_sum double, right_count bigint, right_sum double, "
        "score double, is_valid boolean, tie_break_rank int"
    )
    return with_control_df.groupBy(*group_cols).applyInPandas(_score_group, schema=schema)


def _select_best_split(
    *,
    scored_candidates_df: "DataFrame",
    evaluation_mode: MetricMode,
) -> BestSplit:
    from pyspark.sql import functions as F

    best_candidate_df = (
        scored_candidates_df.where(F.col("is_valid"))
        .orderBy(
            F.col("score").desc(),
            F.col("bin_boundary").asc_nulls_last(),
            F.col("candidate_bin").asc(),
            F.col("tie_break_rank").asc(),
            F.col("feature").asc(),
        )
        .limit(1)
    )
    best_row = next(best_candidate_df.toLocalIterator(), None)
    if best_row is None:
        raise ValueError("No valid split candidates after applying D2.2 constraints.")

    diagnostics: dict[str, Any] = {
        "left_count": int(best_row["left_count"]),
        "left_sum": float(best_row["left_sum"]),
        "right_count": int(best_row["right_count"]),
        "right_sum": float(best_row["right_sum"]),
        "tie_break_rank": int(best_row["tie_break_rank"]),
        "evaluation_mode": evaluation_mode,
    }
    return BestSplit(
        feature=str(best_row["feature"]),
        bin_boundary=(
            float(best_row["bin_boundary"])
            if best_row["bin_boundary"] is not None
            else None
        ),
        candidate_bin=int(best_row["candidate_bin"]),
        nan_direction=str(best_row["nan_direction"]),  # type: ignore[arg-type]
        score=float(best_row["score"]),
        diagnostics=diagnostics,
    )


def _pick_control_treatment_local(treatments: set[str]) -> str | None:
    if not treatments:
        return None
    ranked = sorted(
        treatments,
        key=lambda value: (
            0
            if value.lower() == "control"
            else (1 if value == "0" else 2),
            value,
        ),
    )
    return ranked[0]


def _boundary_sort_value(boundary: float | None) -> float:
    if boundary is None:
        return float("inf")
    if math.isnan(boundary):
        return float("inf")
    return float(boundary)


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
    prefix_sums_df = build_prefix_sums(
        df=df,
        feature_col=feature_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        splits=splits,
    )
    rows = prefix_sums_df.collect()
    if not rows:
        raise ValueError("No split candidates available for driver-collect baseline.")

    candidate_rows: dict[tuple[str, int, float | None], dict[str, dict[str, float]]] = {}
    treatments: set[str] = set()
    for row in rows:
        feature = str(row["feature"])
        candidate_bin = int(row["candidate_bin"])
        bin_boundary = (
            float(row["bin_boundary"]) if row["bin_boundary"] is not None else None
        )
        treatment = str(row["treatment"])
        treatments.add(treatment)

        key = (feature, candidate_bin, bin_boundary)
        if key not in candidate_rows:
            candidate_rows[key] = {}
        candidate_rows[key][treatment] = {
            "left_count_base": float(row["left_count_base"]),
            "left_sum_base": float(row["left_sum_base"]),
            "right_count_base": float(row["right_count_base"]),
            "right_sum_base": float(row["right_sum_base"]),
            "missing_count": float(row["missing_count"]),
            "missing_sum": float(row["missing_sum"]),
        }

    control_treatment = _pick_control_treatment_local(treatments)
    if control_treatment is None:
        raise ValueError("No treatment groups available in baseline input.")

    best_key: tuple[float, float, int, int, str] | None = None
    best_split: BestSplit | None = None

    for feature, candidate_bin, bin_boundary in candidate_rows:
        per_treatment = candidate_rows[(feature, candidate_bin, bin_boundary)]
        if control_treatment not in per_treatment:
            continue

        for nan_direction in ("left", "right"):
            left_counts: dict[str, float] = {}
            right_counts: dict[str, float] = {}
            left_sums: dict[str, float] = {}
            right_sums: dict[str, float] = {}

            for treatment, stats in per_treatment.items():
                missing_count = stats["missing_count"]
                missing_sum = stats["missing_sum"]
                if nan_direction == "left":
                    left_counts[treatment] = stats["left_count_base"] + missing_count
                    left_sums[treatment] = stats["left_sum_base"] + missing_sum
                    right_counts[treatment] = stats["right_count_base"]
                    right_sums[treatment] = stats["right_sum_base"]
                else:
                    left_counts[treatment] = stats["left_count_base"]
                    left_sums[treatment] = stats["left_sum_base"]
                    right_counts[treatment] = stats["right_count_base"] + missing_count
                    right_sums[treatment] = stats["right_sum_base"] + missing_sum

            if not left_counts:
                continue
            if min(left_counts.values()) <= 0.0 or min(right_counts.values()) <= 0.0:
                continue

            total_left_count = sum(left_counts.values())
            total_right_count = sum(right_counts.values())
            if total_left_count < 1 or total_right_count < 1:
                continue

            non_control_treatments = sorted(
                [t for t in per_treatment if t != control_treatment]
            )
            if not non_control_treatments:
                continue

            left_control_rate = (
                left_sums[control_treatment] / left_counts[control_treatment]
            )
            right_control_rate = (
                right_sums[control_treatment] / right_counts[control_treatment]
            )
            left_uplifts = [
                (left_sums[t] / left_counts[t]) - left_control_rate
                for t in non_control_treatments
            ]
            right_uplifts = [
                (right_sums[t] / right_counts[t]) - right_control_rate
                for t in non_control_treatments
            ]

            score = max(
                max(right_uplifts) - min(left_uplifts),
                max(left_uplifts) - min(right_uplifts),
            )
            if score < 0.0:
                continue

            direction_rank = 0 if nan_direction == "right" else 1
            order_key = (
                -float(score),
                _boundary_sort_value(bin_boundary),
                int(candidate_bin),
                direction_rank,
                feature,
            )
            if best_key is not None and order_key >= best_key:
                continue

            best_key = order_key
            best_split = BestSplit(
                feature=feature,
                bin_boundary=bin_boundary,
                candidate_bin=int(candidate_bin),
                nan_direction=nan_direction,  # type: ignore[arg-type]
                score=float(score),
                diagnostics={
                    "left_count": int(total_left_count),
                    "left_sum": float(sum(left_sums.values())),
                    "right_count": int(total_right_count),
                    "right_sum": float(sum(right_sums.values())),
                    "evaluation_mode": "driver_collect",
                },
            )

    if best_split is None:
        raise ValueError("No valid split candidates in driver-collect baseline.")
    return best_split
