"""Public API contracts for collect-less split search (Direction 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


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
    # TODO(D2.1): build_prefix_sums(...)
    # 1) Resolve user-provided splits or derive via approxQuantile.
    # 2) Apply Bucketizer(handleInvalid="keep") for JVM-native binning.
    # 3) Build per-bin/per-treatment tallies with count + sum(outcome).
    # 4) Use Window.orderBy(bin_id) prefix sums to materialize left/right base stats.
    # 5) Return the exact D2.1 schema documented above.
    raise NotImplementedError("D2.1 build_prefix_sums is not implemented yet.")


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

