import pytest

from src.split_search import (
    best_split_driver_collect,
    build_prefix_sums,
    score_candidates_collectless,
)


@pytest.fixture(scope="module")
def spark():
    pyspark = pytest.importorskip("pyspark")

    try:
        spark_session = (
            pyspark.sql.SparkSession.builder.master("local[2]")
            .appName("split-search-correctness-tests")
            .getOrCreate()
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Spark runtime unavailable in this environment: {exc}")

    yield spark_session
    spark_session.stop()


def test_collectless_sql_matches_driver_collect_on_tiny_data(spark) -> None:
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

    prefix_sums_df = build_prefix_sums(
        tiny_df,
        feature_col="feature",
        treatment_col="treatment",
        outcome_col="outcome",
        splits=[-0.2, 0.2],
    )
    collectless = score_candidates_collectless(
        prefix_sums_df,
        evaluation_mode="sql",
    )
    baseline = best_split_driver_collect(
        tiny_df,
        feature_col="feature",
        treatment_col="treatment",
        outcome_col="outcome",
        splits=[-0.2, 0.2],
    )

    assert collectless.feature == baseline.feature
    assert collectless.candidate_bin == baseline.candidate_bin
    assert collectless.nan_direction == baseline.nan_direction
    assert collectless.bin_boundary == pytest.approx(baseline.bin_boundary)
    assert collectless.score == pytest.approx(baseline.score)
