# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Policy Toolkit — Paper Experiments (Directions 1 + 2)
# MAGIC
# MAGIC This notebook runs all six experiments for a single paper covering:
# MAGIC - Direction 1: Arrow-native inference
# MAGIC - Direction 2: collect-less split search
# MAGIC
# MAGIC Output artifacts:
# MAGIC - `<OUT_DIR>/results.csv`
# MAGIC - `<OUT_DIR>/run_metadata.json`
# MAGIC - `<OUT_DIR>/plots/*.png` (optional)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 0 — Setup + Repro Harness Logging (RUN THIS CELL FIRST)

# COMMAND ----------

import datetime as dt
import gc
import json
import math
import os
import platform
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List

import pandas as pd
import pyspark
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.inference import (
    apply_policy_scorer_antipattern,
    apply_policy_scorer_mapinarrow,
    apply_policy_scorer_mapinpandas,
    generate_synthetic_dataframe,
    generate_synthetic_forest,
)
from src.split_search import (
    best_split_driver_collect,
    build_prefix_sums,
    score_candidates_collectless,
)


def now_str() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def dbfs_to_local_path(path: str) -> str:
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path[len("dbfs:/") :]
    return path


def write_json(path: str, obj: Dict[str, Any]) -> None:
    local_path = dbfs_to_local_path(path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def safe_driver_rss_gb() -> float | None:
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / float(1024**3)
    except Exception:
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux returns KB; macOS returns bytes.
            if sys.platform.lower().startswith("darwin"):
                return float(ru) / float(1024**3)
            return float(ru) / float(1024**2)
        except Exception:
            return None


def force_materialize_array_col(df: DataFrame, array_col: str) -> float:
    expr = (
        f"sum(aggregate(transform(`{array_col}`, x -> coalesce(x, 0D)), "
        "0D, (acc, x) -> acc + x)) as __checksum"
    )
    row = df.selectExpr(expr).first()
    if row is None or row["__checksum"] is None:
        return 0.0
    return float(row["__checksum"])


def force_materialize_scalar_col(df: DataFrame, scalar_col: str) -> float:
    row = df.select(
        F.sum(F.coalesce(F.col(scalar_col).cast("double"), F.lit(0.0))).alias(
            "__checksum"
        )
    ).first()
    if row is None or row["__checksum"] is None:
        return 0.0
    return float(row["__checksum"])


def _safe_conf(key: str, default: str = "<unset>") -> str:
    try:
        return spark.conf.get(key)
    except Exception:
        return default


def _discover_cluster_context() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["spark_version"] = spark.version
    out["pyspark_version"] = pyspark.__version__
    out["python_version"] = platform.python_version()
    out["arrow_enabled"] = _safe_conf("spark.sql.execution.arrow.pyspark.enabled")
    out["arrow_max_records_per_batch"] = _safe_conf(
        "spark.sql.execution.arrow.maxRecordsPerBatch"
    )
    out["cluster_usage_tags.clusterName"] = _safe_conf(
        "spark.databricks.clusterUsageTags.clusterName"
    )
    out["cluster_usage_tags.sparkVersion"] = _safe_conf(
        "spark.databricks.clusterUsageTags.sparkVersion"
    )
    out["cluster_usage_tags.nodeTypeId"] = _safe_conf(
        "spark.databricks.clusterUsageTags.nodeTypeId"
    )
    out["cluster_usage_tags.driverNodeTypeId"] = _safe_conf(
        "spark.databricks.clusterUsageTags.driverNodeTypeId"
    )
    out["cluster_usage_tags.clusterWorkers"] = _safe_conf(
        "spark.databricks.clusterUsageTags.clusterWorkers"
    )

    try:
        mem_status = sc._jsc.sc().getExecutorMemoryStatus()
        out["executor_memory_status_entries"] = int(mem_status.size())
        out["executor_count_estimate"] = max(0, int(mem_status.size()) - 1)
    except (Exception, AttributeError):
        # Spark Connect (RemoteContext) doesn't expose _jsc
        out["executor_memory_status_entries"] = None
        out["executor_count_estimate"] = None

    try:
        ctx_json = json.loads(
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
        )
        tags = ctx_json.get("tags", {}) or {}
        out["context_tag.clusterId"] = tags.get("clusterId")
        out["context_tag.clusterName"] = tags.get("clusterName")
        out["context_tag.sparkVersion"] = tags.get("sparkVersion")
        out["context_tag.browserHostName"] = tags.get("browserHostName")
        out["context_tag.notebookPath"] = tags.get("notebookPath")
        out["context_tag.jobId"] = tags.get("jobId")
        out["context_tag.jobRunId"] = tags.get("jobRunId")
    except Exception as exc:
        out["dbutils_context_error"] = str(exc)

    return out


def _collect_job_and_stage_ids(job_group: str) -> tuple[list[int], list[int]]:
    try:
        tracker = sc.statusTracker()
        job_ids = [int(v) for v in tracker.getJobIdsForGroup(job_group)]
        stage_ids: list[int] = []
        for jid in job_ids:
            job_info = tracker.getJobInfo(jid)
            if job_info is None:
                continue
            try:
                ids = list(job_info.stageIds())
            except Exception:
                try:
                    ids = list(job_info.stageIds)
                except Exception:
                    ids = []
            stage_ids.extend(int(v) for v in ids)
        return sorted(set(job_ids)), sorted(set(stage_ids))
    except (Exception, AttributeError):
        # Spark Connect (RemoteContext) doesn't support statusTracker()
        return [], []


# Widgets
run_stamp = now_str()
arrow_default = _safe_conf("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
dbutils.widgets.text("OUT_DIR", "dbfs:/tmp/spark_policy_toolkit_paper/<date_time>/")
dbutils.widgets.text("RUN_EXPERIMENTS", "1,2,3,4,5,6")
dbutils.widgets.text("ARROW_MAX_RECORDS_PER_BATCH", str(arrow_default))
dbutils.widgets.text("SEED", "7")

OUT_DIR_RAW = dbutils.widgets.get("OUT_DIR").strip()
OUT_DIR = OUT_DIR_RAW.replace("<date_time>", run_stamp)
if not OUT_DIR.endswith("/"):
    OUT_DIR += "/"
PLOTS_DIR = OUT_DIR + "plots/"
RESULTS_CSV_PATH = OUT_DIR + "results.csv"
METADATA_JSON_PATH = OUT_DIR + "run_metadata.json"

RUN_EXPERIMENTS_RAW = dbutils.widgets.get("RUN_EXPERIMENTS").strip()
RUN_EXPERIMENTS = {
    int(token.strip())
    for token in RUN_EXPERIMENTS_RAW.split(",")
    if token.strip().isdigit()
}
if not RUN_EXPERIMENTS:
    RUN_EXPERIMENTS = {1, 2, 3, 4, 5, 6}

ARROW_MAX_BATCH = int(dbutils.widgets.get("ARROW_MAX_RECORDS_PER_BATCH"))
SEED = int(dbutils.widgets.get("SEED"))

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(ARROW_MAX_BATCH))

dbutils.fs.mkdirs(OUT_DIR)
dbutils.fs.mkdirs(PLOTS_DIR)

DECLARED_CLUSTER_CONTEXT = {
    "declared_databricks_runtime": "15.4 LTS (Spark 3.5.0)",
    "declared_worker_type": "Standard_D16as_v5 (64 GB, 16 cores), autoscaling min=2 max=40",
    "declared_driver_type": "Standard_D32as_v5 (128 GB, 32 cores)",
}
DISCOVERED_CONTEXT = _discover_cluster_context()

print("OUT_DIR:", OUT_DIR)
print("RUN_EXPERIMENTS:", sorted(RUN_EXPERIMENTS))
print("SEED:", SEED)
print("Arrow conf:", _safe_conf("spark.sql.execution.arrow.pyspark.enabled"), _safe_conf("spark.sql.execution.arrow.maxRecordsPerBatch"))
print("Discovered context:")
print(json.dumps(DISCOVERED_CONTEXT, indent=2, default=str))
print("Declared context:")
print(json.dumps(DECLARED_CLUSTER_CONTEXT, indent=2))

RUN_METADATA: Dict[str, Any] = {
    "run_id": run_stamp,
    "out_dir": OUT_DIR,
    "run_experiments_raw": RUN_EXPERIMENTS_RAW,
    "run_experiments_selected": sorted(RUN_EXPERIMENTS),
    "seed": SEED,
    "discovered_context": DISCOVERED_CONTEXT,
    "declared_cluster_context": DECLARED_CLUSTER_CONTEXT,
    "spark_conf_snapshot": {
        "spark.sql.execution.arrow.pyspark.enabled": _safe_conf(
            "spark.sql.execution.arrow.pyspark.enabled"
        ),
        "spark.sql.execution.arrow.maxRecordsPerBatch": _safe_conf(
            "spark.sql.execution.arrow.maxRecordsPerBatch"
        ),
    },
    "results_csv_path": RESULTS_CSV_PATH,
    "metadata_json_path": METADATA_JSON_PATH,
}
write_json(METADATA_JSON_PATH, RUN_METADATA)

COMMON_RESULT_FIELDS = {
    "run_id": run_stamp,
    "spark_version": DISCOVERED_CONTEXT.get("spark_version"),
    "pyspark_version": DISCOVERED_CONTEXT.get("pyspark_version"),
    "python_version": DISCOVERED_CONTEXT.get("python_version"),
    "executor_count": DISCOVERED_CONTEXT.get("executor_count_estimate"),
    "executor_memory_status_entries": DISCOVERED_CONTEXT.get(
        "executor_memory_status_entries"
    ),
    "arrow_enabled": _safe_conf("spark.sql.execution.arrow.pyspark.enabled"),
    "arrow_max_records_per_batch": _safe_conf(
        "spark.sql.execution.arrow.maxRecordsPerBatch"
    ),
    "cluster_id": DISCOVERED_CONTEXT.get("context_tag.clusterId"),
    "cluster_name": DISCOVERED_CONTEXT.get("context_tag.clusterName"),
    "declared_databricks_runtime": DECLARED_CLUSTER_CONTEXT[
        "declared_databricks_runtime"
    ],
    "declared_worker_type": DECLARED_CLUSTER_CONTEXT["declared_worker_type"],
    "declared_driver_type": DECLARED_CLUSTER_CONTEXT["declared_driver_type"],
}

RESULTS: List[Dict[str, Any]] = []


def append_results(rows: Dict[str, Any] | Iterable[Dict[str, Any]]) -> None:
    if isinstance(rows, dict):
        rows = [rows]
    for row in rows:
        payload = dict(COMMON_RESULT_FIELDS)
        payload["recorded_at_utc"] = now_str()
        payload.update(row)
        RESULTS.append(payload)


def flush_results_to_csv() -> pd.DataFrame:
    pdf = pd.DataFrame(RESULTS)
    local_path = dbfs_to_local_path(RESULTS_CSV_PATH)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    pdf.to_csv(local_path, index=False)
    print(f"Flushed {len(pdf):,} rows -> {RESULTS_CSV_PATH}")
    return pdf


def generate_split_search_dataframe(
    *,
    n_rows: int,
    n_features: int,
    n_treatments: int,
    missing_rate: float,
    seed: int,
) -> DataFrame:
    # Build all feature expressions in one projection to avoid deep withColumn plans.
    feature_exprs = []
    for idx in range(n_features):
        raw = F.rand(seed + idx).cast("double")
        value = (
            F.when(
                F.rand(seed + 10_000 + idx) < F.lit(missing_rate),
                F.lit(None).cast("double"),
            )
            .otherwise(raw)
            .alias(f"x{idx}")
        )
        feature_exprs.append(value)

    t_idx = F.floor(F.rand(seed + 20_000) * F.lit(float(n_treatments))).cast("int")
    x0 = F.coalesce(F.col("x0"), F.lit(0.5))
    p = (
        F.lit(0.10)
        + (
            F.col("treatment_idx").cast("double")
            / F.lit(max(1.0, float(n_treatments) * 10.0))
        )
        + (x0 * F.lit(0.20))
    )
    p = F.greatest(F.lit(0.01), F.least(F.lit(0.95), p))

    return (
        spark.range(n_rows)
        .select(*feature_exprs, t_idx.alias("treatment_idx"))
        .withColumn("treatment", F.col("treatment_idx").cast("string"))
        .withColumn(
            "outcome",
            F.when(F.rand(seed + 30_000) < p, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .drop("treatment_idx")
    )


def unit_interval_equal_width_splits(n_bins: int) -> List[float]:
    if n_bins <= 1:
        return []
    return [float(idx) / float(n_bins) for idx in range(1, n_bins)]


def _run_inference_timed(
    *,
    df: DataFrame,
    method: str,
    forest: List[Dict[str, Any]],
    feature_cols: List[str],
    out_col: str,
    job_group_prefix: str,
) -> Dict[str, Any]:
    job_group = f"{job_group_prefix}_{method}_{int(time.time() * 1000)}"
    try:
        sc.setJobGroup(job_group, f"{job_group_prefix}-{method}")
    except (Exception, AttributeError):
        pass  # Spark Connect (RemoteContext) doesn't support setJobGroup
    rss_before = safe_driver_rss_gb()
    t0 = time.perf_counter()
    checksum = None
    status = "ok"
    notes = ""
    try:
        if method == "mapInArrow":
            out_df = apply_policy_scorer_mapinarrow(
                df=df, forest=forest, feature_cols=feature_cols, out_col=out_col
            )
            checksum = force_materialize_array_col(out_df, out_col)
        elif method == "mapInPandas":
            out_df = apply_policy_scorer_mapinpandas(
                df=df, forest=forest, feature_cols=feature_cols, out_col=out_col
            )
            checksum = force_materialize_array_col(out_df, out_col)
        elif method == "anti_pattern":
            out_df = apply_policy_scorer_antipattern(
                df=df,
                forest=forest,
                feature_cols=feature_cols,
                out_col=out_col,
                simulate_spark_from_json=True,
            )
            checksum = force_materialize_array_col(out_df, out_col)
        else:
            raise ValueError(f"Unsupported method: {method}")
    except Exception as exc:
        status = "failed"
        notes = f"{type(exc).__name__}: {exc}"
        checksum = None
    elapsed = time.perf_counter() - t0
    rss_after = safe_driver_rss_gb()
    job_ids, stage_ids = _collect_job_and_stage_ids(job_group)
    try:
        sc.clearJobGroup()
    except (Exception, AttributeError):
        pass  # Spark Connect (RemoteContext) doesn't support clearJobGroup
    return {
        "status": status,
        "notes": notes,
        "elapsed_s": elapsed,
        "checksum": checksum,
        "driver_rss_gb_before": rss_before,
        "driver_rss_gb_after": rss_after,
        "job_ids_json": json.dumps(job_ids),
        "stage_ids_json": json.dumps(stage_ids),
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 1 — Experiment 1: The Serialization Tax (RUN THIS CELL)

# COMMAND ----------

if 1 in RUN_EXPERIMENTS:
    E1_N_FEATURES = 32
    E1_N_TREES = 50
    E1_N_TREATMENTS = 4
    E1_DEPTH = 7
    E1_MISSING_RATE = 0.10
    E1_MAX_PARTITIONS = 128
    # Keep large-row sweep for Arrow/Pandas methods, but run anti-pattern on small rows only.
    E1_METHOD_ROWS = {
        "mapInArrow": [1_000_000, 5_000_000, 10_000_000],
        "mapInPandas": [1_000_000, 5_000_000, 10_000_000],
        "anti_pattern": [100_000, 250_000],
    }
    E1_METHOD_ORDER = ["mapInArrow", "mapInPandas", "anti_pattern"]
    E1_ROWS_SWEEP = sorted(
        {
            n_rows
            for method_rows in E1_METHOD_ROWS.values()
            for n_rows in method_rows
        }
    )

    e1_forest = generate_synthetic_forest(
        depth=E1_DEPTH,
        n_treatments=E1_N_TREATMENTS,
        n_trees=E1_N_TREES,
        n_features=E1_N_FEATURES,
        missing_routing="random",
        seed=SEED + 101,
    )
    e1_feature_cols = [f"f{i}" for i in range(E1_N_FEATURES)]

    print(f"[E1] method/row matrix: {E1_METHOD_ROWS}")
    for n_rows in E1_ROWS_SWEEP:
        e1_partitions = max(32, min(E1_MAX_PARTITIONS, int(math.ceil(n_rows / 100_000))))
        print(f"[E1] n_rows={n_rows:,}, partitions={e1_partitions}")
        e1_df = generate_synthetic_dataframe(
            spark=spark,
            n_rows=n_rows,
            n_features=E1_N_FEATURES,
            missing_rate=E1_MISSING_RATE,
            seed=SEED + 11,
            partitions=e1_partitions,
        ).cache()
        _ = e1_df.count()

        for method in E1_METHOD_ORDER:
            if n_rows not in E1_METHOD_ROWS[method]:
                continue

            run_info = _run_inference_timed(
                df=e1_df,
                method=method,
                forest=e1_forest,
                feature_cols=e1_feature_cols,
                out_col=f"score_{method}",
                job_group_prefix="E1",
            )
            rows_per_s = (
                float(n_rows) / float(run_info["elapsed_s"])
                if run_info["status"] == "ok" and run_info["elapsed_s"] > 0
                else None
            )
            append_results(
                {
                    "experiment_id": "E1",
                    "method": method,
                    "n_rows": n_rows,
                    "n_features": E1_N_FEATURES,
                    "n_trees": E1_N_TREES,
                    "n_treatments": E1_N_TREATMENTS,
                    "depth": E1_DEPTH,
                    "missing_rate": E1_MISSING_RATE,
                    "partitions": e1_partitions,
                    "elapsed_s": run_info["elapsed_s"],
                    "rows_per_s": rows_per_s,
                    "status": run_info["status"],
                    "notes": run_info["notes"],
                    "checksum": run_info["checksum"],
                    "driver_rss_gb_before": run_info["driver_rss_gb_before"],
                    "driver_rss_gb_after": run_info["driver_rss_gb_after"],
                    "job_ids_json": run_info["job_ids_json"],
                    "stage_ids_json": run_info["stage_ids_json"],
                }
            )

        e1_df.unpersist()

    _ = flush_results_to_csv()
else:
    print("Skipping E1 (not selected).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 2 — Experiment 2: Profiling / GC Overhead (RUN THIS CELL)

# COMMAND ----------

if 2 in RUN_EXPERIMENTS:
    E2_N_ROWS = 50_000_000
    E2_N_FEATURES = 32
    E2_N_TREES = 50
    E2_N_TREATMENTS = 4
    E2_DEPTH = 7
    E2_MISSING_RATE = 0.10
    E2_PARTITIONS = 160

    e2_forest = generate_synthetic_forest(
        depth=E2_DEPTH,
        n_treatments=E2_N_TREATMENTS,
        n_trees=E2_N_TREES,
        n_features=E2_N_FEATURES,
        missing_routing="random",
        seed=SEED + 202,
    )
    e2_feature_cols = [f"f{i}" for i in range(E2_N_FEATURES)]

    print(f"[E2] building df rows={E2_N_ROWS:,}")
    e2_df = generate_synthetic_dataframe(
        spark=spark,
        n_rows=E2_N_ROWS,
        n_features=E2_N_FEATURES,
        missing_rate=E2_MISSING_RATE,
        seed=SEED + 22,
        partitions=E2_PARTITIONS,
    ).cache()
    _ = e2_df.count()

    for method in ["anti_pattern", "mapInArrow"]:
        run_info = _run_inference_timed(
            df=e2_df,
            method=method,
            forest=e2_forest,
            feature_cols=e2_feature_cols,
            out_col=f"score_e2_{method}",
            job_group_prefix="E2",
        )
        rows_per_s = (
            float(E2_N_ROWS) / float(run_info["elapsed_s"])
            if run_info["status"] == "ok" and run_info["elapsed_s"] > 0
            else None
        )
        append_results(
            {
                "experiment_id": "E2",
                "method": method,
                "n_rows": E2_N_ROWS,
                "n_features": E2_N_FEATURES,
                "n_trees": E2_N_TREES,
                "n_treatments": E2_N_TREATMENTS,
                "depth": E2_DEPTH,
                "missing_rate": E2_MISSING_RATE,
                "partitions": E2_PARTITIONS,
                "elapsed_s": run_info["elapsed_s"],
                "rows_per_s": rows_per_s,
                "driver_rss_gb_before": run_info["driver_rss_gb_before"],
                "driver_rss_gb_after": run_info["driver_rss_gb_after"],
                "executor_count": DISCOVERED_CONTEXT.get("executor_count_estimate"),
                "status": run_info["status"],
                "notes": "Fill GC time from Spark UI",
                "gc_time_ratio": None,
                "job_ids_json": run_info["job_ids_json"],
                "stage_ids_json": run_info["stage_ids_json"],
            }
        )

    e2_df.unpersist()
    _ = flush_results_to_csv()
else:
    print("Skipping E2 (not selected).")


# COMMAND ----------

# MAGIC %md
# MAGIC ### E2 Manual Spark UI Instructions (RUN THIS CELL)
# MAGIC
# MAGIC 1. Open Spark UI → Stages for the E2 jobs.
# MAGIC 2. For anti-pattern and mapInArrow stage IDs (from `stage_ids_json`), collect:
# MAGIC    - Executor CPU Time
# MAGIC    - JVM GC Time
# MAGIC 3. Compute `GC_time_ratio = JVM_GC_Time / Executor_CPU_Time`.
# MAGIC 4. Fill these values back into your analysis table.

# COMMAND ----------

e2_rows = [row for row in RESULTS if row.get("experiment_id") == "E2"]
if e2_rows:
    display(pd.DataFrame(e2_rows)[["method", "elapsed_s", "rows_per_s", "stage_ids_json", "notes"]])
else:
    print("No E2 rows found.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 3 — Experiment 3: Driver Bottleneck vs #Features (RUN THIS CELL)

# COMMAND ----------

if 3 in RUN_EXPERIMENTS:
    E3_N_FEATURES_SWEEP = [10, 50, 250, 1000]
    E3_N_ROWS = 500_000
    E3_N_BINS = 32
    E3_N_TREATMENTS = 4
    E3_MISSING_RATE = 0.10
    E3_CANDIDATE_COLLECT_THRESHOLD = 100_000
    E3_MAX_FEATURES_MEASURED = 64
    E3_FIXED_SPLITS = unit_interval_equal_width_splits(E3_N_BINS)

    for n_features in E3_N_FEATURES_SWEEP:
        print(f"[E3] n_features={n_features:,}")
        e3_df = generate_split_search_dataframe(
            n_rows=E3_N_ROWS,
            n_features=n_features,
            n_treatments=E3_N_TREATMENTS,
            missing_rate=E3_MISSING_RATE,
            seed=SEED + 300 + n_features,
        ).cache()
        _ = e3_df.count()

        feature_cols = [f"x{i}" for i in range(n_features)]
        measured_feature_cols = feature_cols[: min(n_features, E3_MAX_FEATURES_MEASURED)]
        measured_feature_count = len(measured_feature_cols)
        projection_factor = float(n_features) / float(max(1, measured_feature_count))

        # A) collect-less
        rss_before = safe_driver_rss_gb()
        t0 = time.perf_counter()
        status = "ok"
        notes = ""
        try:
            for feat in measured_feature_cols:
                prefix = build_prefix_sums(
                    e3_df,
                    feature_col=feat,
                    treatment_col="treatment",
                    outcome_col="outcome",
                    splits=E3_FIXED_SPLITS,
                )
                _ = score_candidates_collectless(prefix, evaluation_mode="sql")
        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
        elapsed_measured = time.perf_counter() - t0
        elapsed_projected = (
            elapsed_measured * projection_factor
            if status == "ok" and projection_factor > 1.0
            else elapsed_measured
        )
        if status == "ok" and projection_factor > 1.0:
            status = "ok_projected"
            notes = (
                f"measured {measured_feature_count}/{n_features} features; "
                f"elapsed projected by factor={projection_factor:.2f}"
            )
        rss_after = safe_driver_rss_gb()
        append_results(
            {
                "experiment_id": "E3",
                "method": "collectless",
                "n_rows": E3_N_ROWS,
                "n_features": n_features,
                "n_bins": E3_N_BINS,
                "n_treatments": E3_N_TREATMENTS,
                "elapsed_s": elapsed_projected,
                "elapsed_s_measured": elapsed_measured,
                "measured_features": measured_feature_count,
                "projection_factor": projection_factor,
                "driver_rss_gb_before": rss_before,
                "driver_rss_gb_after": rss_after,
                "status": status,
                "notes": notes,
            }
        )

        # B) driver-collect baseline
        est_candidate_rows = n_features * max(1, E3_N_BINS - 1) * E3_N_TREATMENTS
        if est_candidate_rows > E3_CANDIDATE_COLLECT_THRESHOLD:
            append_results(
                {
                    "experiment_id": "E3",
                    "method": "driver_collect",
                    "n_rows": E3_N_ROWS,
                    "n_features": n_features,
                    "n_bins": E3_N_BINS,
                    "n_treatments": E3_N_TREATMENTS,
                    "elapsed_s": None,
                    "driver_rss_gb_before": safe_driver_rss_gb(),
                    "driver_rss_gb_after": safe_driver_rss_gb(),
                    "status": "skipped_too_large",
                    "notes": (
                        f"estimated candidate rows={est_candidate_rows:,} > "
                        f"threshold={E3_CANDIDATE_COLLECT_THRESHOLD:,}"
                    ),
                }
            )
        else:
            rss_before = safe_driver_rss_gb()
            t0 = time.perf_counter()
            status = "ok"
            notes = ""
            collected_candidates: list[Any] = []
            try:
                for feat in measured_feature_cols:
                    prefix = build_prefix_sums(
                        e3_df,
                        feature_col=feat,
                        treatment_col="treatment",
                        outcome_col="outcome",
                        splits=E3_FIXED_SPLITS,
                    )
                    collected_candidates.extend(prefix.collect())
            except Exception as exc:
                status = "failed"
                notes = f"{type(exc).__name__}: {exc}"
            elapsed_measured = time.perf_counter() - t0
            elapsed_projected = (
                elapsed_measured * projection_factor
                if status == "ok" and projection_factor > 1.0
                else elapsed_measured
            )
            if status == "ok" and projection_factor > 1.0:
                status = "ok_projected"
                notes = (
                    f"measured {measured_feature_count}/{n_features} features; "
                    f"elapsed and candidate rows projected by factor={projection_factor:.2f}"
                )
            rss_after = safe_driver_rss_gb()
            collected_rows_measured = len(collected_candidates)
            collected_rows_projected = int(round(collected_rows_measured * projection_factor))
            append_results(
                {
                    "experiment_id": "E3",
                    "method": "driver_collect",
                    "n_rows": E3_N_ROWS,
                    "n_features": n_features,
                    "n_bins": E3_N_BINS,
                    "n_treatments": E3_N_TREATMENTS,
                    "elapsed_s": elapsed_projected,
                    "elapsed_s_measured": elapsed_measured,
                    "measured_features": measured_feature_count,
                    "projection_factor": projection_factor,
                    "driver_rss_gb_before": rss_before,
                    "driver_rss_gb_after": rss_after,
                    "status": status,
                    "notes": notes,
                    "collected_candidate_rows": collected_rows_projected,
                    "collected_candidate_rows_measured": collected_rows_measured,
                }
            )
            del collected_candidates
            gc.collect()

        e3_df.unpersist()

    _ = flush_results_to_csv()
else:
    print("Skipping E3 (not selected).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Experiment 4: Collect-less Backend Comparison (RUN THIS CELL)

# COMMAND ----------

if 4 in RUN_EXPERIMENTS:
    E4_N_ROWS = 25_000_000
    E4_N_FEATURES = 1
    E4_N_TREATMENTS = 4
    E4_MISSING_RATE = 0.10
    E4_BINS_SWEEP = [16, 32, 64, 128]

    e4_df = generate_split_search_dataframe(
        n_rows=E4_N_ROWS,
        n_features=E4_N_FEATURES,
        n_treatments=E4_N_TREATMENTS,
        missing_rate=E4_MISSING_RATE,
        seed=SEED + 400,
    ).cache()
    _ = e4_df.count()

    for n_bins in E4_BINS_SWEEP:
        print(f"[E4] bins={n_bins}")
        prefix = build_prefix_sums(
            e4_df,
            feature_col="x0",
            treatment_col="treatment",
            outcome_col="outcome",
            num_quantile_splits=n_bins,
        ).cache()
        _ = prefix.count()

        for mode in ["sql", "mapInPandas"]:
            status = "ok"
            notes = ""
            t0 = time.perf_counter()
            try:
                _ = score_candidates_collectless(prefix, evaluation_mode=mode)
            except Exception as exc:
                status = "failed"
                notes = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - t0
            append_results(
                {
                    "experiment_id": "E4",
                    "evaluation_mode": mode,
                    "n_rows": E4_N_ROWS,
                    "n_bins": n_bins,
                    "n_treatments": E4_N_TREATMENTS,
                    "elapsed_s": elapsed,
                    "status": status,
                    "notes": notes,
                }
            )

        prefix.unpersist()

    e4_df.unpersist()
    _ = flush_results_to_csv()
else:
    print("Skipping E4 (not selected).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 5 — Experiment 5: Parity + Determinism (RUN THIS CELL)

# COMMAND ----------

if 5 in RUN_EXPERIMENTS:
    E5_N_ROWS = 100_000
    E5_N_TREATMENTS = 4
    E5_N_BINS = 32

    e5_df = generate_split_search_dataframe(
        n_rows=E5_N_ROWS,
        n_features=1,
        n_treatments=E5_N_TREATMENTS,
        missing_rate=0.10,
        seed=SEED + 500,
    ).cache()
    _ = e5_df.count()

    e5_prefix = build_prefix_sums(
        e5_df,
        feature_col="x0",
        treatment_col="treatment",
        outcome_col="outcome",
        num_quantile_splits=E5_N_BINS,
    )

    # Extract the bin boundaries so driver_collect uses identical splits
    # (approxQuantile is non-deterministic, so we must reuse the same boundaries).
    e5_splits = sorted([
        float(row["bin_boundary"])
        for row in e5_prefix.select("bin_boundary").distinct().collect()
        if row["bin_boundary"] is not None
    ])

    e5_driver = best_split_driver_collect(
        e5_df.select("x0", "treatment", "outcome"),
        feature_col="x0",
        treatment_col="treatment",
        outcome_col="outcome",
        splits=e5_splits,
    )
    e5_sql = score_candidates_collectless(e5_prefix, evaluation_mode="sql")
    e5_map = score_candidates_collectless(e5_prefix, evaluation_mode="mapInPandas")

    parity_table = pd.DataFrame(
        [
            {
                "Implementation": "driver_collect",
                "Feature": e5_driver.feature,
                "Threshold/BinBoundary": e5_driver.bin_boundary,
                "NaN-direction": e5_driver.nan_direction,
                "Score": e5_driver.score,
            },
            {
                "Implementation": "collectless_sql",
                "Feature": e5_sql.feature,
                "Threshold/BinBoundary": e5_sql.bin_boundary,
                "NaN-direction": e5_sql.nan_direction,
                "Score": e5_sql.score,
            },
            {
                "Implementation": "collectless_mapInPandas",
                "Feature": e5_map.feature,
                "Threshold/BinBoundary": e5_map.bin_boundary,
                "NaN-direction": e5_map.nan_direction,
                "Score": e5_map.score,
            },
        ]
    )
    display(parity_table)

    parity_csv_path = OUT_DIR + "parity_table.csv"
    parity_table.to_csv(dbfs_to_local_path(parity_csv_path), index=False)
    print("Wrote parity table:", parity_csv_path)

    def _assert_same_split(a, b, tol: float = 1e-9) -> None:
        assert a.feature == b.feature, f"feature mismatch: {a.feature} vs {b.feature}"
        assert (
            a.candidate_bin == b.candidate_bin
        ), f"candidate_bin mismatch: {a.candidate_bin} vs {b.candidate_bin}"
        assert (
            a.nan_direction == b.nan_direction
        ), f"nan_direction mismatch: {a.nan_direction} vs {b.nan_direction}"

        if a.bin_boundary is None or b.bin_boundary is None:
            assert a.bin_boundary == b.bin_boundary
        else:
            assert abs(float(a.bin_boundary) - float(b.bin_boundary)) <= tol

        assert abs(float(a.score) - float(b.score)) <= tol, (
            f"score mismatch: {a.score} vs {b.score} (tol={tol})"
        )

    _assert_same_split(e5_driver, e5_sql, tol=1e-9)
    _assert_same_split(e5_driver, e5_map, tol=1e-9)

    det_runs = []
    for _ in range(3):
        x = score_candidates_collectless(e5_prefix, evaluation_mode="sql")
        det_runs.append((x.feature, x.candidate_bin, x.bin_boundary, x.nan_direction, x.score))
    assert det_runs[0] == det_runs[1] == det_runs[2], f"determinism failed: {det_runs}"

    # Direction 1 tiny parity: mapInArrow vs mapInPandas checksum and row-wise tolerance.
    D1_TINY_ROWS = 200_000
    D1_TINY_FEATURES = 8
    D1_TINY_TREATMENTS = 4
    D1_TINY_TREES = 5
    d1_tiny_feature_cols = [f"f{i}" for i in range(D1_TINY_FEATURES)]

    d1_tiny_df = (
        generate_synthetic_dataframe(
            spark=spark,
            n_rows=D1_TINY_ROWS,
            n_features=D1_TINY_FEATURES,
            missing_rate=0.20,
            seed=SEED + 501,
            partitions=16,
        )
        .withColumn("__rid", F.monotonically_increasing_id())
        .cache()
    )
    _ = d1_tiny_df.count()

    d1_tiny_forest = generate_synthetic_forest(
        depth=4,
        n_treatments=D1_TINY_TREATMENTS,
        n_trees=D1_TINY_TREES,
        n_features=D1_TINY_FEATURES,
        missing_routing="random",
        seed=SEED + 502,
    )

    d1_arrow = apply_policy_scorer_mapinarrow(
        d1_tiny_df, d1_tiny_forest, d1_tiny_feature_cols, "score_ar"
    )
    d1_pandas = apply_policy_scorer_mapinpandas(
        d1_tiny_df, d1_tiny_forest, d1_tiny_feature_cols, "score_pd"
    )

    joined = d1_arrow.select("__rid", "score_ar").join(
        d1_pandas.select("__rid", "score_pd"), on="__rid", how="inner"
    )
    checksum_ar = force_materialize_array_col(joined.select("score_ar"), "score_ar")
    checksum_pd = force_materialize_array_col(joined.select("score_pd"), "score_pd")
    mismatch = joined.selectExpr(
        "sum(case when aggregate("
        "zip_with(score_ar, score_pd, (x, y) -> abs(coalesce(x,0D) - coalesce(y,0D))), "
        "0D, (acc, v) -> acc + v) > 1e-9 then 1 else 0 end) as mismatches"
    ).first()["mismatches"]
    assert int(mismatch or 0) == 0, f"Direction 1 parity mismatches={mismatch}"

    append_results(
        [
            {
                "experiment_id": "E5",
                "component": "d2_parity",
                "status": "ok",
                "notes": "driver_collect == collectless_sql == collectless_mapInPandas within 1e-9",
                "score_driver": e5_driver.score,
                "score_sql": e5_sql.score,
                "score_mapinpandas": e5_map.score,
            },
            {
                "experiment_id": "E5",
                "component": "d2_determinism",
                "status": "ok",
                "notes": f"3 runs identical: {det_runs[0]}",
            },
            {
                "experiment_id": "E5",
                "component": "d1_parity",
                "status": "ok",
                "notes": "mapInArrow vs mapInPandas parity passed",
                "checksum_arrow": checksum_ar,
                "checksum_mapinpandas": checksum_pd,
                "mismatch_rows": int(mismatch or 0),
            },
        ]
    )

    d1_tiny_df.unpersist()
    e5_df.unpersist()
    _ = flush_results_to_csv()
else:
    print("Skipping E5 (not selected).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 6 — Experiment 6: Multi-treatment Scaling (RUN THIS CELL)

# COMMAND ----------

if 6 in RUN_EXPERIMENTS:
    E6_TREATMENT_SWEEP = [2, 4, 8, 16]

    # Direction 1 fixed config
    E6_INF_ROWS = 10_000_000
    E6_INF_FEATURES = 32
    E6_INF_TREES = 50
    E6_INF_DEPTH = 7
    E6_INF_MISSING_RATE = 0.10
    E6_INF_PARTITIONS = 128

    # Direction 2 fixed config
    E6_SPLIT_ROWS = 2_000_000
    E6_SPLIT_FEATURES = 1
    E6_SPLIT_BINS = 32
    E6_SPLIT_MISSING_RATE = 0.10

    for n_treatments in E6_TREATMENT_SWEEP:
        print(f"[E6] n_treatments={n_treatments}")

        # Inference component
        e6_inf_df = generate_synthetic_dataframe(
            spark=spark,
            n_rows=E6_INF_ROWS,
            n_features=E6_INF_FEATURES,
            missing_rate=E6_INF_MISSING_RATE,
            seed=SEED + 600 + n_treatments,
            partitions=E6_INF_PARTITIONS,
        ).cache()
        _ = e6_inf_df.count()
        e6_inf_feature_cols = [f"f{i}" for i in range(E6_INF_FEATURES)]
        e6_inf_forest = generate_synthetic_forest(
            depth=E6_INF_DEPTH,
            n_treatments=n_treatments,
            n_trees=E6_INF_TREES,
            n_features=E6_INF_FEATURES,
            missing_routing="random",
            seed=SEED + 700 + n_treatments,
        )

        for method in ["mapInArrow", "mapInPandas"]:
            run_info = _run_inference_timed(
                df=e6_inf_df,
                method=method,
                forest=e6_inf_forest,
                feature_cols=e6_inf_feature_cols,
                out_col=f"score_e6_{method}",
                job_group_prefix="E6_INF",
            )
            rows_per_s = (
                float(E6_INF_ROWS) / float(run_info["elapsed_s"])
                if run_info["status"] == "ok" and run_info["elapsed_s"] > 0
                else None
            )
            append_results(
                {
                    "experiment_id": "E6",
                    "component": "inference",
                    "method": method,
                    "n_treatments": n_treatments,
                    "n_rows": E6_INF_ROWS,
                    "n_features": E6_INF_FEATURES,
                    "n_trees": E6_INF_TREES,
                    "elapsed_s": run_info["elapsed_s"],
                    "rows_per_s": rows_per_s,
                    "status": run_info["status"],
                    "notes": run_info["notes"],
                }
            )

        e6_inf_df.unpersist()

        # Split-search component
        e6_split_df = generate_split_search_dataframe(
            n_rows=E6_SPLIT_ROWS,
            n_features=E6_SPLIT_FEATURES,
            n_treatments=n_treatments,
            missing_rate=E6_SPLIT_MISSING_RATE,
            seed=SEED + 800 + n_treatments,
        ).cache()
        _ = e6_split_df.count()

        e6_prefix = build_prefix_sums(
            e6_split_df,
            feature_col="x0",
            treatment_col="treatment",
            outcome_col="outcome",
            num_quantile_splits=E6_SPLIT_BINS,
        ).cache()
        _ = e6_prefix.count()

        for mode in ["sql", "mapInPandas"]:
            t0 = time.perf_counter()
            status = "ok"
            notes = ""
            try:
                _ = score_candidates_collectless(e6_prefix, evaluation_mode=mode)
            except Exception as exc:
                status = "failed"
                notes = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - t0
            append_results(
                {
                    "experiment_id": "E6",
                    "component": "split_search",
                    "evaluation_mode": mode,
                    "n_treatments": n_treatments,
                    "n_rows": E6_SPLIT_ROWS,
                    "n_features": E6_SPLIT_FEATURES,
                    "n_bins": E6_SPLIT_BINS,
                    "elapsed_s": elapsed,
                    "status": status,
                    "notes": notes,
                }
            )

        e6_prefix.unpersist()
        e6_split_df.unpersist()

    _ = flush_results_to_csv()
else:
    print("Skipping E6 (not selected).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## FINAL SECTION — Consolidate + Export (RUN THIS CELL)

# COMMAND ----------

results_pdf = flush_results_to_csv()

if results_pdf.empty:
    print("No experiment rows were recorded.")
else:
    compact_cols = [
        "experiment_id",
        "component",
        "method",
        "evaluation_mode",
        "n_rows",
        "n_features",
        "n_bins",
        "n_treatments",
        "elapsed_s",
        "rows_per_s",
        "status",
        "notes",
    ]
    compact_cols = [c for c in compact_cols if c in results_pdf.columns]
    compact_view = results_pdf[compact_cols].copy()
    display(compact_view)


plots_written: list[str] = []
try:
    import matplotlib.pyplot as plt  # type: ignore

    # E1: rows vs rows/sec
    e1 = results_pdf[
        (results_pdf["experiment_id"] == "E1") & (results_pdf["status"] == "ok")
    ].copy()
    if not e1.empty and {"method", "n_rows", "rows_per_s"}.issubset(e1.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, g in e1.groupby("method"):
            g = g.sort_values("n_rows")
            ax.plot(g["n_rows"], g["rows_per_s"], marker="o", label=method)
        ax.set_title("E1: Inference Throughput vs Rows")
        ax.set_xlabel("n_rows")
        ax.set_ylabel("rows/sec")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = dbfs_to_local_path(PLOTS_DIR + "e1_rows_vs_throughput.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        plots_written.append(PLOTS_DIR + "e1_rows_vs_throughput.png")

    # E3: driver memory vs features
    e3 = results_pdf[
        (results_pdf["experiment_id"] == "E3")
        & (results_pdf["status"].isin(["ok", "ok_projected"]))
    ].copy()
    if not e3.empty and {"method", "n_features", "driver_rss_gb_before", "driver_rss_gb_after"}.issubset(e3.columns):
        e3["driver_rss_delta_gb"] = (
            pd.to_numeric(e3["driver_rss_gb_after"], errors="coerce")
            - pd.to_numeric(e3["driver_rss_gb_before"], errors="coerce")
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, g in e3.groupby("method"):
            g = g.sort_values("n_features")
            ax.plot(g["n_features"], g["driver_rss_delta_gb"], marker="o", label=method)
        ax.set_title("E3: Driver Memory Delta vs #Features")
        ax.set_xlabel("n_features")
        ax.set_ylabel("driver_rss_delta_gb")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = dbfs_to_local_path(PLOTS_DIR + "e3_driver_memory_vs_features.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        plots_written.append(PLOTS_DIR + "e3_driver_memory_vs_features.png")

    # E4: grouped bar sql vs mapInPandas
    e4 = results_pdf[
        (results_pdf["experiment_id"] == "E4") & (results_pdf["status"] == "ok")
    ].copy()
    if not e4.empty and {"evaluation_mode", "n_bins", "elapsed_s"}.issubset(e4.columns):
        pivot = (
            e4.pivot_table(index="n_bins", columns="evaluation_mode", values="elapsed_s", aggfunc="mean")
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("E4: Split Search Backend Comparison")
        ax.set_xlabel("n_bins")
        ax.set_ylabel("elapsed_s")
        ax.grid(True, axis="y", alpha=0.3)
        p = dbfs_to_local_path(PLOTS_DIR + "e4_backend_comparison.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        plots_written.append(PLOTS_DIR + "e4_backend_comparison.png")

except Exception as exc:
    print("Plotting skipped:", exc)
    print(traceback.format_exc(limit=1))

RUN_METADATA["plots_written"] = plots_written
RUN_METADATA["results_rows"] = len(results_pdf)
RUN_METADATA["completed_at_utc"] = now_str()
write_json(METADATA_JSON_PATH, RUN_METADATA)

print("Results CSV:", RESULTS_CSV_PATH)
print("Metadata JSON:", METADATA_JSON_PATH)
if plots_written:
    print("Plots:")
    for p in plots_written:
        print("-", p)
else:
    print("No plots written.")
