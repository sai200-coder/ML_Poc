from pyspark import pipelines as dp
from pyspark.sql.functions import (
    to_timestamp,
    current_timestamp,
    col
)

# ============================================================
# BRONZE 1: Historical turbine failure logs (RAW, APPEND-ONLY)
# ============================================================
@dp.table(
    name="bronze_failure_logs",
    comment="Raw historical turbine failure logs ingested as-is for ML labeling"
)
@dp.expect("valid_schema", "_rescued_data IS NULL")
def bronze_failure_logs():

    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .load("/Volumes/ml/data/dataset/failure_logs/")
        # Event time (failure start)
        .withColumn(
            "failure_ts",
            to_timestamp(col("failure_timestamp_start"))
        )
        # Ingest time (when Databricks received it)
        .withColumn(
            "ingest_ts",
            current_timestamp()
        )
    )


# ============================================================
# BRONZE 2: Raw SCADA sensor stream (HIGH-VOLUME STREAMING)
# ============================================================
@dp.table(
    name="bronze_scada_raw",
    comment="Raw SCADA sensor events ingested incrementally using Auto Loader",
    partition_cols=["turbine_id"]
)
@dp.expect("valid_schema", "_rescued_data IS NULL")
@dp.expect_or_drop("valid_energy", "energy_kwh IS NOT NULL")
def bronze_scada_raw():

    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("cloudFiles.schemaHints", "turbine_id STRING")
        .option("cloudFiles.maxFilesPerTrigger", "100")
        .load("/Volumes/ml/data/dataset/scada/")
        # Event time from SCADA
        .withColumn(
            "event_ts",
            to_timestamp(col("event_ts"))
        )
        # Ingest time
        .withColumn(
            "ingest_ts",
            current_timestamp()
        )
    )


# ============================================================
# BRONZE 3: Turbine metadata (STATIC REFERENCE DATA)
# ============================================================
@dp.table(
    name="bronze_turbine_metadata",
    comment="Raw turbine metadata (static reference data)"
)
@dp.expect("valid_schema", "_rescued_data IS NULL")
def bronze_turbine_metadata():

    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .load("/Volumes/ml/data/dataset/metadata/")
        .withColumnRenamed(
            "created_at",
            "turbine_createdat"
        )
        .withColumn(
            "ingest_ts",
            current_timestamp()
        )
    )
