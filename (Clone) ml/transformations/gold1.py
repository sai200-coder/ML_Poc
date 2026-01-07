from pyspark import pipelines as dp
from pyspark.sql.functions import (
    col, to_timestamp, when, expr
)


from pyspark.sql import Window
from pyspark.sql.functions import col, avg, stddev, when


@dp.table(
    name="scada_failure_labeled",
    comment="Leakage-safe early failure labels and failure cause",
    
)
def scada_failure_labeled():

    df = spark.read.table("scada_failure_base")

    df = df.withColumn(
        "failure_ts",
        to_timestamp(col("failure_timestamp_start"))
    )

    return (
        df
        # -------- Early warning labels --------
        .withColumn(
            "label_fail_6h",
            when(
                (col("failure_ts").isNotNull()) &
                (col("hour_start") < col("failure_ts")) &
                (col("hour_start") >= expr("failure_ts - INTERVAL 6 HOURS")),
                1
            ).otherwise(0)
        )
        .withColumn(
            "label_fail_12h",
            when(
                (col("failure_ts").isNotNull()) &
                (col("hour_start") < col("failure_ts")) &
                (col("hour_start") >= expr("failure_ts - INTERVAL 12 HOURS")),
                1
            ).otherwise(0)
        )
        .withColumn(
            "label_fail_24h",
            when(
                (col("failure_ts").isNotNull()) &
                (col("hour_start") < col("failure_ts")) &
                (col("hour_start") >= expr("failure_ts - INTERVAL 24 HOURS")),
                1
            ).otherwise(0)
        )
        .withColumn(
            "label_fail_72h",
            when(
                (col("failure_ts").isNotNull()) &
                (col("hour_start") < col("failure_ts")) &
                (col("hour_start") >= expr("failure_ts - INTERVAL 72 HOURS")),
                1
            ).otherwise(0)
        )

        # -------- Failure cause (multiclass) --------
        .withColumn(
            "failure_cause_label",
            when(col("failure_type") == "gearbox", 1)
            .when(col("failure_type") == "generator", 2)
            .when(col("failure_type") == "rotor", 3)
            .when(col("failure_type") == "bearing", 4)
            .otherwise(0)
        )
    )

@dp.materialized_view(
    name="scada_failure_feature_table",
    comment="Leakage-safe rolling features for wind turbine failure prediction (6h/12h/24h/72h)",
    schema="""
        turbine_id STRING NOT NULL,
        hour_start TIMESTAMP NOT NULL,

        power_kw_avg_mean_6h DOUBLE,
        power_kw_avg_mean_12h DOUBLE,
        power_kw_avg_mean_24h DOUBLE,
        power_kw_avg_mean_72h DOUBLE,

        wind_speed_m_s_avg_mean_6h DOUBLE,
        wind_speed_m_s_avg_mean_12h DOUBLE,
        wind_speed_m_s_avg_mean_24h DOUBLE,
        wind_speed_m_s_avg_mean_72h DOUBLE,

        vibration_mm_s_avg_mean_6h DOUBLE,
        vibration_mm_s_avg_mean_12h DOUBLE,
        vibration_mm_s_avg_mean_24h DOUBLE,
        vibration_mm_s_avg_mean_72h DOUBLE,

        gearbox_temp_c_avg_mean_6h DOUBLE,
        gearbox_temp_c_avg_mean_12h DOUBLE,
        gearbox_temp_c_avg_mean_24h DOUBLE,
        gearbox_temp_c_avg_mean_72h DOUBLE,

        generator_speed_rpm_avg_mean_6h DOUBLE,
        generator_speed_rpm_avg_mean_12h DOUBLE,
        generator_speed_rpm_avg_mean_24h DOUBLE,
        generator_speed_rpm_avg_mean_72h DOUBLE,

        generator_temp_c_avg_mean_6h DOUBLE,
        generator_temp_c_avg_mean_12h DOUBLE,
        generator_temp_c_avg_mean_24h DOUBLE,
        generator_temp_c_avg_mean_72h DOUBLE,

        nacelle_temp_c_avg_mean_6h DOUBLE,
        nacelle_temp_c_avg_mean_12h DOUBLE,
        nacelle_temp_c_avg_mean_24h DOUBLE,
        nacelle_temp_c_avg_mean_72h DOUBLE,

        power_kw_avg_stddev_6h DOUBLE,
        power_kw_avg_stddev_12h DOUBLE,
        power_kw_avg_stddev_24h DOUBLE,
        power_kw_avg_stddev_72h DOUBLE,

        wind_speed_m_s_avg_stddev_6h DOUBLE,
        wind_speed_m_s_avg_stddev_12h DOUBLE,
        wind_speed_m_s_avg_stddev_24h DOUBLE,
        wind_speed_m_s_avg_stddev_72h DOUBLE,

        vibration_mm_s_avg_stddev_6h DOUBLE,
        vibration_mm_s_avg_stddev_12h DOUBLE,
        vibration_mm_s_avg_stddev_24h DOUBLE,
        vibration_mm_s_avg_stddev_72h DOUBLE,

        gearbox_temp_c_avg_stddev_6h DOUBLE,
        gearbox_temp_c_avg_stddev_12h DOUBLE,
        gearbox_temp_c_avg_stddev_24h DOUBLE,
        gearbox_temp_c_avg_stddev_72h DOUBLE,

        generator_speed_rpm_avg_stddev_6h DOUBLE,
        generator_speed_rpm_avg_stddev_12h DOUBLE,
        generator_speed_rpm_avg_stddev_24h DOUBLE,
        generator_speed_rpm_avg_stddev_72h DOUBLE,

        generator_temp_c_avg_stddev_6h DOUBLE,
        generator_temp_c_avg_stddev_12h DOUBLE,
        generator_temp_c_avg_stddev_24h DOUBLE,
        generator_temp_c_avg_stddev_72h DOUBLE,

        nacelle_temp_c_avg_stddev_6h DOUBLE,
        nacelle_temp_c_avg_stddev_12h DOUBLE,
        nacelle_temp_c_avg_stddev_24h DOUBLE,
        nacelle_temp_c_avg_stddev_72h DOUBLE,

        label_fail_6h INT,
        label_fail_12h INT,
        label_fail_24h INT,
        label_fail_72h INT,

        health_state INT,
        failure_cause_label INT,

        CONSTRAINT scada_features_pk
          PRIMARY KEY (turbine_id, hour_start TIMESERIES)
    """
)
def scada_failure_feature_table():

    base = spark.read.table("scada_failure_labeled")

    # ---------------------------
    # Health state
    # ---------------------------
    base = base.withColumn(
        "health_state",
        when(col("label_fail_24h") == 1, 1)
        .when(col("failure_ts").isNotNull(), 2)
        .otherwise(0)
    )

    # 🔑 ALL feature columns (updated)
    feature_cols = [
        "power_kw_avg",
        "wind_speed_m_s_avg",
        "vibration_mm_s_avg",
        "gearbox_temp_c_avg",
        "generator_speed_rpm_avg",
        "generator_temp_c_avg",
        "nacelle_temp_c_avg"
    ]

    windows = {
        "6h": 5,
        "12h": 11,
        "24h": 23,
        "72h": 71
    }

    df = base

    for feat in feature_cols:
        for win_name, rows in windows.items():
            w = (
                Window
                .partitionBy("turbine_id")
                .orderBy("hour_start")
                .rowsBetween(-rows, 0)
            )

            df = (
                df
                .withColumn(f"{feat}_mean_{win_name}", avg(col(feat)).over(w))
                .withColumn(f"{feat}_stddev_{win_name}", stddev(col(feat)).over(w))
            )

    return df.select(
        "turbine_id",
        "hour_start",

        *[f"{f}_mean_{w}" for f in feature_cols for w in windows],
        *[f"{f}_stddev_{w}" for f in feature_cols for w in windows],

        "label_fail_6h",
        "label_fail_12h",
        "label_fail_24h",
        "label_fail_72h",
        "health_state",
        "failure_cause_label"
    )
@dp.materialized_view(
    name="model_6h",
    comment="Training dataset for 6-hour failure prediction"
)
def training_model_6h():

    df = spark.read.table("scada_failure_feature_table")

    feature_cols = [
        c for c in df.columns
        if c not in [
            "label_fail_6h",
            "label_fail_12h",
            "label_fail_24h",
            "label_fail_72h",
            "health_state",
            "failure_cause_label"
        ]
    ]

    return df.select(*feature_cols, "label_fail_6h")
@dp.materialized_view(
    name="model_12h",
    comment="Training dataset for 12-hour failure prediction"
)
def training_model_12h():

    df = spark.read.table("scada_failure_feature_table")

    feature_cols = [
        c for c in df.columns
        if c not in [
            "label_fail_6h",
            "label_fail_12h",
            "label_fail_24h",
            "label_fail_72h",
            "health_state",
            "failure_cause_label"
        ]
    ]

    return df.select(*feature_cols, "label_fail_12h")
@dp.materialized_view(
    name="model_24h",
    comment="Training dataset for 24-hour failure prediction"
)
def training_model_24h():

    df = spark.read.table("scada_failure_feature_table")

    feature_cols = [
        c for c in df.columns
        if c not in [
            "label_fail_6h",
            "label_fail_12h",
            "label_fail_24h",
            "label_fail_72h",
            "health_state",
            "failure_cause_label"
        ]
    ]

    return df.select(*feature_cols, "label_fail_24h")
@dp.materialized_view(
    name="model_72h",
    comment="Training dataset for 72-hour failure prediction"
)
def training_model_72h():

    df = spark.read.table("scada_failure_feature_table")

    feature_cols = [
        c for c in df.columns
        if c not in [
            "label_fail_6h",
            "label_fail_12h",
            "label_fail_24h",
            "label_fail_72h",
            "health_state",
            "failure_cause_label"
        ]
    ]

    return df.select(*feature_cols, "label_fail_72h")
@dp.materialized_view(
    name="model_component",
    comment="Training dataset for failure cause (component) classification"
)
def training_model_component():

    df = spark.read.table("scada_failure_feature_table")

    feature_cols = [
        c for c in df.columns
        if c not in [
            "label_fail_6h",
            "label_fail_12h",
            "label_fail_24h",
            "label_fail_72h",
            "health_state",
            "failure_cause_label"
        ]
    ]

    return df.select(*feature_cols, "failure_cause_label")


