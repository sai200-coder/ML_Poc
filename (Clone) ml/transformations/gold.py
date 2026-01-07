from pyspark import pipelines as dp
from pyspark.sql.functions import col, to_timestamp

@dp.table(
    name="scada_failure_base",
    comment="SCADA hourly data joined with historical turbine failures using a time-window join to prevent temporal leakage."
)
def scada_failure_timewindow_joined():
    select_cols = [
        "turbine_id",
        "curtailment_flag_sum",
        "energy_kwh_sum",
        "gearbox_temp_c_avg",
        "generator_speed_rpm_avg",
        "generator_temp_c_avg",
        "nacelle_temp_c_avg",
        "pitch_angle_deg_avg",
        "power_kw_avg",
        "rotor_speed_rpm_avg",
        "shutdown_flag_sum",
        "vibration_mm_s_avg",
        "wind_direction_deg_avg",
        "wind_speed_m_s_avg",
        "yaw_deg_avg",
        "yaw_misalignment_deg_avg",
        "record_count",
        "power_kw_stddev",
        "hour_start",
        "hour_end",
        "age_at_failure_years",
        "degradation_start_ts",
        "downtime_hours",
        "early_warning_window_hours",
        "failure_timestamp_start",
        "failure_id",
        "failure_timestamp_end",
        "failure_type",
        "has_early_warning",
        "is_curtailment_related",
        "is_high_wind_related",
        "is_sudden_failure",
        "related_scada_features",
        "severity",
        "turbine_name",
        "latitude",
        "longitude",
        "country",
        "state",
        "manufacturer",
        "model",
        "rated_power_kw",
        "rotor_diameter_m",
        "hub_height_m",
        "blade_length_m",
        "install_date",
        "manufacturer_date",
        "age_years",
        "env_risk_score"
    ]
    # Read and alias DataFrames
    scada = spark.read.table("ml.data.silver_scada_hourly").alias("scada")
    failure = (
        spark.read.table("ml.data.silver_failure_logs")
        .withColumn("failure_timestamp_start_ts", to_timestamp(col("failure_timestamp_start")))
        .withColumn("failure_timestamp_end_ts", to_timestamp(col("failure_timestamp_end")))
        .withColumn("degradation_start_ts_ts", to_timestamp(col("degradation_start_ts")))
        .alias("failure")
    )
    turbine = spark.read.table("ml.data.silver_turbine_metadata").alias("turbine")

    # Join SCADA and failure logs
    joined = scada.join(
        failure,
        (
            (col("scada.turbine_id") == col("failure.turbine_id")) &
            (col("scada.hour_start") >= col("failure.degradation_start_ts_ts")) &
            (col("scada.hour_start") <= col("failure.failure_timestamp_end_ts"))
        ),
        "left"
    )

    # Drop duplicate turbine_id from failure to avoid ambiguity in next join
    joined = joined.drop(col("failure.turbine_id"))

    # Join with turbine metadata
    joined_with_turbine = joined.join(
        turbine,
        col("scada.turbine_id") == col("turbine.turbine_id"),
        "left"
    ).drop(col("turbine.turbine_id"))

    # Select columns, always from the correct alias
    return joined_with_turbine.select(*[col(c) for c in select_cols])