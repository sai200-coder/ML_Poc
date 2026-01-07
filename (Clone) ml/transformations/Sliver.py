from pyspark import pipelines as dp
from pyspark.sql.functions import col, to_timestamp

@dp.table(
    name="silver_scada_clean",
    comment="Cleaned SCADA events with event-time watermarking and deduplication"
)
def silver_scada_clean():

    df = (
        spark.readStream
        .table("bronze_scada_raw")
        .withColumn("event_ts", to_timestamp("event_ts"))
        .filter(col("event_ts").isNotNull())
        .withWatermark("event_ts", "2 hours")
        .dropDuplicates(["turbine_id", "event_ts"])
    )

    return df
from pyspark.sql.functions import (
    window, avg, sum, min, max, count, stddev
)

@dp.table(
    name="silver_scada_hourly",
    comment="Hourly aggregated SCADA metrics per turbine"
)
def silver_scada_hourly():

    df = spark.readStream.table("silver_scada_clean")

    agg_df = (
        df
        .withWatermark("event_ts", "2 hours")
        .groupBy(
            window(col("event_ts"), "1 hour"),
            col("turbine_id")
        )
        .agg(
            min("event_ts").alias("hour_start"),
            max("event_ts").alias("hour_end"),
            sum("energy_kwh").alias("energy_kwh_sum"),

            avg("gearbox_temp_c").alias("gearbox_temp_c_avg"),
            avg("generator_temp_c").alias("generator_temp_c_avg"),
            avg("generator_speed_rpm").alias("generator_speed_rpm_avg"),
            avg("nacelle_temp_c").alias("nacelle_temp_c_avg"),
            avg("rotor_speed_rpm").alias("rotor_speed_rpm_avg"),
            avg("power_kw").alias("power_kw_avg"),
            stddev("power_kw").alias("power_kw_stddev"),

            avg("vibration_mm_s").alias("vibration_mm_s_avg"),
            avg("wind_speed_m_s").alias("wind_speed_m_s_avg"),
            avg("wind_direction_deg").alias("wind_direction_deg_avg"),
            avg("pitch_angle_deg").alias("pitch_angle_deg_avg"),
            avg("yaw_deg").alias("yaw_deg_avg"),
            avg("yaw_misalignment_deg").alias("yaw_misalignment_deg_avg"),

            sum("curtailment_flag").alias("curtailment_flag_sum"),
            sum("shutdown_flag").alias("shutdown_flag_sum"),

            count("*").alias("record_count")
        )
        .drop("window")
    )

    return agg_df
@dp.table(
    name="silver_failure_logs",
    comment="Cleaned historical failure events"
)
def silver_failure_logs():

    return (
        spark.read
        .table("bronze_failure_logs")
        .filter(col("failure_ts").isNotNull())
        .select(
            "turbine_id",
            "age_at_failure_years",
            "failure_timestamp_start",
            "failure_id",
            "failure_ts",
            "failure_timestamp_end",
            "degradation_start_ts",
            "failure_type",
            "severity",
            "has_early_warning",
            "early_warning_window_hours",
            "downtime_hours",
            "is_curtailment_related",
            "is_high_wind_related",
            "is_sudden_failure",
            "related_scada_features"
        )
    )
@dp.table(
    name="silver_turbine_metadata",
    comment="Cleaned turbine metadata (static reference)"
)
def silver_turbine_metadata():

    return (
        spark.read
        .table("bronze_turbine_metadata")
        .select(
            "turbine_id",
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
        )
    )
