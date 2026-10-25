"""China AQI labels and dataset splits, shared by aqi_labeling_and_splits.ipynb and model_development.ipynb.

Both notebooks used to carry their own copy of these rules, and the copies drifted.
This module is now the single definition of the breakpoints, the validity gates, the
daily label table, the train/test split, and the hourly broadcast of daily labels.

The rules implemented here follow China's HJ 633 standard, not WHO guideline values:

* PM2.5, PM10, SO2, NO2 and CO use a 24-hour mean that needs at least 20 valid hourly
  readings. CO is converted from ug/m3 to mg/m3 before its IAQI is taken.
* O3 uses the day's maximum valid trailing 8-hour mean. Each window needs at least 6
  readings and a day needs at least 14 valid windows ending from 08:00 through 23:00.
  Above 800 ug/m3 the daily maximum 1-hour concentration is used instead, against its
  own breakpoint table.
* Each pollutant IAQI is linearly interpolated between breakpoints; the day's AQI is
  the largest available pollutant IAQI.

`interpolated_columns`, written by etl.ipynb, records which hourly cells that
notebook filled by linear interpolation. Those cells are indistinguishable from
measured readings, so by default they satisfy the validity gates exactly like measured
ones; `build_daily_labels(..., exclude_interpolated=True)` reruns the labels from
measured hours only, for sensitivity analysis.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- China AQI standard -----------------------------------------------------------------

AQI_BREAKPOINTS = [0, 50, 100, 150, 200, 300, 400, 500]

POLLUTANT_BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 115, 150, 250, 350, 500],
    "PM10": [0, 50, 120, 250, 350, 420, 500, 600],
    "SO2": [0, 50, 150, 475, 800, 1600, 2100, 2620],
    "NO2": [0, 40, 80, 180, 280, 565, 750, 940],
    "CO": [0, 2, 4, 14, 24, 36, 48, 60],
    "O3": [0, 100, 160, 215, 265, 800],
}

O3_1H_BREAKPOINTS = [0, 160, 200, 300, 400, 800, 1000, 1200]

AQI_CATEGORIES = {
    1: "Excellent",
    2: "Good",
    3: "Lightly polluted",
    4: "Moderately polluted",
    5: "Heavily polluted",
    6: "Severely polluted",
}

# Lower and upper bound of each category, in category-code order.
AQI_RANGES = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 500)]

POLLUTANTS = list(POLLUTANT_BREAKPOINTS)
DAILY_POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO"]

# --- Validity gates ---------------------------------------------------------------------

MIN_VALID_HOURS = 20
OZONE_WINDOW = "8h"
MIN_WINDOW_READINGS = 6
MIN_OZONE_WINDOWS = 14
OZONE_WINDOW_HOURS = (8, 23)
OZONE_1H_THRESHOLD = 800
CO_UG_TO_MG = 0.001

# Every column name etl.ipynb can record in its `interpolated_columns` output.
INTERPOLATED_COLUMN_NAMES = [*POLLUTANTS, "TEMP", "PRES", "DEWP", "WSPM", "wd"]

REQUIRED_COLUMNS = [*POLLUTANTS, "station", "year", "month", "day", "hour"]

# The daily targets model_development.ipynb trains on: the overall category plus one
# category per pollutant.
TARGET_COLUMNS = ["aqi_category", *[f"{pollutant}_category" for pollutant in POLLUTANTS]]

# Columns broadcast from the daily table onto every hourly row of that station-day.
DAILY_VALUE_COLUMNS = [
    *[f"{pollutant}_daily_avg" for pollutant in DAILY_POLLUTANTS],
    "O3_daily_max_8h_avg",
    "O3_daily_max_1h_avg",
    "CO_daily_avg_mg_m3",
    *[f"{pollutant}_iaqi" for pollutant in POLLUTANTS],
    "aqi",
    "aqi_category_code",
    "aqi_category",
    *[f"{pollutant}_category" for pollutant in POLLUTANTS],
]

STATION_DAY_KEYS = ["_station_key", "_date"]
_RENAMED_DAILY_COLUMNS = {
    **{f"{pollutant}_24h_avg": f"{pollutant}_daily_avg" for pollutant in DAILY_POLLUTANTS},
    "O3_8h_avg": "O3_daily_max_8h_avg",
    "O3_1h_max": "O3_daily_max_1h_avg",
    "CO_mg_m3": "CO_daily_avg_mg_m3",
}


def require_columns(frame, source_name="the source table"):
    """Raise unless every column the AQI rules need is present."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{source_name} is missing required columns: {missing}")


def add_station_day_keys(frame):
    """Return a copy of `frame` with `_timestamp`, `_date` and `_station_key` added.

    Station names are stripped and case-folded so that formatting differences in the
    source table cannot split one station into two groups.
    """
    result = frame.copy()
    parts = result[["year", "month", "day", "hour"]].apply(pd.to_numeric, errors="coerce")
    timestamps = pd.to_datetime(
        {
            "year": parts["year"],
            "month": parts["month"],
            "day": parts["day"],
            "hour": parts["hour"],
        },
        errors="coerce",
    )
    result["_timestamp"] = timestamps
    result["_date"] = timestamps.dt.normalize()
    result["_station_key"] = result["station"].astype("string").str.strip().str.casefold()
    return result


def parse_interpolated_flags(frame):
    """Mark the cells that etl.ipynb filled by linear interpolation.

    `interpolated_columns` holds a comma-space separated list of column names, or the
    literal string "None". Matching is token based, so PM2.5 never matches PM10.
    """
    if "interpolated_columns" not in frame.columns:
        raise KeyError(
            "The source table has no 'interpolated_columns' column, so interpolated "
            "readings cannot be told apart from measured ones. Rebuild all_regions "
            "with etl.ipynb before building labels."
        )
    recorded = frame["interpolated_columns"].fillna("None").astype("string")
    return pd.DataFrame(
        {
            name: recorded.str.contains(
                r"(?:^|, )" + re.escape(name) + r"(?:, |$)", regex=True, na=False
            )
            for name in INTERPOLATED_COLUMN_NAMES
        },
        index=frame.index,
    )


def interpolate_iaqi(concentration, concentration_breakpoints, aqi_breakpoints=None):
    """Interpolate concentrations between the supplied China AQI breakpoints."""
    if aqi_breakpoints is None:
        aqi_breakpoints = AQI_BREAKPOINTS
    # The 8-hour ozone table stops at AQI 200, so it is deliberately shorter than the
    # AQI scale; a shorter concentration table is valid, a longer one is not.
    if len(aqi_breakpoints) < len(concentration_breakpoints):
        raise ValueError(
            "The concentration breakpoint table is longer than the AQI table it maps "
            f"onto: {len(concentration_breakpoints)} vs {len(aqi_breakpoints)}."
        )
    result = pd.Series(np.nan, index=concentration.index, dtype="float64")
    for index in range(len(concentration_breakpoints) - 1):
        bp_lo, bp_hi = concentration_breakpoints[index:index + 2]
        iaqi_lo, iaqi_hi = aqi_breakpoints[index:index + 2]
        in_range = concentration.gt(bp_lo) & concentration.le(bp_hi)
        result.loc[in_range] = (
            (iaqi_hi - iaqi_lo) / (bp_hi - bp_lo) * (concentration.loc[in_range] - bp_lo)
            + iaqi_lo
        )
    result.loc[concentration.eq(0)] = 0
    return result


def category_codes(values):
    """Map AQI values to their 1-6 China category codes, NA where the value is NA.

    Values are compared as float64 so that the mask is a plain numpy boolean array.
    Comparing a nullable Int64 column directly yields a nullable mask, and assigning
    through a mask that contains NA raises.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    codes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    for code, (lower, upper) in enumerate(AQI_RANGES, 1):
        codes.loc[numeric.between(lower, upper)] = code
    return codes


def build_daily_labels(source_frame, exclude_interpolated=False):
    """Return (daily_records, hourly_rows): one AQI per station and calendar day.

    `exclude_interpolated=False` counts interpolated hours as valid, which is how these
    labels have always been built. `True` masks upstream-interpolated pollutant hours
    before the validity gates, so only measured hours can satisfy them. The provenance
    columns describe the same source rows either way, so the two runs stay comparable.
    """
    require_columns(source_frame)
    hourly_df = add_station_day_keys(source_frame)
    hourly_df = hourly_df.loc[
        hourly_df["_timestamp"].notna() & hourly_df["_station_key"].notna()
    ].copy()
    if hourly_df.duplicated(STATION_DAY_KEYS + ["_timestamp"]).any():
        raise ValueError(
            "The source contains duplicate station-hour rows; resolve them before "
            "calculating a daily AQI."
        )

    for pollutant in POLLUTANTS:
        readings = pd.to_numeric(hourly_df[pollutant], errors="coerce")
        hourly_df[pollutant] = readings.mask(readings.lt(0))

    # An interpolated hour is only synthetic if it survived as a usable reading.
    interpolated_reading = pd.DataFrame(
        {
            pollutant: flags.to_numpy() & hourly_df[pollutant].notna().to_numpy()
            for pollutant, flags in parse_interpolated_flags(hourly_df).items()
        },
        index=hourly_df.index,
    )
    for pollutant in POLLUTANTS:
        hourly_df[f"_interp_{pollutant}"] = interpolated_reading[pollutant]
    if exclude_interpolated:
        for pollutant in POLLUTANTS:
            hourly_df[pollutant] = hourly_df[pollutant].mask(interpolated_reading[pollutant])

    daily_group = hourly_df.groupby(STATION_DAY_KEYS, sort=False)
    daily_records = daily_group[["station"]].first()
    daily_records.index.names = STATION_DAY_KEYS

    for pollutant in DAILY_POLLUTANTS:
        valid_hour_count = daily_group[pollutant].count()
        daily_mean = daily_group[pollutant].mean()
        daily_records[f"{pollutant}_24h_avg"] = daily_mean.where(
            valid_hour_count.ge(MIN_VALID_HOURS)
        )

    # Daily ozone is the maximum valid trailing 8-hour mean for each date.
    o3_rolling_parts = []
    for _, station_rows in hourly_df.groupby("_station_key", sort=False):
        ordered = station_rows.sort_values("_timestamp")
        o3_series = pd.Series(
            ordered["O3"].to_numpy(), index=pd.DatetimeIndex(ordered["_timestamp"])
        )
        o3_rolling = o3_series.rolling(
            OZONE_WINDOW, closed="right", min_periods=MIN_WINDOW_READINGS
        ).mean()
        o3_rolling_parts.append(
            pd.DataFrame(
                {
                    "_station_key": ordered["_station_key"].to_numpy(),
                    "_date": ordered["_date"].to_numpy(),
                    "_hour": ordered["_timestamp"].dt.hour.to_numpy(),
                    "O3_8h_avg": o3_rolling.to_numpy(),
                }
            )
        )
    o3_windows = pd.concat(o3_rolling_parts, ignore_index=True)
    o3_windows = o3_windows.loc[o3_windows["_hour"].between(*OZONE_WINDOW_HOURS)]
    o3_daily = o3_windows.groupby(STATION_DAY_KEYS, sort=False)["O3_8h_avg"]
    o3_daily_max = o3_daily.max()
    o3_valid_count = o3_daily.count()
    daily_records["O3_8h_avg"] = o3_daily_max.where(o3_valid_count.ge(MIN_OZONE_WINDOWS))
    daily_records["O3_1h_max"] = daily_group["O3"].max()

    # Provenance: how much of each daily value rests on interpolation rather than
    # measurement. Assigned while the index is still (station, date) so the counts
    # align with the daily rows by key.
    interp_counts = daily_group[
        [f"_interp_{pollutant}" for pollutant in POLLUTANTS]
    ].sum()
    for pollutant in POLLUTANTS:
        daily_records[f"{pollutant}_interpolated_hours"] = interp_counts[
            f"_interp_{pollutant}"
        ]
    for pollutant in DAILY_POLLUTANTS:
        valid_hours = daily_group[pollutant].count()
        measured_hours = valid_hours - interp_counts[f"_interp_{pollutant}"]
        daily_records[f"{pollutant}_valid_hours"] = valid_hours
        daily_records[f"{pollutant}_measured_hours"] = measured_hours
        # True when the day clears the 20-hour gate only because interpolated hours count.
        daily_records[f"{pollutant}_marginal_pass"] = (
            valid_hours.ge(MIN_VALID_HOURS) & measured_hours.lt(MIN_VALID_HOURS)
        )

    daily_records["date"] = daily_records.index.get_level_values("_date")
    daily_records = daily_records.reset_index(drop=True)
    daily_records["CO_mg_m3"] = daily_records["CO_24h_avg"] * CO_UG_TO_MG

    for pollutant in DAILY_POLLUTANTS:
        concentration = (
            daily_records["CO_mg_m3"]
            if pollutant == "CO"
            else daily_records[f"{pollutant}_24h_avg"]
        )
        pollutant_aqi = interpolate_iaqi(concentration, POLLUTANT_BREAKPOINTS[pollutant])
        daily_records[f"{pollutant}_iaqi"] = pollutant_aqi.round().astype("Int64")

    o3_concentration = daily_records["O3_8h_avg"]
    o3_aqi = interpolate_iaqi(o3_concentration, POLLUTANT_BREAKPOINTS["O3"])
    # Above 800 ug/m3, use the daily maximum 1-hour O3 concentration.
    o3_1h_aqi = interpolate_iaqi(daily_records["O3_1h_max"], O3_1H_BREAKPOINTS)
    use_o3_1h = o3_concentration.gt(OZONE_1H_THRESHOLD)
    o3_aqi.loc[use_o3_1h] = o3_1h_aqi.loc[use_o3_1h]
    daily_records["O3_iaqi"] = o3_aqi.round().astype("Int64")

    iaqi_columns = [f"{pollutant}_iaqi" for pollutant in POLLUTANTS]
    daily_records["aqi"] = daily_records[iaqi_columns].max(axis=1, skipna=True).astype("Int64")
    daily_records.loc[daily_records[iaqi_columns].isna().all(axis=1), "aqi"] = pd.NA
    daily_records["aqi_category_code"] = category_codes(daily_records["aqi"])
    daily_records["aqi_category"] = daily_records["aqi_category_code"].map(AQI_CATEGORIES)
    for pollutant in POLLUTANTS:
        daily_records[f"{pollutant}_category"] = category_codes(
            daily_records[f"{pollutant}_iaqi"]
        ).map(AQI_CATEGORIES)

    return daily_records, attach_daily_labels(source_frame, daily_records)


def attach_daily_labels(rows, daily_records, value_columns=None):
    """Broadcast each station-day's daily values onto every hourly row of that day.

    The original hourly rows, their order and their count are preserved.
    """
    if value_columns is None:
        value_columns = DAILY_VALUE_COLUMNS
    daily_lookup = daily_records.rename(columns=_RENAMED_DAILY_COLUMNS).copy()
    missing = [column for column in value_columns if column not in daily_lookup.columns]
    if missing:
        raise KeyError(f"The daily table is missing broadcast columns: {missing}")

    daily_lookup["_station_key"] = (
        daily_lookup["station"].astype("string").str.strip().str.casefold()
    )
    daily_lookup["_date"] = pd.to_datetime(daily_lookup["date"])

    labeled = add_station_day_keys(rows)
    labeled["_row_order"] = range(len(labeled))
    labeled = labeled.merge(
        daily_lookup[[*STATION_DAY_KEYS, *value_columns]],
        on=STATION_DAY_KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    return (
        labeled.sort_values("_row_order")
        .drop(columns=["_row_order", *STATION_DAY_KEYS])
        .reset_index(drop=True)
    )


def split_by_station_day(source_frame, test_fraction=0.20, random_state=42):
    """Split hourly rows into train and test sets that share no station-day.

    Sampling happens within each station, so every station keeps roughly its share of
    rows, but the unit of sampling is the station-day: all 24 hours of a station-day
    land on the same side. This stops hours of one station-day appearing in both sets,
    which the earlier hourly-row split allowed.

    Returns (train, test), each with the original source columns, shuffled and with a
    fresh index.
    """
    frame = add_station_day_keys(source_frame)
    if frame["_timestamp"].isna().any() or frame["_station_key"].isna().any():
        raise ValueError("Rows with unusable station or timestamp values cannot be split.")
    if frame.duplicated([*STATION_DAY_KEYS, "_timestamp"]).any():
        raise ValueError(
            "The source contains duplicate station-hour rows; resolve them before splitting."
        )

    train_parts = []
    test_parts = []
    for _, station_rows in frame.groupby("_station_key", sort=False):
        station_days = station_rows["_date"].drop_duplicates()
        if len(station_days) < 2:
            # A single station-day cannot be present in both sets; retain it in training.
            train_parts.append(station_rows)
            continue
        test_day_count = min(
            len(station_days) - 1, max(1, round(len(station_days) * test_fraction))
        )
        test_days = station_days.sample(n=test_day_count, random_state=random_state)
        is_test = station_rows["_date"].isin(test_days)
        train_parts.append(station_rows.loc[~is_test])
        test_parts.append(station_rows.loc[is_test])

    train = pd.concat(train_parts).sample(frac=1, random_state=random_state)
    test = pd.concat(test_parts).sample(frac=1, random_state=random_state)
    drop = ["_timestamp", *STATION_DAY_KEYS]
    return (
        train.drop(columns=drop).reset_index(drop=True),
        test.drop(columns=drop).reset_index(drop=True),
    )


def station_day_key_set(frame):
    """The set of (station, date) pairs present in `frame`."""
    return set(add_station_day_keys(frame)[STATION_DAY_KEYS].itertuples(index=False, name=None))


def assert_disjoint_station_days(train, test):
    """Raise if any station-day appears in both splits.

    Returns (train_station_days, test_station_days).
    """
    train_keys = station_day_key_set(train)
    test_keys = station_day_key_set(test)
    overlap = train_keys & test_keys
    if overlap:
        raise AssertionError(
            f"{len(overlap)} station-days appear in both the training and test sets, "
            "so the split is leaking daily labels across the boundary."
        )
    return len(train_keys), len(test_keys)
