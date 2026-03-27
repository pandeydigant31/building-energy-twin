"""Data loading and preparation for BDG2 dataset.

Handles loading electricity + weather for a single building,
and estimating RC model inputs (solar gains, internal gains)
from available weather and schedule data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BDG2_ROOT = Path("data/raw/bdg2/buds-lab-building-data-genome-project-2-3d0cbaf")


def load_building_data(
    building_id: str,
    site_id: str,
    data_root: Path = BDG2_ROOT,
) -> pd.DataFrame:
    """Load and merge electricity + weather for a single building.

    Returns DataFrame with DatetimeIndex (hourly) and columns:
        electricity, airTemperature, cloudCoverage, dewTemperature, windSpeed
    """
    elec_path = data_root / "data/meters/cleaned/electricity_cleaned.csv"
    elec = pd.read_csv(
        elec_path, usecols=["timestamp", building_id], parse_dates=["timestamp"]
    )
    elec = elec.set_index("timestamp").rename(columns={building_id: "electricity"})

    weather_path = data_root / "data/weather/weather.csv"
    weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    weather = weather[weather["site_id"] == site_id].set_index("timestamp")
    weather = weather.drop(columns=["site_id"])

    df = elec.join(weather, how="inner")

    # Fill gaps
    df["cloudCoverage"] = df["cloudCoverage"].ffill().bfill().fillna(4.0)
    for col in ["airTemperature", "dewTemperature", "windSpeed"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()

    df = df.dropna(subset=["electricity", "airTemperature"])
    return df


def load_metadata(building_id: str, data_root: Path = BDG2_ROOT) -> dict:
    """Load building metadata as a dict."""
    meta_path = data_root / "data/metadata/metadata.csv"
    meta = pd.read_csv(meta_path)
    row = meta[meta["building_id"] == building_id].iloc[0]
    return row.to_dict()


def estimate_solar_gains(df: pd.DataFrame, sqm: float) -> np.ndarray:
    """Estimate solar heat gains (W) from cloud coverage and time of day.

    Simple model: Q_solar = Q_peak * solar_profile * seasonal * clear_sky
    - Q_peak: based on window area ratio and solar heat gain coefficient
    - solar_profile: sinusoidal from ~6am to ~6pm
    - seasonal: higher in summer
    - clear_sky: scales inversely with cloud coverage (0-9 BDG2 scale)
    """
    hours = df.index.hour.values
    day_of_year = df.index.dayofyear.values

    # Solar hour profile: sine curve, 0 at night
    solar_angle = (hours - 6) / 12 * np.pi
    solar_profile = np.clip(np.sin(solar_angle), 0, None)

    # Seasonal: more solar in summer (peak ~June 21, day 172)
    seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Cloud attenuation (cloud 0=clear, 9=overcast)
    clear_sky = 1.0 - 0.7 * (df["cloudCoverage"].values / 9.0)

    # Peak solar gain: ~12% of floor area as effective solar aperture, 300 W/m2 peak
    Q_peak = 300.0 * sqm * 0.12
    return Q_peak * solar_profile * seasonal * clear_sky


def estimate_internal_gains(df: pd.DataFrame, sqm: float) -> np.ndarray:
    """Estimate internal heat gains (W) from typical office schedule.

    Weekday 7am-7pm: 40 W/m2 (people + lights + equipment)
    Otherwise: 8 W/m2 (standby equipment)
    """
    hours = df.index.hour.values
    dow = df.index.dayofweek.values

    is_occupied = (dow < 5) & (hours >= 7) & (hours < 19)
    return np.where(is_occupied, 40.0 * sqm, 8.0 * sqm)
