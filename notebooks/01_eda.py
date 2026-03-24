"""
Building Energy Digital Twin — Exploratory Data Analysis
========================================================
Dataset: Building Data Genome Project 2 (BDG2)
Source: https://zenodo.org/records/3887306

BDG2 has 3,053 energy meters from 1,636 non-residential buildings,
2 years (2016-2017) of hourly data from 19 sites worldwide.

Key files:
- metadata.csv: building info (sqft, use type, year built)
- weather.csv: outdoor conditions per site (air temp, wind, clouds)
- meters/raw/*.csv: energy readings per meter type (electricity, steam, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %% Setup
DATA_DIR = Path("../data/raw/bdg2")
OUTPUT_DIR = Path("../outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% 1. Discover dataset structure
print("=" * 60)
print("BDG2 Dataset Structure")
print("=" * 60)

if DATA_DIR.exists():
    for item in sorted(DATA_DIR.rglob("*.csv")):
        size_mb = item.stat().st_size / 1e6
        print(f"  {item.relative_to(DATA_DIR)} ({size_mb:.1f} MB)")
else:
    print(f"DATA_DIR not found: {DATA_DIR}")
    print("Checking alternative paths...")
    for alt in [DATA_DIR.parent, DATA_DIR.parent / "building-data-genome-project-2-1.0"]:
        if alt.exists():
            print(f"  Found: {alt}")
            for item in sorted(alt.rglob("*.csv"))[:20]:
                print(f"    {item.relative_to(alt)}")

# %% 2. Load metadata
meta_candidates = list(DATA_DIR.rglob("metadata*.csv"))
if meta_candidates:
    meta = pd.read_csv(meta_candidates[0])
    print(f"\n{'=' * 60}")
    print(f"METADATA: {meta.shape[0]} buildings")
    print(f"{'=' * 60}")
    print(f"Columns: {list(meta.columns)}")
    print(f"\nBuilding types:")
    if "primaryspaceusage" in meta.columns:
        print(meta["primaryspaceusage"].value_counts().head(10))
    print(f"\nSites: {meta['site_id'].nunique() if 'site_id' in meta.columns else 'N/A'}")
    print(f"\nFloor area (sqft):")
    if "sqft" in meta.columns:
        print(meta["sqft"].describe())

    # Plot building type distribution
    if "primaryspaceusage" in meta.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        meta["primaryspaceusage"].value_counts().head(15).plot.barh(ax=ax, color="steelblue")
        ax.set_xlabel("Count")
        ax.set_title("Building Types in BDG2")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "building_types.png", dpi=150)
        plt.show()

# %% 3. Load weather data
weather_candidates = list(DATA_DIR.rglob("weather*.csv"))
if weather_candidates:
    weather = pd.read_csv(weather_candidates[0])
    print(f"\n{'=' * 60}")
    print(f"WEATHER: {weather.shape}")
    print(f"{'=' * 60}")
    print(f"Columns: {list(weather.columns)}")

    # Parse timestamps
    time_col = [c for c in weather.columns if "time" in c.lower()][0]
    weather[time_col] = pd.to_datetime(weather[time_col])

    # Summary per site
    if "site_id" in weather.columns:
        print(f"\nSites: {weather['site_id'].nunique()}")
        for site_id in sorted(weather["site_id"].unique())[:5]:
            site_data = weather[weather["site_id"] == site_id]
            temp_col = [c for c in weather.columns if "airtemp" in c.lower() or "air_temp" in c.lower()]
            if temp_col:
                print(f"  Site {site_id}: {len(site_data)} rows, "
                      f"temp range [{site_data[temp_col[0]].min():.1f}, {site_data[temp_col[0]].max():.1f}] °C")

    # Plot temperature distribution across sites
    temp_col = [c for c in weather.columns if "airtemp" in c.lower() or "air_temp" in c.lower()]
    if temp_col and "site_id" in weather.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        weather.boxplot(column=temp_col[0], by="site_id", ax=ax)
        ax.set_ylabel("Air Temperature (°C)")
        ax.set_title("Temperature Distribution by Site")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "temp_by_site.png", dpi=150)
        plt.show()

# %% 4. Load electricity meter data (largest meter type)
elec_candidates = list(DATA_DIR.rglob("*electricity*.csv"))
if elec_candidates:
    elec = pd.read_csv(elec_candidates[0])
    print(f"\n{'=' * 60}")
    print(f"ELECTRICITY METERS: {elec.shape}")
    print(f"{'=' * 60}")

    # Parse timestamp
    time_col = [c for c in elec.columns if "time" in c.lower() or "date" in c.lower()][0]
    elec[time_col] = pd.to_datetime(elec[time_col])
    elec = elec.set_index(time_col)

    # Building count
    building_cols = [c for c in elec.columns if c != time_col]
    print(f"Buildings with electricity meters: {len(building_cols)}")
    print(f"Time range: {elec.index.min()} to {elec.index.max()}")

    # Missing data per building
    missing_pct = elec[building_cols].isnull().mean().sort_values()
    print(f"\nMissing data distribution:")
    print(f"  <1% missing: {(missing_pct < 0.01).sum()} buildings")
    print(f"  1-10% missing: {((missing_pct >= 0.01) & (missing_pct < 0.10)).sum()} buildings")
    print(f"  >10% missing: {(missing_pct >= 0.10).sum()} buildings")

    # Pick a well-covered building for detailed plots
    best_buildings = missing_pct[missing_pct < 0.01].index.tolist()[:5]
    if best_buildings:
        fig, axes = plt.subplots(len(best_buildings), 1, figsize=(16, 3 * len(best_buildings)), sharex=True)
        if len(best_buildings) == 1:
            axes = [axes]
        for ax, bldg in zip(axes, best_buildings):
            ax.plot(elec.index, elec[bldg], linewidth=0.3, alpha=0.8)
            ax.set_ylabel("kWh")
            ax.set_title(bldg, fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Electricity Consumption — Best-Covered Buildings", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "electricity_timeseries.png", dpi=150)
        plt.show()

    # %% 5. Daily and hourly patterns
    if best_buildings:
        bldg = best_buildings[0]
        series = elec[bldg].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Hourly profile
        hourly = series.groupby(series.index.hour).mean()
        axes[0].bar(hourly.index, hourly.values, color="steelblue")
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel("Mean kWh")
        axes[0].set_title(f"Average Hourly Profile — {bldg}")

        # Weekly profile
        daily = series.groupby(series.index.dayofweek).mean()
        axes[1].bar(daily.index, daily.values, color="coral",
                     tick_label=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        axes[1].set_ylabel("Mean kWh")
        axes[1].set_title(f"Day-of-Week Profile — {bldg}")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "temporal_patterns.png", dpi=150)
        plt.show()

# %% 6. Energy vs Temperature (the RC model's key relationship)
if weather_candidates and elec_candidates and "site_id" in meta.columns:
    # Pick a building and its site's weather
    if best_buildings:
        bldg = best_buildings[0]
        bldg_site = meta.loc[meta.iloc[:, 0] == bldg, "site_id"]
        if len(bldg_site) > 0:
            site = bldg_site.values[0]
            temp_col = [c for c in weather.columns if "airtemp" in c.lower() or "air_temp" in c.lower()]
            if temp_col:
                site_weather = weather[weather["site_id"] == site].set_index(
                    pd.to_datetime(weather[weather["site_id"] == site].iloc[:, 0])
                )
                merged = pd.DataFrame({
                    "energy": elec[bldg],
                    "temp": site_weather[temp_col[0]],
                }).dropna()

                if len(merged) > 100:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(merged["temp"], merged["energy"], s=1, alpha=0.2, c="steelblue")
                    ax.set_xlabel("Outdoor Air Temperature (°C)")
                    ax.set_ylabel("Energy Consumption (kWh)")
                    ax.set_title(f"Energy vs Temperature — {bldg}")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / "energy_vs_temperature.png", dpi=150)
                    plt.show()
                    print(f"\nCorrelation (energy, temp): {merged['energy'].corr(merged['temp']):.3f}")

print("\n✅ EDA complete. Review figures in outputs/figures/")
