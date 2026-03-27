#!/usr/bin/env python3
"""Building Energy Digital Twin — Training Pipeline

Trains a hybrid physics-ML model to predict hourly building electricity.

Architecture:
    E_predicted = E_physics(RC thermal load) + NN_correction(features)

Pipeline:
    1. Load BDG2 data (electricity + weather) for demo building
    2. Run RC model free-floating -> thermal load proxy
    3. Calibrate energy scaling (degree-hours -> kWh via linear regression)
    4. Train correction network on residuals (occupancy, schedules, equipment)
    5. Evaluate: physics-only vs NN-only vs hybrid vs baselines

Usage:
    cd building-energy-twin
    python train.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.physics.rc_model import RC3R2CModel, RC3R2CParams, ZoneInputs
from src.ml.correction import ResidualCorrectionNet, CorrectionTrainer, encode_cyclical
from src.pipeline.data_loader import (
    load_building_data,
    load_metadata,
    estimate_solar_gains,
    estimate_internal_gains,
)

# ──── Configuration ────────────────────────────────────────────────
BUILDING_ID = "Hog_office_Gustavo"
SITE_ID = "Hog"
TRAIN_SLICE = ("2017-01", "2017-09")  # Jan-Sep 2017
TEST_SLICE = ("2017-10", "2017-12")   # Oct-Dec 2017 (fall/winter — good physics test)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Physics
T_HEAT_SETPOINT = 21.0  # degC
T_COOL_SETPOINT = 23.5  # degC
RC_SUBSTEPS = 10         # Euler sub-steps per hour for stability
RC_REF_AREA = 200.0      # m² — default RC params are tuned for this zone size
Q_INT_OCCUPIED = 15.0    # W/m² internal gains for RC sim (reduced from full 40;
                         #   full gains cause unrealistic heating in free-floating mode)
Q_INT_UNOCCUPIED = 3.0   # W/m²
SOLAR_SCALE = 0.3        # scale factor on solar estimate for free-floating sim

# Neural network
N_EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3
HIDDEN_DIMS = [64, 32]
PATIENCE = 15
N_LAGS = 3


# ──── Helpers ──────────────────────────────────────────────────────

def simulate_stable(rc_model, inputs, substeps=RC_SUBSTEPS, **kwargs):
    """Run RC model with sub-stepping for Euler numerical stability.

    Repeats each hourly input `substeps` times with dt/substeps,
    then downsamples back to hourly resolution.
    """
    sub_inputs = ZoneInputs(
        T_outside=np.repeat(inputs.T_outside, substeps),
        Q_solar=np.repeat(inputs.Q_solar, substeps),
        Q_internal=np.repeat(inputs.Q_internal, substeps),
        Q_hvac=np.repeat(inputs.Q_hvac, substeps),
        dt=inputs.dt / substeps,
    )
    result = rc_model.simulate(sub_inputs, **kwargs)
    return {k: v[::substeps] for k, v in result.items()}


def build_features(df, T_freeFloat, E_physics, E_actual):
    """Build feature matrix for the correction network.

    Features: time (cyclical), weather, physics outputs, lagged residuals.
    """
    feat = {}

    # Cyclical time
    h_sin, h_cos = encode_cyclical(df.index.hour.values.astype(float), 24.0)
    d_sin, d_cos = encode_cyclical(df.index.dayofweek.values.astype(float), 7.0)
    feat["hour_sin"] = h_sin
    feat["hour_cos"] = h_cos
    feat["dow_sin"] = d_sin
    feat["dow_cos"] = d_cos

    # Weather
    feat["airTemperature"] = df["airTemperature"].values
    feat["dewTemperature"] = df["dewTemperature"].values
    feat["windSpeed"] = df["windSpeed"].values
    feat["cloudCoverage"] = df["cloudCoverage"].values

    # Physics model outputs
    feat["T_freeFloat"] = T_freeFloat
    feat["E_physics"] = E_physics

    # Lagged residuals (autoregressive)
    residual = E_actual - E_physics
    for lag in range(1, N_LAGS + 1):
        lagged = np.roll(residual, lag)
        lagged[:lag] = 0.0
        feat[f"resid_lag_{lag}"] = lagged

    return pd.DataFrame(feat, index=df.index)


def evaluate(y_true, y_pred):
    """Compute regression metrics."""
    r = y_pred - y_true
    ss_res = np.sum(r ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "mae": float(np.mean(np.abs(r))),
        "max_err": float(np.max(np.abs(r))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "mape": float(np.mean(np.abs(r) / np.clip(np.abs(y_true), 1, None)) * 100),
    }


def train_nn(X_tr, y_tr, X_va, y_va, label="model"):
    """Train a correction network with early stopping. Returns best model."""
    net = ResidualCorrectionNet(X_tr.shape[1], HIDDEN_DIMS)
    trainer = CorrectionTrainer(net, lr=LR)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, N_EPOCHS + 1):
        tl = trainer.train_epoch(X_tr, y_tr, BATCH_SIZE)
        vl = trainer.validate(X_va, y_va)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS}: train={tl:.4f} val={vl:.4f}")

        if wait >= PATIENCE:
            print(f"    Early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    net.load_state_dict(best_state)
    return net, best_val


def predict_nn(net, X_tensor):
    """Run inference on a trained network."""
    net.eval()
    with torch.no_grad():
        return net(X_tensor).numpy()


# ──── Main Pipeline ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Building Energy Digital Twin — Training Pipeline")
    print("=" * 60)

    # ── Stage 1: Load Data ─────────────────────────────────────────
    print(f"\n[1/5] Loading data for {BUILDING_ID}...")

    meta = load_metadata(BUILDING_ID)
    sqm = meta["sqm"]
    print(f"  Building: {BUILDING_ID} ({meta['primaryspaceusage']}, {sqm:.0f} m², {SITE_ID} site)")

    df = load_building_data(BUILDING_ID, SITE_ID)
    print(f"  Loaded: {len(df)} hourly rows ({df.index[0]} to {df.index[-1]})")
    print(f"  Electricity: {df['electricity'].min():.0f}–{df['electricity'].max():.0f} kWh "
          f"(mean {df['electricity'].mean():.0f})")
    print(f"  Temperature: {df['airTemperature'].min():.1f} to {df['airTemperature'].max():.1f} °C")

    train_df = df.loc[TRAIN_SLICE[0]:TRAIN_SLICE[1]]
    test_df = df.loc[TEST_SLICE[0]:TEST_SLICE[1]]
    print(f"  2016 mean={df.loc['2016','electricity'].mean():.0f} vs 2017 mean={df.loc['2017','electricity'].mean():.0f} "
          f"(distribution shift → using 2017 only)")
    print(f"  Train: {len(train_df)} hours ({TRAIN_SLICE[0]} to {TRAIN_SLICE[1]})")
    print(f"  Test:  {len(test_df)} hours ({TEST_SLICE[0]} to {TEST_SLICE[1]})")

    # ── Stage 2: Physics Model (free-floating RC) ──────────────────
    print(f"\n[2/5] Running 3R2C model (free-floating, {RC_SUBSTEPS}x sub-steps)...")

    # Scale RC parameters from reference zone (200 m²) to actual building.
    # Thermal resistance ∝ 1/area (parallel heat paths), capacitance ∝ area.
    ar = sqm / RC_REF_AREA
    params = RC3R2CParams(
        R_wall=0.004 / ar, R_win=0.008 / ar, R_int=0.002 / ar,
        C_wall=5.0e6 * ar, C_int=2.0e6 * ar,
    )
    rc_model = RC3R2CModel(params)
    print(f"  Area-scaled RC params (×{ar:.1f}): UA_total={1/params.R_wall + 1/params.R_win:.0f} W/K")

    T_ff = {}
    for label, subset in [("train", train_df), ("test", test_df)]:
        hours = subset.index.hour.values
        dow = subset.index.dayofweek.values
        is_occ = (dow < 5) & (hours >= 7) & (hours < 19)
        Q_int = np.where(is_occ, Q_INT_OCCUPIED * sqm, Q_INT_UNOCCUPIED * sqm)

        inputs = ZoneInputs(
            T_outside=subset["airTemperature"].values,
            Q_solar=estimate_solar_gains(subset, sqm) * SOLAR_SCALE,
            Q_internal=Q_int,
            Q_hvac=np.zeros(len(subset)),
        )
        result = simulate_stable(rc_model, inputs)
        T_ff[label] = np.clip(result["T_int"], -50, 60)  # safety clip
        print(f"  {label}: T_freeFloat {T_ff[label].min():.1f} to {T_ff[label].max():.1f} °C")

    # ── Stage 3: Calibrate Energy Scaling ──────────────────────────
    print(f"\n[3/5] Calibrating energy model (degree-hours → kWh)...")

    E_actual_train = train_df["electricity"].values
    E_actual_test = test_df["electricity"].values

    # Degree-hours from free-floating temperature
    CDH_train = np.clip(T_ff["train"] - T_COOL_SETPOINT, 0, None)
    HDH_train = np.clip(T_HEAT_SETPOINT - T_ff["train"], 0, None)

    # Occupancy flag — offices use 2-3x more electricity when occupied
    is_occ_train = ((train_df.index.dayofweek < 5) &
                    (train_df.index.hour >= 7) &
                    (train_df.index.hour < 19)).astype(float)

    X_phys_train = np.column_stack([
        CDH_train, HDH_train, is_occ_train, 1 - is_occ_train,
    ])

    energy_reg = LinearRegression(fit_intercept=False, positive=True)
    energy_reg.fit(X_phys_train, E_actual_train)
    c_cool, c_heat, c_occ, c_unocc = energy_reg.coef_

    E_phys_train = energy_reg.predict(X_phys_train)
    m_train = evaluate(E_actual_train, E_phys_train)
    print(f"  E = {c_cool:.1f}*CDH + {c_heat:.1f}*HDH + {c_occ:.0f}*occ + {c_unocc:.0f}*unocc")
    print(f"  Train — RMSE: {m_train['rmse']:.1f}, R²: {m_train['r2']:.3f}")

    # Physics prediction on test
    CDH_test = np.clip(T_ff["test"] - T_COOL_SETPOINT, 0, None)
    HDH_test = np.clip(T_HEAT_SETPOINT - T_ff["test"], 0, None)
    is_occ_test = ((test_df.index.dayofweek < 5) &
                   (test_df.index.hour >= 7) &
                   (test_df.index.hour < 19)).astype(float)
    X_phys_test = np.column_stack([
        CDH_test, HDH_test, is_occ_test, 1 - is_occ_test,
    ])
    E_phys_test = energy_reg.predict(X_phys_test)

    # ── Stage 4: Train Correction Network ──────────────────────────
    print(f"\n[4/5] Training correction network...")

    feat_train = build_features(train_df, T_ff["train"], E_phys_train, E_actual_train)
    feat_test = build_features(test_df, T_ff["test"], E_phys_test, E_actual_test)

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(feat_train.values)
    X_test_np = scaler.transform(feat_test.values)

    # Validation split: last 20% of training
    n_val = len(X_train_np) // 5
    X_tr_np, X_va_np = X_train_np[:-n_val], X_train_np[-n_val:]

    # Targets: residuals for hybrid, actual electricity for NN-only
    resid_train = E_actual_train - E_phys_train
    r_tr = torch.tensor(resid_train[:-n_val], dtype=torch.float32)
    r_va = torch.tensor(resid_train[-n_val:], dtype=torch.float32)
    e_tr = torch.tensor(E_actual_train[:-n_val], dtype=torch.float32)
    e_va = torch.tensor(E_actual_train[-n_val:], dtype=torch.float32)

    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    X_va = torch.tensor(X_va_np, dtype=torch.float32)
    X_te = torch.tensor(X_test_np, dtype=torch.float32)

    input_dim = X_tr.shape[1]
    print(f"  Features: {input_dim} dims | Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    print(f"\n  --- Hybrid correction (learns residuals) ---")
    correction_net, _ = train_nn(X_tr, r_tr, X_va, r_va, "correction")

    print(f"\n  --- NN-only baseline (learns electricity directly) ---")
    nn_direct, _ = train_nn(X_tr, e_tr, X_va, e_va, "nn-only")

    # ── Stage 5: Evaluate ──────────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set ({TEST_SLICE[0]}–{TEST_SLICE[1]})...\n")

    E_hybrid_test = E_phys_test + predict_nn(correction_net, X_te)
    E_nn_test = predict_nn(nn_direct, X_te)

    # Schedule baseline: average by hour-of-week from training data
    train_tmp = train_df.copy()
    train_tmp["how"] = train_df.index.dayofweek * 24 + train_df.index.hour
    how_mean = train_tmp.groupby("how")["electricity"].mean()
    test_how = test_df.index.dayofweek * 24 + test_df.index.hour
    E_sched_test = how_mean.reindex(test_how).values

    E_mean_test = np.full_like(E_actual_test, E_actual_train.mean())

    results = {
        "Mean baseline": evaluate(E_actual_test, E_mean_test),
        "Schedule": evaluate(E_actual_test, E_sched_test),
        "Physics-only": evaluate(E_actual_test, E_phys_test),
        "NN-only": evaluate(E_actual_test, E_nn_test),
        "Hybrid (ours)": evaluate(E_actual_test, E_hybrid_test),
    }

    header = f"  {'Model':<16} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R²':>8}"
    sep = f"  {'─' * 16} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}"
    print(header)
    print(sep)
    for name, m in results.items():
        print(f"  {name:<16} {m['rmse']:>8.1f} {m['mae']:>8.1f} {m['mape']:>7.1f}% {m['r2']:>8.3f}")

    phys_rmse = results["Physics-only"]["rmse"]
    hybrid_rmse = results["Hybrid (ours)"]["rmse"]
    improvement = (1 - hybrid_rmse / phys_rmse) * 100
    print(f"\n  Hybrid improves over physics-only by {improvement:.0f}% (RMSE)")

    # ── Save Artifacts ─────────────────────────────────────────────
    print(f"\n  Saving artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "rc_params.json", "w") as f:
        p = rc_model.params
        json.dump(
            {"R_wall": p.R_wall, "R_win": p.R_win, "R_int": p.R_int,
             "C_wall": p.C_wall, "C_int": p.C_int},
            f, indent=2,
        )

    with open(MODELS_DIR / "energy_scaling.json", "w") as f:
        json.dump(
            {"coef_cooling": float(c_cool), "coef_heating": float(c_heat),
             "baseload_occupied_kWh": float(c_occ),
             "baseload_unoccupied_kWh": float(c_unocc),
             "T_heat_setpoint": T_HEAT_SETPOINT, "T_cool_setpoint": T_COOL_SETPOINT},
            f, indent=2,
        )

    torch.save(
        {"model_state": correction_net.state_dict(),
         "scaler_mean": scaler.mean_.tolist(),
         "scaler_scale": scaler.scale_.tolist(),
         "feature_names": list(feat_train.columns),
         "input_dim": input_dim,
         "hidden_dims": HIDDEN_DIMS},
        MODELS_DIR / "correction_net.pt",
    )

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Figures ────────────────────────────────────────────────────

    # 1. Two-week test windows (summer + winter)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    n_test = len(E_actual_test)
    windows = [
        ("Early Oct", 24 * 7, 24 * 14),
        ("Late Nov", min(24 * 50, n_test - 24 * 14), 24 * 14),
    ]
    for ax, (season, start, span) in zip(axes, windows):
        end = min(start + span, len(E_actual_test))
        sl = slice(start, end)
        t = test_df.index[sl]
        ax.plot(t, E_actual_test[sl], "k-", alpha=0.7, lw=0.8, label="Actual")
        ax.plot(t, E_phys_test[sl], "b--", alpha=0.5, lw=0.8, label="Physics-only")
        ax.plot(t, E_hybrid_test[sl], "r-", alpha=0.7, lw=0.8, label="Hybrid")
        ax.set_ylabel("Electricity (kWh)")
        ax.set_title(f"{BUILDING_ID} — {season}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predictions_test.png", dpi=150)
    plt.close()

    # 2. Scatter: predicted vs actual (3 models)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (pred, name) in zip(axes, [
        (E_phys_test, "Physics-only"),
        (E_nn_test, "NN-only"),
        (E_hybrid_test, "Hybrid"),
    ]):
        ax.scatter(E_actual_test, pred, alpha=0.05, s=2, c="steelblue")
        lo = min(E_actual_test.min(), pred.min())
        hi = max(E_actual_test.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        m = evaluate(E_actual_test, pred)
        ax.set_title(f"{name}\nR²={m['r2']:.3f}  RMSE={m['rmse']:.0f}")
        ax.set_xlabel("Actual (kWh)")
        ax.set_ylabel("Predicted (kWh)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scatter_comparison.png", dpi=150)
    plt.close()

    # 3. Average weekly profile
    fig, ax = plt.subplots(figsize=(12, 5))
    tmp = test_df.copy()
    tmp["how"] = tmp.index.dayofweek * 24 + tmp.index.hour
    tmp["actual"] = E_actual_test
    tmp["physics"] = E_phys_test
    tmp["hybrid"] = E_hybrid_test

    for col, label, style in [
        ("actual", "Actual", "k-"),
        ("physics", "Physics", "b--"),
        ("hybrid", "Hybrid", "r-"),
    ]:
        profile = tmp.groupby("how")[col].mean()
        ax.plot(profile.index, profile.values, style, label=label, lw=1.5)

    ax.set_xlabel("Hour of Week")
    ax.set_ylabel("Mean Electricity (kWh)")
    ax.set_title(f"{BUILDING_ID} — Weekly Profile ({TEST_SLICE[0]}–{TEST_SLICE[1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        ax.axvline(d * 24, color="gray", alpha=0.2, lw=0.5)
        ax.text(d * 24 + 12, ax.get_ylim()[1] * 0.97, name,
                ha="center", va="top", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weekly_profile.png", dpi=150)
    plt.close()

    for p in [
        MODELS_DIR / "rc_params.json",
        MODELS_DIR / "energy_scaling.json",
        MODELS_DIR / "correction_net.pt",
        MODELS_DIR / "metrics.json",
        FIGURES_DIR / "predictions_test.png",
        FIGURES_DIR / "scatter_comparison.png",
        FIGURES_DIR / "weekly_profile.png",
    ]:
        print(f"  {p}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
