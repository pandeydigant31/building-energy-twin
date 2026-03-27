"""Building Energy Digital Twin — Interactive Dashboard

Streamlit app with:
  1. Predictions view (actual vs physics vs hybrid)
  2. What-If Scenario Engine (adjust temp, setpoints, schedule → see energy impact)
  3. Model Performance (scatter, errors, weekly profile)

Usage:
    cd building-energy-twin
    streamlit run dashboards/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.physics.rc_model import RC3R2CModel, RC3R2CParams, ZoneInputs
from src.ml.correction import ResidualCorrectionNet, encode_cyclical
from src.pipeline.data_loader import (
    load_building_data, load_metadata, estimate_solar_gains,
)

# ── Constants (must match train.py) ───────────────────────────
BUILDING_ID = "Hog_office_Gustavo"
SITE_ID = "Hog"
RC_REF_AREA = 200.0
RC_SUBSTEPS = 10
Q_INT_OCC = 15.0   # W/m²
Q_INT_UNOCC = 3.0  # W/m²
SOLAR_SCALE = 0.3
N_LAGS = 3
ELECTRICITY_RATE = 0.12  # $/kWh (US average)

SCHEDULES = {
    "Standard (M-F 7am-7pm)": lambda dow, h: (dow < 5) & (h >= 7) & (h < 19),
    "4-Day Week (M-Th 7am-7pm)": lambda dow, h: (dow < 4) & (h >= 7) & (h < 19),
    "Extended (M-F 6am-10pm)": lambda dow, h: (dow < 5) & (h >= 6) & (h < 22),
    "24/7 Operations": lambda dow, h: np.ones_like(dow, dtype=bool),
    "Reduced (M-F 8am-5pm)": lambda dow, h: (dow < 5) & (h >= 8) & (h < 17),
}


# ── Cached loaders ────────────────────────────────────────────

@st.cache_data
def get_data():
    # Try loading from raw BDG2 data; fall back to bundled parquet (Streamlit Cloud)
    demo_parquet = PROJECT_ROOT / "outputs/demo_data_2017.parquet"
    raw_data_exists = (PROJECT_ROOT / "data/raw/bdg2").exists()

    if raw_data_exists:
        meta = load_metadata(BUILDING_ID)
        df = load_building_data(BUILDING_ID, SITE_ID)
        df = df.loc["2017"]
    elif demo_parquet.exists():
        df = pd.read_parquet(demo_parquet)
        meta = {"primaryspaceusage": "Office", "sqm": 6582.4}
    else:
        st.error("No data found. Run `python train.py` first or ensure demo_data_2017.parquet exists.")
        st.stop()
    return df, meta


@st.cache_data
def get_energy_params():
    with open(PROJECT_ROOT / "outputs/models/energy_scaling.json") as f:
        return json.load(f)


@st.cache_data
def get_metrics():
    with open(PROJECT_ROOT / "outputs/models/metrics.json") as f:
        return json.load(f)


@st.cache_resource
def get_rc_model(sqm):
    ar = sqm / RC_REF_AREA
    params = RC3R2CParams(
        R_wall=0.004 / ar, R_win=0.008 / ar, R_int=0.002 / ar,
        C_wall=5.0e6 * ar, C_int=2.0e6 * ar,
    )
    return RC3R2CModel(params)


@st.cache_resource
def get_correction_net():
    ckpt = torch.load(
        PROJECT_ROOT / "outputs/models/correction_net.pt", weights_only=False
    )
    net = ResidualCorrectionNet(ckpt["input_dim"], ckpt["hidden_dims"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, np.array(ckpt["scaler_mean"]), np.array(ckpt["scaler_scale"])


# ── Prediction engine ─────────────────────────────────────────

def _simulate_rc(rc_model, inputs):
    sub = ZoneInputs(
        T_outside=np.repeat(inputs.T_outside, RC_SUBSTEPS),
        Q_solar=np.repeat(inputs.Q_solar, RC_SUBSTEPS),
        Q_internal=np.repeat(inputs.Q_internal, RC_SUBSTEPS),
        Q_hvac=np.repeat(inputs.Q_hvac, RC_SUBSTEPS),
        dt=inputs.dt / RC_SUBSTEPS,
    )
    r = rc_model.simulate(sub)
    return np.clip(r["T_int"][::RC_SUBSTEPS], -50, 60)


def predict(df, rc_model, ep, net, s_mean, s_scale, sqm,
            temp_offset=0.0, heat_sp=None, cool_sp=None,
            schedule_fn=None, actual=None):
    """Run hybrid prediction. If `actual` is provided, uses real lagged residuals."""
    heat_sp = heat_sp or ep["T_heat_setpoint"]
    cool_sp = cool_sp or ep["T_cool_setpoint"]
    schedule_fn = schedule_fn or SCHEDULES["Standard (M-F 7am-7pm)"]

    mod = df.copy()
    mod["airTemperature"] = mod["airTemperature"] + temp_offset

    hours = mod.index.hour.values
    dow = mod.index.dayofweek.values
    is_occ = schedule_fn(dow, hours).astype(float)

    Q_int = np.where(is_occ > 0.5, Q_INT_OCC * sqm, Q_INT_UNOCC * sqm)
    inputs = ZoneInputs(
        T_outside=mod["airTemperature"].values,
        Q_solar=estimate_solar_gains(mod, sqm) * SOLAR_SCALE,
        Q_internal=Q_int,
        Q_hvac=np.zeros(len(mod)),
    )
    T_ff = _simulate_rc(rc_model, inputs)

    CDH = np.clip(T_ff - cool_sp, 0, None)
    HDH = np.clip(heat_sp - T_ff, 0, None)
    E_phys = (ep["coef_cooling"] * CDH + ep["coef_heating"] * HDH
              + ep["baseload_occupied_kWh"] * is_occ
              + ep["baseload_unoccupied_kWh"] * (1 - is_occ))

    # Build correction features
    feat = np.column_stack([
        *encode_cyclical(hours.astype(float), 24.0),
        *encode_cyclical(dow.astype(float), 7.0),
        mod["airTemperature"].values,
        mod["dewTemperature"].values,
        mod["windSpeed"].values,
        mod["cloudCoverage"].values,
        T_ff,
        E_phys,
        *[np.zeros(len(mod)) for _ in range(N_LAGS)],  # default: zero lags
    ])

    # If we have actuals, compute real lagged residuals
    if actual is not None:
        resid = actual - E_phys
        for lag in range(1, N_LAGS + 1):
            col_idx = 10 + lag - 1  # after the 10 main features
            lagged = np.roll(resid, lag)
            lagged[:lag] = 0.0
            feat[:, col_idx] = lagged

    X = (feat - s_mean) / s_scale
    with torch.no_grad():
        corr = net(torch.tensor(X, dtype=torch.float32)).numpy()

    return {
        "T_ff": T_ff, "E_physics": E_phys, "E_hybrid": E_phys + corr,
        "CDH": CDH, "HDH": HDH, "is_occ": is_occ,
    }


# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Building Energy Digital Twin",
    page_icon="⚡",
    layout="wide",
)

# Load everything
df, meta = get_data()
sqm = meta["sqm"]
ep = get_energy_params()
metrics = get_metrics()
rc_model = get_rc_model(sqm)
net, s_mean, s_scale = get_correction_net()

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.title("Building Digital Twin")
    st.markdown(f"""
    **{BUILDING_ID}**
    - Type: {meta['primaryspaceusage']}
    - Area: {sqm:,.0f} m²
    - Site: {SITE_ID} (Minnesota)
    - Data: 2017 (hourly)
    """)

    st.divider()
    st.subheader("Hybrid Model")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{metrics['Hybrid (ours)']['rmse']:.1f} kWh")
    col2.metric("R²", f"{metrics['Hybrid (ours)']['r2']:.3f}")
    col1.metric("MAE", f"{metrics['Hybrid (ours)']['mae']:.1f} kWh")
    col2.metric("MAPE", f"{metrics['Hybrid (ours)']['mape']:.1f}%")

    st.divider()
    st.caption("Architecture: RC thermal model + neural correction network")


# ── Tabs ──────────────────────────────────────────────────────

tab_pred, tab_whatif, tab_perf = st.tabs([
    "📈 Predictions", "🔮 What-If Scenarios", "📊 Model Performance",
])

# ━━ Tab 1: Predictions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_pred:
    st.header("Predictions vs Actual")

    months = pd.date_range("2017-01-01", "2017-12-01", freq="MS")
    month_labels = [m.strftime("%b %Y") for m in months]
    col1, col2 = st.columns(2)
    start_month = col1.selectbox("From", month_labels, index=0)
    end_month = col2.selectbox("To", month_labels, index=len(month_labels) - 1)

    start_dt = months[month_labels.index(start_month)]
    end_dt = months[month_labels.index(end_month)] + pd.offsets.MonthEnd(1)
    view = df.loc[start_dt:end_dt]

    if len(view) == 0:
        st.warning("No data in selected range.")
    else:
        actual = view["electricity"].values
        res = predict(view, rc_model, ep, net, s_mean, s_scale, sqm, actual=actual)

        show_physics = st.checkbox("Show physics-only", value=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=view.index, y=actual, name="Actual",
            line=dict(color="#333", width=1), opacity=0.8,
        ))
        if show_physics:
            fig.add_trace(go.Scatter(
                x=view.index, y=res["E_physics"], name="Physics-only",
                line=dict(color="#4A90D9", width=1, dash="dot"), opacity=0.6,
            ))
        fig.add_trace(go.Scatter(
            x=view.index, y=res["E_hybrid"], name="Hybrid",
            line=dict(color="#E74C3C", width=1), opacity=0.8,
        ))
        fig.update_layout(
            height=450, xaxis_title="Time", yaxis_title="Electricity (kWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=50, r=20, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Residual
        residual = res["E_hybrid"] - actual
        r_rmse = np.sqrt(np.mean(residual ** 2))
        r_r2 = 1 - np.sum(residual ** 2) / np.sum((actual - actual.mean()) ** 2)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{r_rmse:.1f} kWh")
        c2.metric("R²", f"{r_r2:.3f}")
        c3.metric("Mean Error", f"{np.mean(residual):.1f} kWh")
        c4.metric("Max |Error|", f"{np.max(np.abs(residual)):.0f} kWh")


# ━━ Tab 2: What-If Scenarios ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_whatif:
    st.header("What-If Scenario Engine")
    st.markdown(
        "Adjust building parameters and see the predicted energy impact. "
        "The **physics model** enables physically-grounded extrapolation — "
        "something a pure ML model cannot do."
    )

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Scenario Controls")

        temp_offset = st.slider(
            "Temperature offset (°C)",
            min_value=-10.0, max_value=10.0, value=0.0, step=0.5,
            help="Simulate climate change or weather extremes",
        )
        cool_sp = st.slider(
            "Cooling setpoint (°C)",
            min_value=22.0, max_value=28.0,
            value=float(ep["T_cool_setpoint"]), step=0.5,
        )
        heat_sp = st.slider(
            "Heating setpoint (°C)",
            min_value=16.0, max_value=24.0,
            value=float(ep["T_heat_setpoint"]), step=0.5,
        )
        schedule_name = st.selectbox("Occupancy schedule", list(SCHEDULES.keys()))

        analysis_period = st.selectbox(
            "Analysis period",
            ["Full Year", "Jan-Mar (Winter)", "Apr-Jun (Spring)",
             "Jul-Sep (Summer)", "Oct-Dec (Fall)"],
        )

    # Determine date range
    period_map = {
        "Full Year": ("2017-01", "2017-12"),
        "Jan-Mar (Winter)": ("2017-01", "2017-03"),
        "Apr-Jun (Spring)": ("2017-04", "2017-06"),
        "Jul-Sep (Summer)": ("2017-07", "2017-09"),
        "Oct-Dec (Fall)": ("2017-10", "2017-12"),
    }
    p_start, p_end = period_map[analysis_period]
    period_df = df.loc[p_start:p_end]

    # Baseline prediction (standard conditions)
    baseline = predict(period_df, rc_model, ep, net, s_mean, s_scale, sqm)

    # Scenario prediction
    scenario = predict(
        period_df, rc_model, ep, net, s_mean, s_scale, sqm,
        temp_offset=temp_offset, heat_sp=heat_sp, cool_sp=cool_sp,
        schedule_fn=SCHEDULES[schedule_name],
    )

    with col_result:
        # Summary metrics
        base_total = baseline["E_hybrid"].sum()
        scen_total = scenario["E_hybrid"].sum()
        delta = scen_total - base_total
        delta_pct = (delta / base_total) * 100 if base_total > 0 else 0
        cost_delta = delta * ELECTRICITY_RATE

        st.subheader("Impact Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Energy Change",
            f"{delta:+,.0f} kWh",
            delta=f"{delta_pct:+.1f}%",
            delta_color="inverse",
        )
        m2.metric(
            "Cost Impact",
            f"${cost_delta:+,.0f}",
            delta=f"@ ${ELECTRICITY_RATE}/kWh",
            delta_color="inverse",
        )
        m3.metric(
            "Avg Hourly Change",
            f"{delta / len(period_df):+.1f} kWh/hr",
        )

        # Weekly profile comparison
        st.subheader("Weekly Profile: Baseline vs Scenario")
        base_df = pd.DataFrame({
            "how": period_df.index.dayofweek * 24 + period_df.index.hour,
            "Baseline": baseline["E_hybrid"],
            "Scenario": scenario["E_hybrid"],
        })
        profile = base_df.groupby("how").mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=profile.index, y=profile["Baseline"], name="Baseline",
            line=dict(color="#333", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=profile.index, y=profile["Scenario"], name="Scenario",
            line=dict(color="#E74C3C", width=2),
        ))
        # Day labels
        for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            fig.add_vline(x=d * 24, line_dash="dot", line_color="gray", opacity=0.3)
            fig.add_annotation(x=d * 24 + 12, y=1.02, yref="paper",
                               text=name, showarrow=False, font=dict(size=10, color="gray"))

        fig.update_layout(
            height=350, xaxis_title="Hour of Week", yaxis_title="Electricity (kWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.05),
            margin=dict(l=50, r=20, t=40, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Energy breakdown
        st.subheader("Thermal Load Breakdown")
        bc1, bc2 = st.columns(2)

        for col, data, label in [(bc1, baseline, "Baseline"), (bc2, scenario, "Scenario")]:
            total_cdh = data["CDH"].sum()
            total_hdh = data["HDH"].sum()
            occ_hours = data["is_occ"].sum()
            unocc_hours = len(data["is_occ"]) - occ_hours
            col.markdown(f"**{label}**")
            col.markdown(f"- Cooling degree-hours: {total_cdh:,.0f}")
            col.markdown(f"- Heating degree-hours: {total_hdh:,.0f}")
            col.markdown(f"- Occupied hours: {occ_hours:,.0f}")
            col.markdown(f"- Total energy: {data['E_hybrid'].sum():,.0f} kWh")


# ━━ Tab 3: Model Performance ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_perf:
    st.header("Model Performance Comparison")

    # Metrics table
    st.subheader("Test Set Metrics (Oct-Dec 2017)")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "RMSE (kWh)": f"{m['rmse']:.1f}",
            "MAE (kWh)": f"{m['mae']:.1f}",
            "MAPE (%)": f"{m['mape']:.1f}",
            "R²": f"{m['r2']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # Scatter plots
    st.subheader("Predicted vs Actual")

    test_df = df.loc["2017-10":"2017-12"]
    actual_test = test_df["electricity"].values
    res_test = predict(test_df, rc_model, ep, net, s_mean, s_scale, sqm, actual=actual_test)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Physics-only", "Hybrid"])
    for col_idx, (pred, name) in enumerate([
        (res_test["E_physics"], "Physics"),
        (res_test["E_hybrid"], "Hybrid"),
    ], 1):
        fig.add_trace(go.Scatter(
            x=actual_test, y=pred, mode="markers",
            marker=dict(size=3, color="steelblue", opacity=0.3),
            name=name, showlegend=False,
        ), row=1, col=col_idx)
        lo = min(actual_test.min(), pred.min())
        hi = max(actual_test.max(), pred.max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="red", dash="dash", width=1),
            showlegend=False,
        ), row=1, col=col_idx)
        fig.update_xaxes(title_text="Actual (kWh)", row=1, col=col_idx)
        fig.update_yaxes(title_text="Predicted (kWh)", row=1, col=col_idx)

    fig.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=50))
    st.plotly_chart(fig, use_container_width=True)

    # Error distribution
    st.subheader("Error Distribution (Hybrid)")
    errors = res_test["E_hybrid"] - actual_test
    fig = go.Figure(go.Histogram(
        x=errors, nbinsx=60, marker_color="steelblue", opacity=0.7,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(
        height=300, xaxis_title="Prediction Error (kWh)", yaxis_title="Count",
        margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weekly profile
    st.subheader("Average Weekly Profile")
    tmp = test_df.copy()
    tmp["how"] = tmp.index.dayofweek * 24 + tmp.index.hour
    tmp["Actual"] = actual_test
    tmp["Physics"] = res_test["E_physics"]
    tmp["Hybrid"] = res_test["E_hybrid"]
    profile = tmp.groupby("how")[["Actual", "Physics", "Hybrid"]].mean()

    fig = go.Figure()
    for col_name, color in [("Actual", "#333"), ("Physics", "#4A90D9"), ("Hybrid", "#E74C3C")]:
        fig.add_trace(go.Scatter(
            x=profile.index, y=profile[col_name], name=col_name,
            line=dict(color=color, width=2, dash="dot" if col_name == "Physics" else "solid"),
        ))
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        fig.add_vline(x=d * 24, line_dash="dot", line_color="gray", opacity=0.3)
        fig.add_annotation(x=d * 24 + 12, y=1.02, yref="paper",
                           text=name, showarrow=False, font=dict(size=10, color="gray"))
    fig.update_layout(
        height=350, xaxis_title="Hour of Week", yaxis_title="Mean Electricity (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)
