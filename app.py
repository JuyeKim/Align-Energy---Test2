import time
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import random

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Model / PoC Logic (Single-file)
# -----------------------------

@dataclass
class Measurement:
    ts: datetime
    v_pu: float
    kw: float


def safe_voltage_band(history: list[Measurement], abs_min=0.95, abs_max=1.02):
    """Compute a conservative safe band using mean ± 2*std, then clamp."""
    if len(history) < 10:
        return abs_min, min(1.0, abs_max)

    v = np.array([m.v_pu for m in history], dtype=float)
    v_mean = float(v.mean())
    v_std = float(v.std())

    v_min = max(abs_min, v_mean - 2.0 * v_std)
    v_max = min(abs_max, v_mean + 2.0 * v_std)

    if v_min >= v_max:
        v_min, v_max = abs_min, abs_max

    return float(v_min), float(v_max)


def voltage_sensitivity_index(history: list[Measurement]):
    """
    VSI = ΔkW / ΔV(%)
    Lower VSI => more CVR headroom (site less sensitive to voltage).
    """
    if len(history) < 20:
        return None

    v = np.array([m.v_pu for m in history], dtype=float)
    p = np.array([m.kw for m in history], dtype=float)

    dv_pct = (float(v.max()) - float(v.min())) * 100.0
    dp = float(p.max()) - float(p.min())

    if dv_pct < 1e-6:
        return None
    return dp / dv_pct


def forecast_kw_ewma(history: list[Measurement], alpha=0.25):
    """Simple EWMA forecast for short horizon (5~15 min proxy)."""
    if len(history) < 5:
        return None
    ewma = float(history[0].kw)
    for m in history[1:]:
        ewma = alpha * float(m.kw) + (1 - alpha) * ewma
    return ewma


def decide_target_voltage(
    current: Measurement,
    history: list[Measurement],
    step_pu: float,
    abs_v_min: float,
    abs_v_max: float,
    vsi_threshold: float,
    forecast_rise_pct: float,
):
    safe_min, safe_max = safe_voltage_band(history, abs_min=abs_v_min, abs_max=abs_v_max)
    vsi = voltage_sensitivity_index(history)
    fkw = forecast_kw_ewma(history)

    # Stability check
    stable = True
    if fkw is not None:
        allowed = current.kw * (1.0 + forecast_rise_pct / 100.0)
        stable = fkw <= allowed

    # If we cannot compute VSI yet, be conservative: return to nominal
    if vsi is None:
        nominal = min(1.0, safe_max)
        target = max(safe_min, min(nominal, current.v_pu))
        action = "monitor_only"
        return target, safe_min, safe_max, vsi, fkw, action

    can_ramp_down = (
        (vsi < vsi_threshold)
        and stable
        and (current.v_pu - step_pu >= safe_min)
    )

    if can_ramp_down:
        target = current.v_pu - step_pu
        action = "ramp_down"
    else:
        # gently return toward nominal without overshooting
        nominal = min(1.0, safe_max)
        if current.v_pu < nominal:
            target = min(nominal, current.v_pu + step_pu)
        else:
            target = max(nominal, current.v_pu - step_pu)
        action = "return_nominal"

    target = float(round(target, 4))
    return target, float(round(safe_min, 4)), float(round(safe_max, 4)), vsi, fkw, action


def simulate_measurement(prev: Measurement | None):
    """A simple demo simulator: load varies, voltage reacts slightly."""
    now = datetime.now()
    base_kw = 12500.0
    if prev is None:
        kw = base_kw + random.uniform(-400, 400)
        v = random.uniform(0.975, 1.005)
        return Measurement(ts=now, v_pu=float(round(v, 4)), kw=float(round(kw, 1)))

    # random walk load
    kw = prev.kw + random.uniform(-180, 180)
    kw = max(8000, min(20000, kw))

    # voltage tends to drop a bit when load increases (very rough)
    dv = -(kw - prev.kw) / 250000.0 + random.uniform(-0.0015, 0.0015)
    v = prev.v_pu + dv
    v = max(0.94, min(1.03, v))

    return Measurement(ts=now, v_pu=float(round(v, 4)), kw=float(round(kw, 1)))


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="CVR PoC Demo (TEST2)", layout="wide")
st.title("CVR PoC Demo (TEST2)")
st.caption("Real-time style monitoring • Safe Band • VSI • Forecast • Ramp Control (Simulation)")

with st.sidebar:
    st.header("Control Settings (PoC-safe)")
    step_pu = st.slider("Ramp step (pu)", 0.0005, 0.0100, 0.0020, 0.0005)
    abs_v_min = st.slider("Absolute min voltage (pu)", 0.90, 0.99, 0.95, 0.01)
    abs_v_max = st.slider("Absolute max voltage (pu)", 1.00, 1.05, 1.02, 0.01)
    vsi_th = st.slider("VSI threshold", 0.5, 20.0, 5.0, 0.5)
    forecast_rise_pct = st.slider("Allowable forecast load rise (%)", 0.0, 10.0, 2.0, 0.5)

    st.divider()
    st.header("Run")
    auto = st.toggle("Auto-run (1s)", value=True)
    step_btn = st.button("Step once")
    reset_btn = st.button("Reset data")

# state
if "history" not in st.session_state or reset_btn:
    st.session_state.history = deque(maxlen=300)
    st.session_state.log = deque(maxlen=200)

# one tick per script run (prevents infinite loop issues)
do_tick = False
if step_btn:
    do_tick = True
elif auto:
    do_tick = True

if do_tick:
    prev = st.session_state.history[-1] if len(st.session_state.history) else None
    m = simulate_measurement(prev)
    st.session_state.history.append(m)

    hist_list = list(st.session_state.history)
    target, safe_min, safe_max, vsi, fkw, action = decide_target_voltage(
        current=m,
        history=hist_list,
        step_pu=float(step_pu),
        abs_v_min=float(abs_v_min),
        abs_v_max=float(abs_v_max),
        vsi_threshold=float(vsi_th),
        forecast_rise_pct=float(forecast_rise_pct),
    )

    st.session_state.log.append({
        "Time": m.ts.strftime("%H:%M:%S"),
        "Voltage(pu)": m.v_pu,
        "Target(pu)": target,
        "kW": m.kw,
        "SafeMin(pu)": safe_min,
        "SafeMax(pu)": safe_max,
        "VSI": None if vsi is None else float(round(vsi, 2)),
        "Forecast(kW)": None if fkw is None else float(round(fkw, 1)),
        "Action": action
    })

df = pd.DataFrame(list(st.session_state.log))

# KPIs
top1, top2, top3, top4, top5 = st.columns(5)
if len(df) > 0:
    last = df.iloc[-1]
    top1.metric("Current Voltage (pu)", f"{last['Voltage(pu)']:.4f}")
    top2.metric("Target Voltage (pu)", f"{last['Target(pu)']:.4f}")
    top3.metric("Load (kW)", f"{last['kW']:.0f}")
    top4.metric("VSI", "-" if pd.isna(last["VSI"]) else f"{last['VSI']:.2f}")
    top5.metric("Safe Band (pu)", f"{last['SafeMin(pu)']:.4f} ~ {last['SafeMax(pu)']:.4f}")
else:
    top1.metric("Current Voltage (pu)", "-")
    top2.metric("Target Voltage (pu)", "-")
    top3.metric("Load (kW)", "-")
    top4.metric("VSI", "-")
    top5.metric("Safe Band (pu)", "-")

left, right = st.columns([2, 1])

with left:
    st.subheader("Voltage vs Target")
    if len(df) >= 2:
        chart_df = df[["Time", "Voltage(pu)", "Target(pu)"]].set_index("Time")
        st.line_chart(chart_df, height=360)
    st.info("PoC-safe behavior: ramp-based control • conservative safe band • no aggressive jumps")

with right:
    st.subheader("Operational Log (latest 25)")
    st.dataframe(df.tail(25), use_container_width=True, height=420)

st.caption("Demo mode: simulation only. For real Modbus/AVR control, we can add a separate 'Live' branch after the demo is stable.")

# auto rerun (one tick per run)
if auto:
    time.sleep(1)
    st.rerun()
