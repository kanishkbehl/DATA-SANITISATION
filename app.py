# app.py — Streamlit Telemetry Visualizer & Exporter
# Run with:  streamlit run app.py

import io
import json
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------- Constants --------------------
CANDIDATE_TS = ["timestamp", "time", "datetime", "date_time", "ts", "logged_at"]
CANDIDATE_SPEED = ["speed", "vehicle_speed", "veh_speed", "gps_speed", "spd"]
CANDIDATE_ALT = ["altitude", "elevation", "gps_alt", "alt", "elev"]
CANDIDATE_PACK_V = ["batteryVoltage", "packVoltage", "voltage", "busVoltage", "capacitorVoltage"]
G = 9.80665  # m/s^2

# -------------------- Utilities --------------------

def read_any(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(upload)
    elif name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(upload)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    # exact match first
    for name in candidates:
        if name in cols:
            return cols[name]
    # substring match
    for name in candidates:
        for c in cols:
            if name in c:
                return cols[c]
    return None


def to_datetime_safe(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt


def to_seconds(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts, errors="coerce")
    t0 = t.iloc[0]
    return (t - t0).dt.total_seconds().to_numpy()


def seconds_between(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts, errors="coerce")
    dt = (t - t.iloc[0]).dt.total_seconds().fillna(0).to_numpy()
    ddt = np.diff(dt, prepend=dt[0])
    ddt = np.where(ddt <= 0, np.nan, ddt)
    if np.isnan(ddt[0]):
        pos = ddt[1:][~np.isnan(ddt[1:]) & (ddt[1:] > 0)]
        ddt[0] = pos.min() if pos.size else 0.1
    return ddt


def safe_num(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    return pd.to_numeric(s, errors="coerce")

# -------------------- Core processing --------------------

def sanitize(df: pd.DataFrame, timestamp_col: Optional[str], speed_col: Optional[str], min_speed: float):
    if timestamp_col is None:
        timestamp_col = find_col(df, CANDIDATE_TS)
    if speed_col is None:
        speed_col = find_col(df, CANDIDATE_SPEED)
    if timestamp_col is None:
        raise ValueError("Could not detect a timestamp column. Use the sidebar to set it.")
    if speed_col is None:
        raise ValueError("Could not detect a speed column. Use the sidebar to set it.")

    df = df.copy()
    df[timestamp_col] = to_datetime_safe(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col])
    df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce")
    df = df.dropna(subset=[speed_col])

    df_active = df[df[speed_col] > min_speed].copy()
    df_active = df_active.sort_values(timestamp_col)
    return df_active, timestamp_col, speed_col


def add_altitude_and_energy(
    df: pd.DataFrame,
    timestamp_col: str,
    speed_col: str,
    mass_kg: float = 1500.0,
    altitude_col: Optional[str] = None,
    nominal_voltage: Optional[float] = None,
) -> pd.DataFrame:
    """Add distance, electrical energy, and (if available) altitude-derived metrics."""
    df = df.copy()

    # Time step
    tsec = to_seconds(df[timestamp_col])
    dt = np.diff(tsec, prepend=tsec[0])
    dt[dt <= 0] = np.nan
    if np.isnan(dt[0]):
        mins = dt[1:][dt[1:] > 0]
        dt[0] = mins.min() if mins.size else 0.1

    # Distance from speed (auto units)
    spd = pd.to_numeric(df[speed_col], errors="coerce").to_numpy()
    p95 = np.nanpercentile(spd, 95)
    spd_mps = spd if p95 < 40 else spd / 3.6  # if mostly <40, assume m/s; else km/h
    dist_m = np.cumsum(np.nan_to_num(spd_mps * dt, nan=0.0))
    df["distance_m"] = dist_m
    df["distance_km"] = dist_m / 1000.0

    # Electrical features
    def _add_electrical(df_inner: pd.DataFrame):
        pack_v_col = find_col(df_inner, CANDIDATE_PACK_V)
        if "batteryCurrent" in df_inner.columns:
            i = pd.to_numeric(df_inner["batteryCurrent"], errors="coerce").to_numpy()
            # Auto-fix current sign: if median < 0, flip so discharge is positive
            if np.nanmedian(i) < 0:
                i = -i
            if pack_v_col is not None:
                v = pd.to_numeric(df_inner[pack_v_col], errors="coerce").to_numpy()
            elif nominal_voltage is not None:
                v = np.full_like(i, nominal_voltage, dtype=float)
            else:
                v = None
            if v is not None:
                elec_power_w = v * i
                df_inner["elec_power_w"] = elec_power_w
                df_inner["elec_energy_wh"] = np.cumsum(
                    np.nan_to_num(elec_power_w * dt / 3600.0, nan=0.0)
                )
                # Negative power is regen
                df_inner["elec_energy_recov_wh"] = np.cumsum(
                    np.nan_to_num(np.where(elec_power_w < 0, elec_power_w, 0.0) * dt / 3600.0, nan=0.0)
                )
        return df_inner

    df = _add_electrical(df)
    if "elec_energy_wh" in df.columns:
        dist_km = np.maximum(df["distance_km"].to_numpy(), 1e-6)
        df["wh_per_km"] = df["elec_energy_wh"] / dist_km

    # Altitude features (optional)
    if altitude_col is None:
        altitude_col = find_col(df, CANDIDATE_ALT)

    if altitude_col is not None and altitude_col in df.columns:
        alt = pd.to_numeric(df[altitude_col], errors="coerce").to_numpy()
        alt_sm = (
            pd.Series(alt)
            .interpolate(limit_direction="both")
            .rolling(5, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        df["altitude_smoothed_m"] = alt_sm

        dalt = np.gradient(alt_sm)
        ddist = np.gradient(dist_m)
        grade = np.where(np.abs(ddist) > 0.5, (dalt / ddist) * 100.0, 0.0)
        grade = pd.Series(grade).rolling(5, min_periods=1, center=True).mean().to_numpy()
        df["grade_pct"] = grade

        with np.errstate(divide="ignore", invalid="ignore"):
            vvert = np.gradient(alt_sm) / np.gradient(tsec)
        vvert = (
            pd.Series(vvert)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .rolling(3, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        df["vertical_speed_mps"] = vvert

        grav_power_w = mass_kg * G * vvert
        df["grav_power_w"] = grav_power_w
        df["grav_energy_wh"] = np.cumsum(np.nan_to_num(grav_power_w * dt / 3600.0, nan=0.0))
        df["elev_gain_m"] = np.cumsum(np.maximum(np.gradient(alt_sm), 0.0))

        if "soc" in df.columns:
            soc = pd.to_numeric(df["soc"], errors="coerce").to_numpy()
            elev_gain_cum = np.maximum.accumulate(pd.Series(df["elev_gain_m"]).fillna(0.0).to_numpy())
            elev_gain_100m = np.maximum(elev_gain_cum, 1e-6) / 100.0
            df["soc_drop_per_100m"] = (np.nanmax(soc) - soc) / elev_gain_100m

    return df


def add_basic_metrics(df: pd.DataFrame, timestamp_col: str, speed_col: str) -> pd.DataFrame:
    out = df.copy()
    ddt = seconds_between(out[timestamp_col])
    out["dt_s"] = ddt

    spd = safe_num(out[speed_col]).to_numpy()
    p95 = np.nanpercentile(spd, 95)
    spd_mps = spd if p95 < 40 else spd / 3.6
    out["speed_mps"] = spd_mps
    out["is_driving"] = np.where(spd_mps > 0.3, 1, 0)

    if "elec_energy_wh" in out.columns:
        if "elec_energy_recov_wh" in out.columns:
            out["net_energy_wh"] = out["elec_energy_wh"] + out["elec_energy_recov_wh"]
        else:
            out["net_energy_wh"] = out["elec_energy_wh"]
    else:
        if "elec_power_w" in out.columns:
            out["net_energy_wh"] = np.cumsum(
                np.nan_to_num(safe_num(out["elec_power_w"]) * out["dt_s"] / 3600.0, nan=0.0)
            )

    if "wh_per_km" not in out.columns and "net_energy_wh" in out.columns and "distance_km" in out.columns:
        dist_km = np.maximum(out["distance_km"].to_numpy(), 1e-6)
        out["wh_per_km"] = out["net_energy_wh"] / dist_km

    return out


def summarize_run(df: pd.DataFrame) -> dict:
    summary = {}

    total_time_s = float(np.nansum(df.get("dt_s", pd.Series(dtype=float)).to_numpy())) if "dt_s" in df.columns else None
    driving_time_s = float(np.nansum((df.get("is_driving", 0) * df.get("dt_s", 0)).to_numpy())) if "is_driving" in df.columns else None
    idle_time_s = (total_time_s - driving_time_s) if (total_time_s is not None and driving_time_s is not None) else None

    summary["time_total_s"] = total_time_s
    summary["time_driving_s"] = driving_time_s
    summary["time_idle_s"] = idle_time_s

    summary["distance_km"] = float(np.nanmax(df["distance_km"])) if "distance_km" in df.columns and df["distance_km"].size else None

    if "speed" in df.columns:
        spd = safe_num(df["speed"]).to_numpy()
        summary["speed_avg_kmh"] = float(np.nanmean(spd))
        summary["speed_p95_kmh"] = float(np.nanpercentile(spd, 95))
        summary["speed_max_kmh"] = float(np.nanmax(spd))
    elif "speed_mps" in df.columns:
        spd_kmh = df["speed_mps"] * 3.6
        summary["speed_avg_kmh"] = float(np.nanmean(spd_kmh))
        summary["speed_p95_kmh"] = float(np.nanpercentile(spd_kmh, 95))
        summary["speed_max_kmh"] = float(np.nanmax(spd_kmh))
    else:
        summary["speed_avg_kmh"] = summary["speed_p95_kmh"] = summary["speed_max_kmh"] = None

    used = df["elec_energy_wh"].to_numpy()[-1] if "elec_energy_wh" in df.columns and df["elec_energy_wh"].size else None
    recov = df["elec_energy_recov_wh"].to_numpy()[-1] if "elec_energy_recov_wh" in df.columns and df["elec_energy_recov_wh"].size else None
    net = df["net_energy_wh"].to_numpy()[-1] if "net_energy_wh" in df.columns and df["net_energy_wh"].size else (used + (recov or 0) if used is not None else None)

    summary["energy_used_Wh"] = float(used) if used is not None else None
    summary["energy_recovered_Wh"] = float(recov) if recov is not None else None
    summary["energy_net_Wh"] = float(net) if net is not None else None
    summary["regen_ratio"] = (float(abs(recov) / used) if (used is not None and recov is not None and used > 1e-6) else None)

    if "wh_per_km" in df.columns and df["wh_per_km"].size:
        wpk = safe_num(df["wh_per_km"]).to_numpy()
        summary["wh_per_km_end"] = float(wpk[-1])
        summary["wh_per_km_median"] = float(np.nanmedian(wpk))
    else:
        summary["wh_per_km_end"] = summary["wh_per_km_median"] = None

    if "soc" in df.columns and df["soc"].size:
        soc = safe_num(df["soc"]).to_numpy()
        soc_drop_pct = np.nanmax(soc) - np.nanmin(soc)
        summary["soc_start_pct"] = float(soc[0])
        summary["soc_end_pct"] = float(soc[-1])
        summary["soc_drop_pct"] = float(soc_drop_pct)
        summary["est_usable_capacity_Wh"] = (float(net / (soc_drop_pct / 100.0)) if (net is not None and soc_drop_pct > 0.1) else None)
    else:
        summary["soc_start_pct"] = summary["soc_end_pct"] = summary["soc_drop_pct"] = summary["est_usable_capacity_Wh"] = None

    for col in ["controllerTemperature", "motorTemperature", "maxCellTemp", "minCellTemp", "maxCellVoltage", "minCellVoltage"]:
        if col in df.columns and df[col].size:
            series = safe_num(df[col]).to_numpy()
            summary[f"{col}_max"] = float(np.nanmax(series))
            summary[f"{col}_avg"] = float(np.nanmean(series))

    for col in ["driveCurrentLimit", "regenCurrentLimit", "rmsCurrent", "batteryCurrent"]:
        if col in df.columns and df[col].size:
            series = safe_num(df[col]).to_numpy()
            summary[f"{col}_max"] = float(np.nanmax(series))
            summary[f"{col}_avg"] = float(np.nanmean(series))

    for col in ["altitude_smoothed_m", "grade_pct", "vertical_speed_mps", "grav_energy_wh", "elev_gain_m"]:
        if col in df.columns and df[col].size:
            summary[f"{col}_max"] = float(np.nanmax(safe_num(df[col]).to_numpy()))
            summary[f"{col}_end"] = float(safe_num(df[col]).to_numpy()[-1])

    return summary

# -------------------- Plot helpers --------------------

def plot_all(df: pd.DataFrame, timestamp_col: str):
    required_columns = {
        "soc": "SOC (%)",
        "batteryCurrent": "Battery Current (A)",
        "motorRPM": "Motor RPM",
        "controllerTemperature": "Controller Temp (°C)",
        "motorTemperature": "Motor Temp (°C)",
        "brake": "Brake (%)",
        "throttle": "Throttle (%)",
    }
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns for core plots: " + ", ".join(missing))

    ts = df[timestamp_col]
    fig, axes = plt.subplots(6, 1, figsize=(18, 18), sharex=True)

    axes[0].plot(ts, pd.to_numeric(df["soc"], errors="coerce"), label="soc", color="tab:blue")
    axes[0].set_ylabel(required_columns["soc"]); axes[0].legend(loc="upper right"); axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(ts, pd.to_numeric(df["batteryCurrent"], errors="coerce"), label="batteryCurrent", color="tab:orange")
    axes[1].set_ylabel(required_columns["batteryCurrent"]); axes[1].legend(loc="upper right"); axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(ts, pd.to_numeric(df["motorRPM"], errors="coerce"), label="motorRPM", color="tab:green")
    axes[2].set_ylabel(required_columns["motorRPM"]); axes[2].legend(loc="upper right"); axes[2].grid(True, linestyle="--", alpha=0.3)

    axes[3].plot(ts, pd.to_numeric(df["controllerTemperature"], errors="coerce"), label="controllerTemperature", color="tab:red")
    axes[3].plot(ts, pd.to_numeric(df["motorTemperature"], errors="coerce"), label="motorTemperature", color="tab:purple")
    axes[3].set_ylabel(f'{required_columns["controllerTemperature"]} / {required_columns["motorTemperature"]}')
    axes[3].legend(loc="upper right"); axes[3].grid(True, linestyle="--", alpha=0.3)

    axes[4].plot(ts, pd.to_numeric(df["brake"], errors="coerce"), label="brake", color="tab:brown")
    axes[4].set_ylabel(required_columns["brake"]); axes[4].legend(loc="upper right"); axes[4].grid(True, linestyle="--", alpha=0.3)

    axes[5].plot(ts, pd.to_numeric(df["throttle"], errors="coerce"), label="throttle", color="tab:cyan")
    axes[5].set_ylabel(required_columns["throttle"]); axes[5].legend(loc="upper right"); axes[5].grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Vehicle Active Periods (Idle Removed)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_energy_efficiency(df: pd.DataFrame, timestamp_col: str):
    ts = df[timestamp_col]
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))
    if "wh_per_km" in df.columns:
        ax1.plot(ts, df["wh_per_km"], label="Wh/km", color="tab:blue")
        ax1.set_ylabel("Wh/km")
    if "net_energy_wh" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(ts, df["net_energy_wh"], label="Net Energy (Wh)", color="tab:red", alpha=0.7)
        ax2.set_ylabel("Net Energy (Wh)")
    ax1.set_xlabel("Time")
    ax1.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Efficiency & Net Energy", fontsize=14)
    fig.tight_layout()
    return fig


def plot_altitude_block(df: pd.DataFrame, timestamp_col: str):
    if "altitude_smoothed_m" not in df.columns:
        return None
    ts = df[timestamp_col]
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    axes[0].plot(ts, df["altitude_smoothed_m"], label="Altitude (m)", color="tab:blue")
    axes[0].set_ylabel("Altitude (m)"); axes[0].legend(loc="upper right"); axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(ts, df["grade_pct"], label="Grade (%)", color="tab:orange")
    axes[1].set_ylabel("Grade (%)"); axes[1].legend(loc="upper right"); axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(ts, df["vertical_speed_mps"], label="Vertical Speed (m/s)", color="tab:green")
    axes[2].set_ylabel("Vert Speed (m/s)"); axes[2].legend(loc="upper right"); axes[2].grid(True, linestyle="--", alpha=0.3)

    has_any_power = False
    if "elec_power_w" in df.columns:
        axes[3].plot(ts, df["elec_power_w"], label="Electrical Power (W)", color="tab:red"); has_any_power = True
    if "grav_power_w" in df.columns:
        axes[3].plot(ts, df["grav_power_w"], label="Gravitational Power (W)", color="tab:purple", alpha=0.7); has_any_power = True
    if has_any_power:
        axes[3].set_ylabel("Power (W)"); axes[3].legend(loc="upper right"); axes[3].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[3].axis("off")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Altitude & Grade Derived Metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Telemetry Visualizer", layout="wide")
st.title("Telemetry Visualizer & Exporter")

with st.sidebar:
    st.header("Options")
    min_speed = st.number_input("Min speed to keep rows (units auto)", value=0.0, step=0.1)
    mass_kg = st.number_input("Vehicle mass (kg)", value=1500.0, step=10.0)
    nominal_voltage = st.number_input("Nominal pack voltage (optional)", value=0.0, step=1.0)
    use_nominal_voltage = st.checkbox("Force nominal voltage when pack voltage column missing", value=False)
    alt_col_override = st.text_input("Altitude column name (optional)")
    ts_col_override = st.text_input("Timestamp column name (optional)")
    speed_col_override = st.text_input("Speed column name (optional)")

upload = st.file_uploader("Upload CSV or Excel telemetry file", type=["csv", "xlsx", "xls"]) 

if upload is not None:
    try:
        df = read_any(upload)
        st.write("### Preview", df.head())

        df_active, ts_col, speed_col = sanitize(
            df,
            timestamp_col=ts_col_override or None,
            speed_col=speed_col_override or None,
            min_speed=min_speed,
        )
        st.success(f"Detected timestamp column: {ts_col} — speed column: {speed_col}. Active rows: {len(df_active)}/{len(df)}")

        df_active = add_altitude_and_energy(
            df_active,
            ts_col,
            speed_col,
            mass_kg=mass_kg,
            altitude_col=(alt_col_override or None),
            nominal_voltage=(nominal_voltage if use_nominal_voltage and nominal_voltage > 0 else None),
        )
        df_active = add_basic_metrics(df_active, ts_col, speed_col)

        # --- Plots ---
        core_fig = plot_all(df_active, ts_col)
        st.pyplot(core_fig)
        energy_fig = plot_energy_efficiency(df_active, ts_col)
        st.pyplot(energy_fig)
        alt_fig = plot_altitude_block(df_active, ts_col)
        if alt_fig is not None:
            st.pyplot(alt_fig)

        # --- Exports ---
        derived_csv = df_active.to_csv(index=False).encode("utf-8")
        summary = summarize_run(df_active)
        summary_json = json.dumps(summary, indent=2).encode("utf-8")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="Download derived metrics CSV",
                data=derived_csv,
                file_name="derived_metrics.csv",
                mime="text/csv",
            )
        with col2:
            st.download_button(
                label="Download summary JSON",
                data=summary_json,
                file_name="summary.json",
                mime="application/json",
            )
        with col3:
            # Prepare JSONL rows
            jsonl_buf = io.StringIO()
            for _, row in df_active.iterrows():
                obj = row.to_dict()
                if isinstance(obj.get(ts_col), (pd.Timestamp, )):
                    obj[ts_col] = pd.Timestamp(obj[ts_col]).isoformat()
                for k, v in list(obj.items()):
                    if isinstance(v, (np.floating, )):
                        obj[k] = float(v)
                    elif isinstance(v, (np.integer, )):
                        obj[k] = int(v)
                jsonl_buf.write(json.dumps(obj) + "\n")
            st.download_button(
                label="Download per-row JSON Lines",
                data=jsonl_buf.getvalue().encode("utf-8"),
                file_name="rows.jsonl",
                mime="application/x-ndjson",
            )

        # Plot image downloads
        def fig_to_png_bytes(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            buf.seek(0)
            return buf

        core_png = fig_to_png_bytes(core_fig)
        energy_png = fig_to_png_bytes(energy_fig)
        st.download_button("Download core plots PNG", data=core_png, file_name="core_plots.png", mime="image/png")
        st.download_button("Download energy/efficiency PNG", data=energy_png, file_name="energy_plots.png", mime="image/png")
        if alt_fig is not None:
            alt_png = fig_to_png_bytes(alt_fig)
            st.download_button("Download altitude plots PNG", data=alt_png, file_name="altitude_plots.png", mime="image/png")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a file to begin.")
