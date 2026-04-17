import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from monitoring.psi import compute_psi_all_features
from monitoring.statistical_tests import run_all_tests
from monitoring.performance_tracker import (
    get_performance_all_windows, compute_degradation, get_health_status
)
from monitoring.shap_drift import compute_shap_drift
from monitoring.alerting import run_all_alerts
from monitoring.evidently_reports import (
    generate_drift_report as generate_evidently_report,
)
from models.isolation_forest import score_window as iso_score
from models.autoencoder import score_window as ae_score
from reports.drift_report import generate_drift_report

DB_PATH = BASE_DIR / "data" / "modelwatch.db"
SAVED_DIR = BASE_DIR / "models" / "saved"
REPORT_DIR = BASE_DIR / "reports" / "generated"
BACKGROUND = "#F4F0E6"
SURFACE = "#FFFDF8"
SURFACE_ALT = "#F7F1E6"
INK = "#1A1814"
MUTED = "#746E64"
BORDER = "#E3DCCF"
ACCENT = "#0E8C82"
ACCENT_DEEP = "#0B5E58"
ACCENT_SOFT = "#DDF6F2"
WARNING = "#B96D2C"
DANGER = "#D64C43"
SUCCESS = "#17785F"
PLUM = "#7656B8"

st.set_page_config(
    page_title="ModelWatch",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE,
    font=dict(
        family="Space Grotesk, sans-serif",
        color=INK,
        size=12,
    ),
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(
        bgcolor=INK,
        bordercolor=INK,
        font=dict(
            family="IBM Plex Mono, monospace",
            color="#FFFFFF",
            size=11,
        ),
    ),
)

AXIS_STYLE = dict(
    gridcolor="#EFE7D7",
    linecolor=BORDER,
    tickfont=dict(color=MUTED, size=11),
)

EXTERNAL_FIG_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE,
    font=dict(
        family="Space Grotesk, sans-serif",
        color=INK,
        size=12,
    ),
    margin=dict(l=40, r=30, t=50, b=40),
)

STATUS_COLORS = {
    "GREEN": SUCCESS,
    "AMBER": WARNING,
    "RED": DANGER,
    "HEALTHY": SUCCESS,
    "MILD": WARNING,
    "MODERATE": DANGER,
    "SEVERE": "#7A261F",
    "PROMOTE CHALLENGER": SUCCESS,
    "KEEP CHAMPION": WARNING,
    "RETRAIN URGENTLY": DANGER,
    "RETRAIN SOON": WARNING,
    "MONITOR": PLUM,
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(232, 191, 96, 0.20), transparent 18%),
        radial-gradient(circle at 92% 10%, rgba(14, 140, 130, 0.16), transparent 18%),
        linear-gradient(180deg, #F7F3EA 0%, #F0E9DD 100%);
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.block-container {
    padding: 1.25rem 2.6rem 4rem !important;
    max-width: 1680px !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFDF8 0%, #F4EEE2 100%) !important;
    border-right: 1px solid #E6DDCF !important;
}
[data-testid="metric-container"] {
    background: rgba(255,253,248,0.92);
    border: 0.5px solid #E7DECF;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 14px 40px rgba(26, 24, 20, 0.04);
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #1A1814 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #746E64 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0E8C82 0%, #146B63 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    padding: 0.72rem 1.25rem !important;
    box-shadow: 0 16px 34px rgba(14, 140, 130, 0.22);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0B7B72 0%, #105C56 100%) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #0E8C82 !important;
    border-bottom: 2px solid #0E8C82 !important;
}
.page-title {
    font-size: 2.15rem;
    font-weight: 700;
    color: #1A1814;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}
.page-subtitle {
    font-size: 0.92rem;
    color: #746E64;
    margin-bottom: 0.15rem;
    max-width: 64rem;
}
.mw-hero {
    background:
        radial-gradient(circle at 0% 0%, rgba(232, 191, 96, 0.22), transparent 28%),
        linear-gradient(135deg, rgba(255, 253, 248, 0.95) 0%, rgba(245, 239, 226, 0.98) 100%);
    border: 1px solid #E7DECF;
    border-radius: 24px;
    padding: 1.35rem 1.5rem 1.55rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 20px 50px rgba(26, 24, 20, 0.06);
}
.mw-kicker {
    display: inline-block;
    background: rgba(14, 140, 130, 0.08);
    border: 1px solid rgba(14, 140, 130, 0.18);
    color: #0E8C82;
    border-radius: 999px;
    padding: 0.32rem 0.7rem;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.9rem;
}
.mw-hero-grid {
    display: grid;
    grid-template-columns: 1.8fr 1fr;
    gap: 1rem;
    align-items: start;
}
.mw-hero-side {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(231, 222, 207, 0.95);
    border-radius: 18px;
    padding: 1rem 1.05rem;
}
.mw-side-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #746E64;
    margin-bottom: 0.4rem;
}
.mw-side-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1A1814;
    font-family: 'IBM Plex Mono', monospace;
}
.mw-shell {
    background: rgba(255,253,248,0.84);
    border: 0.5px solid #E7DECF;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 12px 28px rgba(26, 24, 20, 0.04);
    margin-bottom: 1rem;
}
.mw-section-title {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #746E64;
    margin-bottom: 0.5rem;
}
.mw-note {
    background: linear-gradient(135deg, rgba(14, 140, 130, 0.08), rgba(118, 86, 184, 0.06));
    border: 1px solid rgba(14, 140, 130, 0.15);
    border-radius: 18px;
    padding: 0.95rem 1.05rem;
    color: #403A33;
    font-size: 0.84rem;
}
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #E7DECF;
}
hr { border-color: #E7DECF !important; }
</style>
""",
    unsafe_allow_html=True,
)


def header(title: str, subtitle: str):
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def db_check():
    if not DB_PATH.exists():
        st.error("Database not found. Run python data/loader.py")
        st.stop()


def models_check():
    if not (SAVED_DIR / "lgbm_baseline.joblib").exists():
        st.error("Models not trained. Run python models/lgbm_model.py")
        st.stop()


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#888888")
    return (
        f"<span style='background:{color}22;color:{color};"
        f"border:0.5px solid {color}55;border-radius:20px;"
        f"padding:4px 12px;font-weight:600;font-size:0.78rem;"
        f"font-family:IBM Plex Mono,monospace'>{status}</span>"
    )


def hero_banner(title: str, subtitle: str, selected_label: str, status: str, stat_lines: list[str]):
    stat_html = "".join(
        f"<div style='display:flex;justify-content:space-between;gap:1rem;margin:0.32rem 0;'>"
        f"<span style='color:{MUTED};font-size:0.78rem'>{label}</span>"
        f"<span style='color:{INK};font-family:IBM Plex Mono,monospace;font-size:0.78rem;font-weight:600'>{value}</span>"
        f"</div>"
        for label, value in stat_lines
    )
    st.markdown(
        f"""
        <div class='mw-hero'>
          <div class='mw-kicker'>Wide Monitoring Canvas</div>
          <div class='mw-hero-grid'>
            <div>
              <div class='page-title'>{title}</div>
              <div class='page-subtitle'>{subtitle}</div>
            </div>
            <div class='mw-hero-side'>
              <div class='mw-side-label'>{selected_label}</div>
              <div style='margin-bottom:0.75rem'>{status_badge(status)}</div>
              {stat_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=1800)
def perf_cached() -> pd.DataFrame:
    return get_performance_all_windows()


@st.cache_data(show_spinner=False, ttl=1800)
def degraded_cached() -> pd.DataFrame:
    return compute_degradation(perf_cached())


@st.cache_data(show_spinner=False, ttl=1800)
def psi_cached(window_id: int) -> pd.DataFrame:
    return compute_psi_all_features(window_id)


@st.cache_data(show_spinner=False, ttl=1800)
def stat_cached(window_id: int) -> pd.DataFrame:
    return run_all_tests(window_id)


@st.cache_data(show_spinner=False, ttl=1800)
def iso_cached(window_id: int) -> dict:
    return iso_score(window_id)


@st.cache_data(show_spinner=False, ttl=1800)
def ae_cached(window_id: int) -> dict:
    return ae_score(window_id)


@st.cache_data(show_spinner=False, ttl=1800)
def shap_cached(window_id: int) -> dict:
    return compute_shap_drift(window_id)


@st.cache_data(show_spinner=False, ttl=1800)
def window_frame(window_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(
        "SELECT * FROM credit_records WHERE window_id = ?",
        conn,
        params=(window_id,),
    )
    conn.close()
    return df


window_options = {
    "Window 2 (Mild Drift)": 2,
    "Window 3 (Moderate Drift)": 3,
    "Window 4 (Severe Drift)": 4,
}

with st.sidebar:
    st.markdown("""
    <div style='padding:0.2rem 0 1.2rem'>
      <div style='font-size:1.3rem;margin-bottom:6px'>🔭</div>
      <div style='font-size:1.1rem;font-weight:700;color:#1a1a18;
                  letter-spacing:-0.02em'>ModelWatch</div>
      <div style='font-size:0.68rem;color:#888;
                  text-transform:uppercase;letter-spacing:0.07em'>
        ML Monitoring Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        [
            "📊  Overview",
            "📈  Data Drift",
            "🎯  Model Performance",
            "🧠  Deep Drift",
            "🔍  SHAP Drift",
            "📝  Drift Report",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    selected_window_name = st.selectbox(
        "Monitoring Window",
        list(window_options.keys()),
        index=2,
        key="window_selector"
    )
    selected_window = window_options[selected_window_name]

    st.divider()

    try:
        perf_df = get_performance_all_windows()
        for w_name, w_id in window_options.items():
            status = get_health_status(perf_df, w_id)
            color = STATUS_COLORS.get(status, "#888")
            short = w_name.split(" ")[0] + " " + w_name.split(" ")[1]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;font-size:0.8rem'>"
                f"<span style='color:#888'>{short}</span>"
                f"<span style='color:{color};font-weight:600;"
                f"font-family:IBM Plex Mono,monospace'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    except Exception:
        pass

    st.divider()
    st.markdown("""
    <div style='font-size:0.68rem;color:#aaa;line-height:1.9'>
      Built by <b style='color:#0E8C82'>Yashwanth</b><br>
      UCI Credit Card Default<br>
      LightGBM | LSTM | Autoencoder<br>
      Isolation Forest | Evidently AI
    </div>
    """, unsafe_allow_html=True)

if "Overview" in page:
    db_check()
    models_check()

    try:
        perf_df = perf_cached()
        degraded = degraded_cached()
        selected_status = get_health_status(perf_df, selected_window)
        selected_psi = psi_cached(selected_window)
        selected_iso = iso_cached(selected_window)
        selected_ae = ae_cached(selected_window)
        selected_shap = shap_cached(selected_window)
        forecast_path = SAVED_DIR / "lstm_forecast_result.joblib"
        forecast = joblib.load(forecast_path) if forecast_path.exists() else None
        hero_banner(
            "ModelWatch Command Deck",
            "A wide production monitoring surface for credit-risk drift, degradation, behaviour shift, and retraining readiness.",
            selected_window_name,
            selected_status,
            [
                ("Current AUC", f"{degraded[degraded['window_id'] == selected_window].iloc[0]['auc_roc']:.3f}"),
                ("Max PSI", f"{selected_psi['psi'].max():.3f}"),
                ("Anomaly Rate", f"{selected_iso['anomaly_rate']:.1f}%"),
                ("Next Move", forecast["recommendation"] if forecast else "FORECAST PENDING"),
            ],
        )

        cols = st.columns(4)
        window_labels = [
            ("Window 1", "Baseline", 1),
            ("Window 2", "Mild Drift", 2),
            ("Window 3", "Moderate Drift", 3),
            ("Window 4", "Severe Drift", 4),
        ]

        for col, (label, sublabel, w_id) in zip(cols, window_labels):
            row = degraded[degraded["window_id"] == w_id].iloc[0]
            status = get_health_status(perf_df, w_id)
            color = STATUS_COLORS.get(status, "#888")
            auc = row["auc_roc"]
            deg = row.get("auc_roc_degradation_pct", 0)
            stripe = f"linear-gradient(135deg, {color} 0%, rgba(255,255,255,0.0) 110%)"

            col.markdown(f"""
            <div style='background:
                        radial-gradient(circle at top left, {color}22, transparent 30%),
                        #FFFDF8;
                        border:0.5px solid #E7DECF;
                        border-radius:18px;padding:1.05rem 1rem 1.1rem;
                        box-shadow:0 16px 36px rgba(26,24,20,0.04);
                        text-align:left;min-height:145px'>
              <div style='font-size:0.68rem;font-weight:600;
                          text-transform:uppercase;letter-spacing:.07em;
                          color:#888;margin-bottom:6px'>{label}</div>
              <div style='font-size:0.82rem;color:{MUTED};margin-bottom:0.8rem'>{sublabel}</div>
              <div style='font-size:1.55rem;font-weight:700;
                          font-family:IBM Plex Mono,monospace;
                          color:{INK}'>{auc:.3f}</div>
              <div style='font-size:0.78rem;color:{color};
                          font-family:IBM Plex Mono,monospace;
                          margin:0.4rem 0 0.8rem'>{deg:+.1f}% vs baseline</div>
              <div>{status_badge(status)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        chart_col, signal_col = st.columns([1.9, 1.0], gap="large")
        with chart_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perf_df["window_id"],
                y=perf_df["auc_roc"],
                mode="lines+markers",
                name="AUC-ROC",
                line=dict(color=ACCENT, width=3),
                marker=dict(size=10, color=ACCENT, line=dict(color=SURFACE, width=2)),
                fill="tozeroy",
                fillcolor="rgba(14, 140, 130, 0.10)",
                text=[f"AUC={v:.3f}" for v in perf_df["auc_roc"]],
                textposition="top center"
            ))
            fig.add_hline(
                y=0.75,
                line_dash="dash",
                line_color=WARNING,
                annotation_text="Performance floor",
                annotation_font=dict(color=WARNING, size=11)
            )
            fig.update_layout(
                **CHART_THEME,
                height=370,
                title="Model quality trajectory across all monitoring windows",
                xaxis_title="Window",
                yaxis_title="AUC-ROC",
                xaxis=dict(
                    tickvals=[1, 2, 3, 4],
                    ticktext=["W1\nBaseline", "W2\nMild",
                              "W3\nModerate", "W4\nSevere"],
                    **AXIS_STYLE
                ),
                yaxis=dict(range=[0.5, 1.0], **AXIS_STYLE),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with signal_col:
            st.markdown("<div class='mw-shell'>"
                        "<div class='mw-section-title'>Signal Stack</div>"
                        f"<div style='font-size:1.9rem;font-weight:700;color:{INK};margin-bottom:0.2rem'>"
                        f"{selected_status}</div>"
                        f"<div style='color:{MUTED};font-size:0.82rem;margin-bottom:0.9rem'>"
                        f"Selected window health based on degradation and drift intensity.</div>"
                        f"<div style='display:grid;gap:0.65rem'>"
                        f"<div class='mw-note'><b>PSI max</b><br><span style='font-family:IBM Plex Mono, monospace;font-size:1.12rem'>{selected_psi['psi'].max():.3f}</span></div>"
                        f"<div class='mw-note'><b>Anomaly rate</b><br><span style='font-family:IBM Plex Mono, monospace;font-size:1.12rem'>{selected_iso['anomaly_rate']:.1f}%</span></div>"
                        f"<div class='mw-note'><b>AE drift ratio</b><br><span style='font-family:IBM Plex Mono, monospace;font-size:1.12rem'>{selected_ae['drift_ratio']:.2f}x</span></div>"
                        f"<div class='mw-note'><b>SHAP rank corr</b><br><span style='font-family:IBM Plex Mono, monospace;font-size:1.12rem'>{selected_shap['spearman_correlation']:.3f}</span></div>"
                        f"</div></div>",
                        unsafe_allow_html=True)

        st.markdown(f"#### Active alerts - {selected_window_name}")
        with st.spinner("Loading alerts..."):
            alerts = run_all_alerts(
                window_id=selected_window,
                psi_df=selected_psi,
                perf_df=perf_df,
                anomaly_result=selected_iso,
                ae_result=selected_ae,
                shap_result=selected_shap,
            )

        red_count = sum(1 for a in alerts if a["level"] == "RED")
        amber_count = sum(1 for a in alerts if a["level"] == "AMBER")

        a1, a2, a3 = st.columns(3)
        a1.metric("RED Alerts", str(red_count))
        a2.metric("AMBER Alerts", str(amber_count))
        a3.metric("Total Alerts", str(len(alerts)))

        for alert in alerts[:6]:
            color = "#FCEBEB" if alert["level"] == "RED" else "#FAEEDA"
            border_color = "#F09595" if alert["level"] == "RED" else "#FAC775"
            st.markdown(f"""
            <div style='background:{color};
                        border:0.5px solid {border_color};
                        border-left:4px solid {"#E24B4A" if alert["level"]=="RED" else "#BA7517"};
                        border-radius:8px;padding:0.7rem 1rem;
                        margin-bottom:0.5rem;font-size:0.83rem'>
              <b>[{alert["level"]}] {alert["type"]}</b>:
              {alert["message"][:120]}...
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Overview error: {e}")

elif "Data Drift" in page:
    db_check()
    models_check()

    try:
        psi_df = psi_cached(selected_window)
        stat_df = stat_cached(selected_window)
        hero_banner(
            "Distribution Shift Atlas",
            "Feature-by-feature drift diagnostics across PSI, statistical tests, and Evidently HTML reports.",
            selected_window_name,
            "RED" if psi_df["psi"].max() >= 0.25 else "AMBER" if psi_df["psi"].max() >= 0.10 else "GREEN",
            [
                ("Max PSI", f"{psi_df['psi'].max():.3f}"),
                ("Mean PSI", f"{psi_df['psi'].mean():.3f}"),
                ("Drifted Features", f"{int(stat_df['drift_detected'].sum())}/{len(stat_df)}"),
                ("Worst Feature", psi_df.iloc[0]["feature"]),
            ],
        )

        with st.spinner("Computing drift metrics..."):
            pass

        st.markdown("#### PSI heatmap - all features × all windows")
        psi_data = {}
        for w in [2, 3, 4]:
            df = psi_cached(w)
            psi_data[f"W{w}"] = df.set_index("feature")["psi"]

        psi_matrix = pd.DataFrame(psi_data)
        psi_matrix = psi_matrix.sort_values("W4", ascending=False)

        fig_heat = go.Figure(go.Heatmap(
            z=psi_matrix.values,
            x=["Window 2\n(Mild)",
               "Window 3\n(Moderate)",
               "Window 4\n(Severe)"],
            y=psi_matrix.index.tolist(),
            colorscale=[
                [0.0, "#E1F5EE"],
                [0.1, "#FAEEDA"],
                [0.25, "#FCEBEB"],
                [1.0, "#7F1F1F"]
            ],
            text=[[f"{v:.3f}" for v in row]
                  for row in psi_matrix.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(
                title="PSI",
                tickvals=[0, 0.1, 0.25, 0.5, 1.0],
                ticktext=["0", "0.10\nMonitor",
                          "0.25\nAlert", "0.50", "1.0+"]
            ),
            xgap=3, ygap=3
        ))
        fig_heat.update_layout(
            **EXTERNAL_FIG_THEME,
            height=550,
            title="Population Stability Index by feature and window"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        spotlight_features = psi_df.head(4)["feature"].tolist()
        spotlight = []
        for w in [1, 2, 3, 4]:
            frame = window_frame(w)
            for feature in spotlight_features:
                if feature in frame.columns:
                    spotlight.append({
                        "window_id": w,
                        "feature": feature,
                        "mean_value": frame[feature].mean(),
                    })
        spotlight_df = pd.DataFrame(spotlight)
        if not spotlight_df.empty:
            st.markdown("#### Feature trajectory spotlight")
            fig_spot = px.line(
                spotlight_df,
                x="window_id",
                y="mean_value",
                color="feature",
                markers=True,
                line_shape="spline",
                color_discrete_sequence=[ACCENT, WARNING, DANGER, PLUM],
            )
            fig_spot.update_traces(line=dict(width=3), marker=dict(size=9))
            fig_spot.update_layout(
                **CHART_THEME,
                height=340,
                title="Mean feature value movement across monitoring windows",
                xaxis=dict(
                    tickvals=[1, 2, 3, 4],
                    ticktext=["Baseline", "Mild", "Moderate", "Severe"],
                    **AXIS_STYLE,
                ),
                yaxis=dict(**AXIS_STYLE),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )
            st.plotly_chart(fig_spot, use_container_width=True)

        st.divider()

        st.markdown(f"#### Top drifted features - {selected_window_name}")

        tab_psi, tab_stat, tab_evidently = st.tabs([
            "PSI Analysis", "Statistical Tests", "Evidently Report"
        ])

        with tab_psi:
            display_psi = psi_df[[
                "feature", "psi", "status"
            ]].head(15).copy()
            st.dataframe(
                display_psi,
                use_container_width=True,
                hide_index=True
            )

            fig_bar = go.Figure(go.Bar(
                y=psi_df["feature"].head(10),
                x=psi_df["psi"].head(10),
                orientation="h",
                marker_color=[
                    "#E24B4A" if s == "RED"
                    else "#BA7517" if s == "AMBER"
                    else "#1D9E75"
                    for s in psi_df["status"].head(10)
                ],
                text=[f"{v:.3f}" for v in psi_df["psi"].head(10)],
                textposition="outside"
            ))
            fig_bar.add_vline(
                x=0.10, line_dash="dot",
                line_color="#BA7517",
                annotation_text="Amber 0.10"
            )
            fig_bar.add_vline(
                x=0.25, line_dash="dash",
                line_color="#E24B4A",
                annotation_text="Red 0.25"
            )
            fig_bar.update_layout(
                **CHART_THEME,
                height=380,
                title="PSI by feature",
                xaxis_title="PSI Score",
                yaxis=dict(autorange="reversed", **AXIS_STYLE),
                xaxis=dict(**AXIS_STYLE),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_stat:
            st.dataframe(
                stat_df[[
                    "feature", "method", "statistic",
                    "p_value", "js_divergence", "drift_detected"
                ]],
                use_container_width=True,
                hide_index=True
            )

        with tab_evidently:
            report_path = REPORT_DIR / f"drift_report_window_{selected_window}.html"
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(
                    html_content,
                    height=600,
                    scrolling=True
                )
            else:
                st.info(
                    "No Evidently HTML report is bundled for this window yet. "
                    "Generate it live from the dashboard.",
                    icon="📊"
                )
                if st.button(
                    "Generate Evidently Report",
                    key=f"generate_evidently_{selected_window}",
                ):
                    with st.spinner("Building Evidently HTML report..."):
                        try:
                            result = generate_evidently_report(
                                selected_window,
                                save_html=True,
                            )
                            st.success(
                                f"Evidently report ready: "
                                f"{result['n_drifted']}/{result['n_features']} features drifted.",
                                icon="✅",
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Evidently generation failed: {exc}")

    except Exception as e:
        st.error(f"Data drift error: {e}")

elif "Model Performance" in page:
    db_check()
    models_check()

    try:
        perf_df = perf_cached()
        degraded = degraded_cached()

        row = degraded[
            degraded["window_id"] == selected_window
        ].iloc[0]
        baseline = perf_df[perf_df["window_id"] == 1].iloc[0]
        hero_banner(
            "Performance Observatory",
            "Window-by-window quality, degradation pressure, and challenger readiness from the production LightGBM system.",
            selected_window_name,
            get_health_status(perf_df, selected_window),
            [
                ("Current AUC", f"{row['auc_roc']:.4f}"),
                ("AUC Delta", f"{row['auc_roc_degradation_pct']:+.1f}%"),
                ("F1 Score", f"{row['f1']:.4f}"),
                ("KS Stat", f"{row['ks_stat']:.4f}"),
            ],
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC-ROC", f"{row['auc_roc']:.4f}",
                  delta=f"{row['auc_roc']-baseline['auc_roc']:+.4f}")
        m2.metric("F1 Score", f"{row['f1']:.4f}",
                  delta=f"{row['f1']-baseline['f1']:+.4f}")
        m3.metric("Precision", f"{row['precision']:.4f}",
                  delta=f"{row['precision']-baseline['precision']:+.4f}")
        m4.metric("Recall", f"{row['recall']:.4f}",
                  delta=f"{row['recall']-baseline['recall']:+.4f}")
        m5.metric("KS Stat", f"{row['ks_stat']:.4f}",
                  delta=f"{row['ks_stat']-baseline['ks_stat']:+.4f}")

        st.divider()

        metrics_to_plot = ["auc_roc", "f1", "precision", "recall"]
        colors_plot = [ACCENT, SUCCESS, "#185FA5", WARNING]

        fig_lines = go.Figure()
        for metric, color in zip(metrics_to_plot, colors_plot):
            if metric in perf_df.columns:
                fig_lines.add_trace(go.Scatter(
                    x=perf_df["window_id"],
                    y=perf_df[metric],
                    mode="lines+markers",
                    name=metric.upper().replace("_", " "),
                    line=dict(color=color, width=2),
                    marker=dict(size=8, color=color),
                ))

        fig_lines.update_layout(
            **CHART_THEME,
            height=380,
            title="Performance metrics across monitoring windows",
            xaxis_title="Window",
            yaxis_title="Score",
            xaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=["Baseline", "Mild", "Moderate", "Severe"],
                **AXIS_STYLE
            ),
            yaxis=dict(range=[0.3, 1.0], **AXIS_STYLE),
            legend=dict(
                bgcolor="#FFFFFF",
                bordercolor="#E8E8E8",
                borderwidth=1
            )
        )
        st.plotly_chart(fig_lines, use_container_width=True)

        cc_path = SAVED_DIR / "champion_challenger_result.joblib"
        if cc_path.exists():
            st.markdown("#### Champion-Challenger comparison")
            cc = joblib.load(cc_path)

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Champion AUC", f"{cc['champion_auc']:.4f}")
            cc2.metric("Challenger AUC", f"{cc['challenger_auc']:.4f}",
                       delta=f"{cc['auc_improvement']:+.4f}")
            cc3.metric("Improvement", f"{cc['auc_improvement']:+.4f}")
            decision_label = (
                "PROMOTE CHALLENGER"
                if cc["decision"] == "PROMOTE CHALLENGER"
                else "KEEP CHAMPION"
            )
            cc4.markdown(f"""
            <div style='padding:0.8rem 0'>
              {status_badge(decision_label)}
            </div>
            """, unsafe_allow_html=True)

            if cc["decision"] == "PROMOTE CHALLENGER":
                st.success(
                    f"The challenger model retrained on Windows "
                    f"{cc['train_windows']} achieves AUC "
                    f"{cc['challenger_auc']:.4f} vs champion's "
                    f"{cc['champion_auc']:.4f} on Window "
                    f"{cc['test_window']}. "
                    f"Retraining is recommended.",
                    icon="✅"
                )

    except Exception as e:
        st.error(f"Performance error: {e}")

elif "Deep Drift" in page:
    db_check()
    models_check()

    try:
        with st.spinner("Running deep drift analysis..."):
            ae = ae_cached(selected_window)
            iso = iso_cached(selected_window)
        hero_banner(
            "Deep Drift Lab",
            "Unsupervised drift detection through reconstruction error, anomaly density, and sequence forecasting.",
            selected_window_name,
            ae["status"],
            [
                ("AE Mean Error", f"{ae['mean_error']:.6f}"),
                ("AE Ratio", f"{ae['drift_ratio']:.2f}x"),
                ("Anomaly Rate", f"{iso['anomaly_rate']:.1f}%"),
                ("Records Flagged", f"{iso['n_anomalies']:,}"),
            ],
        )

        st.markdown("#### Autoencoder reconstruction error")
        ae_data = {}
        for w in [1, 2, 3, 4]:
            r = ae_cached(w)
            ae_data[w] = r["mean_error"]

        fig_ae = go.Figure()
        fig_ae.add_trace(go.Bar(
            x=list(ae_data.keys()),
            y=list(ae_data.values()),
            marker_color=[
                "#1D9E75", "#BA7517", "#E24B4A", "#7F1F1F"
            ],
            text=[f"{v:.4f}" for v in ae_data.values()],
            textposition="outside"
        ))
        fig_ae.add_hline(
            y=ae_score(1)["mean_error"] * 1.5,
            line_dash="dash",
            line_color="#BA7517",
            annotation_text="Amber threshold (1.5x baseline)",
            annotation_font=dict(color="#BA7517", size=10)
        )
        fig_ae.update_layout(
            **CHART_THEME,
            height=320,
            title="Autoencoder reconstruction error by window",
            xaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=["Baseline", "Mild", "Moderate", "Severe"],
                **AXIS_STYLE
            ),
            yaxis=dict(**AXIS_STYLE),
            showlegend=False
        )
        st.plotly_chart(fig_ae, use_container_width=True)

        a1, a2, a3 = st.columns(3)
        a1.metric("Mean Error", f"{ae['mean_error']:.6f}")
        a2.metric("Drift Ratio", f"{ae['drift_ratio']:.2f}x")
        a3.metric("Status", ae["status"])

        st.divider()

        st.markdown("#### Isolation Forest anomaly detection")
        iso_data = {}
        for w in [1, 2, 3, 4]:
            r = iso_cached(w)
            iso_data[w] = r["anomaly_rate"]

        fig_iso = go.Figure(go.Bar(
            x=list(iso_data.keys()),
            y=list(iso_data.values()),
            marker_color=[
                "#1D9E75", "#BA7517", "#E24B4A", "#7F1F1F"
            ],
            text=[f"{v:.1f}%" for v in iso_data.values()],
            textposition="outside"
        ))
        fig_iso.add_hline(
            y=8.0, line_dash="dot",
            line_color="#BA7517",
            annotation_text="Amber 8%"
        )
        fig_iso.add_hline(
            y=15.0, line_dash="dash",
            line_color="#E24B4A",
            annotation_text="Red 15%"
        )
        fig_iso.update_layout(
            **CHART_THEME,
            height=320,
            title="Anomaly rate by window (%)",
            xaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=["Baseline", "Mild", "Moderate", "Severe"],
                **AXIS_STYLE
            ),
            yaxis=dict(**AXIS_STYLE),
            showlegend=False
        )
        st.plotly_chart(fig_iso, use_container_width=True)

        st.divider()
        st.markdown("#### LSTM PSI forecast - Window 5")
        lstm_path = SAVED_DIR / "lstm_forecast_result.joblib"
        if lstm_path.exists():
            forecast = joblib.load(lstm_path)
            f1, f2, f3 = st.columns(3)
            f1.metric("Predicted Max PSI",
                      f"{forecast['predicted_max_psi']:.4f}")
            f2.metric("Predicted Mean PSI",
                      f"{forecast['predicted_mean_psi']:.4f}")
            f3.metric("Recommendation",
                      forecast["recommendation"])

            rec = forecast["recommendation"]
            if "URGENT" in rec:
                st.error(
                    f"**{rec}** - LSTM forecasts continued severe drift "
                    f"in the next monitoring window. "
                    f"Schedule retraining immediately.",
                    icon="🚨"
                )
            elif "SOON" in rec:
                st.warning(
                    f"**{rec}** - Drift expected to continue. "
                    f"Plan retraining for next sprint.",
                    icon="⚠️"
                )
            else:
                st.success(
                    f"**{rec}** - Drift appears to be stabilising.",
                    icon="✅"
                )

    except Exception as e:
        st.error(f"Deep drift error: {e}")

elif "SHAP Drift" in page:
    db_check()
    models_check()

    try:
        with st.spinner("Computing SHAP importance drift..."):
            shap_result = shap_cached(selected_window)

        corr = shap_result["spearman_correlation"]
        status = shap_result["status"]
        hero_banner(
            "Behaviour Shift Studio",
            "A ranked view of how the model's internal decision logic drifts as the production data distribution moves.",
            selected_window_name,
            status,
            [
                ("Rank Corr", f"{corr:.3f}"),
                ("Top Baseline", shap_result.get("baseline_top3", ["n/a"])[0]),
                ("Top Current", shap_result.get("current_top3", ["n/a"])[0]),
                ("Biggest Mover", next(iter(shap_result.get("top_movers", {"n/a": 0}).keys()))),
            ],
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Spearman Correlation", f"{corr:.3f}")
        s2.metric("Drift Status", status)
        s3.markdown(f"""
        <div style='padding:0.8rem 0'>
          {status_badge(status)}
        </div>""", unsafe_allow_html=True)

        st.info(
            "Spearman rank correlation measures how similar the "
            "feature importance rankings are between baseline and "
            "the selected window. 1.0 = identical, 0.6 = severe drift.",
            icon="ℹ️"
        )

        st.markdown("#### Features with largest importance rank changes")
        movers = shap_result.get("top_movers", {})
        if movers:
            mover_df = pd.DataFrame([
                {"feature": k, "rank_change": v}
                for k, v in movers.items()
            ]).sort_values("rank_change", ascending=False)

            fig_movers = go.Figure(go.Bar(
                x=mover_df["rank_change"],
                y=mover_df["feature"],
                orientation="h",
                marker_color=ACCENT,
                opacity=0.85,
                text=[f"{v:.1f}" for v in
                      mover_df["rank_change"]],
                textposition="outside"
            ))
            fig_movers.update_layout(
                **CHART_THEME,
                height=340,
                title="Feature importance rank shift (positions moved)",
                xaxis_title="Rank positions moved",
                yaxis=dict(autorange="reversed", **AXIS_STYLE),
                xaxis=dict(**AXIS_STYLE),
                showlegend=False
            )
            st.plotly_chart(fig_movers, use_container_width=True)

        col_base, col_curr = st.columns(2)
        col_base.markdown(
            f"**Baseline top features:**  "
            f"{', '.join(shap_result.get('baseline_top3', []))}"
        )
        col_curr.markdown(
            f"**Window {selected_window} top features:**  "
            f"{', '.join(shap_result.get('current_top3', []))}"
        )

    except Exception as e:
        st.error(f"SHAP drift error: {e}")

elif "Drift Report" in page:
    db_check()
    models_check()

    try:
        perf_df = perf_cached()
        degraded = degraded_cached()
        row = degraded[
            degraded["window_id"] == selected_window
        ].iloc[0]
        hero_banner(
            "Executive Drift Narrative",
            "Generate a plain-English monitoring brief that turns metrics, anomalies, behaviour drift, and retraining signals into one story.",
            selected_window_name,
            get_health_status(perf_df, selected_window),
            [
                ("Baseline AUC", f"{perf_df[perf_df.window_id == 1]['auc_roc'].iloc[0]:.4f}"),
                ("Current AUC", f"{row['auc_roc']:.4f}"),
                ("Delta", f"{row['auc_roc_degradation_pct']:+.1f}%"),
                ("Window", f"W{selected_window}"),
            ],
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Baseline AUC",
                  f"{perf_df[perf_df.window_id == 1]['auc_roc'].iloc[0]:.4f}")
        s2.metric("Current AUC",
                  f"{row['auc_roc']:.4f}",
                  delta=f"{row['auc_roc_degradation_pct']:+.1f}%")
        status = get_health_status(perf_df, selected_window)
        s3.metric("Health Status", status)
        s4.metric("Window", f"{selected_window} vs Baseline")

        st.divider()

        if st.button(
            "✨  Generate Drift Report", type="primary"
        ):
            with st.spinner("Gemini is writing the report..."):
                psi_df = psi_cached(selected_window)
                iso = iso_cached(selected_window)
                ae = ae_cached(selected_window)
                shap = shap_cached(selected_window)
                alerts = run_all_alerts(
                    window_id=selected_window,
                    psi_df=psi_df,
                    perf_df=perf_df,
                    anomaly_result=iso,
                    ae_result=ae,
                    shap_result=shap,
                )

                cc_result = joblib.load(
                    SAVED_DIR / "champion_challenger_result.joblib"
                ) if (SAVED_DIR /
                      "champion_challenger_result.joblib").exists() else None

                lstm_result = joblib.load(
                    SAVED_DIR / "lstm_forecast_result.joblib"
                ) if (SAVED_DIR /
                      "lstm_forecast_result.joblib").exists() else None

                perf_summary = {
                    "baseline_auc": round(float(
                        perf_df[perf_df.window_id == 1]["auc_roc"].iloc[0]
                    ), 4),
                    "current_auc": round(float(row["auc_roc"]), 4),
                    "degradation_pct": round(float(
                        row["auc_roc_degradation_pct"]
                    ), 1),
                    "status": status
                }

                report = generate_drift_report(
                    window_id=selected_window,
                    psi_results={
                        "top_features": psi_df.head(5).to_dict("records")
                    },
                    perf_results=perf_summary,
                    ae_results=ae,
                    iso_results=iso,
                    shap_results=shap,
                    alerts=alerts,
                    cc_results=cc_result,
                    lstm_forecast=lstm_result,
                )

            st.success("Report generated", icon="✅")

            st.markdown(
                f"""<div style='background:{SURFACE};
                    border:0.5px solid {BORDER};
                    border-left:4px solid {ACCENT};
                    border-radius:18px;
                    padding:1.5rem 1.8rem;
                    line-height:1.9;
                    font-size:0.88rem;
                    color:{INK};
                    white-space:pre-wrap'>{report}</div>""",
                unsafe_allow_html=True
            )

            st.download_button(
                "⬇  Download Report (.txt)",
                report,
                file_name=f"drift_report_window_{selected_window}.txt",
                mime="text/plain"
            )
            st.caption(
                "Gemini 2.5 Flash · UCI Credit Card Default · "
                "ModelWatch v1.0"
            )

        else:
            st.info(
                "Select a monitoring window from the sidebar, "
                "then click Generate Drift Report.",
                icon="📝"
            )

    except Exception as e:
        st.error(f"Report error: {e}")
