"""
HAVM Governance Dashboard
=========================
Interactive Streamlit dashboard for monitoring healthcare AI model
assumptions in real time.

Launch with::

    streamlit run dashboard.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Ensure project root is on sys.path so local imports work ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader
from drift_detection import detect_drift
from ood_detection import train_ood_detector, detect_ood
from uncertainty_monitor import assess_uncertainty
from fairness_monitor import assess_fairness
from train_model import train_and_save_model

# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HAVM Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Import Google Font ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Status Cards ──────────────────────────────────────────────────────── */
.status-card {
    padding: 1.2rem 1.4rem;
    border-radius: 0.75rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 5px solid;
    margin-bottom: 1rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}
.status-ok    { border-color: #00d4aa; }
.status-alert { border-color: #ff4757; }
.status-warning { border-color: #ffa502; }

.card-title {
    font-size: 0.78rem;
    color: #9ea0b5;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
    font-weight: 600;
}
.card-metric {
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    line-height: 1.1;
}
.card-metric.ok    { color: #00d4aa; }
.card-metric.alert { color: #ff4757; }
.card-metric.warn  { color: #ffa502; }

.card-desc {
    font-size: 0.78rem;
    color: #7a7c8e;
    line-height: 1.4;
}

/* ── Section Header ────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e0ee;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid #2d2d4a;
    padding-bottom: 0.4rem;
}

/* ── Dashboard title ───────────────────────────────────────────────────── */
.dash-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4aa, #6c5ce7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.dash-subtitle {
    font-size: 0.85rem;
    color: #7a7c8e;
    margin-bottom: 1.5rem;
}

/* ── Plotly chart backgrounds ──────────────────────────────────────────── */
.stPlotlyChart {
    border-radius: 0.5rem;
}

/* ── Footer ────────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #5a5c6e;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid #2d2d4a;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data & model loading (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading training data …")
def load_train_data() -> pd.DataFrame:
    """Load the full training CSV."""
    if not os.path.isfile(config.TRAIN_DATA_PATH):
        data_loader.generate_and_save_data()
    return pd.read_csv(config.TRAIN_DATA_PATH)


@st.cache_data(show_spinner="Loading incoming data …")
def load_incoming_data() -> pd.DataFrame:
    """Load the incoming / production CSV."""
    if not os.path.isfile(config.INCOMING_DATA_PATH):
        data_loader.generate_and_save_data()
    return pd.read_csv(config.INCOMING_DATA_PATH)


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load the persisted Random Forest model."""
    if not os.path.isfile(config.MODEL_PATH):
        train_and_save_model()
    return joblib.load(config.MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Helper – build a status card
# ══════════════════════════════════════════════════════════════════════════════

def status_card(title: str, icon: str, metric: str, description: str,
                status: str = "ok") -> str:
    """Return an HTML status card.  *status* is 'ok', 'alert', or 'warning'."""
    css_class = {"ok": "status-ok", "alert": "status-alert",
                 "warning": "status-warning"}[status]
    metric_class = {"ok": "ok", "alert": "alert", "warning": "warn"}[status]
    return f"""
    <div class="status-card {css_class}">
        <div class="card-title">{title}</div>
        <div class="card-metric {metric_class}">{icon} {metric}</div>
        <div class="card-desc">{description}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Plotly theming helper
# ══════════════════════════════════════════════════════════════════════════════

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c0c0d0"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the dark HAVM theme to a Plotly figure."""
    fig.update_layout(**_PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#2a2a40", zerolinecolor="#2a2a40")
    fig.update_yaxes(gridcolor="#2a2a40", zerolinecolor="#2a2a40")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 HAVM Controls")
    st.markdown("---")

    if st.button("🔄 Regenerate Data & Retrain", use_container_width=True):
        data_loader.generate_and_save_data()
        train_and_save_model()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if st.button("▶️ Run All Monitors", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Thresholds")

    drift_pval = st.slider(
        "Drift p-value threshold",
        min_value=0.01, max_value=0.10, value=0.05, step=0.01,
        help="KS test significance level. Lower = stricter."
    )
    ood_contam = st.slider(
        "OOD contamination",
        min_value=0.05, max_value=0.20, value=0.10, step=0.01,
        help="Expected outlier proportion in training data."
    )
    uncertainty_thresh = st.slider(
        "Uncertainty threshold",
        min_value=0.50, max_value=0.80, value=0.60, step=0.05,
        help="Minimum prediction confidence."
    )
    fairness_thresh = st.slider(
        "Fairness disparity threshold",
        min_value=0.05, max_value=0.20, value=0.10, step=0.01,
        help="Maximum tolerated metric disparity across groups."
    )

    st.markdown("---")
    st.info(
        "**HAVM** continuously validates the assumptions "
        "underlying deployed healthcare AI models by monitoring "
        "data drift, out-of-distribution inputs, prediction "
        "uncertainty, and demographic fairness.",
        icon="ℹ️",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load data & model
# ══════════════════════════════════════════════════════════════════════════════

train_df = load_train_data()
incoming_df = load_incoming_data()
model = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# Run all monitors
# ══════════════════════════════════════════════════════════════════════════════

drift_report = detect_drift(
    train_df, incoming_df,
    features=config.FEATURE_COLUMNS,
    p_threshold=drift_pval,
)

ood_model = train_ood_detector(
    train_df, features=config.FEATURE_COLUMNS, contamination=ood_contam,
)
ood_report = detect_ood(ood_model, incoming_df, features=config.FEATURE_COLUMNS)

uncertainty_report = assess_uncertainty(
    model, incoming_df[config.FEATURE_COLUMNS], threshold=uncertainty_thresh,
)

fairness_report = assess_fairness(
    model, incoming_df,
    sensitive_attr=config.SENSITIVE_ATTRIBUTE,
    features=config.FEATURE_COLUMNS,
    target=config.TARGET_COLUMN,
    disparity_threshold=fairness_thresh,
)


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="dash-title">Healthcare Assumption Validation &amp; Monitoring Framework</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="dash-subtitle">Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    f'  •  Training samples: {len(train_df)}  •  Incoming samples: {len(incoming_df)}</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Status Cards Row
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns(4)

# ── Drift ────────────────────────────────────────────────────────────────────
drift_pct = drift_report["summary"]["drift_percentage"]
drift_status = "alert" if drift_report["overall_drift"] else "ok"
drift_icon = "🚨" if drift_report["overall_drift"] else "✅"
with c1:
    st.markdown(
        status_card(
            "Data Drift", drift_icon,
            f"{drift_pct:.1f}%",
            f"{drift_report['summary']['drifted_features']}/{drift_report['summary']['total_features']} features drifted",
            drift_status,
        ),
        unsafe_allow_html=True,
    )

# ── OOD ──────────────────────────────────────────────────────────────────────
ood_pct = ood_report["ood_percentage"]
ood_status = "alert" if ood_report["ood_detected"] else "ok"
ood_icon = "🚨" if ood_report["ood_detected"] else "✅"
with c2:
    st.markdown(
        status_card(
            "Out-of-Distribution", ood_icon,
            f"{ood_pct:.1f}%",
            f"{ood_report['ood_count']} OOD samples detected",
            ood_status,
        ),
        unsafe_allow_html=True,
    )

# ── Uncertainty ──────────────────────────────────────────────────────────────
unc_pct = uncertainty_report["uncertain_percentage"]
unc_status = "alert" if uncertainty_report["high_uncertainty"] else "ok"
unc_icon = "🚨" if uncertainty_report["high_uncertainty"] else "✅"
with c3:
    st.markdown(
        status_card(
            "Uncertainty", unc_icon,
            f"{unc_pct:.1f}%",
            f"Mean confidence: {uncertainty_report['mean_confidence']:.2%}",
            unc_status,
        ),
        unsafe_allow_html=True,
    )

# ── Fairness ─────────────────────────────────────────────────────────────────
fair_status = "alert" if fairness_report["fairness_violated"] else "ok"
fair_icon = "🚨" if fairness_report["fairness_violated"] else "✅"
with c4:
    st.markdown(
        status_card(
            "Fairness", fair_icon,
            f"{fairness_report['max_disparity_value']:.2%}",
            f"Max disparity on {fairness_report['max_disparity_metric']}",
            fair_status,
        ),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Detailed Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_drift, tab_ood, tab_unc, tab_fair, tab_data = st.tabs([
    "📊 Data Drift", "🔍 Out-of-Distribution",
    "🎯 Uncertainty", "⚖️ Fairness", "📋 Data Overview",
])


# ── Tab 1: Data Drift ────────────────────────────────────────────────────────
with tab_drift:
    st.markdown('<div class="section-header">Kolmogorov-Smirnov Drift Test per Feature</div>',
                unsafe_allow_html=True)

    drift_df = pd.DataFrame(drift_report["results"])

    # Bar chart – KS statistic per feature
    colors = ["#ff4757" if d else "#00d4aa" for d in drift_df["drift_detected"]]
    fig = go.Figure(go.Bar(
        x=drift_df["feature"],
        y=drift_df["statistic"],
        marker_color=colors,
        text=drift_df["statistic"].round(4),
        textposition="outside",
    ))
    fig.update_layout(
        title="KS Statistic by Feature",
        xaxis_title="Feature",
        yaxis_title="KS Statistic",
        height=400,
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)

    # Results table
    display_df = drift_df.copy()
    display_df["drift_detected"] = display_df["drift_detected"].map({True: "🔴 YES", False: "🟢 No"})
    display_df.columns = ["Feature", "KS Statistic", "p-value", "Drift Detected"]
    st.dataframe(
        display_df.style.format({"KS Statistic": "{:.6f}", "p-value": "{:.6f}"}),
        use_container_width=True, hide_index=True,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Features", drift_report["summary"]["total_features"])
    col_b.metric("Drifted Features", drift_report["summary"]["drifted_features"])
    col_c.metric("Drift Percentage", f"{drift_pct:.1f}%")


# ── Tab 2: Out-of-Distribution ───────────────────────────────────────────────
with tab_ood:
    st.markdown('<div class="section-header">Isolation Forest Anomaly Detection</div>',
                unsafe_allow_html=True)

    scores = ood_report["anomaly_scores"]

    # Histogram of anomaly scores
    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=40,
        marker_color="#6c5ce7",
        opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#ff4757",
                  annotation_text="Decision boundary", annotation_position="top right")
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score (negative = outlier)",
        yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Samples", ood_report["total_count"])
    col_b.metric("OOD Samples", ood_report["ood_count"])
    col_c.metric("OOD Percentage", f"{ood_pct:.1f}%")

    # Show sample OOD records
    if ood_report["ood_indices"]:
        st.markdown("**Sample OOD Records**")
        ood_samples = incoming_df.iloc[ood_report["ood_indices"][:20]].copy()
        ood_samples.insert(0, "Anomaly Score",
                           [round(float(scores[i]), 4) for i in ood_report["ood_indices"][:20]])
        st.dataframe(ood_samples, use_container_width=True, hide_index=True)


# ── Tab 3: Uncertainty ───────────────────────────────────────────────────────
with tab_unc:
    st.markdown('<div class="section-header">Prediction Confidence Analysis</div>',
                unsafe_allow_html=True)

    confidences = uncertainty_report["confidences"]

    # Histogram of confidences
    fig = go.Figure(go.Histogram(
        x=confidences,
        nbinsx=30,
        marker_color="#00d4aa",
        opacity=0.85,
    ))
    fig.add_vline(x=uncertainty_thresh, line_dash="dash", line_color="#ff4757",
                  annotation_text=f"Threshold ({uncertainty_thresh:.0%})",
                  annotation_position="top left")
    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Confidence (max class probability)",
        yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Predictions", uncertainty_report["total_count"])
    col_b.metric("Uncertain", uncertainty_report["uncertain_count"])
    col_c.metric("Mean Confidence", f"{uncertainty_report['mean_confidence']:.2%}")
    col_d.metric("Min Confidence", f"{uncertainty_report['min_confidence']:.2%}")

    # Pie chart – confident vs uncertain
    fig_pie = go.Figure(go.Pie(
        labels=["Confident", "Uncertain"],
        values=[
            uncertainty_report["total_count"] - uncertainty_report["uncertain_count"],
            uncertainty_report["uncertain_count"],
        ],
        marker=dict(colors=["#00d4aa", "#ff4757"]),
        hole=0.5,
        textinfo="label+percent",
    ))
    fig_pie.update_layout(title="Confident vs Uncertain Predictions", height=350)
    st.plotly_chart(_apply_theme(fig_pie), use_container_width=True)

    # Table of uncertain predictions
    if uncertainty_report["uncertain_indices"]:
        st.markdown("**Uncertain Predictions (top 20)**")
        unc_idx = uncertainty_report["uncertain_indices"][:20]
        unc_df = incoming_df.iloc[unc_idx][config.FEATURE_COLUMNS].copy()
        unc_df.insert(0, "Confidence", [round(float(confidences[i]), 4) for i in unc_idx])
        unc_df.insert(1, "Prediction", [int(uncertainty_report["predictions"][i]) for i in unc_idx])
        st.dataframe(unc_df, use_container_width=True, hide_index=True)


# ── Tab 4: Fairness ──────────────────────────────────────────────────────────
with tab_fair:
    st.markdown('<div class="section-header">Demographic Fairness Analysis (Gender)</div>',
                unsafe_allow_html=True)

    group_metrics = fairness_report["group_metrics"]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    group_names = list(group_metrics.keys())

    # Grouped bar chart
    fig = go.Figure()
    colors = ["#6c5ce7", "#00d4aa"]
    for i, group in enumerate(group_names):
        fig.add_trace(go.Bar(
            name=group,
            x=metric_names,
            y=[group_metrics[group][m] for m in metric_names],
            marker_color=colors[i % len(colors)],
            text=[f"{group_metrics[group][m]:.3f}" for m in metric_names],
            textposition="outside",
        ))
    fig.update_layout(
        title="Performance Metrics by Demographic Group",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group",
        height=420,
        yaxis=dict(range=[0, 1.15]),
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)

    # Per-group metrics table
    st.markdown("**Per-Group Metrics**")
    rows = []
    for group, metrics in group_metrics.items():
        rows.append({
            "Group": group,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Count": metrics["count"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Disparity table
    st.markdown("**Disparity Metrics**")
    disp_rows = []
    for metric, value in fairness_report["disparities"].items():
        status_txt = "🔴 Exceeds" if value > fairness_thresh else "🟢 Within limit"
        disp_rows.append({"Metric": metric, "Disparity": value, "Status": status_txt})
    st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)

    if fairness_report["fairness_violated"]:
        st.error(
            f"⚠️ Fairness violation detected! Maximum disparity of "
            f"**{fairness_report['max_disparity_value']:.4f}** on "
            f"**{fairness_report['max_disparity_metric']}** exceeds "
            f"threshold ({fairness_thresh:.2f}).",
            icon="🚨",
        )
    else:
        st.success("All metric disparities are within the acceptable threshold.", icon="✅")


# ── Tab 5: Data Overview ─────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-header">Dataset Statistics & Feature Distributions</div>',
                unsafe_allow_html=True)

    col_train, col_inc = st.columns(2)
    with col_train:
        st.markdown("**Training Data** (`describe()`)")
        st.dataframe(train_df.describe().round(2), use_container_width=True)
    with col_inc:
        st.markdown("**Incoming Data** (`describe()`)")
        st.dataframe(incoming_df.describe().round(2), use_container_width=True)

    # Overlay histograms for key drifted features
    st.markdown("**Feature Distribution Comparison** (Training vs Incoming)")
    compare_features = ["age", "chol", "trestbps", "thalach"]
    cols = st.columns(2)
    for i, feat in enumerate(compare_features):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=train_df[feat], name="Training", opacity=0.6,
            marker_color="#6c5ce7", nbinsx=25,
        ))
        fig.add_trace(go.Histogram(
            x=incoming_df[feat], name="Incoming", opacity=0.6,
            marker_color="#ff6b81", nbinsx=25,
        ))
        fig.update_layout(
            title=f"{feat}",
            barmode="overlay",
            height=300,
            xaxis_title=feat,
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        with cols[i % 2]:
            st.plotly_chart(_apply_theme(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="footer">
        <strong>HAVM</strong> — Healthcare Assumption Validation &amp; Monitoring Framework
        &nbsp;•&nbsp; v1.0.0 &nbsp;•&nbsp; {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        <br>Built with Streamlit · Scikit-Learn · Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
