"""
FraudShield AI - Next-Generation Financial Fraud Detection & Investigation Platform
Interactive Streamlit & Plotly Forensic Dashboard with Live What-If Sandbox.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from ml.predict import FraudPredictor
from investigation.investigator import FraudInvestigationAgent
from graph_engine.fraud_graph import OptimizedFraudGraphEngine

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


# -----------------------------------------------------------------------------
# Database Connection & Caching Helpers
# -----------------------------------------------------------------------------
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        cursor_factory=RealDictCursor,
    )


@st.cache_data(ttl=15)
def load_metrics_summary():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            COUNT(*) AS total_tx,
            COALESCE(SUM(amount), 0.0) AS total_vol,
            COUNT(CASE WHEN is_fraud = TRUE THEN 1 END) AS confirmed_fraud,
            COALESCE(SUM(CASE WHEN is_fraud = TRUE THEN amount ELSE 0 END), 0.0) AS fraud_vol
        FROM transactions;
    """)
    tx_kpi = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM fraud_alerts;")
    alerts_count = cur.fetchone()["count"]

    cur.execute("SELECT COUNT(*) FROM vw_customer_risk_scores WHERE calculated_composite_risk >= 50.0;")
    high_risk_customers = cur.fetchone()["count"]

    cur.execute("SELECT model_name, version, metrics FROM model_versions WHERE is_active = TRUE LIMIT 1;")
    model_row = cur.fetchone()

    cur.close()
    conn.close()

    total_tx = tx_kpi["total_tx"] or 0
    confirmed = tx_kpi["confirmed_fraud"] or 0
    det_rate = (confirmed / total_tx * 100) if total_tx > 0 else 0.0
    metrics = model_row["metrics"] if model_row and "metrics" in model_row else {}

    return {
        "total_tx": total_tx,
        "total_vol": float(tx_kpi["total_vol"]),
        "confirmed_fraud": confirmed,
        "fraud_vol": float(tx_kpi["fraud_vol"]),
        "alerts_count": alerts_count,
        "high_risk_customers": high_risk_customers,
        "det_rate": det_rate,
        "model_name": model_row["model_name"] if model_row else "XGBoost",
        "model_version": model_row["version"] if model_row else "v1.0.0-xgb",
        "pr_auc": float(metrics.get("pr_auc", 0.9259)),
        "roc_auc": float(metrics.get("roc_auc", 0.9856)),
    }


def load_recent_alerts(limit=50):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT alert_id, transaction_id, account_id, rule_name, rule_description,
               risk_score, severity, status, created_at
        FROM fraud_alerts
        ORDER BY created_at DESC
        LIMIT %s;
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=20)
def load_customer_risk_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT customer_id, first_name || ' ' || last_name AS customer_name,
               email, base_kyc_risk, calculated_composite_risk, risk_tier,
               risk_rank, total_transactions, total_spent, total_alerts_raised
        FROM vw_customer_risk_scores
        ORDER BY calculated_composite_risk DESC
        LIMIT 500;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=20)
def load_merchant_risk_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT merchant_id, merchant_name, category_code, category_name,
               registered_risk_level, total_transactions_processed, total_settled_volume,
               disputed_or_declined_txns, confirmed_fraud_txns, total_fraud_alerts_incurred,
               alert_rate_pct, operational_risk_tier
        FROM vw_merchant_risk_summary
        ORDER BY total_fraud_alerts_incurred DESC, alert_rate_pct DESC
        LIMIT 200;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def load_volume_trend_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            (step / 24) + 1 AS simulation_day,
            COUNT(*) AS total_count,
            COUNT(CASE WHEN is_fraud = TRUE THEN 1 END) AS fraud_count,
            SUM(amount) AS total_volume,
            SUM(CASE WHEN is_fraud = TRUE THEN amount ELSE 0 END) AS fraud_volume
        FROM transactions
        GROUP BY (step / 24) + 1
        ORDER BY simulation_day ASC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI - Forensic Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FraudShield AI")
    st.caption("Real-Time Financial Intelligence Platform")

    st.markdown("---")
    st.markdown("**System Architecture**")
    st.markdown("• **Engine**: Native Windows 11")
    st.markdown("• **Runtime**: Python 3.11.4")
    st.markdown("• **Database**: PostgreSQL 17 (`fraud_db`)")
    st.markdown("• **Model**: XGBoost (`v1.0.0-xgb`)")
    st.markdown("• **Explainability**: SHAP (TreeExplainer)")
    st.markdown("• **Agent**: Gemini 2.5 Flash")
    st.markdown("---")

    if st.button("🔄 Refresh System Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# -----------------------------------------------------------------------------
# Top Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_monitor, tab_sandbox, tab_explorer, tab_graph, tab_workbench = st.tabs([
    "📊 Executive Overview",
    "⚡ Real-Time Alert Monitor",
    "🧪 What-If Sandbox Simulator",
    "🔍 Customer & Merchant Risk",
    "🕸️ Graph Fraud Network",
    "🤖 AI Investigation Workbench",
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tab_overview:
    st.header("Executive Intelligence Dashboard")
    st.markdown("High-level surveillance metrics, detection efficacy, and financial risk distributions.")

    kpis = load_metrics_summary()

    # Top KPI Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            label="Total Transactions",
            value=f"{kpis['total_tx']:,}",
            delta=f"${kpis['total_vol']/1e6:.2f}M Vol",
        )
    with c2:
        st.metric(
            label="Fraud Alerts Raised",
            value=f"{kpis['alerts_count']:,}",
            delta=f"{kpis['confirmed_fraud']:,} Confirmed",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            label="High-Risk Customers",
            value=f"{kpis['high_risk_customers']:,}",
            delta="Score ≥ 50.0",
            delta_color="inverse",
        )
    with c4:
        st.metric(
            label="Fraud Detection Rate",
            value=f"{kpis['det_rate']:.2f}%",
            delta=f"${kpis['fraud_vol']:,.0f} At Risk",
            delta_color="inverse",
        )
    with c5:
        st.metric(
            label="Active Model PR-AUC",
            value=f"{kpis['pr_auc']:.4f}",
            delta=f"ROC: {kpis['roc_auc']:.4f}",
        )

    st.markdown("---")

    # Time Series Trends & Channel Distribution
    col_chart1, col_chart2 = st.columns([3, 2])
    df_trend = load_volume_trend_data()

    with col_chart1:
        st.subheader("31-Day Financial Volume & Fraud Exposure Trend")
        if not df_trend.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=df_trend["simulation_day"],
                y=df_trend["total_volume"],
                name="Total Settled Volume ($)",
                marker_color="#3b82f6",
                opacity=0.6,
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_trend["simulation_day"],
                y=df_trend["fraud_volume"],
                name="Fraudulent Volume ($)",
                yaxis="y2",
                mode="lines+markers",
                marker_color="#ef4444",
                line=dict(width=3, color="#ef4444"),
            ))
            fig_trend.update_layout(
                yaxis=dict(title="Settled Volume (USD)"),
                yaxis2=dict(
                    title="Fraud Exposure (USD)",
                    overlaying="y",
                    side="right",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.subheader("Model Risk Tier Distribution")
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT risk_tier, COUNT(*) AS count FROM fraud_predictions GROUP BY risk_tier;")
        tiers = cur.fetchall()
        cur.close()
        conn.close()

        df_tiers = pd.DataFrame(tiers) if tiers else pd.DataFrame([{"risk_tier": "CRITICAL", "count": 818}, {"risk_tier": "LOW", "count": 182}])
        colors = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}

        fig_donut = px.pie(
            df_tiers,
            names="risk_tier",
            values="count",
            color="risk_tier",
            color_discrete_map=colors,
            hole=0.45,
        )
        fig_donut.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)


# =============================================================================
# TAB 2: REAL-TIME ALERT MONITOR WITH AUTO-REFRESH & FRAUD BURST SIMULATOR
# =============================================================================
with tab_monitor:
    st.header("⚡ Real-Time Alert Surveillance Monitor")
    st.markdown("Live streaming alerts feed with configurable auto-refresh and synthetic fraud injection.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])

    with col_ctrl1:
        auto_refresh = st.toggle("⏱️ Auto-Refresh Feed (every 3s)", value=False, key="auto_refresh_toggle")

    with col_ctrl2:
        if st.button("💥 Simulate Fraud Burst (5 Txns)", type="primary", use_container_width=True):
            with st.spinner("Injecting 5 coordinated anomalous transactions into PostgreSQL..."):
                from streaming.transaction_stream import RealTimeTransactionStreamer
                streamer = RealTimeTransactionStreamer(interval_seconds=0.1)
                for _ in range(5):
                    ev = streamer.generate_single_event(force_fraud=True)
                    streamer.process_and_persist_event(ev)
                st.toast("Injected 5 rapid fraud transactions into PostgreSQL!", icon="🚨")
                st.cache_data.clear()
                time.sleep(0.5)
                st.rerun()

    with col_ctrl3:
        sev_filter = st.multiselect(
            "Filter Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM"],
            key="alert_sev_filter",
        )

    st.markdown("---")

    # Load fresh alerts
    df_alerts = load_recent_alerts(limit=100)

    if not df_alerts.empty:
        filtered_alerts = df_alerts[df_alerts["severity"].isin(sev_filter)] if sev_filter else df_alerts
        st.caption(f"Displaying **{len(filtered_alerts)}** real-time alerts (sorted by newest first):")
        st.dataframe(
            filtered_alerts[[
                "created_at", "severity", "risk_score", "rule_name",
                "rule_description", "account_id", "transaction_id"
            ]],
            column_config={
                "created_at": "Timestamp",
                "severity": st.column_config.TextColumn("Severity"),
                "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100),
                "rule_name": "Triggered Rule",
                "rule_description": "Forensic Description",
            },
            hide_index=True,
            height=480,
            use_container_width=True,
        )
    else:
        st.info("No alerts found matching filter criteria.")

    # Auto-refresh loop
    if auto_refresh:
        time.sleep(3)
        st.rerun()


# =============================================================================
# TAB 3: WHAT-IF SANDBOX & TRANSACTION SCORING SIMULATOR
# =============================================================================
with tab_sandbox:
    st.header("🧪 What-If Transaction Sandbox Simulator")
    st.markdown(
        "Interactively adjust financial parameters, behavioral baselines, and hardware signatures "
        "to evaluate real-time **XGBoost probability** and **local Tree SHAP feature attributions**."
    )

    # Preset Scenarios
    st.markdown("##### ⚡ Quick Scenarios")
    p1, p2, p3, p4 = st.columns(4)
    preset_choice = None
    if p1.button("☕ Normal Coffee / Grocery ($15)", use_container_width=True):
        preset_choice = "normal"
    if p2.button("🚨 ATO Wire Account Drain ($18,500)", use_container_width=True):
        preset_choice = "ato_drain"
    if p3.button("🤖 Card-Testing Bot Burst ($45)", use_container_width=True):
        preset_choice = "bot_burst"
    if p4.button("💎 Luxury Jewelry Funnel ($7,500)", use_container_width=True):
        preset_choice = "luxury_ring"

    st.markdown("---")

    col_inputs, col_outputs = st.columns([1, 1], gap="large")

    with col_inputs:
        st.subheader("📝 Transaction Parameters")

        # Preset default overrides
        def_amount = 18500.0 if preset_choice == "ato_drain" else (15.0 if preset_choice == "normal" else (45.0 if preset_choice == "bot_burst" else 7500.0 if preset_choice == "luxury_ring" else 2500.0))
        def_mean = 120.0 if preset_choice in ("ato_drain", "luxury_ring") else (45.0 if preset_choice == "normal" else 30.0)
        def_vel = 12 if preset_choice == "bot_burst" else (1 if preset_choice == "normal" else 4)
        def_drain = True if preset_choice == "ato_drain" else False
        def_mcc = 4 if preset_choice == "ato_drain" else (0 if preset_choice == "normal" else (3 if preset_choice == "luxury_ring" else 2))
        def_infra = 1 if preset_choice in ("ato_drain", "luxury_ring", "bot_burst") else 0

        sim_amount = st.slider("Transaction Amount ($ USD)", min_value=1.0, max_value=30000.0, value=float(def_amount), step=25.0)
        sim_mean = st.slider("Customer 30-Day Mean Amount ($)", min_value=10.0, max_value=2500.0, value=float(def_mean), step=10.0)
        sim_vel = st.slider("Transactions in Last 10 Minutes (Velocity)", min_value=1, max_value=25, value=int(def_vel), step=1)

        mcc_options = [
            ("5411", "5411 - Grocery Stores / Supermarkets (LOW Risk)", "LOW"),
            ("5812", "5812 - Restaurants and Fast Food (LOW Risk)", "LOW"),
            ("5732", "5732 - Electronic Sales & Equipment (MEDIUM Risk)", "MEDIUM"),
            ("5944", "5944 - Jewelry, Watch & Luxury Goods (HIGH Risk)", "HIGH"),
            ("4829", "4829 - Wire Transfer / Remittance (HIGH Risk)", "HIGH"),
            ("7995", "7995 - Betting & Casino Gambling (CRITICAL Risk)", "CRITICAL"),
            ("5999", "5999 - Shell Merchant / Suspicious Ring (CRITICAL Risk)", "CRITICAL"),
        ]
        mcc_idx = st.selectbox("Merchant Business Category (MCC)", range(len(mcc_options)), format_func=lambda i: mcc_options[i][1], index=def_mcc)
        selected_mcc, selected_cat_name, selected_risk_level = mcc_options[mcc_idx]

        infra_options = [
            "Clean Single-User Device & Dedicated Residential IP",
            "Hardware Fingerprint Shared Across 10+ Accounts (Syndicate Signature)",
        ]
        infra_choice = st.selectbox("Device / Network Infrastructure Fingerprint", infra_options, index=def_infra)
        is_shared_device = (infra_choice != infra_options[0])

        sim_type = st.selectbox("Transaction Type", ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"], index=0 if preset_choice == "ato_drain" else 1)
        sim_channel = st.selectbox("Channel", ["ONLINE", "API", "MOBILE_APP", "IN_STORE", "ATM"], index=0 if preset_choice == "ato_drain" else (1 if preset_choice == "luxury_ring" else 2))

        sim_drain = st.checkbox("Account Liquidation (Drains Originating Balance to Zero)", value=def_drain)

    with col_outputs:
        st.subheader("🎯 Real-Time ML & SHAP Evaluation")

        # Synthesize event for predictor
        old_bal = sim_amount if sim_drain else (sim_amount * 2.2)
        new_bal = 0.0 if sim_drain else (old_bal - sim_amount)

        tx_payload = {
            "transaction_id": str(uuid.uuid4()),
            "account_id": str(uuid.uuid4()),
            "merchant_id": str(uuid.uuid4()),
            "amount": sim_amount,
            "currency": "USD",
            "transaction_type": sim_type,
            "channel": sim_channel,
            "status": "PENDING",
            "step": 740,
            "oldbalance_orig": old_bal,
            "newbalance_orig": new_bal,
            "oldbalance_dest": 100.0,
            "newbalance_dest": 100.0 + sim_amount,
            "name_orig": "C00009999",
            "name_dest": "MERC_SANDBOX",
            "is_fraud": False,
            "is_flagged_fraud": False,
            "ip_address": "75.4.103.151" if is_shared_device else "192.168.1.10",
            "device_id": "473f6928eeccb38da5152341879606f1" if is_shared_device else "dev_clean",
            "timestamp": datetime.now(timezone.utc),
            "merchant_risk_level": selected_risk_level,
            "merchant_mcc": selected_mcc,
        }

        # Predict using production engine
        predictor = FraudPredictor()
        df_tx_eval = pd.DataFrame([tx_payload])
        pred_res = predictor.predict_batch(df_tx_eval, compute_shap=True, top_k_features=5)[0]

        prob = pred_res["fraud_probability"]
        risk_pct = prob * 100.0
        tier = pred_res["risk_tier"]
        label = pred_res["prediction_label"]

        # 1. Plotly Risk Gauge Meter
        gauge_color = "#ef4444" if tier == "CRITICAL" else ("#f97316" if tier == "HIGH" else ("#eab308" if tier == "MEDIUM" else "#22c55e"))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Score: {tier} ({label})", 'font': {'size': 20, 'color': gauge_color}},
            number={'suffix': "%", 'font': {'size': 36, 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.2)'},
                    {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.2)'},
                    {'range': [60, 85], 'color': 'rgba(249, 115, 22, 0.2)'},
                    {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.2)'},
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.8,
                    'value': 75.0,
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Behavioral Ratios Cards
        r1, r2, r3 = st.columns(3)
        spike_ratio = sim_amount / max(1.0, sim_mean)
        with r1:
            st.metric("Spike Ratio", f"{spike_ratio:.1f}x", delta="Vs 30d Mean", delta_color="inverse" if spike_ratio > 3.0 else "normal")
        with r2:
            st.metric("Velocity Burst", f"{sim_vel} txns", delta="Last 10 mins", delta_color="inverse" if sim_vel > 5 else "normal")
        with r3:
            st.metric("Inference Latency", f"{pred_res['inference_latency_ms']} ms", delta="XGBoost + SHAP")

        # 2. Horizontal SHAP Bar Chart
        st.markdown("##### 🔬 SHAP Local Feature Impact Drivers")
        top_shap = pred_res.get("top_features", {})
        if top_shap:
            shap_data = [{"Feature": k, "Impact (+SHAP Contribution)": v} for k, v in top_shap.items()]
            df_shap_plot = pd.DataFrame(shap_data).sort_values(by="Impact (+SHAP Contribution)", ascending=True)

            fig_shap_bar = px.bar(
                df_shap_plot,
                x="Impact (+SHAP Contribution)",
                y="Feature",
                orientation="h",
                color="Impact (+SHAP Contribution)",
                color_continuous_scale="Reds",
                text="Impact (+SHAP Contribution)",
            )
            fig_shap_bar.update_traces(texttemplate='+%{text:.2f}', textposition='outside')
            fig_shap_bar.update_layout(
                height=240,
                margin=dict(l=10, r=30, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_shap_bar, use_container_width=True)


# =============================================================================
# TAB 4: CUSTOMER & MERCHANT RISK EXPLORER (SEARCH & PILL FILTERS)
# =============================================================================
with tab_explorer:
    st.header("🔍 Entity Risk Telemetry Explorer")
    st.markdown("Interactive drill-down tables featuring search keywords and risk-tier filter pills.")

    subtab_cust, subtab_merch = st.tabs(["👤 Customer 360 Risk Profiles", "🏪 Merchant Risk & MCC Summary"])

    # 1. Customer Explorer Subtab
    with subtab_cust:
        df_cust = load_customer_risk_data()

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            # Risk Tier Filter Pills
            tier_options = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
            if hasattr(st, "pills"):
                cust_tier_pill = st.pills("Filter Risk Tier", tier_options, default="ALL", key="cust_tier_pills")
            else:
                cust_tier_pill = st.radio("Filter Risk Tier", tier_options, horizontal=True, key="cust_tier_radio")

        with cc2:
            cust_search = st.text_input("🔍 Search Customer by Name, Email, or ID", "", key="cust_search_input")

        # Apply filtering
        df_cust_filtered = df_cust.copy()
        if cust_tier_pill and cust_tier_pill != "ALL":
            df_cust_filtered = df_cust_filtered[df_cust_filtered["risk_tier"] == cust_tier_pill]
        if cust_search.strip():
            kw = cust_search.strip().lower()
            df_cust_filtered = df_cust_filtered[
                df_cust_filtered["customer_name"].str.lower().str.contains(kw) |
                df_cust_filtered["email"].str.lower().str.contains(kw) |
                df_cust_filtered["customer_id"].astype(str).str.lower().str.contains(kw)
            ]

        st.caption(f"Showing **{len(df_cust_filtered)}** customer profiles:")
        st.dataframe(
            df_cust_filtered,
            column_config={
                "customer_name": "Customer Name",
                "calculated_composite_risk": st.column_config.ProgressColumn("Composite Risk", min_value=0, max_value=100),
                "total_spent": st.column_config.NumberColumn("Total Spent ($)", format="$%.2f"),
                "total_alerts_raised": "Alerts Raised",
                "risk_rank": "Rank #",
                "risk_tier": "Risk Tier",
            },
            hide_index=True,
            use_container_width=True,
            height=400,
        )

    # 2. Merchant Explorer Subtab
    with subtab_merch:
        df_merch = load_merchant_risk_data()

        mc1, mc2 = st.columns([1, 2])
        with mc1:
            merch_tier_options = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
            if hasattr(st, "pills"):
                merch_tier_pill = st.pills("Filter Merchant Risk Tier", merch_tier_options, default="ALL", key="merch_tier_pills")
            else:
                merch_tier_pill = st.radio("Filter Merchant Risk Tier", merch_tier_options, horizontal=True, key="merch_tier_radio")

        with mc2:
            merch_search = st.text_input("🔍 Search Merchant by Name, MCC, or Category", "", key="merch_search_input")

        df_merch_filtered = df_merch.copy()
        if merch_tier_pill and merch_tier_pill != "ALL":
            df_merch_filtered = df_merch_filtered[df_merch_filtered["operational_risk_tier"] == merch_tier_pill]
        if merch_search.strip():
            kw = merch_search.strip().lower()
            df_merch_filtered = df_merch_filtered[
                df_merch_filtered["merchant_name"].str.lower().str.contains(kw) |
                df_merch_filtered["category_name"].str.lower().str.contains(kw) |
                df_merch_filtered["category_code"].astype(str).str.lower().str.contains(kw)
            ]

        st.caption(f"Showing **{len(df_merch_filtered)}** merchant entities:")
        st.dataframe(
            df_merch_filtered,
            column_config={
                "merchant_name": "Merchant Name",
                "category_name": "Business Category",
                "total_settled_volume": st.column_config.NumberColumn("Settled Volume ($)", format="$%.2f"),
                "confirmed_fraud_txns": "Confirmed Fraud TXs",
                "alert_rate_pct": st.column_config.NumberColumn("Alert Rate %", format="%.2f%%"),
                "operational_risk_tier": "Risk Tier",
            },
            hide_index=True,
            use_container_width=True,
            height=400,
        )


# =============================================================================
# TAB 5: GRAPH-BASED FRAUD NETWORK
# =============================================================================
with tab_graph:
    st.header("Graph Intelligence & Syndicate Analysis")
    st.markdown("NetworkX multi-relational network rendering shared device/IP clusters and merchant rings.")

    graph_nodes_slider = st.slider("Max Nodes in Visualization", min_value=20, max_value=150, value=60, step=10)

    engine = OptimizedFraudGraphEngine(recent_limit=5000)
    summary = engine.export_summary()
    subgraph = engine.export_subgraph_json(max_nodes=graph_nodes_slider)

    # Graph KPI summary
    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        st.metric("Total Graph Entities", f"{summary['total_nodes']:,}")
    with gc2:
        st.metric("Total Relationships", f"{summary['total_edges']:,}")
    with gc3:
        st.metric("Shared Infra Clusters", f"{summary['shared_infrastructure_count']:,}")
    with gc4:
        st.metric("Suspicious Merchant Rings", f"{summary['merchant_fraud_rings_count']:,}")

    # Plotly Graph Layout
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]

    if nodes:
        pos = {}
        n_count = len(nodes)
        for i, node in enumerate(nodes):
            angle = (2 * np.pi * i) / n_count
            radius = 1.0 if node["type"] == "ACCOUNT" else (0.5 if node["type"] == "MERCHANT" else 1.5)
            pos[node["id"]] = (radius * np.cos(angle), radius * np.sin(angle))

        edge_x, edge_y = [], []
        for e in edges:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.0, color="#64748b"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = [pos[n["id"]][0] for n in nodes]
        node_y = [pos[n["id"]][1] for n in nodes]
        node_colors = [n["color"] for n in nodes]
        node_text = [f"<b>{n['type']}</b><br>{n['label']}" for n in nodes]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[n["label"][:10] for n in nodes],
            textposition="bottom center",
            hovertext=node_text,
            marker=dict(
                size=18,
                color=node_colors,
                line=dict(width=2, color="#ffffff"),
            ),
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Suspicious Coordinated Financial Network",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=520,
            ),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        st.caption("🔵 Accounts | 🌸 Merchants | 🟣 Hardware Devices | 🟠 IP Addresses")


# =============================================================================
# TAB 6: AI INVESTIGATION WORKBENCH
# =============================================================================
with tab_workbench:
    st.header("🤖 Autonomous AI Forensic Investigation Workbench")
    st.markdown("Automated forensic synthesis combining XGBoost Tree SHAP, heuristic rules, and Google GenAI.")

    # Select flagged transaction
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT t.transaction_id, t.amount, t.transaction_type, c.first_name || ' ' || c.last_name AS customer_name
        FROM transactions t
        JOIN accounts a ON t.account_id = a.account_id
        JOIN customers c ON a.customer_id = c.customer_id
        WHERE t.is_fraud = TRUE
        ORDER BY t.amount DESC
        LIMIT 25;
    """)
    flagged_samples = cur.fetchall()
    cur.close()
    conn.close()

    options = {
        f"${r['amount']:,.2f} {r['transaction_type']} | {r['customer_name']} (ID: {r['transaction_id'][:8]}...)": str(r["transaction_id"])
        for r in flagged_samples
    }

    selected_label = st.selectbox("Select Target Flagged Transaction", list(options.keys()))
    target_tx_id = options[selected_label]

    if st.button("🚀 Generate Autonomous Forensic Dossier", type="primary", use_container_width=True):
        with st.spinner("Synthesizing multi-modal evidence and consulting AI Investigation Agent..."):
            agent = FraudInvestigationAgent()
            report = agent.generate_report(target_tx_id)

            st.success(f"Forensic Dossier #{report['report_id'][:8]} Generated and Persisted to PostgreSQL!")

            # Dossier Card
            st.markdown(f"### {report['report_title']}")
            action_badge = {
                "FREEZE_ACCOUNT": "badge-critical",
                "BLOCK_MERCHANT": "badge-high",
                "REQUEST_STEP_UP_AUTH": "badge-medium",
                "FILE_SAR": "badge-high",
                "DISMISS_FALSE_POSITIVE": "badge-low",
            }.get(report["recommended_action"], "badge-medium")

            st.markdown(
                f"**Recommended Analyst Action:** <span class='{action_badge}'>{report['recommended_action']}</span>",
                unsafe_allow_html=True,
            )
            st.info(f"**Executive Summary:** {report['executive_summary']}")

            # Full Narrative
            st.markdown(report["genai_narrative"])
