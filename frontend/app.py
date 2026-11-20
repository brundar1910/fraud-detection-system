"""
FraudShield AI - Next-Generation Financial Fraud Detection & Investigation Platform
Interactive Streamlit & Plotly Forensic Dashboard.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

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


@st.cache_data(ttl=20)
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


@st.cache_data(ttl=30)
def load_customer_risk_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT customer_id, first_name || ' ' || last_name AS customer_name,
               email, base_kyc_risk, calculated_composite_risk, risk_tier,
               risk_rank, total_transactions, total_spent, total_alerts_raised
        FROM vw_customer_risk_scores
        ORDER BY calculated_composite_risk DESC
        LIMIT 250;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
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
        LIMIT 100;
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
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
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

    if st.button("🔄 Refresh System Data"):
        st.cache_data.clear()
        st.rerun()


# -----------------------------------------------------------------------------
# Top Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_monitor, tab_explorer, tab_graph, tab_workbench = st.tabs([
    "📊 Executive Overview",
    "⚡ Real-Time Monitor & Simulator",
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
        # Query risk tiers
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
# TAB 2: REAL-TIME ALERT MONITOR & SIMULATOR
# =============================================================================
with tab_monitor:
    st.header("Real-Time Surveillance Feed & Live Simulator")
    st.markdown("Live event stream monitor integrated directly with the XGBoost + SHAP scoring pipeline.")

    col_sim, col_feed = st.columns([1, 2])

    with col_sim:
        st.subheader("🧪 Live Scoring Simulator")
        st.markdown("Inject a live transaction directly into the pipeline:")

        sim_type = st.selectbox("Transaction Scenario", [
            "Normal In-Store Payment ($45.00)",
            "High-Volume Account Drain Spike ($14,500.00)",
            "Coordinated Shell Merchant Transfer ($6,200.00)",
            "Rapid Velocity Card-Testing ($125.00)",
        ])

        custom_amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=450.0, step=50.0)
        custom_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
        custom_channel = st.selectbox("Channel", ["ONLINE", "MOBILE_APP", "IN_STORE", "ATM", "API"])

        if st.button("⚡ Score Transaction Live", type="primary"):
            # Execute live prediction
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT account_id, account_number, balance FROM accounts LIMIT 1;")
            sample_acc = cur.fetchone()
            cur.close()
            conn.close()

            acc_id = str(sample_acc["account_id"])
            predictor = FraudPredictor()

            tx_event = {
                "transaction_id": str(os.urandom(16).hex()),
                "account_id": acc_id,
                "merchant_id": None,
                "amount": custom_amount,
                "currency": "USD",
                "transaction_type": custom_type,
                "channel": custom_channel,
                "status": "PENDING",
                "step": 744,
                "oldbalance_orig": float(sample_acc["balance"]),
                "newbalance_orig": max(0.0, float(sample_acc["balance"]) - custom_amount),
                "oldbalance_dest": 0.0,
                "newbalance_dest": 0.0,
                "name_orig": sample_acc["account_number"],
                "name_dest": "MERC_SIMULATED",
                "is_fraud": False,
                "is_flagged_fraud": False,
                "ip_address": "127.0.0.1",
                "device_id": "dev_simulator",
                "timestamp": datetime.now(timezone.utc),
                "merchant_risk_level": "LOW",
                "merchant_mcc": "0000",
            }

            df_sim = pd.DataFrame([tx_event])
            pred = predictor.predict_batch(df_sim, compute_shap=True)[0]

            st.success("Transaction Evaluated Successfully!")
            st.metric("Fraud Probability", f"{pred['fraud_probability']*100:.1f}%")
            st.markdown(f"**Risk Tier:** `{pred['risk_tier']}` | **Verdict:** `{pred['prediction_label']}`")

            # SHAP Bar Chart
            if pred["top_features"]:
                st.markdown("**Top Mathematical Drivers (SHAP Impact):**")
                shap_df = pd.DataFrame([
                    {"Feature": k, "Impact (+SHAP)": v}
                    for k, v in pred["top_features"].items()
                ])
                fig_shap = px.bar(
                    shap_df,
                    x="Impact (+SHAP)",
                    y="Feature",
                    orientation="h",
                    color="Impact (+SHAP)",
                    color_continuous_scale="Reds",
                )
                fig_shap.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_shap, use_container_width=True)

    with col_feed:
        st.subheader("🚨 Live Alerts Feed")
        sev_filter = st.multiselect(
            "Filter by Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"],
        )

        df_alerts = load_recent_alerts(limit=100)
        if not df_alerts.empty:
            filtered_alerts = df_alerts[df_alerts["severity"].isin(sev_filter)]
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
                height=450,
                use_container_width=True,
            )
        else:
            st.info("No active fraud alerts detected.")


# =============================================================================
# TAB 3: CUSTOMER & MERCHANT RISK EXPLORER
# =============================================================================
with tab_explorer:
    st.header("Entity Risk Telemetry Explorer")
    st.markdown("Detailed investigation views on individual customer portfolios and registered merchant categories.")

    subtab_cust, subtab_merch = st.tabs(["👤 Customer 360 Risk Profiles", "🏪 Merchant Risk & MCC Summary"])

    with subtab_cust:
        df_cust = load_customer_risk_data()
        cust_tier = st.selectbox("Filter Customer Tier", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

        if cust_tier != "ALL":
            df_cust_view = df_cust[df_cust["risk_tier"] == cust_tier]
        else:
            df_cust_view = df_cust

        st.dataframe(
            df_cust_view,
            column_config={
                "customer_name": "Customer Name",
                "calculated_composite_risk": st.column_config.ProgressColumn("Composite Risk", min_value=0, max_value=100),
                "total_spent": st.column_config.NumberColumn("Total Spent ($)", format="$%.2f"),
                "total_alerts_raised": "Alerts Raised",
                "risk_rank": "Rank #",
            },
            hide_index=True,
            use_container_width=True,
            height=380,
        )

    with subtab_merch:
        df_merch = load_merchant_risk_data()
        st.dataframe(
            df_merch,
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
            height=380,
        )


# =============================================================================
# TAB 4: GRAPH-BASED FRAUD NETWORK
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
        # Create deterministic 2D circular/spring layout positions
        pos = {}
        n_count = len(nodes)
        for i, node in enumerate(nodes):
            angle = (2 * np.pi * i) / n_count
            radius = 1.0 if node["type"] == "ACCOUNT" else (0.5 if node["type"] == "MERCHANT" else 1.5)
            pos[node["id"]] = (radius * np.cos(angle), radius * np.sin(angle))

        edge_x, edge_y = [], []
        edge_colors = []
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
# TAB 5: AI INVESTIGATION WORKBENCH
# =============================================================================
with tab_workbench:
    st.header("Autonomous AI Forensic Investigation Workbench")
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

    if st.button("🚀 Generate Autonomous Forensic Dossier", type="primary"):
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
