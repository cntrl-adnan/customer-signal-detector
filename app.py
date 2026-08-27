"""
Intelligent Customer Signal Detector - Streamlit Operations Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List

from src.models import AnalyzedCustomerRecord
from src.data_loader import DataLoader
from src.signal_analyzer import SignalAnalyzer

# Page configuration
st.set_page_config(
    page_title="Intelligent Customer Signal Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; }
    .sub-title { font-size: 0.95rem; color: #475569; margin-bottom: 1.5rem; }
    .badge-critical { background-color: #ffe4e6; color: #be123c; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-high { background-color: #fef3c7; color: #b45309; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-medium { background-color: #e0e7ff; color: #4338ca; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-low { background-color: #d1fae5; color: #047857; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "analyzed_records" not in st.session_state:
        st.session_state["analyzed_records"] = []
    if "validation_errors" not in st.session_state:
        st.session_state["validation_errors"] = []
    if "raw_df" not in st.session_state:
        # Load sample benchmark by default
        st.session_state["raw_df"] = pd.read_csv("data/sample_customers.csv")


init_session_state()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("🛡️ Signal Detector")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Data Ingestion")
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Sample Benchmark (14 Accounts)", "Upload Custom CSV"]
)

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state["raw_df"] = pd.read_csv(uploaded_file)
else:
    st.session_state["raw_df"] = pd.read_csv("data/sample_customers.csv")

st.sidebar.info(f"Loaded **{len(st.session_state['raw_df'])}** customer rows.")

st.sidebar.subheader("2. AI Engine Mode")
mock_mode = st.sidebar.toggle("Use Offline Mock Mode", value=False, help="Runs deterministic offline simulation without requiring live Gemini API network calls.")

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("🚀 Run Signal Analysis", type="primary", use_container_width=True)

# -------------------------------------------------------------
# RUN ANALYSIS PIPELINE
# -------------------------------------------------------------
if analyze_btn or len(st.session_state["analyzed_records"]) == 0:
    with st.spinner("Correlating multi-signal text interactions & behavioral telemetry..."):
        try:
            records, errors = DataLoader.load_from_dataframe(st.session_state["raw_df"])
            st.session_state["validation_errors"] = errors

            analyzer = SignalAnalyzer(mock_mode=mock_mode)
            analyzed = analyzer.analyze_batch(records)
            st.session_state["analyzed_records"] = analyzed
            st.toast(f"Successfully analyzed {len(analyzed)} customer accounts!", icon="✅")
        except Exception as e:
            st.error(f"Error during analysis: {e}")

analyzed_records: List[AnalyzedCustomerRecord] = st.session_state["analyzed_records"]
validation_errors = st.session_state["validation_errors"]

# -------------------------------------------------------------
# MAIN DASHBOARD VIEW
# -------------------------------------------------------------
st.markdown('<div class="main-title">Intelligent Customer Signal Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Proactive multi-signal early warning intelligence for customer operations & retention teams</div>', unsafe_allow_html=True)

# Validation Error Alert (if any rows failed)
if validation_errors:
    with st.expander(f"⚠️ {len(validation_errors)} Row Validation Errors Detected", expanded=False):
        st.dataframe(pd.DataFrame(validation_errors), use_container_width=True)

# 1. TOP KPI SUMMARY METRICS
critical_count = sum(1 for r in analyzed_records if r.risk_band == "Critical")
high_count = sum(1 for r in analyzed_records if r.risk_band == "High")
medium_count = sum(1 for r in analyzed_records if r.risk_band == "Medium")
low_count = sum(1 for r in analyzed_records if r.risk_band == "Low")
needs_review_count = sum(1 for r in analyzed_records if r.semantic_status == "needs_review")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Monitored Accounts", len(analyzed_records))
col2.metric("Critical (≥75)", critical_count, help="Urgent operational escalation required")
col3.metric("High (50-74)", high_count, help="Priority outreach required")
col4.metric("Medium (25-49)", medium_count)
col5.metric("Low (<25)", low_count)
col6.metric("Needs Review", needs_review_count, help="Confidence below 0.60 or unparsed")

st.markdown("---")

# 2. VISUAL CHARTS & SUMMARY
chart_col1, chart_col2 = st.columns([1, 2])

with chart_col1:
    st.subheader("Risk Band Distribution")
    band_counts = pd.DataFrame([
        {"Band": "Critical", "Count": critical_count, "Color": "#e11d48"},
        {"Band": "High", "Count": high_count, "Color": "#f59e0b"},
        {"Band": "Medium", "Count": medium_count, "Color": "#6366f1"},
        {"Band": "Low", "Count": low_count, "Color": "#10b981"},
    ])
    fig = px.bar(
        band_counts,
        x="Band",
        y="Count",
        color="Band",
        color_discrete_map={"Critical": "#e11d48", "High": "#f59e0b", "Medium": "#6366f1", "Low": "#10b981"},
        text="Count"
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Multi-Signal Risk Correlation (Scatter Matrix)")
    scatter_data = []
    for r in analyzed_records:
        sentiment_val = 0
        if r.semantic_signal:
            sentiment_map = {"very_negative": -1.0, "negative": -0.5, "neutral": 0.0, "positive": 0.8}
            sentiment_val = sentiment_map.get(r.semantic_signal.sentiment, 0.0)

        scatter_data.append({
            "Customer ID": r.customer_id,
            "Name": r.customer_name,
            "Risk Score": r.risk_score,
            "Risk Band": r.risk_band,
            "Usage Change %": r.usage_change_pct,
            "Sentiment": sentiment_val,
            "Churn Threat": "Yes" if (r.semantic_signal and r.semantic_signal.churn_intent) else "No",
            "Primary Issue": r.semantic_signal.issue_type if r.semantic_signal else "Unknown"
        })
    if scatter_data:
        df_scatter = pd.DataFrame(scatter_data)
        fig_scatter = px.scatter(
            df_scatter,
            x="Sentiment",
            y="Usage Change %",
            color="Risk Band",
            size="Risk Score",
            hover_name="Name",
            hover_data=["Customer ID", "Risk Score", "Primary Issue", "Churn Threat"],
            color_discrete_map={"Critical": "#e11d48", "High": "#f59e0b", "Medium": "#6366f1", "Low": "#10b981"}
        )
        fig_scatter.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No analyzed records to plot yet.")

st.markdown("---")

# 3. PRIORITIZED QUEUE TABLE
st.subheader("Prioritized Customer Action Queue")

filter_col1, filter_col2 = st.columns([1, 3])
with filter_col1:
    band_filter = st.selectbox("Filter by Risk Tier:", ["All Tiers", "Critical", "High", "Medium", "Low"])

display_list = analyzed_records
if band_filter != "All Tiers":
    display_list = [r for r in display_list if r.risk_band == band_filter]

table_data = []
for r in display_list:
    top_contrib_text = ", ".join([c.signal_name for c in r.contributors[:2]]) if r.contributors else "none"
    table_data.append({
        "Customer ID": r.customer_id,
        "Customer Name": r.customer_name,
        "Risk Score": r.risk_score,
        "Risk Band": r.risk_band,
        "Issue Category": r.semantic_signal.issue_type if r.semantic_signal else "unknown",
        "Churn Intent": "🚨 True" if (r.semantic_signal and r.semantic_signal.churn_intent) else "False",
        "Top Contributors": top_contrib_text,
        "Analysis Source": r.analysis_source,
        "Semantic Status": r.semantic_status,
        "Suggested Action": r.suggested_action[:45] + "..." if len(r.suggested_action) > 45 else r.suggested_action,
    })

st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=280)

# 4. CUSTOMER DETAIL DRILL-DOWN INSPECTOR
st.markdown("---")
st.subheader("🔍 Customer Deep-Dive & Action Breakdown")

customer_ids = [r.customer_id for r in analyzed_records]

if customer_ids:
    selected_id = st.selectbox("Select an Account to Inspect:", customer_ids)
    selected_record = next((r for r in analyzed_records if r.customer_id == selected_id), None)
else:
    st.info("Run an analysis to inspect individual accounts.")
    selected_record = None

if selected_record:
    d_col1, d_col2 = st.columns([1, 1])

    with d_col1:
        st.markdown(f"### {selected_record.customer_name} (`{selected_record.customer_id}`)")
        
        # Risk Badge
        st.markdown(
            f"**Deterministic Risk Score**: `{selected_record.risk_score}/100` &nbsp;|&nbsp; "
            f"**Tier**: `{selected_record.risk_band}` &nbsp;|&nbsp; "
            f"**Semantic Status**: `{selected_record.semantic_status}`"
        )
        
        # Controlled Action Box
        st.info(
            f"**Suggested prototype action** (needs human review):\n\n"
            f"👉 {selected_record.suggested_action}"
        )

        if selected_record.semantic_status == "unavailable":
            st.warning(
                "**Semantic analysis unavailable** — this score is based on "
                "structured telemetry alone. No AI interpretation was applied.\n\n"
                f"`{selected_record.semantic_error or 'unknown error'}`"
            )
        elif selected_record.semantic_status == "needs_review":
            st.warning(
                "**Needs review** — model confidence was below the 0.60 threshold, "
                "so semantic signals were excluded from the score."
            )

        # Human Rationale Box
        st.success(f"**Explainable Rationale**:\n\n{selected_record.rationale}")

        # Raw Transcript
        with st.expander("📄 Raw Interaction Transcript", expanded=False):
            st.code(selected_record.interaction_text, language="text")

    with d_col2:
        st.markdown("#### Point-by-Point Score Breakdown")
        if selected_record.contributors:
            contrib_df = pd.DataFrame([
                {
                    "Signal": c.signal_name,
                    "Rule Triggered": c.rule_description,
                    "Points Awarded": f"+{c.points} pts"
                }
                for c in selected_record.contributors
            ])
            st.table(contrib_df)
        else:
            st.write("No positive risk points triggered.")

        # Structured Telemetry Summary
        st.markdown("#### Behavioral Telemetry")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Usage Change", f"{selected_record.usage_change_pct}%")
        t_col2.metric("Payment", selected_record.payment_status.capitalize())
        t_col3.metric("30d Issues", selected_record.issue_count_30d)
        t_col4.metric("CSAT", f"{selected_record.csat_score}/5" if selected_record.csat_score else "N/A")