import os
import sys
import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from orchestrator import Orchestrator
from tools import generate_pdf_report

st.set_page_config(
    page_title="Smart Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e5fc8 0%, #2d8cf0 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.3rem 0;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #2d8cf0;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        color: black;
    }
    .anomaly-box {
        background: #fff4e5;
        border-left: 4px solid #f5a623;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        color: black;
    }
    .rec-box {
        background: #edfff2;
        border-left: 4px solid #2ec86a;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar
with st.sidebar:
    st.title("🤖 Smart Data Analyst")
    st.caption("Powered by LangChain + Groq LLaMA3")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at https://console.groq.com",
    )

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    run_btn = st.button("🚀 Run Analysis", width="stretch", type="primary")

    st.divider()
    st.markdown(
        "**Pipeline stages:**\n"
        "1. 🧹 Data Cleaning\n"
        "2. 📊 Statistical Analysis\n"
        "3. 📈 Visualization\n"
        "4. 🧠 AI Insights (Groq)"
    )


# Main content 
st.title("📊 InsightFlow AI")

if not uploaded_file:
    st.info("👆 Upload a CSV file in the sidebar to get started.")
    st.markdown(
        """
        ### What this tool does
        1. **Cleans** your data (missing values, duplicates, outliers)
        2. **Analyses** distributions, correlations & group patterns
        3. **Generates** interactive Plotly charts automatically
        4. **Explains** findings in plain English using Groq AI
        5. **Exports** a PDF summary report

        ### Sample datasets to try
        - [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
        - [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
        - [Car prices](https://raw.githubusercontent.com/dsrscientist/dataset1/master/CarPrice_Assignment.csv)
        """
    )
    st.stop()

# Load CSV
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Quick preview before running
with st.expander("📄 Raw data preview", expanded=False):
    st.dataframe(raw_df.head(20), width="stretch")
    st.caption(f"{len(raw_df):,} rows × {len(raw_df.columns)} columns")

# Run pipeline 
if run_btn:
    if not groq_key:
        st.warning("⚠️ No Groq API key provided. AI insights will use fallback mode.")

    with st.spinner("Running multi-agent pipeline..."):
        orchestrator = Orchestrator(groq_api_key=groq_key or "no-key")
        result = orchestrator.run(raw_df)

    st.session_state["result"] = result

# Display results
if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]

if not result.success:
    st.error(f"Pipeline failed:\n\n{result.error_message}")
    st.stop()

# Summary metrics 
st.subheader("📋 Pipeline Summary")
c1, c2, c3, c4, c5 = st.columns(5)
cr = result.cleaning_report
an = result.analysis

with c1:
    st.metric("Rows (clean)", f"{cr.get('rows_after', 0):,}")
with c2:
    st.metric("Duplicates removed", cr.get("duplicates_removed", 0))
with c3:
    st.metric("Outliers capped", cr.get("outliers_capped", 0))
with c4:
    st.metric("Charts generated", len(result.figures))
with c5:
    st.metric("Total time", f"{result.elapsed_seconds}s")

st.divider()

# Tabs 
tab_data, tab_stats, tab_charts, tab_insights, tab_report = st.tabs(
    ["🧹 Clean Data", "📊 Statistics", "📈 Charts", "🧠 AI Insights", "📄 Report"]
)

# Tab 1: Clean Data 
with tab_data:
    st.markdown("### Cleaned Dataset")
    st.dataframe(result.cleaned_df, width="stretch", height=350)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Cleaning Report**")
        st.json(cr)
    with col_r:
        renamed = cr.get("columns_renamed", [])
        if renamed:
            st.markdown("**Renamed Columns**")
            for r in renamed:
                st.code(r, language=None)
        conversions = cr.get("type_conversions", [])
        if conversions:
            st.markdown("**Type Conversions**")
            for c in conversions:
                st.code(c, language=None)

# Tab 2: Statistics 
with tab_stats:
    col_types = an.get("column_types", {})
    numeric_cols = col_types.get("numeric", [])
    cat_cols = col_types.get("categorical", [])

    if numeric_cols:
        st.markdown("### Descriptive Statistics")
        desc_df = pd.DataFrame(an["descriptive_stats"]).T
        st.dataframe(desc_df.style.format("{:.3f}"), width="stretch")

    if len(numeric_cols) >= 2:
        st.markdown("### Top Correlations")
        corr_df = pd.DataFrame(an.get("top_correlations", []))
        if not corr_df.empty:
            st.dataframe(corr_df, width="stretch")

    if an.get("group_by_analysis"):
        ga = an["group_by_analysis"]
        st.markdown(f"### Group-By Analysis (by `{ga['group_by_col']}`)")
        for num_col, vals in list(ga["means_per_group"].items())[:5]:
            st.caption(f"Mean {num_col} per group")
            gdf = pd.DataFrame(
                [{"group": k, "mean": v} for k, v in vals.items()]
            ).sort_values("mean", ascending=False)
            st.dataframe(gdf, width="stretch", hide_index=True)

    if an.get("key_patterns"):
        st.markdown("### Detected Patterns")
        for p in an["key_patterns"]:
            st.markdown(f"- {p}")

# Tab 3: Charts 
with tab_charts:
    if not result.figures:
        st.info("No charts were generated. Ensure your CSV has numeric or categorical columns.")
    else:
        st.markdown(f"### {len(result.figures)} Charts Generated")
        # Render in 2-column grid
        for i in range(0, len(result.figures), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(result.figures):
                    with col:
                        st.plotly_chart(result.figures[idx], width="stretch")

# Tab 4: AI Insights 
with tab_insights:
    ins = result.insights

    st.markdown("### 📝 Executive Summary")
    st.info(ins.get("executive_summary", "No summary generated."))

    key_insights = ins.get("key_insights", [])
    if key_insights:
        st.markdown("### 💡 Key Insights")
        for insight in key_insights:
            st.markdown(
                f'<div class="insight-box">💡 {insight}</div>',
                unsafe_allow_html=True,
            )

    anomalies = ins.get("anomalies", [])
    if anomalies:
        st.markdown("### ⚠️ Anomalies & Noteworthy Findings")
        for a in anomalies:
            st.markdown(
                f'<div class="anomaly-box">⚠️ {a}</div>',
                unsafe_allow_html=True,
            )

    recommendations = ins.get("recommendations", [])
    if recommendations:
        st.markdown("### ✅ Recommendations")
        for r in recommendations:
            st.markdown(
                f'<div class="rec-box">→ {r}</div>',
                unsafe_allow_html=True,
            )

    # Stage timing debug info
    with st.expander("⏱ Stage Timings"):
        for stage, secs in result.stage_times.items():
            st.write(f"`{stage}`: {secs}s")

# Tab 5: Report Download 
with tab_report:
    st.markdown("### 📄 Download Summary Report")
    st.caption("Generates a PDF with cleaning summary, statistics, and AI insights.")

    if st.button("📥 Generate PDF Report"):
        pdf_bytes = generate_pdf_report(
            cleaning_report=result.cleaning_report,
            analysis=result.analysis,
            insights=result.insights,
        )
        if pdf_bytes:
            st.download_button(
                label="⬇️ Download Report PDF",
                data=pdf_bytes,
                file_name="data_analysis_report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("fpdf2 not installed. Run `pip install fpdf2` to enable PDF export.")

    # Always offer CSV download of cleaned data
    csv_bytes = result.cleaned_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
