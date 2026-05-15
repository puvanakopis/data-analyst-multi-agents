import io
from datetime import datetime
from typing import Any

try:
    from fpdf import FPDF

    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False


def generate_pdf_report(
    cleaning_report: dict,
    analysis: dict,
    insights: dict,
    filename: str = "data_analysis_report",
) -> bytes | None:
 
    if not _FPDF_AVAILABLE:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header 
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_fill_color(30, 90, 200)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "Smart Data Analysis Report", ln=True, fill=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(6)

    # Section helper 
    def sanitize(text: str) -> str:
        return text.replace("•", "-").replace("→", "->").replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').encode("latin-1", "replace").decode("latin-1")

    def section_title(title: str):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(220, 230, 255)
        pdf.cell(0, 9, sanitize(title), ln=True, fill=True)
        pdf.ln(2)

    def body(text: str):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize(text))
        pdf.ln(1)

    # Cleaning Report 
    section_title("1. Data Cleaning Summary")
    body(
        f"Rows before cleaning : {cleaning_report.get('rows_before', '?')}\n"
        f"Rows after cleaning  : {cleaning_report.get('rows_after', '?')}\n"
        f"Duplicates removed   : {cleaning_report.get('duplicates_removed', 0)}\n"
        f"Outliers capped      : {cleaning_report.get('outliers_capped', 0)}\n"
        f"Missing values fixed : {cleaning_report.get('missing_values_handled', False)}"
    )

    renamed = cleaning_report.get("columns_renamed", [])
    if renamed:
        body("Renamed columns:\n" + "\n".join(f"  - {r}" for r in renamed))

    # Dataset Overview
    section_title("2. Dataset Overview")
    shape = analysis.get("shape", {})
    col_types = analysis.get("column_types", {})
    body(
        f"Rows: {shape.get('rows', '?')}    Columns: {shape.get('cols', '?')}\n"
        f"Numeric columns    : {', '.join(col_types.get('numeric', [])) or 'None'}\n"
        f"Categorical columns: {', '.join(col_types.get('categorical', [])) or 'None'}\n"
        f"Datetime columns   : {', '.join(col_types.get('datetime', [])) or 'None'}"
    )

    # Descriptive Stats
    section_title("3. Descriptive Statistics")
    desc = analysis.get("descriptive_stats", {})
    for col, stats in list(desc.items())[:10]:
        body(
            f"{col}: mean={stats.get('mean', 0):.2f}  "
            f"std={stats.get('std', 0):.2f}  "
            f"min={stats.get('min', 0):.2f}  "
            f"max={stats.get('max', 0):.2f}"
        )

    # Key Patterns
    patterns = analysis.get("key_patterns", [])
    if patterns:
        section_title("4. Detected Patterns")
        for p in patterns:
            body(f"- {p}")

    # AI Insights
    section_title("5. AI-Generated Insights")
    body(insights.get("executive_summary", ""))

    key_insights = insights.get("key_insights", [])
    if key_insights:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Key Insights:", ln=True)
        for insight in key_insights:
            body(f"  - {insight}")

    anomalies = insights.get("anomalies", [])
    if anomalies:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Anomalies:", ln=True)
        for a in anomalies:
            body(f"  ! {a}")

    # Footer
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150)
    pdf.cell(0, 8, "Smart Data Analysis Pipeline - AI-Powered Report", align="C")

    return bytes(pdf.output())
