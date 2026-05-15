import json
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are a senior data scientist providing concise, actionable insights 
to a business audience. Analyze the provided dataset statistics and patterns, then respond 
ONLY with valid JSON in the exact structure below — no markdown fences, no preamble.

{
  "executive_summary": "<2-3 sentence business overview>",
  "key_insights": [
    "<insight 1>",
    "<insight 2>",
    "<insight 3>",
    "<insight 4>",
    "<insight 5>"
  ],
  "anomalies": ["<anomaly or noteworthy finding>"],
  "recommendations": ["<actionable recommendation>"]
}

Be specific. Reference column names and actual numbers from the data provided."""


class InsightGeneratorAgent:

    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )

    def run(self, analysis: dict[str, Any], df_head: str) -> dict[str, Any]:
        """
        Main entry point.
        
        Args:
            analysis: output dict from DataAnalysisAgent
            df_head:  string representation of df.head() for context
        
        Returns:
            dict with keys: executive_summary, key_insights, anomalies, recommendations
        """
        prompt = self._build_prompt(analysis, df_head)
        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            return self._parse_response(response.content)
        except Exception as e:
            return self._fallback_insights(analysis, str(e))

    # Private helpers

    def _build_prompt(self, analysis: dict, df_head: str) -> str:
        shape = analysis.get("shape", {})
        desc_stats = analysis.get("descriptive_stats", {})
        top_corr = analysis.get("top_correlations", [])
        patterns = analysis.get("key_patterns", [])
        value_counts = analysis.get("value_counts", {})
        group_analysis = analysis.get("group_by_analysis", {})

        lines = [
            f"Dataset shape: {shape.get('rows', '?')} rows × {shape.get('cols', '?')} columns",
            "",
            "Column types:",
            f"  Numeric:     {analysis['column_types']['numeric']}",
            f"  Categorical: {analysis['column_types']['categorical']}",
            f"  Datetime:    {analysis['column_types']['datetime']}",
            "",
        ]

        if desc_stats:
            lines.append("Descriptive Statistics (mean | std | min | max):")
            for col, stats in list(desc_stats.items())[:8]:
                lines.append(
                    f"  {col}: mean={stats.get('mean', 0):.2f} | "
                    f"std={stats.get('std', 0):.2f} | "
                    f"min={stats.get('min', 0):.2f} | "
                    f"max={stats.get('max', 0):.2f}"
                )
            lines.append("")

        if top_corr:
            lines.append("Top Correlations:")
            for p in top_corr:
                lines.append(f"  {p['col_a']} ↔ {p['col_b']}: r={p['correlation']:.3f}")
            lines.append("")

        if value_counts:
            lines.append("Top Category Values:")
            for col, counts in list(value_counts.items())[:3]:
                top3 = list(counts.items())[:3]
                lines.append(f"  {col}: {top3}")
            lines.append("")

        if group_analysis:
            g_col = group_analysis.get("group_by_col", "")
            lines.append(f"Group-By Analysis (grouped by '{g_col}'):")
            for num_col, vals in list(group_analysis.get("means_per_group", {}).items())[:3]:
                top_groups = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:3]
                lines.append(f"  {num_col}: {top_groups}")
            lines.append("")

        if patterns:
            lines.append("Detected Patterns:")
            for p in patterns:
                lines.append(f"  • {p}")
            lines.append("")

        lines.append("Sample rows (head):")
        lines.append(df_head)

        return "\n".join(lines)

    def _parse_response(self, content: str) -> dict[str, Any]:
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("```").strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {
                "executive_summary": clean[:500],
                "key_insights": [clean],
                "anomalies": [],
                "recommendations": [],
            }

    def _fallback_insights(self, analysis: dict, error: str) -> dict[str, Any]:
        patterns = analysis.get("key_patterns", [])
        shape = analysis.get("shape", {})
        return {
            "executive_summary": (
                f"Dataset contains {shape.get('rows', '?')} rows and "
                f"{shape.get('cols', '?')} columns. "
                f"(LLM unavailable: {error[:100]})"
            ),
            "key_insights": patterns[:5] if patterns else ["No patterns detected."],
            "anomalies": [],
            "recommendations": [],
        }
