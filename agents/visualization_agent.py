import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any


CHART_PALETTE = px.colors.qualitative.Plotly


class VisualizationAgent:

    def run(self, df: pd.DataFrame, analysis: dict[str, Any]) -> list[go.Figure]:

        figures: list[go.Figure] = []
        numeric_cols: list[str] = analysis["column_types"]["numeric"]
        categorical_cols: list[str] = analysis["column_types"]["categorical"]
        datetime_cols: list[str] = analysis["column_types"]["datetime"]

        # 1. Distribution histograms for each numeric column (up to 6)
        for col in numeric_cols[:6]:
            fig = self._histogram(df, col)
            if fig:
                figures.append(fig)

        # 2. Bar charts for categorical columns (up to 3)
        for col in categorical_cols[:3]:
            fig = self._bar_chart(df, col)
            if fig:
                figures.append(fig)

        # 3. Pie chart for first categorical col with ≤10 unique values
        for col in categorical_cols:
            if df[col].nunique() <= 10:
                fig = self._pie_chart(df, col)
                if fig:
                    figures.append(fig)
                break

        # 4. Scatter plot for top correlated pair
        top_corr = analysis.get("top_correlations", [])
        if top_corr:
            pair = top_corr[0]
            fig = self._scatter(df, pair["col_a"], pair["col_b"], categorical_cols)
            if fig:
                figures.append(fig)

        # 5. Correlation heatmap (if ≥3 numeric cols)
        if len(numeric_cols) >= 3:
            fig = self._correlation_heatmap(analysis)
            if fig:
                figures.append(fig)

        # 6. Box plots for numeric cols grouped by best categorical col (up to 3)
        group_analysis = analysis.get("group_by_analysis", {})
        group_col = group_analysis.get("group_by_col")
        if group_col and numeric_cols:
            for num_col in numeric_cols[:3]:
                fig = self._box_plot(df, num_col, group_col)
                if fig:
                    figures.append(fig)

        # 7. Time-series line chart if datetime column exists
        if datetime_cols and numeric_cols:
            fig = self._time_series(df, datetime_cols[0], numeric_cols[0])
            if fig:
                figures.append(fig)

        return figures

    # Private chart builders

    def _histogram(self, df: pd.DataFrame, col: str) -> go.Figure | None:
        try:
            fig = px.histogram(
                df, x=col,
                nbins=30,
                title=f"Distribution of {col.replace('_', ' ').title()}",
                color_discrete_sequence=[CHART_PALETTE[0]],
                template="plotly_white",
            )
            fig.update_layout(bargap=0.05)
            return fig
        except Exception:
            return None

    def _bar_chart(self, df: pd.DataFrame, col: str) -> go.Figure | None:
        try:
            vc = df[col].value_counts().head(15).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(
                vc, x=col, y="count",
                title=f"Frequency of {col.replace('_', ' ').title()}",
                color=col,
                color_discrete_sequence=CHART_PALETTE,
                template="plotly_white",
            )
            fig.update_layout(showlegend=False)
            return fig
        except Exception:
            return None

    def _pie_chart(self, df: pd.DataFrame, col: str) -> go.Figure | None:
        try:
            vc = df[col].value_counts().head(10)
            fig = px.pie(
                values=vc.values,
                names=vc.index,
                title=f"Share of {col.replace('_', ' ').title()}",
                color_discrete_sequence=CHART_PALETTE,
                template="plotly_white",
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            return fig
        except Exception:
            return None

    def _scatter(
        self, df: pd.DataFrame, x: str, y: str, categorical_cols: list[str]
    ) -> go.Figure | None:
        try:
            color_col = categorical_cols[0] if categorical_cols else None
            if color_col and df[color_col].nunique() > 15:
                color_col = None
            fig = px.scatter(
                df, x=x, y=y, color=color_col,
                title=f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}",
                trendline="ols",
                color_discrete_sequence=CHART_PALETTE,
                template="plotly_white",
                opacity=0.7,
            )
            return fig
        except Exception:
            return None

    def _correlation_heatmap(self, analysis: dict) -> go.Figure | None:
        try:
            corr_dict = analysis.get("correlation_matrix", {})
            if not corr_dict:
                return None
            corr_df = pd.DataFrame(corr_dict)
            fig = go.Figure(
                go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns.tolist(),
                    y=corr_df.index.tolist(),
                    colorscale="RdBu",
                    zmid=0,
                    text=corr_df.round(2).values,
                    texttemplate="%{text}",
                    showscale=True,
                )
            )
            fig.update_layout(
                title="Correlation Heatmap",
                template="plotly_white",
                xaxis_tickangle=-30,
            )
            return fig
        except Exception:
            return None

    def _box_plot(self, df: pd.DataFrame, num_col: str, group_col: str) -> go.Figure | None:
        try:
            fig = px.box(
                df, x=group_col, y=num_col,
                title=f"{num_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}",
                color=group_col,
                color_discrete_sequence=CHART_PALETTE,
                template="plotly_white",
            )
            fig.update_layout(showlegend=False)
            return fig
        except Exception:
            return None

    def _time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure | None:
        try:
            ts = df[[date_col, value_col]].dropna().sort_values(date_col)
            fig = px.line(
                ts, x=date_col, y=value_col,
                title=f"{value_col.replace('_', ' ').title()} Over Time",
                template="plotly_white",
                color_discrete_sequence=[CHART_PALETTE[2]],
            )
            return fig
        except Exception:
            return None
