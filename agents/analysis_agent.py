import pandas as pd
import numpy as np
from typing import Any


class DataAnalysisAgent:

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        analysis: dict[str, Any] = {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols,
            },
            "descriptive_stats": {},
            "missing_summary": {},
            "correlation_matrix": {},
            "top_correlations": [],
            "group_by_analysis": {},
            "value_counts": {},
            "key_patterns": [],
        }

        # Descriptive statistics 
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            analysis["descriptive_stats"] = {
                col: {k: float(v) for k, v in stats.items()}
                for col, stats in desc.items()
            }

        # Missing value summary 
        missing = df.isna().sum()
        analysis["missing_summary"] = {
            col: int(cnt) for col, cnt in missing[missing > 0].items()
        }

        # Correlation matrix
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(3)
            analysis["correlation_matrix"] = corr.to_dict()

            pairs = []
            for i, col_a in enumerate(numeric_cols):
                for col_b in numeric_cols[i + 1:]:
                    val = corr.loc[col_a, col_b]
                    pairs.append((col_a, col_b, float(val)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            analysis["top_correlations"] = [
                {"col_a": a, "col_b": b, "correlation": r}
                for a, b, r in pairs[:5]
            ]

        # Categorical value counts 
        for col in categorical_cols[:5]: 
            vc = df[col].value_counts().head(10).to_dict()
            analysis["value_counts"][col] = {str(k): int(v) for k, v in vc.items()}

        # Group-by analysis 
        if categorical_cols and numeric_cols:
            group_col = self._best_group_col(df, categorical_cols)
            if group_col:
                grouped = (
                    df.groupby(group_col)[numeric_cols]
                    .mean()
                    .round(3)
                    .to_dict()
                )
                analysis["group_by_analysis"] = {
                    "group_by_col": group_col,
                    "means_per_group": {
                        num_col: {str(k): float(v) for k, v in vals.items()}
                        for num_col, vals in grouped.items()
                    },
                }

        # Key patterns (heuristic insights) 
        analysis["key_patterns"] = self._extract_patterns(df, analysis)

        return analysis

    #  Private helpers

    def _best_group_col(self, df: pd.DataFrame, cat_cols: list) -> str | None:
        for col in cat_cols:
            n = df[col].nunique()
            if 2 <= n <= 20:
                return col
        return None

    def _extract_patterns(self, df: pd.DataFrame, analysis: dict) -> list[str]:
        patterns = []

        # Skewness
        for col, stats in analysis["descriptive_stats"].items():
            mean = stats.get("mean", 0)
            median = stats.get("50%", 0)
            if abs(mean - median) / (abs(mean) + 1e-9) > 0.2:
                direction = "right" if mean > median else "left"
                patterns.append(f"'{col}' is {direction}-skewed (mean={mean:.2f}, median={median:.2f})")

        # Strong correlations
        for pair in analysis["top_correlations"]:
            r = pair["correlation"]
            if abs(r) > 0.7:
                strength = "strong positive" if r > 0 else "strong negative"
                patterns.append(
                    f"'{pair['col_a']}' and '{pair['col_b']}' have a {strength} correlation ({r:.2f})"
                )

        # Dominant category
        for col, counts in analysis["value_counts"].items():
            total = sum(counts.values())
            if total > 0:
                top_cat, top_cnt = next(iter(counts.items()))
                pct = top_cnt / total * 100
                if pct > 50:
                    patterns.append(
                        f"'{col}' is dominated by '{top_cat}' ({pct:.1f}% of records)"
                    )

        return patterns
