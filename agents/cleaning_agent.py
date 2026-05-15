import pandas as pd
import numpy as np
import re
from typing import Tuple


class DataCleaningAgent:

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
 
        report = {
            "rows_before": len(df),
            "cols_before": len(df.columns),
            "missing_values_handled": False,
            "duplicates_removed": 0,
            "outliers_capped": 0,
            "columns_renamed": [],
            "type_conversions": [],
            "rows_after": 0,
            "cols_after": 0,
        }

        df = df.copy()

        # Step 1: Normalize column names
        df, renamed = self._normalize_columns(df)
        report["columns_renamed"] = renamed

        # Step 2: Remove duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        report["duplicates_removed"] = before - len(df)

        # Step 3: Fix data types
        df, conversions = self._fix_types(df)
        report["type_conversions"] = conversions

        # Step 4: Handle missing values
        df = self._handle_missing(df)
        report["missing_values_handled"] = True

        # Step 5: Detect & cap outliers via IQR on numeric columns
        df, capped = self._handle_outliers(df)
        report["outliers_capped"] = capped

        report["rows_after"] = len(df)
        report["cols_after"] = len(df.columns)

        return df, report


    # Private helpers 

    def _normalize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        renamed = []
        new_cols = {}
        for col in df.columns:
            clean = re.sub(r"[^\w]", "_", col.strip().lower())
            clean = re.sub(r"_+", "_", clean).strip("_")
            if clean != col:
                renamed.append(f"{col} → {clean}")
            new_cols[col] = clean
        df = df.rename(columns=new_cols)
        return df, renamed

    def _fix_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        conversions = []
        for col in df.select_dtypes(include="object").columns:
            # Try numeric first
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df), 1) > 0.7:
                df[col] = converted
                conversions.append(f"{col}: object → numeric")
                continue
            # Try datetime
            try:
                converted_dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if converted_dt.notna().sum() / max(len(df), 1) > 0.7:
                    df[col] = converted_dt
                    conversions.append(f"{col}: object → datetime")
            except Exception:
                pass
        return df, conversions

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.columns:
            missing = df[col].isna().sum()
            if missing == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].ffill().bfill()
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
  
        capped_total = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            before = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            capped_total += int(before)
        return df, capped_total
