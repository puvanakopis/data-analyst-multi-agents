import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from agents import (
    DataCleaningAgent,
    DataAnalysisAgent,
    VisualizationAgent,
    InsightGeneratorAgent,
)


@dataclass
class PipelineResult:
    # Raw inputs
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Agent outputs
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cleaning_report: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    insights: dict = field(default_factory=dict)

    # Status
    success: bool = False
    error_message: str = ""
    elapsed_seconds: float = 0.0
    stage_times: dict = field(default_factory=dict)


class Orchestrator:

    def __init__(self, groq_api_key: str):
        self.cleaning_agent = DataCleaningAgent()
        self.analysis_agent = DataAnalysisAgent()
        self.visualization_agent = VisualizationAgent()
        self.insight_agent = InsightGeneratorAgent(groq_api_key=groq_api_key)

    def run(self, df: pd.DataFrame) -> PipelineResult:

        result = PipelineResult(raw_df=df)
        pipeline_start = time.time()

        # Stage 1: Data Cleaning 
        t0 = time.time()
        try:
            result.cleaned_df, result.cleaning_report = self.cleaning_agent.run(df)
            result.stage_times["cleaning"] = round(time.time() - t0, 2)
        except Exception as e:
            result.error_message = f"Cleaning stage failed: {e}\n{traceback.format_exc()}"
            result.elapsed_seconds = time.time() - pipeline_start
            return result

        # Stage 2: Data Analysis 
        t0 = time.time()
        try:
            result.analysis = self.analysis_agent.run(result.cleaned_df)
            result.stage_times["analysis"] = round(time.time() - t0, 2)
        except Exception as e:
            result.error_message = f"Analysis stage failed: {e}\n{traceback.format_exc()}"
            result.elapsed_seconds = time.time() - pipeline_start
            return result

        # Stage 3: Visualization 
        t0 = time.time()
        try:
            result.figures = self.visualization_agent.run(result.cleaned_df, result.analysis)
            result.stage_times["visualization"] = round(time.time() - t0, 2)
        except Exception as e:
            result.stage_times["visualization"] = round(time.time() - t0, 2)
            result.error_message += f"\nVisualization warning: {e}"

        # Stage 4: Insight Generation
        t0 = time.time()
        try:
            df_head_str = result.cleaned_df.head(5).to_string(index=False)
            result.insights = self.insight_agent.run(result.analysis, df_head_str)
            result.stage_times["insights"] = round(time.time() - t0, 2)
        except Exception as e:
            result.stage_times["insights"] = round(time.time() - t0, 2)
            result.insights = {
                "executive_summary": "Insight generation encountered an error.",
                "key_insights": [],
                "anomalies": [],
                "recommendations": [str(e)],
            }

        result.success = True
        result.elapsed_seconds = round(time.time() - pipeline_start, 2)
        return result
