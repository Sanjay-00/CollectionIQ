import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_concentration_treemap
from report_agent.charts import fig_to_base64


def compute_concentration_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    try:
        fig = compute_concentration_treemap(df_curr)
        png = fig_to_base64(fig, width=1100, height=420)
        if not png:
            return None
        return {"image": png}
    except Exception:
        return None
