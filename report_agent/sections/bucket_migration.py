import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.roll_rate import compute_roll_rate_matrix, compute_roll_rate_kpis


def compute_bucket_migration_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict | None:
    try:
        if df_prev is None or len(df_prev) == 0:
            return None
        matrix, meta = compute_roll_rate_matrix(df_curr, df_prev)
        if meta["matched_count"] == 0:
            return None
        # Serialize matrix to plain dict for report builder
        matrix_dict = {row: matrix.loc[row].to_dict() for row in matrix.index}
        return {
            "matrix":            matrix_dict,
            "buckets":           matrix.columns.tolist(),
            "roll_forward_rate": meta["roll_forward_rate"],
            "cure_rate":         meta["cure_rate"],
            "npa_formation_rate": meta["npa_formation_rate"],
            "matched_count":     meta["matched_count"],
            "new_entries":       meta["new_entries"],
            "exits":             meta["exits"],
        }
    except Exception:
        return None
