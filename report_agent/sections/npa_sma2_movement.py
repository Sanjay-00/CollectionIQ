import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_npa_sma2_comparison


def _delta_pct(c: int, p: int | None) -> tuple[int | None, float | None]:
    if p is None:
        return None, None
    d = c - p
    pct = round(d / p * 100, 1) if p > 0 else (100.0 if d > 0 else 0.0)
    return d, pct


def _portfolio_row(df_curr: pd.DataFrame, df_prev: pd.DataFrame, has_prev: bool) -> dict | None:
    if "curr_bucket" not in df_curr.columns:
        return None
    npa_c  = int((df_curr["curr_bucket"] == "NPA").sum())
    sma2_c = int((df_curr["curr_bucket"] == "SMA-2").sum())
    npa_p = sma2_p = None
    if has_prev and "curr_bucket" in df_prev.columns:
        npa_p  = int((df_prev["curr_bucket"] == "NPA").sum())
        sma2_p = int((df_prev["curr_bucket"] == "SMA-2").sum())

    npa_d,  npa_dpct  = _delta_pct(npa_c,  npa_p)
    sma2_d, sma2_dpct = _delta_pct(sma2_c, sma2_p)

    # Sequence fixed per business request: curr/prev pairs, then deltas, then %change.
    return {
        "npa_current":     npa_c,
        "npa_prev":        npa_p,
        "sma2_current":    sma2_c,
        "sma2_prev":       sma2_p,
        "npa_delta":       npa_d,
        "sma2_delta":      sma2_d,
        "npa_pct_change":  npa_dpct,
        "sma2_pct_change": sma2_dpct,
    }


def _top_movers(df: pd.DataFrame, id_col: str, n: int = 5) -> tuple[list, list]:
    """id_col is the row-identity column for this dimension -- "Unit" for
    branch rows, "MNT NAME" for executive rows (compute_npa_sma2_comparison's
    own column names for each). Fewer than 2n total valid rows means "worst"
    and "best" (two independent, unrelated .head(n) sorts) can overlap --
    drop duplicates (by identity) so the same entity never appears in both
    lists. Same fix as overdue_demand.py/branch_performance.py's own
    top5/bottom5 sections."""
    if df.empty or "NPA Δ" not in df.columns:
        return [], []
    valid = df.dropna(subset=["NPA Δ"])
    worst = valid.sort_values("NPA Δ", ascending=False).head(n).to_dict("records")
    if id_col in valid.columns:
        worst_ids = {r[id_col] for r in worst}
        valid = valid[~valid[id_col].isin(worst_ids)]
    best = valid.sort_values("NPA Δ", ascending=True).head(n).to_dict("records")
    return worst, best


def compute_npa_sma2_movement(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None) -> dict | None:
    """Dedicated month-over-month NPA & SMA-2 movement section.

    Portfolio-level totals plus a region breakdown and branch top-movers,
    all built from analysis.portfolio_intelligence.compute_npa_sma2_comparison
    (the same source the dashboard's NPA/SMA-2 comparison table uses) so the
    report can't drift from the dashboard's numbers.
    """
    try:
        if df_curr is None or len(df_curr) == 0:
            return None

        has_prev = df_prev is not None and len(df_prev) > 0
        empty_prev = df_curr.iloc[0:0]

        portfolio = _portfolio_row(df_curr, df_prev if has_prev else empty_prev, has_prev)
        comparison = compute_npa_sma2_comparison(df_curr, df_prev if has_prev else empty_prev)

        region_rows = comparison.get("region", pd.DataFrame()).to_dict("records")
        branch_worst, branch_best = _top_movers(comparison.get("branch", pd.DataFrame()), "Unit")
        exec_worst, exec_best     = _top_movers(comparison.get("executive", pd.DataFrame()), "MNT NAME")

        if (
            portfolio is None and not region_rows
            and not branch_worst and not branch_best
            and not exec_worst and not exec_best
        ):
            return None

        return {
            "portfolio":     portfolio,
            "region":        region_rows,
            "branch_worst":  branch_worst,
            "branch_best":   branch_best,
            "exec_worst":    exec_worst,
            "exec_best":     exec_best,
            "has_prev":      has_prev,
        }
    except Exception:
        return None
