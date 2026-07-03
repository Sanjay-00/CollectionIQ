import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.portfolio_intelligence import compute_good_customers


def compute_good_customers_section(df_curr: pd.DataFrame, df_prev: pd.DataFrame = None, n: int = 10) -> dict | None:
    """Loyal, high-quality customers (tenure completed, no lifetime collection shortfall) -
    refinance / relationship-management / retention targets, a positive counterpart to the risk lists.

    Criteria (config.py GOOD_CUSTOMER_MIN_TENURE_PCT / GOOD_CUSTOMER_MIN_LCC_PCT) are strict by
    design - LCC% >= 100 means zero lifetime shortfall. A distressed portfolio can legitimately
    have none, so this always returns a dict (never None) and the report renders an explicit
    "no accounts meet criteria" message instead of silently dropping the section - matching the
    dashboard's Section 8 behavior, which does the same on the identical computation.
    """
    try:
        if df_curr is None or len(df_curr) == 0:
            return None
        df = compute_good_customers(df_curr)
        return {"rows": df.head(n).to_dict("records"), "total": len(df)}
    except Exception:
        return None
