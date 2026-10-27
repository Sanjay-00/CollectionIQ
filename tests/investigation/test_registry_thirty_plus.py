"""One 30+ definition (spec 9.5 decision 3): AI Query's registry metric equals
the investigation's rate, group by group."""
import pytest

from agents.plan_executor import execute_plan
from compiler.core import compile_logical
from investigation.metrics import get_metric
from investigation.synthetic import make_book
from investigation.tools.change import side_counts


def test_ai_query_thirty_plus_matches_the_investigation_by_region():
    _, df = make_book(6000, seed=60, churn=0.1)
    df = df.copy()
    df["Arrears / EMI"] = df["curr_bucket"].map({"STD": 0.0, "1-30 DPD": 0.5, "SMA-1": 1.5, "SMA-2": 2.5, "NPA": 4.0})
    df.loc[df.index[:200], "Loan Status"] = "CLOSED"            # non-running loans are outside both
    ir1 = {"intent": "aggregation", "view": None, "filters": [], "dimensions": ["region"],
           "measures": [{"metric": "thirty_plus_pct", "alias": "m"}], "metrics": [], "having": [],
           "order_by": [], "limit": None, "display_columns": [], "time": None}
    plan, errs = compile_logical(ir1, list(df.columns))
    assert errs == []
    out, err = execute_plan(df, plan)
    assert err == ""
    metric = get_metric("thirty_plus_rate")
    for region, rate in zip(out["RegionName"], out["m"]):
        assert rate == pytest.approx(side_counts(df[df["RegionName"] == region], metric)["rate"], abs=0.01)
    assert (df["curr_bucket"] == "NA").any()                    # no-EMI loans present, and left out of the base
