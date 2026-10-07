"""Exposure: where the money sits and with whom. Fleet operators,
the largest accounts, SOH concentration, and the action lists built on
them (repossession candidates, good customers). Pure pandas, no LLM."""


import pandas as pd
import plotly.graph_objects as go

from utils import (
    BUCKET_SCORE, CUSTOMER_LOAN_COUNT, CUSTOMER_SOH, add_customer_loan_count, as_of_or_today, fleet_loan_mask,
    loan_flags, segment_column, to_num, unit_metrics,
)
from config import (
    REPOSSESSION_WINDOW_MONTHS, GOOD_CUSTOMER_MIN_TENURE_PCT, GOOD_CUSTOMER_MIN_LCC_PCT, REPOSSESSION_BUCKETS,
    REPOSSESSION_EXCLUDE_STATUSES,
)


def _soh_cr(df: pd.DataFrame) -> float:
    return round(to_num(df, "SOH").sum() / 1e7, 2)


def compute_concentration_treemap(df_curr: pd.DataFrame) -> go.Figure:
    """
    Two-level treemap: Region → Branch.
    Size = SOH (Cr). Color = NPA% (green→red).

    Values are built leaf-up (branch → region → root) so each parent value
    equals the exact sum of its children  -  avoiding the branchvalues="total"
    blank-render bug caused by rounding drift when _soh_cr rounds independently
    at each level.
    """
    if "RegionName" not in df_curr.columns or "Unit" not in df_curr.columns:
        return go.Figure()

    ids, labels, parents, values, colors, texts = ["portfolio"], ["Portfolio"], [""], [0.0], [0.0], [""]
    if df_curr.empty:
        return go.Figure()
    flags = loan_flags(df_curr)
    branches = unit_metrics(df_curr, ["RegionName", "Unit"], flags=flags)
    if branches.empty:
        return go.Figure()
    region_npa = unit_metrics(df_curr, ["RegionName"], flags=flags).set_index("RegionName")["NPA%"]

    portfolio_total = 0.0
    for region, rb in branches.groupby("RegionName", sort=True):
        r_id  = f"r_{region}"
        r_npa = float(region_npa.get(region, 0.0))

        # Branches first, accumulating their exact (unrounded) SOH for the region
        branch_sum = 0.0
        branch_buf: list[tuple] = []
        for _, b in rb.iterrows():
            branch = b["Unit"]
            b_soh = float(b["SOH"] / 1e7)
            b_npa = float(b["NPA%"])
            b_id  = f"b_{region}_{branch}"
            branch_buf.append((b_id, str(branch), r_id, b_soh, b_npa,
                                f"{branch}<br>SOH ₹{b_soh:.1f}Cr<br>NPA {b_npa:.1f}%"))
            branch_sum += b_soh

        # Region value = exact branch sum (no independent rounding)
        ids.append(r_id); labels.append(str(region)); parents.append("portfolio")
        values.append(branch_sum); colors.append(r_npa)
        texts.append(f"{region}<br>SOH ₹{branch_sum:.1f}Cr<br>NPA {r_npa:.1f}%")
        portfolio_total += branch_sum

        for (bid, bl, bp, bsoh, bnpa, btxt) in branch_buf:
            ids.append(bid); labels.append(bl); parents.append(bp)
            values.append(bsoh); colors.append(bnpa); texts.append(btxt)

    # Root value = exact sum of region values
    values[0] = portfolio_total

    if len(ids) < 3:
        return go.Figure()

    leaf_npas = [c for c in colors[1:] if isinstance(c, (int, float))]
    cmax = max(max(leaf_npas, default=0), 1.0)

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        customdata=texts,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="%{label}",
        textfont=dict(size=12),
        marker=dict(
            colors=colors,
            # NPA has no "safe" level, so never green: pale pink (low) -> dark red (high).
            colorscale=[[0, "#fef2f2"], [0.3, "#fca5a5"], [0.6, "#ef4444"], [1.0, "#991b1b"]],
            cmin=0, cmax=cmax,
            showscale=True,
            colorbar=dict(title="NPA %", thickness=12, len=0.6),
        ),
        branchvalues="total",
        pathbar=dict(visible=True),
    ))
    fig.update_layout(
        title=dict(text="Concentration Map: Region → Branch  (size = SOH, color = NPA%)", font=dict(size=13, color="#000")),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        paper_bgcolor="white",
    )
    return fig


def _grouped_mode(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Vectorized per-group mode, matching Series.mode().iat[0]'s own tie-break
    (pandas' mode() returns every tied value sorted ascending; .iat[0] takes
    the smallest) -- one groupby+sort+drop_duplicates pass over the WHOLE
    column, replacing a Python-level .mode() call paid once per group (each
    itself an O(n log n) sort) -- confirmed via profiling to be the dominant
    cost of compute_fleet_exposure on a real ~60k-row file (~1,600+ fleet
    customers x 2 mode() calls each, ~2.7s). Groups with no non-null
    value_col rows are simply absent from the result -- callers already
    handle a missing key via .get(...)/reindex + fillna("")."""
    sub = df[[group_col, value_col]].dropna(subset=[value_col])
    if sub.empty:
        return pd.Series(dtype=object)
    counts = sub.groupby([group_col, value_col], observed=True).size().rename("_n").reset_index()
    counts = counts.sort_values([group_col, "_n", value_col], ascending=[True, False, True])
    winners = counts.drop_duplicates(subset=[group_col], keep="first")
    return winners.set_index(group_col)[value_col]


def compute_fleet_exposure(df_curr: pd.DataFrame, top_n: int | None = 20) -> dict:
    """
    Customers (identified by Cust Mob No) with 3+ loans  -  fleet operators.
    Returns summary dict + top_df (fleet customers ranked by total SOH).
    NOTE: Cust Mob No may not be unique across branches  -  treat counts as approximate.

    Loans with a blank/missing Cust Mob No are excluded before grouping (see
    excluded_blank_mobile_loans below) -- utils.py::clean_mobile normalizes a
    missing mobile number to "" (not NaN), and without this exclusion every
    such loan groups under the single key "" and gets reported as one
    fictitious "fleet operator" combining unrelated customers' exposure.
    Confirmed on real production data: 175 loans with no mobile number on
    file collapsed into a single phantom "fleet operator" with ~47.9 Cr of
    combined SOH -- by far the largest entry in the table, and pure noise.
    """
    empty_dict = {
        "count": 0, "total_soh_cr": 0.0, "npa_operators": 0, "top_df": pd.DataFrame(),
        "excluded_blank_mobile_loans": 0,
    }
    if "Cust Mob No" not in df_curr.columns or "Loan No" not in df_curr.columns:
        return empty_dict

    has_mobile = df_curr["Cust Mob No"].astype(str).str.strip() != ""
    excluded_blank_mobile_loans = int((~has_mobile).sum())
    df_curr = df_curr[has_mobile]
    if df_curr.empty:
        return {**empty_dict, "excluded_blank_mobile_loans": excluded_blank_mobile_loans}

    # Fleet status comes from the customer's loans across the whole upload
    # (utils.CUSTOMER_LOAN_COUNT), so a sidebar filter can't demote a fleet
    # operator whose other loans are filtered out; "Loans" below still counts
    # only the loans in view, consistent with the SOH shown beside it.
    fleet_df = df_curr[fleet_loan_mask(df_curr)]
    fleet_customers = pd.Index(fleet_df["Cust Mob No"].unique())
    if fleet_df.empty:
        return {**empty_dict, "excluded_blank_mobile_loans": excluded_blank_mobile_loans}

    n_operators = fleet_customers.nunique()
    total_soh   = _soh_cr(fleet_df)

    # Fleet operators with at least 1 NPA loan
    npa_ops = 0
    if "curr_bucket" in fleet_df.columns:
        has_npa = fleet_df[fleet_df["curr_bucket"] == "NPA"].groupby("Cust Mob No").size()
        npa_ops = len(has_npa)

    # Top fleet customers by SOH -- vectorized (see _grouped_mode's docstring
    # for why this replaced a per-customer Python loop). first_rows uses
    # .head(1) per group (NOT .first(), which silently skips NaN and would
    # disagree with the original .iloc[0] on a customer whose first row's
    # Cust Name happens to be blank).
    fleet_df = fleet_df.copy()
    fleet_df["_soh_num"] = to_num(fleet_df, "SOH")
    fleet_df["_is_npa"]  = (fleet_df["curr_bucket"] == "NPA") if "curr_bucket" in fleet_df.columns else False

    grp_loans  = fleet_df.groupby("Cust Mob No")["Loan No"].nunique()
    grp_soh    = (fleet_df.groupby("Cust Mob No")["_soh_num"].sum() / 1e7).round(2)
    grp_npa    = fleet_df.groupby("Cust Mob No")["_is_npa"].sum().astype(int)
    first_rows = fleet_df.groupby("Cust Mob No", sort=False).head(1).set_index("Cust Mob No")
    grp_name   = first_rows["Cust Name"] if "Cust Name" in first_rows.columns else pd.Series(dtype=object)
    # A fleet customer's loans CAN span multiple regions/branches (Cust Mob
    # No isn't guaranteed unique per branch, per this function's own
    # docstring) -- the most common (mode) region/branch is shown as a
    # single representative value, not necessarily every branch this
    # customer touches. Same convention report_agent's own fleet section
    # used to compute independently; now the single source of truth.
    grp_region = _grouped_mode(fleet_df, "Cust Mob No", "RegionName") if "RegionName" in fleet_df.columns else pd.Series(dtype=object)
    grp_unit   = _grouped_mode(fleet_df, "Cust Mob No", "Unit") if "Unit" in fleet_df.columns else pd.Series(dtype=object)

    # "Cust Name" missing entirely -> str(mob) for every customer (matches the
    # original per-row `... if "Cust Name" in grp.columns else str(mob)`).
    # "Cust Name" present but blank for a given customer's first row -> keep
    # that blank as-is, same as the original `.iloc[0]` (no per-row fallback).
    customer_col = grp_name.reindex(fleet_customers) if "Cust Name" in fleet_df.columns else pd.Series(fleet_customers, index=fleet_customers)

    top_df = pd.DataFrame({
        "Customer":       customer_col,
        "Mobile":         pd.Series(fleet_customers, index=fleet_customers).astype(str),
        "Region":         grp_region.reindex(fleet_customers).fillna(""),
        "Unit":           grp_unit.reindex(fleet_customers).fillna(""),
        "Loans":          grp_loans.reindex(fleet_customers),
        "NPA Loans":      grp_npa.reindex(fleet_customers).fillna(0).astype(int),
        "Total SOH (Cr)": grp_soh.reindex(fleet_customers),
    })
    top_df = top_df.sort_values("Total SOH (Cr)", ascending=False)
    top_df = (top_df if top_n is None else top_df.head(top_n)).reset_index(drop=True)
    # Python's str(), not pandas' .astype(str) -- under pandas 3.x's default
    # string dtype, .astype(str) preserves NaN as a null instead of the
    # literal string "nan" that the original code's `str(cust_name)` (a
    # per-row Python builtin call) always produced. Only 20 rows by now
    # (after head(20)), so a per-row .apply(str) here is negligible cost --
    # this is about matching Python's str() semantics exactly, not performance.
    top_df["Customer"] = top_df["Customer"].apply(str)

    return {
        "count": n_operators,
        "total_soh_cr": total_soh,
        "npa_operators": npa_ops,
        "top_df": top_df,
        "excluded_blank_mobile_loans": excluded_blank_mobile_loans,
    }


def compute_top_accounts(df_curr: pd.DataFrame, n: int = 20) -> tuple[pd.DataFrame, dict]:
    """
    Top N accounts by SOH among DELINQUENT accounts (any non-STD bucket)  -  the
    largest single exposures actually at risk, not just the largest loans overall
    (a big healthy STD loan isn't a collections priority).

    Returns (top_df, summary): total SOH of these N accounts, what % of the
    whole portfolio's SOH they represent (single-borrower concentration), and
    how many are already NPA  -  the most severe subset of an already-delinquent
    list.
    """
    empty_summary = {"total_soh_cr": 0.0, "pct_of_portfolio": 0.0, "npa_count": 0}
    if df_curr.empty or "SOH" not in df_curr.columns or "curr_bucket" not in df_curr.columns:
        return pd.DataFrame(), empty_summary

    # NA = missing Arrears/EMI data (not a real delinquency state) - exclude it
    # alongside STD, same as VALID_BUCKETS elsewhere in this module.
    delinquent = df_curr[~df_curr["curr_bucket"].isin(["STD", "NA"])]
    if delinquent.empty:
        return pd.DataFrame(), empty_summary

    seg_col = segment_column(delinquent)
    cols = [c for c in [
        "Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit", seg_col, "MNT NAME",
        "curr_bucket", "SOH", "Closing Arrears", "Arrears / EMI", "VehEMI Accrued", "LCC%",
        "Loan Status", "Ag_Date", "Non Starter", "CoLending_Loans",
    ] if c and c in delinquent.columns]

    top_df = (
        delinquent[cols]
        .assign(SOH=lambda d: to_num(d, "SOH"))
        .sort_values("SOH", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    total_portfolio_soh = to_num(df_curr, "SOH").sum()
    top_soh = top_df["SOH"].sum()
    npa_count = int((top_df["curr_bucket"] == "NPA").sum())
    summary = {
        "total_soh_cr": round(top_soh / 1e7, 2),
        "pct_of_portfolio": round(top_soh / total_portfolio_soh * 100, 1) if total_portfolio_soh > 0 else 0.0,
        "npa_count": npa_count,
    }
    return top_df, summary


# ── Repossession Analysis ─────────────────────────────────────────────────────

_REPO_DISPLAY_COLS = [
    "Loan No", "Cust Name", "Cust Mob No", "RegionName", "Unit", "MNT NAME",
    "curr_bucket", "Ag_Date", "SOH", "LCC%", "Arrears / EMI",
    "Closing Arrears", "Loan Amount", "Vehicle Description", "FUEL_TYPE",
    "Veh ID", "Last Receipt Date", "Last Receipt Amount",
]


def compute_repossession_list(df_curr: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """
    Accounts eligible for repossession:
      - curr_bucket in config.REPOSSESSION_BUCKETS (default SMA-2, NPA)
      - Ag_Date within config.REPOSSESSION_WINDOW_MONTHS (collateral value left)
      - Loan Status not in config.REPOSSESSION_EXCLUDE_STATUSES (S&S = the
        vehicle is already seized and sold)

    as_of: the report's OWN reporting month/date, same reasoning as
    compute_product_analysis's as_of -- defaults to the real wall-clock date
    only when the caller has no reporting date to hand. Without this, the
    "still within the repossession window" cutoff silently measured backward
    from whatever day the code happened to RUN, not the month the upload is
    reporting on -- wrong for any retroactive/historical analysis.

    Returns cleaned DataFrame. Caller sorts for the 3 views.
    """
    if df_curr.empty:
        return pd.DataFrame()

    cutoff = as_of_or_today(as_of) - pd.DateOffset(months=REPOSSESSION_WINDOW_MONTHS)

    bucket_mask = pd.Series(False, index=df_curr.index)
    if "curr_bucket" in df_curr.columns:
        bucket_mask = df_curr["curr_bucket"].isin(REPOSSESSION_BUCKETS)
    if "Loan Status" in df_curr.columns:
        status = df_curr["Loan Status"].astype(str).str.strip().str.upper()
        bucket_mask &= ~status.isin([s.upper() for s in REPOSSESSION_EXCLUDE_STATUSES])

    date_mask = pd.Series(True, index=df_curr.index)
    if "Ag_Date" in df_curr.columns:
        ag = pd.to_datetime(df_curr["Ag_Date"], errors="coerce")
        date_mask = ag >= cutoff

    repo_df = df_curr[bucket_mask & date_mask].copy()
    if repo_df.empty:
        return pd.DataFrame()

    for col in ["SOH", "LCC%", "Arrears / EMI"]:
        if col in repo_df.columns:
            repo_df[col] = to_num(repo_df, col)
    if "Ag_Date" in repo_df.columns:
        repo_df["Ag_Date"] = pd.to_datetime(repo_df["Ag_Date"], errors="coerce").dt.date

    cols = [c for c in _REPO_DISPLAY_COLS if c in repo_df.columns]
    return repo_df[cols].reset_index(drop=True)


# ── Good Customers ────────────────────────────────────────────────────────────

_GOOD_CUST_DISPLAY_COLS = [
    "Loan No", "Cust Name", "RegionName", "Unit",
    "Ag_Date", "Tenure", "VehEMI Accrued",
    "Arrears against Inst+Exp", "SOH", "Arrears / EMI",
    "LCC%", "Tenure Completed %",
]


def compute_good_customers(df_curr: pd.DataFrame) -> pd.DataFrame:
    """
    Loyal / high-quality customers eligible for refinance or relationship management.
    Criteria:
      - VehEMI Accrued / Tenure >= 70%  (completed at least 70% of loan term)
      - LCC% >= 100%                     (collected everything ever due, no lifetime shortfall)
    Sorted by SOH ascending (lowest exposure first - easiest refinance targets).
    """
    if df_curr.empty:
        return pd.DataFrame()

    df = df_curr.copy()

    # Tenure completed %
    if "VehEMI Accrued" in df.columns and "Tenure" in df.columns:
        emi_paid = to_num(df, "VehEMI Accrued")
        tenure   = to_num(df, "Tenure")
        df["Tenure Completed %"] = (emi_paid / tenure * 100).round(1)
        tenure_mask = (df["Tenure Completed %"] >= GOOD_CUSTOMER_MIN_TENURE_PCT) & tenure.notna() & (tenure > 0)
    else:
        df["Tenure Completed %"] = None
        tenure_mask = pd.Series(False, index=df.index)

    # LCC% hard cutoff
    lcc_mask = to_num(df, "LCC%") >= GOOD_CUSTOMER_MIN_LCC_PCT

    good_df = df[tenure_mask & lcc_mask].copy()
    if good_df.empty:
        return pd.DataFrame()

    # Numeric clean-up for display
    for col in ["SOH", "Arrears / EMI", "LCC%", "Arrears against Inst+Exp"]:
        if col in good_df.columns:
            good_df[col] = to_num(good_df, col)

    good_df = good_df.sort_values("SOH", ascending=True)

    cols = [c for c in _GOOD_CUST_DISPLAY_COLS if c in good_df.columns]
    return good_df[cols].reset_index(drop=True)


# ── Large customers ─────────────────────────────────────────────────────────
# Large customers: borrowers whose loans add up to a big SOH, and which of
# them have a loan behind on payment.
# 
# A customer is matched by mobile number (Cust Mob No), the same rule as fleet
# operators. "Exposure" is the customer's total SOH across the WHOLE upload
# (utils.CUSTOMER_SOH, worked out at load time), so a sidebar filter on one
# branch never hides a customer whose other loans sit in another branch. The
# loans listed are the ones in view.

_COLLECTED = "Month Collection (Excluding Reserve Collection)"

# Loan-level columns, in reading order: who, where, how bad, how much, proof.
LOAN_COLS = [
    "Cust Name", "Cust Mob No", "Loan No", "RegionName", "Unit", "MNT NAME", "curr_bucket", "Delinquent",
    "SOH", "POS", "Arrears / EMI", "Closing Arrears", "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "Month Due-Inst", _COLLECTED, "Last Receipt Date", "Last Receipt Amount",
    "Ag_Date", "Loan Amount", "Make", "Vehicle Description", "Veh ID",
]


def _join(values: pd.Series) -> str:
    seen = [str(v) for v in dict.fromkeys(values.dropna()) if str(v).strip() and str(v).lower() != "nan"]
    return ", ".join(seen)


def large_customers(df: pd.DataFrame, min_soh_cr: float) -> dict:
    """{"customers": one row per customer, largest exposure first;
        "loans": every loan of those customers in view, the customer's
                 delinquent loans first;
        "largest_cr": the biggest customer exposure in the upload (so an empty
                      result can say how far off the threshold is)}."""
    empty = {"customers": pd.DataFrame(), "loans": pd.DataFrame(), "largest_cr": 0.0}
    if df.empty or "Cust Mob No" not in df.columns or "Loan No" not in df.columns:
        return empty
    if CUSTOMER_SOH not in df.columns:          # hand-built frames: count within themselves
        df = add_customer_loan_count(df)
    if CUSTOMER_SOH not in df.columns:
        return empty
    exposure = pd.to_numeric(df[CUSTOMER_SOH], errors="coerce")
    largest = float(exposure.max() / 1e7) if exposure.notna().any() else 0.0
    d = df[exposure >= min_soh_cr * 1e7].copy()
    if d.empty:
        return {**empty, "largest_cr": round(largest, 2)}

    f = loan_flags(d)
    mob = d["Cust Mob No"].astype(str).str.strip()
    d["_soh"], d["_delq"], d["_npa"] = f["soh"], f["delinquent"], f["npa"]
    d["_delq_soh"] = f["soh"].where(f["delinquent"], 0.0)
    d["_arrears"] = to_num(d, "Closing Arrears", fill=0)
    d["_score"] = d["curr_bucket"].map(BUCKET_SCORE) if "curr_bucket" in d.columns else 0
    g = d.groupby(mob, sort=False)

    worst = g["_score"].max()
    score_to_bucket = {v: k for k, v in BUCKET_SCORE.items()}
    cust = pd.DataFrame({
        "Customer": g["Cust Name"].first() if "Cust Name" in d.columns else "",
        "Mobile": g["Cust Mob No"].first(),
        "Exposure (Cr)": (g[CUSTOMER_SOH].first() / 1e7).round(2),
        "Loans in Book": g[CUSTOMER_LOAN_COUNT].first() if CUSTOMER_LOAN_COUNT in d.columns else g["Loan No"].nunique(),
        "Loans": g["Loan No"].nunique(),
        "Delinquent": g["_delq"].sum().astype(int),
        "NPA": g["_npa"].sum().astype(int),
        "SOH (Cr)": (g["_soh"].sum() / 1e7).round(2),
        "Delinquent SOH (Cr)": (g["_delq_soh"].sum() / 1e7).round(2),
        "Overdue (Closing Arrears)": g["_arrears"].sum().round(0),
        "Worst Bucket": worst.map(score_to_bucket).fillna("-"),
        "Branches": g["Unit"].agg(_join) if "Unit" in d.columns else "",
        "Regions": g["RegionName"].agg(_join) if "RegionName" in d.columns else "",
        "Executives": g["MNT NAME"].agg(_join) if "MNT NAME" in d.columns else "",
    })
    cust["Delinquent %"] = (cust["Delinquent"] / cust["Loans"] * 100).round(2)
    cust["Delinquent SOH %"] = (cust["Delinquent SOH (Cr)"] / cust["SOH (Cr)"].where(cust["SOH (Cr)"] > 0) * 100) \
        .round(2).fillna(0.0)
    cust["Has Delinquent Loan"] = cust["Delinquent"] > 0
    cust = cust.sort_values(["Exposure (Cr)", "Delinquent SOH (Cr)"], ascending=False, kind="stable").reset_index(drop=True)

    # Loans: grouped by customer (largest exposure first), delinquent loans
    # on top within a customer, then by SOH.
    order = {m: i for i, m in enumerate(cust["Mobile"].astype(str).str.strip())}
    d["Delinquent"] = d["_delq"].map({True: "Yes", False: "No"})
    d["_order"] = mob.map(order)
    loans = d.sort_values(["_order", "_delq", "_soh"], ascending=[True, False, False], kind="stable")
    loans = loans[[c for c in LOAN_COLS if c in loans.columns]].reset_index(drop=True)
    return {"customers": cust, "loans": loans, "largest_cr": round(largest, 2)}
