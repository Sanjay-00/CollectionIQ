"""Views  -  the FAST-PATH vocabulary layer.

Companion to registry/ontology.py (CONCEPTS/METRICS) and registry/semantic_model.py
(DIMENSIONS): a fixed dict the planner picks a NAME from, never a place it authors
logic. Each entry maps a natural-language-matchable "known view" to a precomputed
analysis/ function that ALREADY answers that exact business question -- some of
these (Concern Score composite ranking, quartile executive tiers, narrative
synthesis) encode logic the general IR-1 compiler cannot express at all via its
filter/measure/dimension algebra, so this isn't just a shortcut, it's the only
path to a correct answer for that class of question.

Only views whose analysis/ function returns a DataFrame, a (DataFrame, dict) tuple,
or a flat dict are included -- anything returning a Plotly Figure/HTML string is
out of scope (the AI Query tab's result renderer expects tabular/KPI data).

Schema per entry:
    label        human-readable name, for the prompt catalog
    description  what question this answers, for the planner to semantically match
    fn           dotted path to the analysis/ function, resolved lazily (keeps this
                 module -- and the compiler/registry layer generally -- from eagerly
                 importing all of analysis/, and avoids import cycles)
    inputs       which QueryState-derived DataFrames feed the function, in call order
    params       {name: {"type", "default"}} -- runtime parameters the planner may
                 override (e.g. top_delinquent_accounts.n); empty dict if fixed
    requires     QueryState fields that must be non-empty for this view to be usable
                 (e.g. "df_prev") -- view_node falls through to the compiler path if missing
    filterable   whether an ad-hoc row-level "filters" list may be layered on top
                 (view_node pre-filters the input DataFrame(s) before calling fn)
    output       selects the normalizer in _NORMALIZERS that turns the function's
                 raw return value into (result_df, result_kpis, result_rankings)
    cache_key    key into the `precomputed_views` dict app.py threads through
                 QueryState -- used to serve the literal cached object when no
                 filter/non-default param requires a fresh call
"""
import importlib

import pandas as pd

VIEWS: dict[str, dict] = {
    "top_delinquent_accounts": {
        "label": "Top Delinquent Accounts by SOH",
        "description": (
            "Top N accounts by SOH (exposure) among currently delinquent accounts (any "
            "non-STD, non-NA bucket), with a summary of their share of total portfolio "
            "SOH and how many are already NPA. Use for 'top N delinquent/at-risk "
            "customers/accounts by SOH/exposure/outstanding'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_top_accounts",
        "inputs": ["df_curr"],
        "params": {"n": {"type": "int", "default": 20}},
        "requires": [],
        "filterable": True,
        "output": "df_dict_tuple",
        "cache_key": "pi_top_accounts",
        "grain": "customer",
    },
    "fleet_operators": {
        "label": "Fleet Operator Exposure",
        "description": (
            "Customers holding 3+ loans (fleet operators), ranked by total SOH, with a "
            "count / NPA-operator summary. Use for 'fleet customers/operators', "
            "'multi-loan customers', 'customers with multiple vehicles'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_fleet_exposure",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": True,
        "output": "dict_with_top_df",
        "cache_key": "pi_fleet",
        "grain": "customer",
    },
    "executive_scorecard": {
        "label": "Executive Performance Scorecard",
        "description": (
            "Per-field-executive collection scorecard: performance Tier (top/mid/bottom "
            "quartile), Strike Rate%, Collection%, roll forward/backward%, NPA/SMA-2 "
            "count and %. Use for 'executive scorecard', 'how are our executives "
            "performing', 'rank executives'."
        ),
        "fn": "analysis.executive_scorecard.compute_executive_scorecard",
        "inputs": ["df_curr"],
        "params": {"min_accounts": {"type": "int", "default": 5}},
        "requires": [],
        "filterable": True,
        "output": "df",
        "cache_key": "scorecard_df",
        "grain": "executive",
    },
    "roll_rate_matrix": {
        "label": "Bucket Roll-Rate Migration",
        "description": (
            "Month-over-month bucket migration matrix (which buckets accounts moved "
            "from/to), plus roll-forward rate, roll-backward rate, and NPA formation "
            "rate. Use for 'roll rate', 'bucket migration', 'how many accounts moved "
            "buckets'. Requires a previous-period file."
        ),
        "fn": "analysis.roll_rate.compute_roll_rate_matrix",
        "inputs": ["df_curr", "df_prev"],
        "params": {},
        "requires": ["df_prev"],
        "filterable": False,   # a migration crosstab isn't row-filterable the same way
        "output": "matrix_tuple",
        "cache_key": "rr_matrix",
        "grain": "matrix",
    },
    "region_scorecard": {
        "label": "Region Delinquency Scorecard",
        "description": (
            "Per-region snapshot: NPA% (current + previous + change), Collection%, "
            "Hard Bucket%, SOH, roll rates, and an Improving/Stable/Worsening status "
            "label. Use for 'region scorecard', 'how are regions performing', 'which "
            "regions are worsening'. Requires a previous-period file for the trend/status."
        ),
        "fn": "analysis.portfolio_intelligence.compute_region_scorecard",
        "inputs": ["df_curr", "df_prev"],
        "params": {},
        "requires": [],
        "filterable": True,
        "output": "df",
        "cache_key": "pi_region",
        "grain": "region",
    },
    "branch_quadrant": {
        "label": "Branch Performance Quadrant",
        "description": (
            "Branch risk ranking by a weighted Concern Score (Collection%, Hard "
            "Bucket%, Roll Forward%, chronic defaulters) with a healthy/watch/"
            "underperforming/intervene-now quadrant label. Use for 'branch quadrant', "
            "'which branches need intervention', 'branch concern score/ranking'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_branch_quadrant",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": True,
        "output": "tuple_df_fig",
        "cache_key": "pi_branch",
        "grain": "branch",
    },
    "executive_recovery": {
        "label": "Executive Recovery Ranking",
        "description": (
            "Executives ranked by net accounts rescued from high-risk buckets (rescued "
            "minus slipped) between the previous and current period. Use for 'which "
            "executives are rescuing the most accounts', 'executive recovery ranking'. "
            "Requires a previous-period file."
        ),
        "fn": "analysis.portfolio_intelligence.compute_executive_recovery",
        "inputs": ["df_curr"],
        "params": {},
        "requires": ["df_prev"],
        "filterable": True,
        "output": "df",
        "cache_key": "pi_exec",
        "grain": "executive",
    },
    "risk_indicators": {
        "label": "Early Warning Risk Indicators",
        "description": (
            "8+ month-over-month early-warning signals: SMA-1/SMA-2/NPA pool %, fresh "
            "NPA formation, chronic defaulters, non-starters, co-lending at risk, each "
            "with an Improving/Stable/Worsening direction. Use for 'risk indicators', "
            "'early warning signals', 'what's getting worse'. Requires a previous-period file."
        ),
        "fn": "analysis.portfolio_intelligence.compute_risk_indicators",
        "inputs": ["df_curr", "df_prev", "rr_meta"],
        "params": {},
        "requires": ["df_prev"],
        "filterable": False,   # signal list, not a row-level frame to pre-filter
        "output": "list_of_dicts",
        "render": "risk_indicator_table",   # UI hint: dedicated signal-table renderer, not a raw grid
        "cache_key": "pi_risk",
        "grain": "signal",
    },
    "repossession_list": {
        "label": "Repossession-Eligible Accounts",
        "description": (
            "Accounts eligible for repossession: recent loans (agreement date within "
            "18 months) already in SMA-2 or NPA. Use for 'repossession list', "
            "'accounts to repossess', 'vehicles to seize'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_repossession_list",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": True,
        "output": "df",
        "cache_key": "pi_repo_df",
        "grain": "customer",
    },
    "good_customers": {
        "label": "Good Customers (Refinance Candidates)",
        "description": (
            "Loyal, low-risk customers eligible for refinance/relationship management: "
            "tenure completed >=70% and LCC%=100 (no lifetime shortfall). Use for 'good "
            "customers', 'refinance candidates', 'loyal customers'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_good_customers",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": True,
        "output": "df",
        "cache_key": "pi_good_customers",
        "grain": "customer",
    },
    "npa_sma2_by_region": {
        "label": "NPA/SMA-2 Comparison by Region",
        "description": (
            "Region-wise NPA and SMA-2 counts with month-over-month deltas. Use for "
            "'NPA by region vs last month', 'region NPA change'. Requires a "
            "previous-period file."
        ),
        "fn": "analysis.portfolio_intelligence.compute_npa_sma2_comparison",
        "inputs": ["df_curr", "df_prev"],
        "params": {},
        "requires": ["df_prev"],
        "filterable": False,   # source dict is pre-grouped, not a row-level frame
        "output": "dict_subkey_df",
        "subkey": "region",
        "cache_key": "pi_npa_sma2_cmp",
        "grain": "region",
    },
    "npa_sma2_by_branch": {
        "label": "NPA/SMA-2 Comparison by Branch",
        "description": (
            "Branch-wise NPA and SMA-2 counts with month-over-month deltas. Use for "
            "'NPA by branch vs last month', 'branch NPA change'. Requires a "
            "previous-period file."
        ),
        "fn": "analysis.portfolio_intelligence.compute_npa_sma2_comparison",
        "inputs": ["df_curr", "df_prev"],
        "params": {},
        "requires": ["df_prev"],
        "filterable": False,
        "output": "dict_subkey_df",
        "subkey": "branch",
        "cache_key": "pi_npa_sma2_cmp",
        "grain": "branch",
    },
    "npa_sma2_by_executive": {
        "label": "NPA/SMA-2 Comparison by Executive",
        "description": (
            "Executive-wise NPA and SMA-2 counts with month-over-month deltas. Use for "
            "'NPA by executive vs last month', 'executive NPA change'. Requires a "
            "previous-period file."
        ),
        "fn": "analysis.portfolio_intelligence.compute_npa_sma2_comparison",
        "inputs": ["df_curr", "df_prev"],
        "params": {},
        "requires": ["df_prev"],
        "filterable": False,
        "output": "dict_subkey_df",
        "subkey": "executive",
        "cache_key": "pi_npa_sma2_cmp",
        "grain": "executive",
    },
    "segment_analysis": {
        "label": "Delinquency by Vehicle Segment",
        "description": (
            "Delinquency breakdown by vehicle segment: accounts, SMA-2%, NPA%, "
            "Collection%, SOH. Use for 'which segment has highest NPA', 'delinquency "
            "by vehicle segment/product'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_product_analysis",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": False,
        "output": "dict_subkey_df",
        "subkey": "segment",
        "cache_key": "pi_product",
        "grain": "segment",
    },
    "fuel_analysis": {
        "label": "Delinquency by Fuel Type",
        "description": (
            "Delinquency breakdown by fuel type: accounts, SMA-2%, NPA%, Collection%, "
            "SOH. Use for 'delinquency by fuel type', 'diesel vs petrol vs EV performance'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_product_analysis",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": False,
        "output": "dict_subkey_df",
        "subkey": "fuel",
        "cache_key": "pi_product",
        "grain": "segment",
    },
    "vintage_analysis": {
        "label": "Delinquency by Disbursement Vintage",
        "description": (
            "Delinquency breakdown by disbursement cohort (month of loan agreement): "
            "accounts, NPA/SMA-2 count and %, Collection%, SOH, average loan size. Use "
            "for 'vintage analysis', 'which disbursement cohort is worst', 'delinquency "
            "by loan age/cohort'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_product_analysis",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": False,
        "output": "dict_subkey_df",
        "subkey": "vintage",
        "cache_key": "pi_product",
        "grain": "segment",
    },
    "source_analysis": {
        "label": "Delinquency by Sourcing Channel",
        "description": (
            "Delinquency breakdown by sourcing dealer/DSA: accounts, SMA-2%, NPA%, "
            "Collection%, SOH. Use for 'which dealer/DSA brings the most delinquency', "
            "'sourcing channel risk'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_product_analysis",
        "inputs": ["df_curr"],
        "params": {},
        "requires": [],
        "filterable": False,
        "output": "dict_subkey_df",
        "subkey": "source",
        "cache_key": "pi_product",
        "grain": "segment",
    },
    "pulse_kpis": {
        "label": "Portfolio Pulse KPI Summary",
        "description": (
            "Top-line portfolio KPI cards: total accounts, SOH, SMA-2, NPA, Hard "
            "Bucket%, Collection%, Strike%, each with a month-over-month delta. Use "
            "for 'portfolio pulse', 'overall portfolio summary/health', 'top-line KPIs'."
        ),
        "fn": "analysis.portfolio_intelligence.compute_pulse_kpis",
        "inputs": ["df_curr", "df_prev"],
        "params": {},
        "requires": [],
        "filterable": False,   # a fixed KPI card list, not a row-level frame
        "output": "list_of_dicts",
        "render": "kpi_cards",   # UI hint: render as KPI cards, not a raw grid
        "cache_key": "pi_pulse_kpis",
        "grain": "portfolio",
    },
    "good_bad_summary": {
        "label": "Good/Bad Portfolio Narrative",
        "description": (
            "A synthesized narrative of the top improving ('good') and top concerning "
            "('bad') signals across regions, branches, executives, and risk indicators "
            "this period. Use for 'what's good and bad this month', 'portfolio "
            "narrative summary'. Only available from the cached dashboard computation."
        ),
        # No fresh-callable inputs: compute_good_bad composes from OTHER views'
        # outputs (region/branch/exec DataFrames + risk indicators), not raw
        # df_curr/df_prev. A cache miss safely TypeErrors and falls through --
        # see _call_view_fn in graph.py.
        "fn": "analysis.portfolio_intelligence.compute_good_bad",
        "inputs": [],
        "params": {},
        "requires": [],
        "filterable": False,
        "output": "good_bad_dict",
        "cache_key": "pi_good_bad",
        "grain": "signal",
    },
}


# ── Output normalization ────────────────────────────────────────────────────
# Each analysis/ function returns a different shape; QueryState/analyze_node/the
# UI only understand (result_df, result_kpis, result_rankings). One tiny
# normalizer per shape class -- explicit and legible, not a generic reflection
# unpacker, matching how graph.py::execute_node already hand-builds these fields.

def _normalize_df_dict_tuple(raw) -> tuple[pd.DataFrame, dict, dict]:
    df, kpis = raw
    return df, dict(kpis or {}), {}


def _normalize_dict_with_top_df(raw) -> tuple[pd.DataFrame, dict, dict]:
    raw = raw or {}
    df = raw.get("top_df", pd.DataFrame())
    kpis = {k: v for k, v in raw.items() if k != "top_df"}
    return df, kpis, {}


def _normalize_df(raw) -> tuple[pd.DataFrame, dict, dict]:
    df = raw if raw is not None else pd.DataFrame()
    return df, {"Count": len(df)}, {}


def _normalize_matrix_tuple(raw) -> tuple[pd.DataFrame, dict, dict]:
    matrix, meta = raw
    return matrix, dict(meta or {}), {}


def _normalize_tuple_df_fig(raw) -> tuple[pd.DataFrame, dict, dict]:
    df = raw[0] if raw else pd.DataFrame()
    return df, {"Count": len(df)}, {}


def _normalize_list_of_dicts(raw) -> tuple[pd.DataFrame, dict, dict]:
    rows = raw or []
    df = pd.DataFrame(rows)
    return df, {"Count": len(rows)}, {}


def _normalize_dict_subkey_df(raw, subkey: str) -> tuple[pd.DataFrame, dict, dict]:
    raw = raw or {}
    df = raw.get(subkey, pd.DataFrame())
    return df, {"Count": len(df)}, {}


def _normalize_good_bad_dict(raw) -> tuple[pd.DataFrame, dict, dict]:
    raw = raw or {}
    good = raw.get("good") or []
    bad = raw.get("bad") or []
    rows = [{"Type": "Good", "Signal": s} for s in good] + [{"Type": "Bad", "Signal": s} for s in bad]
    df = pd.DataFrame(rows)
    return df, {"Good Count": len(good), "Bad Count": len(bad)}, {}


_NORMALIZERS = {
    "df_dict_tuple": lambda raw, spec: _normalize_df_dict_tuple(raw),
    "dict_with_top_df": lambda raw, spec: _normalize_dict_with_top_df(raw),
    "df": lambda raw, spec: _normalize_df(raw),
    "matrix_tuple": lambda raw, spec: _normalize_matrix_tuple(raw),
    "tuple_df_fig": lambda raw, spec: _normalize_tuple_df_fig(raw),
    "list_of_dicts": lambda raw, spec: _normalize_list_of_dicts(raw),
    "dict_subkey_df": lambda raw, spec: _normalize_dict_subkey_df(raw, spec["subkey"]),
    "good_bad_dict": lambda raw, spec: _normalize_good_bad_dict(raw),
}


def normalize_view_output(spec: dict, raw) -> tuple[pd.DataFrame, dict, dict]:
    """Turn a raw analysis/ function return value into (result_df, result_kpis,
    result_rankings), dispatched on spec['output']. Raises on an unregistered
    output kind -- a VIEWS entry with a typo'd `output` field should fail loudly
    at call time, not silently degrade to an empty result."""
    fn = _NORMALIZERS.get(spec["output"])
    if fn is None:
        raise ValueError(f"no normalizer registered for view output kind '{spec['output']}'")
    return fn(raw, spec)


def resolve_view_fn(view_name: str):
    """Lazily import and return the analysis/ callable a VIEWS entry points to."""
    spec = VIEWS[view_name]
    module_path, func_name = spec["fn"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
