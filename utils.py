import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

YELLOW = "#FFC000"

# Columns that MUST exist for calculations to work
CRITICAL_COLS = [
    "Loan No", "RegionName", "Unit", "Loan Status", "Ag_Date",
    "Arrears / EMI", "Month Receipt Amount", "Month Collection (Excluding Reserve Collection)",
    "Net Collection Demand Inst+Exp+BC",
    "POS", "LCC%", "Strike", "Closing Arrears",
    "Month Due-Inst", "Month Due-Exp", "Total Cum Collection",
]

# Known column name variations across different LCC extracts
COL_ALIASES = {
    
    "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by":
        "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by R":
        "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "Cum Coll (Inst+Exp+BC)": "Cum Coll (Inst+Exp)",
}

# All expected columns (used for reference only — missing ones show a warning, not error)
REQUIRED_COLS = [
    "SNo", "Loan No", "CHANNEL", "BU", "StateName", "Zone", "RegionName", "Unit",
    "Ag_Date", "SRC Code", "SRC Name", "MNT CODE", "MNT NAME", "Due Dt", "Tenure",
    "Loan Status", "Loan Amount", "Veh ID", "Cust Name", "Guar Name", "Cust Mob No",
    "Guar Mob No", "Segment", "SegmentName", "Make", "Vehicle Description",
    "Year Of Manufacture", "Arrear Opening", "ARREARS OPEN AGAINST INST",
    "ARREARS OPEN AGAINST EXP", "ARREARS OPEN AGAINST BC", "ARREARS OPEN AGAINST PC",
    "OPENING RESERVE COLLECTION", "Month Due-Inst", "Month Due-Exp", "MONTH DUE (BC)",
    "MONTH DUE PC", "Month Receipt Amount", "MONTHCOLL INST", "MONTHCOLL EXP",
    "MONTHCOLL BC", "MONTHCOLL PC", "Month Collection (Excluding Reserve Collection)",
    "Closing Arrears", "UN-CLEARED CHEQUE FOR THE MONTH/Amount Not remitted by RE",
    "Cum Due-Inst", "Cum Due-Exp", "CUM DUE (BC)", "Cum Due PC", "Cum Coll (Inst+Exp)",
    "Total Cum Collection", "ARREARS AGAINST INST", "ARREARS AGAINST EXP",
    "ARREARS AGAINST BC", "ARREARS AGAINST PC", "CLOSING RESERVE COLLECTION",
    "Arrears against Inst+Exp", "Uncleared Cheque/Amount Not remitted by RE", "LCC%",
    "Arrears / EMI", "DelinquencyDays", "VehEMI Accrued", "ClosingPC", "POS", "scheme",
    "Non Starter", "Strike", "RCEndors(>90Days)", "RTO / INSURANCE",
    "NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC",
    "NET COLLECTION", "NET COLLECTION EXCLUDING RESERVE COLL", "Last Receipt Date",
    "Last Receipt Amount", "ParentLDueDate", "No Coll 3 Months and >6 EMI",
    "NACHStatus", "SaleType", "CoLending_Loans", "CUSTOMER_STATUS", "LGL_FLAG",
    "LGL_DESCRIPTION", "TyreFlag", "FUEL_TYPE",
]

BUCKET_ORDER = ["STD", "1-30 DPD", "SMA-1", "SMA-2", "NPA", "NA"]
BUCKET_SCORE = {"STD": 0, "1-30 DPD": 1, "SMA-1": 2, "SMA-2": 3, "NPA": 4, "NA": -1}


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v = pd.to_numeric(df["Arrears / EMI"], errors="coerce")
    df["curr_bucket"] = np.select(
        [v.isna(), v <= 0, v < 1, v < 2, v < 3],
        ["NA",     "STD",  "1-30 DPD", "SMA-1", "SMA-2"],
        default="NPA",
    )
    df["curr_score"] = df["curr_bucket"].map(BUCKET_SCORE)
    # SOH = Sum of Hire = POS + Closing Arrears = total exposure if customer defaults
    _pos = pd.to_numeric(df["POS"], errors="coerce").fillna(0) if "POS" in df.columns else pd.Series(0.0, index=df.index)
    _arr = pd.to_numeric(df["Closing Arrears"], errors="coerce").fillna(0) if "Closing Arrears" in df.columns else pd.Series(0.0, index=df.index)
    df["SOH"] = _pos + _arr
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded column names to our standard names.

    Pass 1 — Strip leading/trailing spaces from every column.
    Pass 2 — Apply COL_ALIASES (exact known truncations / variations).
    Pass 3 — Case-insensitive exact match (handles capitalisation differences).
    Pass 4 — Prefix match for truncated long names.
              Targets sorted longest-first so the more-specific column wins
              when a shorter column name is a prefix of a longer one
              (e.g. "Net Collection Demand Inst+Exp+BC" beats "NET Collection
              Demand Inst+Exp" when the file column is ambiguously truncated).
    """
    all_standard = list(dict.fromkeys(CRITICAL_COLS + REQUIRED_COLS))  # critical first, no dupes

    # Pass 1 — strip spaces
    df.columns = pd.Index([str(c).strip() for c in df.columns])

    # Pass 2 — known aliases
    df.rename(columns={k: v for k, v in COL_ALIASES.items() if k in df.columns}, inplace=True)

    # Build case-insensitive lookup; longest targets first so more-specific
    # columns win over shorter prefix-collision siblings.
    target_ci: dict[str, str] = {}
    for t in sorted(all_standard, key=len, reverse=True):
        tl = t.lower()
        if tl not in target_ci:
            target_ci[tl] = t

    rename_map: dict[str, str] = {}
    claimed: set[str] = {c for c in df.columns if c in all_standard}

    for col in df.columns:
        if col in all_standard:
            claimed.add(col)
            continue

        col_lower = col.lower()

        # Pass 3 — case-insensitive exact match
        if col_lower in target_ci and target_ci[col_lower] not in claimed:
            target = target_ci[col_lower]
            rename_map[col] = target
            claimed.add(target)
            continue

        # Pass 4 — prefix match (file col is truncated version of target)
        # Skip short columns (< 15 chars) to avoid false matches.
        if len(col) < 15:
            continue

        for tl, target in sorted(target_ci.items(), key=lambda x: len(x[0]), reverse=True):
            if target in claimed:
                continue
            if len(col_lower) < len(tl) and tl.startswith(col_lower):
                rename_map[col] = target
                claimed.add(target)
                break

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    return df


@__import__("streamlit").cache_data(show_spinner=False)
def load_and_validate(file) -> tuple[pd.DataFrame, list[str]]:
    try:
        fname = getattr(file, "name", "").lower()
        if fname.endswith(".xlsb"):
            engine = "pyxlsb"
        elif fname.endswith(".xls"):
            engine = "xlrd"
        else:
            engine = "openpyxl"
        df = pd.read_excel(file, engine=engine, sheet_name=0)
    except Exception as e:
        return None, [f"Could not read file: {e}"]

    # Normalize column names (strips spaces, fixes capitalisation, maps truncated names)
    df = _normalize_columns(df)

    # If critical columns are missing from sheet 0, try a sheet named "LCC"
    missing_critical = [c for c in CRITICAL_COLS if c not in df.columns]
    if missing_critical:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            xl   = pd.ExcelFile(file, engine=engine)
            lcc  = next((s for s in xl.sheet_names if str(s).strip().upper() == "LCC"), None)
            if lcc:
                if hasattr(file, "seek"):
                    file.seek(0)
                df = pd.read_excel(file, engine=engine, sheet_name=lcc)
                df = _normalize_columns(df)
                missing_critical = [c for c in CRITICAL_COLS if c not in df.columns]
        except Exception:
            pass  # fall through to the error below

    # Hard-fail if critical columns still missing after sheet fallback
    if missing_critical:
        return None, [
            f"Missing critical column(s): {', '.join(missing_critical)}"
        ]

    _EXCEL_EPOCH = pd.Timestamp("1899-12-30")
    for date_col in ["Ag_Date", "Last Receipt Date", "ParentLDueDate"]:
        if date_col not in df.columns:
            continue
        col = df[date_col]
        if pd.api.types.is_numeric_dtype(col):
            # xlsb files store dates as Excel serial numbers (days since 1899-12-30)
            numeric = pd.to_numeric(col, errors="coerce")
            df[date_col] = _EXCEL_EPOCH + pd.to_timedelta(numeric, unit="D")
            df[date_col] = df[date_col].where(numeric.notna() & (numeric > 0), pd.NaT)
        else:
            df[date_col] = pd.to_datetime(col, errors="coerce")

    # Due Dt is a numeric EMI due day (5, 10, 15, 20) — keep as number
    df["Due Dt"] = pd.to_numeric(df["Due Dt"], errors="coerce")

    # Additive cash flows - zero is the correct default when missing
    for col in ["Month Receipt Amount", "Month Collection (Excluding Reserve Collection)", "NET COLLECTION", "Cum Coll (Inst+Exp)", "Total Cum Collection"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # These must NOT be filled - missing means unknown, not zero (fillna(0) distorts ratios)
    for col in [
        "NET Collection Demand Inst+Exp", "Net Collection Demand Inst+Exp+BC",
        "POS", "Arrears / EMI",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["LCC%"] = pd.to_numeric(df["LCC%"], errors="coerce")
    df["Strike"] = df["Strike"].astype(str).str.strip().str.upper()
    if "Unit" in df.columns:
        df["Unit"] = df["Unit"].astype(str).str.strip().str.upper()

    df = assign_buckets(df)
    return df, []


def apply_filters(df: pd.DataFrame, region: str, branch: str, status: str) -> pd.DataFrame:
    if region != "All":
        df = df[df["RegionName"] == region]
    if branch != "All":
        df = df[df["Unit"] == branch]
    if status != "All":
        df = df[df["Loan Status"] == status]
    return df


def _safe_pct(num, den):
    if den == 0:
        return 0.0
    return round(num / den * 100, 2)


def _mom_pct(curr, prev):
    if prev == 0:
        return 0.0
    return round((curr - prev) / abs(prev) * 100, 2)


def compute_metrics(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    def _calc(df):
        n_accounts = df["Loan No"].nunique()
        demand = df["Net Collection Demand Inst+Exp+BC"].sum(min_count=1)
        demand = 0.0 if pd.isna(demand) else demand
        collection = df["Month Collection (Excluding Reserve Collection)"].sum()
        _soh_col = "SOH" if "SOH" in df.columns else "POS"
        pos = df[_soh_col].sum(min_count=1)
        pos = 0.0 if pd.isna(pos) else pos
        cum_coll_total = df["Total Cum Collection"].sum()
        cum_coll_inst_exp = df["Cum Coll (Inst+Exp)"].sum()

        strike_valid = df[df["Strike"].isin(["Y", "N"])]
        strike_pct = _safe_pct(
            (strike_valid["Strike"] == "Y").sum(),
            len(strike_valid),
        )

        npa_pct = _safe_pct(
            df[df["curr_bucket"] == "NPA"]["Loan No"].nunique(),
            n_accounts,
        )
        hard_pct = _safe_pct(
            df[df["Arrears / EMI"] >= 6]["Loan No"].nunique(),
            n_accounts,
        )
        lcc_avg = df["LCC%"].mean()
        lcc_avg = round(lcc_avg, 2) if not pd.isna(lcc_avg) else 0.0
        cmd_pct = _safe_pct(cum_coll_total, cum_coll_inst_exp)

        return {
            "Month Demand": demand,
            "Total Collection": collection,
            "Collection %": _safe_pct(collection, demand),
            "Strike %": strike_pct,
            "NPA %": npa_pct,
            "Hard Bucket %": hard_pct,
            "Count": n_accounts,
            "SOH": pos,
            "LCC%": lcc_avg,
            "CMD %": cmd_pct,
        }

    curr = _calc(df_curr)
    prev = _calc(df_prev)
    result = {}
    for k in curr:
        result[k] = (curr[k], _mom_pct(curr[k], prev[k]))
    return result


def fmt_value(val, kind: str) -> str:
    if kind == "money":
        if abs(val) >= 1_00_00_000:
            return f"₹{val / 1_00_00_000:.2f}Cr"
        if abs(val) >= 1_00_000:
            return f"₹{val / 1_00_000:.2f}L"
        return f"₹{val:,.0f}"
    if kind == "pct":
        return f"{val:.2f}"
    if kind == "count":
        if abs(val) >= 1_00_00_000:
            return f"{val / 1_00_00_000:.2f}Cr"
        if abs(val) >= 1_00_000:
            return f"{val / 1_00_000:.2f}L"
        return f"{int(val):,}"
    return str(val)


def build_status_bar_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df.groupby("curr_bucket")["Loan No"]
        .nunique()
        .reindex(BUCKET_ORDER, fill_value=0)
    )
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=YELLOW,
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(size=12, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Loan Count by DPD Bucket", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def build_branch_bar_chart(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return go.Figure()

    grp = df.groupby("Unit").agg(
        demand=("Net Collection Demand Inst+Exp+BC", "sum"),
        collection=("Month Collection (Excluding Reserve Collection)", "sum"),
    )
    grp["coll_pct"] = grp.apply(
        lambda r: _safe_pct(r["collection"], r["demand"]), axis=1
    )
    grp = grp.sort_values("coll_pct", ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        x=grp["coll_pct"].tolist(),
        y=grp.index.tolist(),
        orientation="h",
        marker_color=YELLOW,
        text=[f"{v:.0f}" for v in grp["coll_pct"]],
        textposition="outside",
        textfont=dict(size=11, color="#000000"),
    ))
    max_val = grp["coll_pct"].max() if len(grp) > 0 else 100
    fig.update_layout(
        title=dict(text="Collection % by Branch", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            range=[0, max_val * 1.12],
            title=dict(text="Collection %", font=dict(color="#000000")),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def build_closing_pc_chart(df: pd.DataFrame) -> go.Figure:
    """Arrears exposure by DPD bucket — SUM(Closing Arrears) per bucket.
    Shows how much money is stuck at each risk level."""
    if len(df) == 0 or "Closing Arrears" not in df.columns:
        return go.Figure()

    df = df.copy()
    df["Closing Arrears"] = pd.to_numeric(df["Closing Arrears"], errors="coerce").fillna(0)

    exposure = (
        df.groupby("curr_bucket")["Closing Arrears"]
        .sum()
        .reindex(BUCKET_ORDER, fill_value=0)
    )

    # Format labels in Indian units
    def _fmt(v):
        if v >= 1_00_00_000:
            return f"₹{v/1_00_00_000:.1f}Cr"
        if v >= 1_00_000:
            return f"₹{v/1_00_000:.1f}L"
        return f"₹{v:,.0f}"

    labels = [_fmt(v) for v in exposure.values]

    # Colour-code buckets: STD=green, 1-30=yellow, SMA-1=orange, SMA-2=orangered, NPA=red, NA=grey
    bucket_colors = {
        "STD":      "#16a34a",
        "1-30 DPD": YELLOW,
        "SMA-1":    "#f97316",
        "SMA-2":    "#ef4444",
        "NPA":      "#991b1b",
        "NA":       "#9ca3af",
    }
    colors = [bucket_colors.get(b, YELLOW) for b in exposure.index]

    fig = go.Figure(go.Bar(
        x=exposure.index.tolist(),
        y=exposure.values.tolist(),
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=12, color="#000000"),
    ))
    fig.update_layout(
        title=dict(text="Closing Arrears by DPD Bucket", font=dict(size=14, color="#000000")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(color="#000000")),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig


def _kpi_card_html(label: str, value: str, mom: float, unit: str = "", inverse: bool = False) -> str:
    arrow = "&#9650;" if mom >= 0 else "&#9660;"
    color = ("#CC0000" if mom >= 0 else "#00A651") if inverse else ("#00A651" if mom >= 0 else "#CC0000")
    mom_str = f"{abs(mom):.2f}%"
    return (
        f'<div style="border:2px solid {YELLOW};border-radius:8px;padding:16px 12px;'
        f'background:#fff;text-align:center;min-width:130px;flex:1;">'
        f'<div style="font-size:13px;font-weight:600;color:#333;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:700;color:#000;line-height:1.1;">{value}{unit}</div>'
        f'<div style="font-size:12px;margin-top:6px;">'
        f'MoM %: <span style="color:{color};font-weight:600;">{arrow} {mom_str}</span>'
        f'</div></div>'
    )


def build_html_export(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    metrics: dict,
    fig_status: go.Figure,
    fig_branch: go.Figure,
    fig_trend: go.Figure,
    filters: dict,
    *,
    curr_month: str = "",
    alerts: list = None,
    scorecard_df=None,
    roll_rate_meta: dict = None,
) -> str:
    import datetime as _dt

    def _fig_html(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    KPI_TOP = ["Month Demand", "Total Collection", "Collection %", "Strike %", "NPA %", "Hard Bucket %"]
    KPI_BOT = ["Count", "SOH", "LCC%", "CMD %"]
    KINDS = {
        "Month Demand": "money", "Total Collection": "money", "Collection %": "pct",
        "Strike %": "pct", "NPA %": "pct", "Hard Bucket %": "pct",
        "Count": "count", "SOH": "money", "LCC%": "pct", "CMD %": "pct",
    }
    INVERSE = {"NPA %", "Hard Bucket %"}

    def _card(label, value, mom, unit="", inverse=False):
        arrow = "&#9650;" if mom >= 0 else "&#9660;"
        color = ("#CC0000" if mom >= 0 else "#00A651") if inverse else ("#00A651" if mom >= 0 else "#CC0000")
        return (
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-bottom:3px solid {YELLOW};'
            f'border-radius:10px;padding:16px 14px;min-width:120px;flex:1;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px;">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:#111827;line-height:1;letter-spacing:-0.5px;">{value}{unit}</div>'
            f'<div style="font-size:11px;margin-top:8px;color:#9ca3af;">MoM <span style="color:{color};font-weight:700;">{arrow} {abs(mom):.2f}%</span></div>'
            f'</div>'
        )

    def _cards(keys, style=""):
        cards = "".join(
            _card(k, fmt_value(metrics[k][0], KINDS[k]), metrics[k][1],
                  unit="%" if KINDS[k] == "pct" else "",
                  inverse=k in INVERSE)
            for k in keys
        )
        return f'<div style="display:flex;gap:10px;flex-wrap:wrap;{style}">{cards}</div>'

    def _section(title):
        return (
            f'<div style="display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;'
            f'color:#111827;margin:28px 0 14px 0;">'
            f'<span style="width:4px;height:18px;background:{YELLOW};border-radius:2px;display:inline-block;flex-shrink:0;"></span>'
            f'{title}</div>'
        )

    filter_info = " | ".join(f"<b>{k}:</b> {v}" for k, v in filters.items() if v != "All") or "All data"
    generated_at = _dt.datetime.now().strftime("%d %b %Y, %I:%M %p")
    month_label = curr_month or filters.get("Year Month", "")

    # ── Smart Alerts section ──────────────────────────────────────────────────
    alerts_html = ""
    if alerts:
        SEVERITY_COLOR = {"critical": "#dc2626", "high": "#f97316", "medium": "#d97706"}
        cards_html = ""
        for alert in alerts:
            is_clear = alert["count"] == 0
            color = "#16a34a" if is_clear else SEVERITY_COLOR.get(alert["severity"], "#d97706")
            pos_fmt = fmt_value(alert["pos"], "money") if not is_clear else "—"
            arr_fmt = fmt_value(alert["closing_arrears"], "money") if not is_clear else "—"
            cards_html += (
                f'<div style="background:#fff;border:1px solid #e5e7eb;border-left:4px solid {color};'
                f'border-radius:10px;padding:14px 16px;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                f'<span style="font-size:18px;">{"✅" if is_clear else alert["icon"]}</span>'
                f'<span style="font-size:13px;font-weight:700;color:{color};">{alert["title"]}</span></div>'
                f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">{alert["subtitle"]}</div>'
                f'<div style="display:flex;gap:14px;">'
                f'<div><div style="font-size:9px;color:#9ca3af;font-weight:600;text-transform:uppercase;">Accounts</div>'
                f'<div style="font-size:22px;font-weight:800;color:{color};">{alert["count"]}</div></div>'
                f'<div><div style="font-size:9px;color:#9ca3af;font-weight:600;text-transform:uppercase;">POS</div>'
                f'<div style="font-size:16px;font-weight:700;color:#111;">{pos_fmt}</div></div>'
                f'<div><div style="font-size:9px;color:#9ca3af;font-weight:600;text-transform:uppercase;">Arrears</div>'
                f'<div style="font-size:16px;font-weight:700;color:{color};">{arr_fmt}</div></div>'
                f'</div>'
                f'<div style="font-size:11px;color:#6b7280;font-style:italic;margin-top:8px;border-top:1px solid #f3f4f6;padding-top:6px;">'
                f'{"✓ All clear" if is_clear else alert["action"]}</div>'
                f'</div>'
            )
        alerts_html = (
            _section("Smart Alerts") +
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">{cards_html}</div>'
        )

    # ── Executive Scorecard section ───────────────────────────────────────────
    scorecard_html = ""
    if scorecard_df is not None and len(scorecard_df) > 0:
        top5 = scorecard_df[scorecard_df["Tier"] == "top"].head(5)
        bot5 = scorecard_df[scorecard_df["Tier"] == "bottom"].sort_values("Collection %").head(5)

        def _exec_rows(sub):
            rows = ""
            for _, row in sub.iterrows():
                coll = row["Collection %"]
                coll_color = "#16a34a" if coll > 100 else "#d97706" if coll >= 90 else "#dc2626"
                rows += (
                    f'<tr style="border-bottom:1px solid #f3f4f6;">'
                    f'<td style="padding:7px 10px;font-size:12px;font-weight:600;color:#111;">{row["Executive (Branch)"]}</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row["Accounts"]}</td>'
                    f'<td style="padding:7px 10px;font-size:13px;font-weight:800;color:{coll_color};text-align:center;">{coll}%</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row["Strike Rate %"]}%</td>'
                    f'<td style="padding:7px 10px;font-size:12px;text-align:center;">{row["NPA %"]}%</td>'
                    f'</tr>'
                )
            return rows

        def _exec_table(title, sub, color):
            return (
                f'<div style="flex:1;">'
                f'<div style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px;padding:5px 10px;background:{color}18;border-radius:6px;">{title}</div>'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr style="background:#f9fafb;">'
                f'<th style="padding:7px 10px;text-align:left;font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Executive</th>'
                f'<th style="padding:7px 10px;text-align:center;font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Accounts</th>'
                f'<th style="padding:7px 10px;text-align:center;font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Coll %</th>'
                f'<th style="padding:7px 10px;text-align:center;font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Strike %</th>'
                f'<th style="padding:7px 10px;text-align:center;font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">NPA %</th>'
                f'</tr></thead>'
                f'<tbody>{_exec_rows(sub)}</tbody>'
                f'</table></div>'
            )

        scorecard_html = (
            _section(f"Executive Performance ({len(scorecard_df)} Executives)") +
            f'<div style="display:flex;gap:16px;background:#fff;border:1px solid #e5e7eb;'
            f'border-radius:10px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            + _exec_table("Top Performers", top5, "#16a34a")
            + f'<div style="width:1px;background:#e5e7eb;flex-shrink:0;"></div>'
            + _exec_table("Need Attention", bot5, "#dc2626")
            + f'</div>'
        )

    # ── Roll-Rate section ─────────────────────────────────────────────────────
    roll_html = ""
    if roll_rate_meta and roll_rate_meta.get("matched_count", 0) > 0:
        rr = roll_rate_meta
        rr_items = [
            ("Roll-Forward Rate", f'{rr["roll_forward_rate"]:.1f}%',  "#dc2626", "Accounts that worsened"),
            ("Roll-Backward Rate", f'{rr["roll_backward_rate"]:.1f}%',  "#16a34a", "Returned to STD"),
            ("NPA Formation",     f'{rr["npa_formation_rate"]:.1f}%', "#991b1b", "New NPA this month"),
            ("Matched Accounts",  f'{rr["matched_count"]:,}',         "#111827", "In both months"),
        ]
        rr_cards = "".join(
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-top:3px solid {c};'
            f'border-radius:10px;padding:14px 16px;flex:1;box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;">{lbl}</div>'
            f'<div style="font-size:24px;font-weight:800;color:{c};">{val}</div>'
            f'<div style="font-size:11px;color:#9ca3af;margin-top:4px;">{tip}</div>'
            f'</div>'
            for lbl, val, c, tip in rr_items
        )
        roll_html = (
            _section("Bucket Migration / Roll-Rate") +
            f'<div style="display:flex;gap:10px;">{rr_cards}</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shriram Finance – Regional Collection Dashboard {month_label}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Inter',Arial,sans-serif;background:#f2f2f2;color:#111827;}}
  .banner{{height:4px;background:linear-gradient(90deg,{YELLOW},#FFD740,{YELLOW});background-size:200%;animation:shimmer 3s linear infinite;}}
  @keyframes shimmer{{0%{{background-position:-200% 0}}100%{{background-position:200% 0}}}}
  .header{{background:#fff;border-bottom:3px solid {YELLOW};padding:14px 32px;display:flex;align-items:center;gap:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
  .logo-box{{background:#111;border-radius:8px;padding:8px 14px;}}
  .logo-main{{font-size:16px;font-weight:900;color:{YELLOW};letter-spacing:2px;line-height:1;}}
  .logo-sub{{font-size:9px;color:#888;letter-spacing:1px;margin-top:2px;}}
  .header-title{{font-size:17px;font-weight:700;color:#111;}}
  .header-sub{{font-size:11px;color:#999;margin-top:2px;}}
  .month-badge{{margin-left:auto;background:{YELLOW};color:#000;font-size:11px;font-weight:800;padding:5px 14px;border-radius:20px;letter-spacing:1px;white-space:nowrap;}}
  .content{{padding:20px 32px 40px;max-width:1400px;margin:0 auto;}}
  .meta-bar{{display:flex;align-items:center;justify-content:space-between;background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 16px;margin-bottom:20px;font-size:12px;color:#6b7280;}}
  .chart-card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:6px;box-shadow:0 2px 6px rgba(0,0,0,0.06);}}
  .footer{{text-align:center;padding:20px;font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;margin-top:8px;}}
</style>
</head>
<body>
<div class="banner"></div>
<div class="header">
  <div class="logo-box">
    <div class="logo-main">SHRIRAM</div>
    <div class="logo-sub">FINANCE</div>
  </div>
  <div>
    <div class="header-title">Regional Collection Dashboard</div>
    <div class="header-sub">Credit &amp; Collection Risk Monitoring &nbsp;&middot;&nbsp; CollectionIQ</div>
  </div>
  {f'<div class="month-badge">{month_label}</div>' if month_label else ''}
</div>

<div class="content">
  <div class="meta-bar">
    <span>&#128269; <b>Filters:</b> &nbsp;{filter_info}</span>
    <span>Generated: {generated_at}</span>
  </div>

  {_section("Key Performance Indicators")}
  {_cards(KPI_TOP, "margin-bottom:10px;")}
  {_cards(KPI_BOT)}

  {_section("Portfolio Analysis")}
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
    <div class="chart-card">{_fig_html(fig_status)}</div>
    <div class="chart-card">{_fig_html(fig_branch)}</div>
    <div class="chart-card">{_fig_html(fig_trend)}</div>
  </div>

  {alerts_html}
  {scorecard_html}
  {roll_html}
</div>

<div class="footer">
  Generated by <b>CollectionIQ</b> &nbsp;&middot;&nbsp; Shriram Finance Regional Collection Dashboard &nbsp;&middot;&nbsp; {generated_at}
</div>
</body>
</html>"""
    return html
