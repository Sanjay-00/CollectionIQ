import io

import pandas as pd
import streamlit as st

from ui.components import _bump_data_version


@st.cache_data(show_spinner=False)
def _fetch_sample_from_github():
    import urllib.request
    from utils import load_and_validate

    BASE  = "https://raw.githubusercontent.com/Sanjay-00/CollectionIQ/main/sample_data/"
    FILES = {"curr": "Current_Month_Demo.xlsx", "prev": "Previous_Month_Demo.xlsx"}

    def _load(fname):
        with urllib.request.urlopen(BASE + fname, timeout=20) as resp:
            buf = io.BytesIO(resp.read())
        buf.name = fname
        # Route through the same pipeline every uploaded file goes through, so
        # the sample data gets identical normalization (mobile-number cleanup,
        # duplicate Loan No handling, etc.) instead of a hand-duplicated copy
        # that silently drifts out of sync. Call the unwrapped function directly:
        # this whole outer function is already @st.cache_data'd, and Streamlit's
        # cache hasher chokes on a BytesIO with a fake .name (tries os.path.getmtime
        # on it as if it were a real file on disk).
        df, errs = load_and_validate.__wrapped__(buf)
        if errs:
            raise ValueError(errs[0])
        return df

    return _load(FILES["curr"]), _load(FILES["prev"])


def render_landing() -> None:
    """Render the pre-upload landing page. Caller should call st.stop() after this."""
    st.markdown(
        '<div style="margin-top:12px;text-align:center;margin-bottom:40px;">'
        '<div style="font-size:12px;font-weight:800;letter-spacing:4px;color:#FFC000;'
        'text-transform:uppercase;margin-bottom:12px;">COLLECTION INTELLIGENCE PLATFORM</div>'
        '<div style="font-size:26px;font-weight:800;color:#111;margin-bottom:10px;'
        'letter-spacing:-0.5px;line-height:1.2;">Upload your LCC extract to activate the dashboard</div>'
        '<div style="font-size:13px;color:#999;font-weight:400;letter-spacing:0.2px;">'
        'Supports .xlsx, .xls and .xlsb &nbsp;&#183;&nbsp; Reliable &nbsp;&#183;&nbsp; No data persisted, in-memory only'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # One card per tab in the nav bar -- previously a hand-picked subset of 5
    # that had drifted out of date as the app grew to 8 tabs (Migration,
    # Portfolio Intelligence, and Business were entirely unmentioned here even
    # though each is a full tab with real analytical depth). Mirroring the
    # actual tab list 1:1 means this can't silently go stale the same way
    # again -- adding a 9th tab is now an obvious gap to notice here too.
    _FEATURES = [
        ("#FFBF00", "DASHBOARD",              "Collection %, POS, demand, strike rate and NPA with month-on-month movement."),
        ("#111111", "SCORECARD",              "Every field executive ranked by collection efficiency, strike rate and roll rates."),
        ("#FFBF00", "SMART ALERTS",           "Flagged non-starters, co-lending risk, easy settlements and insurance-driven arrears."),
        ("#111111", "MIGRATION",              "Bucket migration matrix with roll-forward / roll-backward rates between two periods."),
        ("#FFBF00", "PORTFOLIO INTELLIGENCE", "Pulse KPIs, risk indicators, concentration map, repossession and good-customer lists."),
        ("#111111", "BUSINESS",               "New advances this month, trend over time, and by region/branch/executive breakdown."),
        ("#FFBF00", "AI QUERY ENGINE",        "Ask any question in plain English. Query priority loans or filter NPA loans by region."),
        ("#111111", "MONTHLY REPORT",         "Board-ready HTML report with AI narrative, branch league table and a five-point action plan."),
    ]
    for row_start in (0, 4):
        for col, (color, title, desc) in zip(st.columns(4), _FEATURES[row_start:row_start + 4]):
            with col:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;'
                    f'padding:20px 16px;border-top:3px solid {color};margin-bottom:12px;">'
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:{color};'
                    f'text-transform:uppercase;margin-bottom:8px;">{title}</div>'
                    f'<div style="font-size:12px;color:#555;line-height:1.6;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div style="border-top:1px solid #e5e7eb;margin-top:36px;margin-bottom:16px;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align:center;font-size:11px;font-weight:700;color:#9ca3af;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">No file? Try the built-in sample</div>',
        unsafe_allow_html=True,
    )

    _, col_s, _ = st.columns([2, 1, 2])
    with col_s:
        if st.button("Fill Sample Data", type="primary", width='stretch'):
            try:
                with st.spinner("Fetching sample data from GitHub..."):
                    _dc, _dp = _fetch_sample_from_github()
                st.session_state["df_curr_raw"]       = _dc
                st.session_state["df_prev_raw"]       = _dp
                st.session_state["_sample_loaded"]    = True
                st.session_state["_set_sample_dates"] = True
                _bump_data_version()
                st.rerun()
            except Exception as e:
                st.error(f"Could not fetch sample data: {e}")
