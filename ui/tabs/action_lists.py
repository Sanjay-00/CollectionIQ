"""Action Lists tab: every loan-level list in one place (analysis/action_lists.py).
An overview of all lists, then the chosen list's loans with branch / top-X
filters and a download carrying every column of the upload."""
import pandas as pd
import streamlit as st

from ui.components import _dl_btn, _safe_df, html_table, list_controls, section_label, takeaway

PICK_KEY = "al_pick"      # other tabs open a list by setting this to its name


def _label(e: dict) -> str:
    return f"{e['name']} ({e['count']:,})"


def render_action_lists_tab(lists: list[dict], df_curr: pd.DataFrame) -> None:
    from analysis.action_lists import GROUPS, RAW_NAMES, distinct_loans
    if not lists:
        st.info("No loan is on any action list for this view.")
        return
    takeaway(f"<b>{distinct_loans(lists, GROUPS[0]):,}</b> loans to call this week; "
             f"<b>{distinct_loans(lists):,}</b> loans across all {len(lists)} lists "
             "(a loan can be on more than one).")

    overview = pd.DataFrame([{"Group": e["group"], "List": e["name"], "Loans": e["count"], "SOH (Cr)": e["soh_cr"],
                              "Change": None if e["prev"] is None else e["count"] - e["prev"],
                              "What to do": e["action"]} for e in lists])
    overview["Group"] = overview["Group"].where(overview["Group"] != overview["Group"].shift(), "")
    st.markdown(html_table(overview, [
        {"key": "Group", "bold": True}, {"key": "List"}, {"key": "Loans", "fmt": "int"},
        {"key": "SOH (Cr)", "label": "SOH", "fmt": "cr"},
        {"key": "Change", "label": "vs last month", "fmt": "count_change",
         "help": "Change in the number of loans since last month's file (risk flags only)."},
        {"key": "What to do"},
    ]), unsafe_allow_html=True)

    section_label("Open a List", "18px")
    names = [e["name"] for e in lists]
    if st.session_state.get(PICK_KEY) not in names:
        st.session_state[PICK_KEY] = names[0]
    name = st.selectbox("List", names, key=PICK_KEY,
                        format_func=lambda n: next(f"{e['group']}: {_label(e)}" for e in lists if e["name"] == n))
    e = next(x for x in lists if x["name"] == name)
    st.caption(f"{e['action']} Largest SOH first.")
    view = list_controls(f"al_{names.index(name)}", e["loans"], "Branch", "loans")
    soh = view["SOH"].sum() / 1e7 if "SOH" in view.columns else 0.0
    st.caption(f"These {len(view):,} loans: ₹{soh:,.2f} Cr SOH.")
    st.dataframe(_safe_df(view), use_container_width=True, hide_index=True, height=min(38 + 35 * len(view), 520))
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
    _dl_btn(view.rename(columns=RAW_NAMES), f"{slug}.xlsx", f"dl_al_{slug}", full_source=df_curr)
