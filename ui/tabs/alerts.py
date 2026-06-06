import pandas as pd
import streamlit as st

from utils import fmt_value
from ui.components import _dl_btn, _safe_df

_SEVERITY_STYLE = {
    "critical": ("#dc2626", "#fff5f5"),
    "high":     ("#f97316", "#fff7ed"),
    "medium":   ("#d97706", "#fffbea"),
}
_CLEAR_STYLE = ("#16a34a", "#f0fdf4")


def render_alerts_tab(df_curr: pd.DataFrame, alerts: list) -> None:
    n_triggered = sum(1 for a in alerts if a["count"] > 0)

    # ── Summary bar ─────────────────────────────────────────────────────────
    if n_triggered == 0:
        st.success("All clear - no risk flags triggered on the current portfolio.")
    else:
        critical = sum(1 for a in alerts if a["count"] > 0 and a["severity"] == "critical")
        high     = sum(1 for a in alerts if a["count"] > 0 and a["severity"] == "high")
        medium   = sum(1 for a in alerts if a["count"] > 0 and a["severity"] == "medium")
        parts = []
        if critical: parts.append(f'<span style="color:#dc2626;font-weight:800;">{critical} Critical</span>')
        if high:     parts.append(f'<span style="color:#f97316;font-weight:800;">{high} High</span>')
        if medium:   parts.append(f'<span style="color:#d97706;font-weight:800;">{medium} Medium</span>')
        st.markdown(
            f'<div class="filter-bar" style="background:#fff5f5;border-color:#fecaca;">'
            f'🚨 <strong>{n_triggered} alert{"s" if n_triggered > 1 else ""} active:</strong>&nbsp; '
            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Risk Flag Details</div>', unsafe_allow_html=True)

    # ── Alert cards ──────────────────────────────────────────────────────────
    for i in range(0, len(alerts), 2):
        row_alerts = alerts[i:i + 2]
        cols = st.columns(len(row_alerts))
        for col, alert in zip(cols, row_alerts):
            is_clear = alert["count"] == 0
            title_color, bg = _CLEAR_STYLE if is_clear else _SEVERITY_STYLE.get(alert["severity"], ("#d97706", "#fffbea"))
            border      = f"border-left:4px solid {title_color};"
            count_color = "#16a34a" if is_clear else title_color
            pos_fmt     = fmt_value(alert["pos"], "money")
            pc_fmt      = fmt_value(alert["closing_arrears"], "money")

            with col:
                st.markdown(f"""
                <div style="background:{bg};border-radius:12px;{border}
                            padding:16px 18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:4px;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="font-size:20px;">{"✅" if is_clear else alert["icon"]}</span>
                    <span style="font-size:14px;font-weight:700;color:{title_color};">{alert["title"]}</span>
                  </div>
                  <div style="font-size:12px;color:#666;margin-bottom:10px;">{alert["subtitle"]}</div>
                  <div style="display:flex;gap:16px;margin-bottom:10px;">
                    <div>
                      <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Accounts</div>
                      <div style="font-size:26px;font-weight:800;color:{count_color};">{alert["count"]}</div>
                    </div>
                    <div>
                      <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">POS</div>
                      <div style="font-size:22px;font-weight:800;color:#111;">{pos_fmt if not is_clear else "—"}</div>
                    </div>
                    <div>
                      <div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;">Closing Arrears</div>
                      <div style="font-size:22px;font-weight:800;color:{title_color};">{pc_fmt if not is_clear else "—"}</div>
                    </div>
                  </div>
                  <div style="font-size:11px;color:#555;font-style:italic;border-top:1px solid rgba(0,0,0,0.08);
                              padding-top:8px;">
                    {"✓ All clear - no accounts flagged" if is_clear else f"💬 {alert['action']}"}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if not is_clear:
                    with st.expander(f"View {alert['count']} accounts"):
                        display_df = alert["df"].loc[:, ~alert["df"].columns.duplicated()]
                        st.dataframe(
                            _safe_df(display_df.reset_index(drop=True)),
                            use_container_width=True,
                            hide_index=True,
                            height=min(300, 40 + alert["count"] * 35),
                        )
                        _dl_btn(
                            display_df.reset_index(drop=True),
                            f"alert_{alert['title'].replace(' ', '_')}.xlsx",
                            f"dl_alert_{alert['title']}",
                        )
