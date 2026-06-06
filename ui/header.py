import base64
import pathlib

import streamlit as st


def render_header() -> None:
    logo_path = pathlib.Path(__file__).parent.parent / "assets" / "shriram_logo.jpg"
    logo_b64  = base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else ""
    logo_html = (
        f'<img src="data:image/jpeg;base64,{logo_b64}" '
        f'style="height:52px;width:auto;object-fit:contain;display:block;" alt="Shriram Finance">'
        if logo_b64 else
        '<div class="dash-logo-box"><div class="dash-logo-main">SHRIRAM</div>'
        '<div class="dash-logo-sub">FINANCE</div></div>'
    )
    st.markdown('<div class="top-banner"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="dash-header">
      <div style="flex-shrink:0;">{logo_html}</div>
      <div>
        <div class="dash-title">Regional Collection Dashboard</div>
        <div class="dash-subtitle">Credit &amp; Collection Risk Monitoring System</div>
      </div>
      <div class="dash-badge">CollectionIQ</div>
    </div>
    """, unsafe_allow_html=True)
