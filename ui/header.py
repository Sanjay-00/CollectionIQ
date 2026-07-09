import base64
import pathlib

import streamlit as st


def render_header() -> None:
    logo_path = pathlib.Path(__file__).parent.parent / "assets" / "logo.jpeg"
    logo_b64  = base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else ""
    logo_html = (
        f'<img src="data:image/jpeg;base64,{logo_b64}" '
        f'style="height:68px;width:auto;object-fit:contain;display:block;" alt="CollectionIQ">'
        if logo_b64 else
        '<div class="dash-logo-box"><div class="dash-logo-main">COLLECTION</div>'
        '<div class="dash-logo-sub">IQ</div></div>'
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
