"""Run with: streamlit run app.py"""
import os
from pathlib import Path
import streamlit as st
from app.ui import home, add_layout, layout_editor

st.set_page_config(page_title="CAD-to-Rendering Research Tool", page_icon="▧", layout="wide")
st.markdown("""<style>
 .block-container {padding-top:4.5rem;padding-bottom:2rem;max-width:1800px}
 h1 {letter-spacing:-.045em} h3 {letter-spacing:-.02em}
 [data-testid="stMetricValue"] {font-size:1.5rem}
 div[data-testid="stVerticalBlock"] {gap:.7rem}
</style>""", unsafe_allow_html=True)
workspace = Path(os.environ.get("CAD_WORKSPACE", Path(__file__).resolve().parent / "workspace"))
workspace.mkdir(parents=True, exist_ok=True)
page = st.session_state.get("page", "home")
try:
    if page == "add":
        add_layout.render(workspace)
    elif page == "editor" and st.session_state.get("active_layout"):
        layout_editor.render(workspace)
    else:
        home.render(workspace)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.caption("The last successfully saved JSON remains on disk. Reopen the Layout to reload it.")
