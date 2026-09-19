import streamlit as st
from app.layout.manager import create


def render(workspace):
    st.subheader("Add new Layout")
    st.caption("One source drawing. A persistent research workspace.")
    with st.form("add_layout", clear_on_submit=False):
        name = st.text_input("Layout name", placeholder="Living Room Test 01")
        source = st.file_uploader("Source CAD · DXF only", type=["dxf"])
        submitted = st.form_submit_button("Create Layout", type="primary")
    if submitted:
        if source is None:
            st.error("Upload a DXF file first.")
        else:
            try:
                root, _ = create(workspace, name, source.name, source.getvalue())
                st.session_state.active_layout = root.name
                st.session_state.page = "editor"
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    if st.button("Back to Layouts"):
        st.session_state.page = "home"
        st.rerun()
