from pathlib import PurePosixPath
import streamlit as st
from app.layout.discovery import discover


def render(workspace):
    st.caption("FROM PLAN TO PERSPECTIVE / RESEARCH WORKSPACE")
    st.title("CAD-to-Rendering Research Tool")
    st.write("Preserve the drawing. Annotate the space.")
    st.divider()
    st.subheader("Layout")
    layouts, labels, errors = discover(workspace)
    for error in errors:
        st.warning(error)
    selected = st.selectbox("Select existing Layout", list(labels), format_func=labels.get,
                            placeholder="No Layouts yet", disabled=not layouts)
    a, b, _ = st.columns([1, 1, 3])
    if a.button("Open / Edit Layout", type="primary", disabled=not layouts, width="stretch"):
        st.session_state.active_layout = selected
        st.session_state.pop(f"camera_error_{selected}", None)
        st.session_state.page = "editor"
        st.rerun()
    if b.button("+ Add Layout", width="stretch"):
        st.session_state.page = "add"
        st.rerun()
    if selected:
        state = next(item for item in layouts if item["layout"]["id"] == selected)
        live = [e for e in state["entities"].values() if not e.get("orphaned")]
        with st.container(border=True):
            st.subheader(state["layout"]["name"])
            st.caption(state["layout"]["id"])
            a, b, c = st.columns(3)
            a.metric("Status", state["layout"]["status"].capitalize())
            b.metric("Tagged", f"{sum(bool(e.get('semantic')) for e in live)} / {len(live)}")
            c.metric("Cameras", len(state["cameras"]))
            st.text(f"Source: {PurePosixPath(state['source']['cad_file']).name}")
            st.caption(f"Last modified: {state['layout']['updated_at']}")
    else:
        st.info("Create your first Layout by uploading a DXF drawing.")
    st.caption("V0.3 · Local persistence · CAD annotation · Live camera perspective")
