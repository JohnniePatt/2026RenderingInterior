"""Prompt-only UI extension; API configuration and generation remain on their existing page."""
import streamlit as st
from services.prompt_builder import compile_prompt


def render_prompt_panel(root, state, camera_id, conditions, references, appearance, strictness="Very Strict (nanobanana)"):
    result = compile_prompt(state, camera_id, conditions, references, appearance, root, strictness=strictness)
    for warning in result.warnings:
        st.warning(warning)
    for error in result.errors:
        st.error(error)
    for label, text, key in [("Spatial / Condition Prompt", result.spatial_prompt, "spatial"),
                             ("Final Prompt", result.final_prompt, "final")]:
        widget_key = f"compiled_{key}_{root.name}"
        st.session_state[widget_key] = text
        st.text_area(label, key=widget_key, height=240, disabled=True)
    st.caption("Spatial instructions update from the selected camera and exported maps. Edit Appearance Prompt for materials, lighting and style.")
    with st.expander("Prompt Builder Debug"):
        st.json(result.debug)
    if not result.errors:
        st.download_button("Download exact final prompt", result.final_prompt,
                           file_name=f"{camera_id}_prompt.txt", mime="text/plain", key=f"prompt_download_{root.name}")
    return result
