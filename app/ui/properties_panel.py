import hashlib
import streamlit as st
from app.annotation.tagging import CATEGORIES, apply, clear, store_reference
from app.cad.units import from_mm
from app.layout.persistence import resolve
from app.layout.schema import VOID_CATEGORIES
from app.geometry.proxy import compute_furniture_colors


def render(root, state, selected, tool):
    st.subheader("Properties")
    if not selected:
        st.caption("Click a CAD edge to select an entity. Hold Shift to add more.")
        return
    record = state["entities"][selected[0]]
    context = hashlib.sha256((root.name + '|'.join(selected) + tool).encode()).hexdigest()[:12]
    st.caption(f"{len(selected)} selected · {record['dxf_type']} · handle {record['dxf_handle']}")
    st.caption(f"Layer: {record.get('layer', '0')} (metadata only)")
    if len(selected) > 1:
        st.info("Apply replaces properties on every selected entity with the values below.")
    semantic_default = tool.lower() if tool != "Select" else record.get("semantic") or "wall"
    semantics = ["wall", "floor", "furniture", "void"]
    semantic = st.selectbox("Semantic", semantics, index=semantics.index(semantic_default),
                            format_func=str.capitalize, key=f"semantic_{context}")
    unit = state["source"]["units"]
    suffix = f"{context}_{semantic}"
    same = record.get("semantic") == semantic
    def val(key, default):
        return record.get(key, default) if same else default
    # Keep drafts in Python widget state across mode changes; only Apply writes research state.
    # A Streamlit form would leave unsubmitted edits exclusively in the browser and lose them on mode reruns.
    with st.container(border=True):
        props = {}
        if semantic == "void":
            cat_options = ["door", "window", "opening"]
            cat_labels = {"door": "Door", "window": "Window", "opening": "Other Opening"}
            current_cat = val("category", "door")
            if current_cat not in cat_options:
                current_cat = "door"
            selected_cat = st.selectbox(
                "Opening type",
                cat_options,
                index=cat_options.index(current_cat),
                format_func=lambda x: cat_labels.get(x, str(x).capitalize()),
                key=f"opening_cat_{suffix}"
            )
            props["category"] = selected_cat
            if selected_cat == "window":
                default_sill = val("sill_height", val("base_elevation", from_mm(900, unit)))
                default_opening = val("opening_height", val("height", from_mm(1200, unit)))
                props["sill_height"] = st.number_input(
                    f"Sill Height ({unit})",
                    value=float(default_sill),
                    format="%.3f",
                    key=f"sill_{suffix}"
                )
                props["opening_height"] = st.number_input(
                    f"Opening Height ({unit})",
                    min_value=0.001,
                    value=float(default_opening),
                    format="%.3f",
                    key=f"opening_h_{suffix}"
                )
            else:
                default_base = val("base_elevation", val("sill_height", 0.0))
                default_opening = val("opening_height", val("height", from_mm(2100, unit)))
                props["base_elevation"] = st.number_input(
                    f"Base Elevation ({unit})",
                    value=float(default_base),
                    format="%.3f",
                    key=f"base_elev_{suffix}"
                )
                props["opening_height"] = st.number_input(
                    f"Opening Height ({unit})",
                    min_value=0.001,
                    value=float(default_opening),
                    format="%.3f",
                    key=f"opening_h_{suffix}"
                )
        if semantic in ("wall", "furniture"):
            props["height"] = st.number_input(f"Height ({unit})", min_value=0.0,
                value=float(val("height", from_mm({"wall": 2800, "furniture": 820}[semantic], unit))), format="%.3f", key=f"height_{suffix}")
        if semantic == "wall":
            props["base_elevation"] = st.number_input(f"Base elevation ({unit})", value=float(val("base_elevation", 0)), format="%.3f", key=f"base_{suffix}")
        if semantic == "floor":
            props["elevation"] = st.number_input(f"Elevation ({unit})", value=float(val("elevation", 0)), format="%.3f", key=f"elevation_{suffix}")
            props["thickness"] = st.number_input(f"Thickness ({unit})", min_value=0.0, value=float(val("thickness", from_mm(150, unit))), format="%.3f", key=f"thickness_{suffix}")
        if semantic == "furniture":
            f_colors = compute_furniture_colors(state)
            color_hex = f_colors.get(selected[0]) if selected else None
            if color_hex:
                st.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:0.85rem;'><span style='display:inline-block;width:14px;height:14px;border-radius:3px;background-color:{color_hex};border:1px solid rgba(255,255,255,0.25);'></span><span>Block set color: <code style='color:{color_hex};font-weight:bold;'>{color_hex}</code></span></div>", unsafe_allow_html=True)
            category = val("category", "sofa")
            categories = list(dict.fromkeys(CATEGORIES + [category]))
            props["category"] = st.selectbox("Category", categories, index=categories.index(category), key=f"category_{suffix}")
            custom = st.text_input("Custom category (optional)", key=f"custom_{suffix}")
            if custom.strip():
                props["category"] = custom.strip()
            upload = st.file_uploader("Furniture reference", type=["png", "jpg", "jpeg", "webp"], key=f"upload_{suffix}")
            remove = st.checkbox("Remove reference image", key=f"remove_{suffix}")
        props["description"] = st.text_area("Description", value=val("description", ""), key=f"description_{suffix}")
        props["material"] = st.text_input("Material", value=val("material", "") or "", key=f"material_{suffix}") or None
        props["notes"] = st.text_area("Notes", value=val("notes", ""), key=f"notes_{suffix}")
        submitted = st.button("Apply & autosave", type="primary", disabled=unit.startswith("unknown"), key=f"apply_{suffix}")
    if submitted:
        if semantic == "furniture":
            if remove:
                props["reference_image"] = None
            elif upload:
                props["reference_image"] = store_reference(root, upload.name, upload.getvalue())
        apply(root, state, selected, semantic, props)
        st.rerun()
    if record.get("reference_image"):
        image = resolve(root, record["reference_image"])
        if image.is_file():
            st.image(str(image), caption="Saved furniture reference")
        else:
            st.warning(f"Reference image missing: {record['reference_image']}")
    if st.button("Clear annotation", key=f"clear_{context}"):
        clear(root, state, selected)
        st.rerun()
