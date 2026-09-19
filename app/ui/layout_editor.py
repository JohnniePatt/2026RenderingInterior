from copy import deepcopy
import hashlib
from pathlib import Path
import streamlit as st
from app.cad.entities import load_for_editor
from app.cad.units import COMMON_UNITS
from app.cad.units import from_mm
from app.geometry.camera import save_cameras
from app.geometry.conditioning import save_conditioning_outputs
from app.geometry.proxy import build_proxy, compute_furniture_colors
from app.layout.persistence import save
from app.ui.properties_panel import render as properties
from app.ui.viewer import cad_viewer


def render(workspace):
    st.markdown("""<style>
      .block-container {padding-top:2.2rem !important;padding-bottom:2rem !important;max-width:1850px !important}
      iframe[title="app.ui.viewer.cad_viewer"] {border-radius:8px;border:none;width:100%}
      [data-testid="stColumn"]:has(iframe[title="app.ui.viewer.cad_viewer"]) {
          position: sticky !important;
          top: 2.8rem !important;
          align-self: flex-start !important;
          z-index: 90 !important;
      }
    </style>""", unsafe_allow_html=True)
    root = Path(workspace) / st.session_state.active_layout
    top = st.columns([1, 6, 1])
    if top[0].button("← Layouts"):
        st.session_state.page = "home"
        st.rerun()
    try:
        state, geometry, warnings = load_for_editor(root)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.info("Restore the missing or damaged file, then reopen this Layout. Annotations have not been discarded.")
        return
    top[1].subheader(state["layout"]["name"])
    top[1].caption(f"{root.name} · {state['layout']['status'].capitalize()} · ● Saved · {state['layout']['updated_at']}")
    if top[2].button("Save Layout"):
        save(root, state)
        st.rerun()
    for warning in warnings:
        st.warning(warning)
    if st.session_state.get(f"export_success_{root.name}"):
        st.success(st.session_state.pop(f"export_success_{root.name}"))
    selection_key = f"selection_{root.name}"
    valid = {item["id"] for item in geometry}
    selected = [i for i in st.session_state.get(selection_key, []) if i in valid]
    mode = st.radio("Editing mode", ["Annotation Mode", "Camera Mode"], horizontal=True, key=f"mode_{root.name}")
    camera_mode = mode == "Camera Mode"

    if camera_mode:
        view = st.container()
        # Render draft widgets even while collapsed, so changing mode does not delete widget state.
        with st.expander("Annotation properties · drafts retained", expanded=False):
            tools, panel = st.columns([1, 3], gap="small")
    else:
        left_col, view = st.columns([1, 2.7], gap="medium")
        tools = left_col
        panel = left_col

    with tools:
        if not camera_mode:
            st.caption("TOOLS")
            tool = st.radio("Annotation tool", ["Select", "Wall", "Floor", "Furniture", "Void"], label_visibility="collapsed")
            live = [e for e in state["entities"].values() if not e.get("orphaned")]
            tagged = sum(bool(e.get("semantic")) for e in live)
            st.caption(f"{tagged} / {len(live)} tagged")
            st.progress(tagged / max(1, len(live)))
            st.divider()
        else:
            st.caption("TOOLS")
            tool = st.radio("Annotation tool", ["Select", "Wall", "Floor", "Furniture", "Void"], label_visibility="collapsed", key=f"draft_tool_{root.name}")

    with panel:
        properties(root, state, selected, tool)
        if not camera_mode:
            with st.expander("Layout settings"):
                with st.form(f"settings_{root.name}"):
                    name = st.text_input("Layout name", state["layout"]["name"])
                    current_units = state["source"]["units"]
                    options = list(dict.fromkeys([*COMMON_UNITS, current_units]))
                    unit = st.selectbox("Drawing units", options, index=options.index(current_units))
                    st.caption("Declares existing CAD coordinates; does not rescale geometry or saved dimensions.")
                    if st.form_submit_button("Apply settings"):
                        candidate = deepcopy(state)
                        candidate["layout"]["name"] = name.strip()
                        candidate["source"].update(units=unit, units_confirmed=True)
                        save(root, candidate)
                        st.rerun()

    proxy = build_proxy(state, geometry)
    proxy_version = hashlib.sha256(repr(proxy).encode()).hexdigest() if proxy else None

    with view:
        version = hashlib.sha256(repr(geometry).encode()).hexdigest()
        entity_colors = compute_furniture_colors(state)
        auto_ceiling = bool(state.get("proxy_settings", {}).get("auto_ceiling", True))
        tagged_entities = state.get("entities", {})
        from app.geometry.homography import get_floor_elevation
        floor_elevation = get_floor_elevation(state)
        event = cad_viewer(entities=geometry, tags={k:v.get("semantic") for k,v in state["entities"].items()},
                           entity_colors=entity_colors,
                           units=state["source"]["units"], selected_ids=selected, geometry_version=version,
                           mode="camera" if camera_mode else "annotation",
                           cameras=state["cameras"], active_camera=state["active_camera"],
                           revision=state["layout"]["updated_at"], proxy=proxy, proxy_version=proxy_version,
                           units_known=state["source"]["units"] in COMMON_UNITS[1:],
                           unit_scale=from_mm(1, state["source"]["units"]),
                           camera_error=st.session_state.get(f"camera_error_{root.name}"),
                           acknowledged_event=st.session_state.get(f"event_{root.name}"),
                           auto_ceiling=auto_ceiling,
                           tagged_entities=tagged_entities,
                           floor_elevation=floor_elevation,
                           key=f"viewer_{root.name}", default=None)
        if event and event.get("event_id") != st.session_state.get(f"event_{root.name}"):
            st.session_state[f"event_{root.name}"] = event.get("event_id")
            if event.get("type") == "cameras":
                try:
                    if state["source"]["units"] not in COMMON_UNITS[1:]:
                        raise ValueError("Confirm drawing units before editing cameras")
                    save_cameras(root, state, event.get("cameras"), event.get("active_camera"), event.get("expected_revision"))
                    st.session_state.pop(f"camera_error_{root.name}", None)
                except (ValueError, OSError) as exc:
                    st.session_state[f"camera_error_{root.name}"] = str(exc)
            elif event.get("type") == "export_conditioning":
                try:
                    res = save_conditioning_outputs(root, state, event)
                    st.session_state[f"export_success_{root.name}"] = f"Conditioning outputs exported to {res['output_dir']}"
                    st.session_state.pop(f"camera_error_{root.name}", None)
                except Exception as exc:
                    st.session_state[f"camera_error_{root.name}"] = f"Export failed: {exc}"
            else:
                st.session_state[selection_key] = [i for i in event.get("selected_ids", []) if i in valid]
            st.rerun()

    def render_proxy_settings():
        with st.expander("Proxy geometry settings and diagnostics"):
            st.caption("Line walls default to vertical surfaces. A nonzero thickness below is an explicit proxy assumption, not a CAD measurement.")
            with st.form(f"proxy_settings_{root.name}"):
                thickness = st.number_input(f"Line wall proxy thickness ({state['source']['units']})", min_value=0.0,
                    value=float(state.get("proxy_settings", {}).get("line_wall_thickness", 0)), format="%.3f")
                styles = ["Outer Boundary (CAD contour)", "Oriented Box (OBB)"]
                current_style = state.get("proxy_settings", {}).get("furniture_style", "outer_boundary")
                idx = 1 if current_style == "box" else 0
                style_choice = st.radio("Furniture proxy style", styles, index=idx,
                    help="Outer Boundary keeps curved and contoured CAD outlines. Oriented Box allocates simple bounding boxes for each sub-object.")
                st.divider()
                st.markdown("**Ceiling settings & description (ฝ้าเพดาน)**")
                auto_ceiling = st.checkbox("Automatic ceiling (สร้างฝ้าเพดานอัตโนมัติ)",
                    value=bool(state.get("proxy_settings", {}).get("auto_ceiling", True)),
                    help="Automatically caps the room with a ceiling based on the maximum wall height.")
                
                wall_tops = [float(rec.get("base_elevation", 0)) + float(rec.get("height", 0))
                             for rec in state["entities"].values()
                             if rec.get("semantic") == "wall" and not rec.get("orphaned") and float(rec.get("height", 0)) > 0]
                auto_h = max(wall_tops) if wall_tops else 2.8
                existing_h = state.get("proxy_settings", {}).get("ceiling_height") or state.get("ceiling", {}).get("elevation")
                h_val = float(existing_h) if existing_h is not None else 0.0
                ceiling_height_input = st.number_input(
                    f"Ceiling elevation override ({state['source']['units']}) [0.0 = auto ({auto_h:.3f} {state['source']['units']})]",
                    min_value=0.0, value=h_val, format="%.3f",
                    help="Set 0.0 to automatically match highest wall top. Set a positive number to override ceiling elevation."
                )
                
                existing_desc = state.get("ceiling", {}).get("description") or state.get("proxy_settings", {}).get("ceiling_description", "")
                ceiling_desc = st.text_area(
                    "Ceiling overwrite description (คำอธิบายฝ้าเพดาน)",
                    value=existing_desc,
                    help="Custom description / AI rendering prompt for the generated ceiling."
                )
                
                existing_mat = state.get("ceiling", {}).get("material") or state.get("proxy_settings", {}).get("ceiling_material", "")
                ceiling_mat = st.text_input(
                    "Ceiling material (วัสดุฝ้าเพดาน)",
                    value=existing_mat or "",
                    help="Material specification (e.g. gypsum, acoustic tile, wood slat)."
                )

                if st.form_submit_button("Save proxy settings"):
                    candidate = deepcopy(state)
                    ps = candidate.setdefault("proxy_settings", {})
                    ps["line_wall_thickness"] = thickness
                    ps["furniture_style"] = "box" if "Oriented Box" in style_choice else "outer_boundary"
                    ps["auto_ceiling"] = auto_ceiling
                    ps["ceiling_height"] = ceiling_height_input if ceiling_height_input > 0 else None
                    ps["ceiling_description"] = ceiling_desc.strip()
                    ps["ceiling_material"] = ceiling_mat.strip() or None
                    if "ceiling" not in candidate:
                        candidate["ceiling"] = {}
                    candidate["ceiling"]["enabled"] = auto_ceiling
                    candidate["ceiling"]["description"] = ceiling_desc.strip()
                    candidate["ceiling"]["material"] = ceiling_mat.strip() or None
                    if ceiling_height_input > 0:
                        candidate["ceiling"]["elevation"] = ceiling_height_input
                        candidate["ceiling"]["source"] = "explicit_ceiling_height"
                    save(root, candidate)
                    st.rerun()
            for message in proxy["warnings"]:
                st.warning(message)
            st.caption(f"{len(proxy['meshes'])} CAD-derived proxy parts · geometry stays in drawing units")

    def render_inventory():
        with st.expander("Entity inventory / keyboard selection"):
            inventory_key = f"inventory_{root.name}"
            st.session_state[inventory_key] = selected
            def change_selection():
                st.session_state[selection_key] = st.session_state[inventory_key]
            st.multiselect("Selected entities", list(valid), key=inventory_key,
                format_func=lambda i: f"{i} · {state['entities'][i]['dxf_type']} · {state['entities'][i].get('layer', '0')}",
                on_change=change_selection)
            inv_items = [{"id": k, **v} for k, v in state["entities"].items()]
            c_info = state.get("ceiling") or {}
            if state.get("proxy_settings", {}).get("auto_ceiling", True) or c_info.get("enabled"):
                c_elev = c_info.get("elevation")
                c_desc = c_info.get("description") or state.get("proxy_settings", {}).get("ceiling_description", "")
                c_mat = c_info.get("material") or state.get("proxy_settings", {}).get("ceiling_material")
                inv_items.append({
                    "id": "auto_ceiling",
                    "semantic": "ceiling",
                    "dxf_handle": "(generated)",
                    "dxf_type": "GENERATED",
                    "layer": "ceiling",
                    "elevation": c_elev,
                    "thickness": c_info.get("thickness", from_mm(50, state["source"]["units"])),
                    "description": c_desc,
                    "material": c_mat,
                    "orphaned": False
                })
            st.dataframe(inv_items, width="stretch")
        st.download_button("Export layout.json", data=(root / "layout.json").read_bytes(), file_name=f"{root.name}.json", mime="application/json")

    if camera_mode:
        render_proxy_settings()
        render_inventory()
    else:
        with left_col:
            render_proxy_settings()
            render_inventory()
