"""UI Component for Part 7B — Spatial Measurement and Metric Reconstruction."""
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image
import streamlit as st

from services.validation.measurement_service import (
    generate_debug_overlay,
    load_measurements,
    measure_all_entities,
    measure_entity,
    render_topdown_planar_solution,
    save_entity_overlay,
    save_measurements,
)
from services.validation.vertical_metrology import propose_vertical_observations


def render_measurement_dashboard(root: Path, state: dict, context: dict, camera: dict):
    st.subheader("Part 7B · Spatial Measurement & Metric Reconstruction")
    st.caption(
        "Measure raw spatial errors between AI-generated objects and CAD metric ground truth. "
        "Produces metric planar errors (m), image-space mask metrics, and single-view vertical metrology (m). "
        "No SAS score is calculated at this stage."
    )

    doc = context.get("doc") or {}
    correspondences = doc.get("correspondences", [])
    if not correspondences:
        st.info("No Part 7A correspondences found. Run and review Part 7A correspondence first.")
        return

    # Check resolution alignment
    if not context.get("aligned", True):
        st.error(
            "Resolution mismatch between CAD conditioning and render. "
            "Metric homography and metrology are blocked until coordinate alignment is restored."
        )
        return

    eval_folder = context["evaluation"]
    session_key = f"spatial_7b_{root.name}_{context.get('camera_id')}_{context['render'].stem}"
    if session_key not in st.session_state:
        # Load from disk if available
        existing = load_measurements(context)
        st.session_state[session_key] = {
            "measurements": existing,
            "manual_observations": {},
        }
    app_state = st.session_state[session_key]

    # Batch compute button
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.write(
            f"Evaluated camera: **{context.get('camera_id')}** · Render: **{context['render'].name}**"
        )
    with top_col2:
        if st.button("Run Spatial Measurements", type="primary", use_container_width=True):
            with st.spinner("Computing planar and vertical spatial errors…"):
                summary = measure_all_entities(
                    context,
                    camera,
                    manual_observations_map=app_state.get("manual_observations", {}),
                )
                save_measurements(context, summary)
                app_state["measurements"] = summary
                st.success("Measurements computed and saved to measurements.json!")
                st.rerun()

    current_data = app_state.get("measurements")
    if not current_data:
        # Automatically run once if not yet computed
        summary = measure_all_entities(
            context,
            camera,
            manual_observations_map=app_state.get("manual_observations", {}),
        )
        save_measurements(context, summary)
        app_state["measurements"] = summary
        current_data = summary

    # Display overview metrics table
    entities_data = current_data.get("entities", [])
    table_rows = []
    for ent in entities_data:
        eid = ent.get("entity_id")
        name = context["entities"].get(eid, {}).get("description") or context["entities"].get(eid, {}).get("category") or eid
        pl = ent.get("planar", {})
        pos_err = pl.get("position_error_m") if isinstance(pl, dict) else None
        im = ent.get("image_space", {})
        iou = im.get("mask_iou") if isinstance(im, dict) else None
        vt = ent.get("vertical", {})
        h_err = vt.get("height_error_m") if isinstance(vt, dict) else None
        h_pct = vt.get("height_error_percent") if isinstance(vt, dict) else None

        table_rows.append({
            "Entity": f"{name} ({eid})",
            "Semantic": ent.get("semantic", "-"),
            "Status": ent.get("measurement_status", "-"),
            "Position Error (m)": f"{pos_err:.3f}" if pos_err is not None else "-",
            "Mask IoU": f"{iou:.3f}" if iou is not None else "-",
            "Height Error (m)": f"{h_err:.3f}" if h_err is not None else "-",
            "Height Error (%)": f"{h_pct:.1f}%" if h_pct is not None else "-",
        })

    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    st.write("---")
    st.subheader("🔍 Detailed Entity Metric Inspection")

    # Select entity to inspect in detail
    visible_eids = [e["entity_id"] for e in context.get("visible", [])]
    if not visible_eids:
        st.info("No visible entities to inspect.")
        return

    def fmt_eid(k):
        m = context["entities"].get(k, {})
        d = m.get("description") or m.get("category") or k
        return f"{d} · {k}"

    selected_eid = st.selectbox("Inspect Entity", visible_eids, format_func=fmt_eid, key=f"insp_{session_key}")
    selected_ent_data = next((e for e in entities_data if e.get("entity_id") == selected_eid), None)

    if not selected_ent_data:
        st.info("No measurement data for this entity.")
        return

    # 3-column metric cards
    col_pl, col_im, col_vt = st.columns(3)

    # 1. Planar Card
    with col_pl:
        st.markdown("### 📏 Planar Position (XY)")
        pl = selected_ent_data.get("planar")
        if isinstance(pl, dict) and pl.get("status") == "evaluated":
            st.metric("Position Error", f"{pl['position_error_m']:.3f} m")
            st.caption(f"CAD XY: `{pl['ground_truth_xy']}`")
            st.caption(f"Generated XY: `{pl['generated_xy']}`")
            st.caption(f"Method: `{pl['ground_contact_method']}`")
        elif pl == "not_applicable":
            st.info("Planar position not applicable for this entity class.")
        else:
            reason = pl.get("reason") if isinstance(pl, dict) else "Could not evaluate"
            st.warning(f"Cannot evaluate: {reason}")

    # 2. Image-space Mask Card
    with col_im:
        st.markdown("### 🖼️ Mask Metrics (Image-space)")
        im = selected_ent_data.get("image_space")
        if isinstance(im, dict):
            st.metric("Mask IoU", f"{im['mask_iou']:.3f}")
            disp = im.get("centroid_displacement_px")
            st.caption(f"Centroid Disp: `{disp:.1f} px`" if disp is not None else "-")
            w_err = im.get("bbox_width_error_percent")
            h_err = im.get("bbox_height_error_percent")
            st.caption(f"BBox W/H Err: `{w_err}%` / `{h_err}%`" if w_err is not None else "-")
            a_err = im.get("projected_area_error_percent")
            st.caption(f"Area Error: `{a_err}%`" if a_err is not None else "-")
        else:
            st.info("Image-space mask not applicable.")

    # 3. Vertical Metrology Card
    with col_vt:
        st.markdown("### 📐 Single-View Metrology (Z)")
        vt = selected_ent_data.get("vertical")
        if isinstance(vt, dict) and vt.get("status") == "evaluated":
            h_err = vt.get("height_error_m", 0.0)
            h_pct = vt.get("height_error_percent", 0.0)
            st.metric("Height Error", f"{h_err:.3f} m", delta=f"{h_pct:.1f}% error", delta_color="inverse")
            st.caption(f"CAD Height: `{vt.get('cad_height_m')} m`")
            st.caption(f"Reconstructed: `{vt.get('reconstructed_height_m')} m`")
            st.caption(f"Observation: `{vt.get('observation_method')}`")
        elif vt == "not_applicable":
            st.info("Vertical height not applicable for this entity class.")
        else:
            reason = vt.get("reason") if isinstance(vt, dict) else "Could not evaluate"
            st.warning(f"Cannot evaluate: {reason}")

    # Openings card if applicable
    op = selected_ent_data.get("openings")
    if isinstance(op, dict) and op.get("evaluated"):
        st.markdown("#### 🚪 Opening Dimensions (CAD)")
        st.write(f"Opening Height: **{op.get('opening_height_cad_m')} m** · Sill Height: **{op.get('sill_height_cad_m')} m**")

    st.write("---")

    # Visual Inspection & Landmark Tuning Section
    st.markdown("### 🎯 Diagnostic Visual Overlay & Landmark Tuning")

    col_view, col_tune = st.columns([3, 2])

    corr_record = next(
        (r for r in correspondences if r.get("entity_id") == selected_eid), None
    )
    mask_rel = corr_record.get("generated_mask") if corr_record else None
    gen_mask = None
    if mask_rel and (context["evaluation"] / mask_rel).is_file():
        gen_mask = np.asarray(Image.open(context["evaluation"] / mask_rel).convert("L")) > 0

    gt_mask = (context["ids"] == selected_ent_data.get("instance_id")) if selected_ent_data.get("instance_id") is not None else None

    # Current landmarks
    planar_dict = selected_ent_data.get("planar") if isinstance(selected_ent_data.get("planar"), dict) else {}
    vert_dict = selected_ent_data.get("vertical") if isinstance(selected_ent_data.get("vertical"), dict) else {}

    g_contact = planar_dict.get("generated_contact_pixel")
    top_pt = vert_dict.get("top_pixel")
    base_pt = vert_dict.get("base_pixel")

    with col_tune:
        st.markdown("**Landmark Adjustment**")
        st.caption("Inspect and fine-tune observation points for Single-View Metrology:")

        auto_obs = propose_vertical_observations(gen_mask) if gen_mask is not None else None
        curr_base = base_pt or (auto_obs["base_pixel"] if auto_obs else [0.0, 0.0])
        curr_top = top_pt or (auto_obs["top_pixel"] if auto_obs else [0.0, 0.0])

        b_col1, b_col2 = st.columns(2)
        new_base_u = b_col1.number_input("Base X (px)", value=float(curr_base[0]), step=1.0, key=f"b_x_{selected_eid}")
        new_base_v = b_col2.number_input("Base Y (px)", value=float(curr_base[1]), step=1.0, key=f"b_y_{selected_eid}")

        t_col1, t_col2 = st.columns(2)
        new_top_u = t_col1.number_input("Top X (px)", value=float(curr_top[0]), step=1.0, key=f"t_x_{selected_eid}")
        new_top_v = t_col2.number_input("Top Y (px)", value=float(curr_top[1]), step=1.0, key=f"t_y_{selected_eid}")

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Apply & Recalculate", type="primary", use_container_width=True):
                obs_data = {
                    "base_pixel": [new_base_u, new_base_v],
                    "top_pixel": [new_top_u, new_top_v],
                    "observation_method": "human_corrected",
                }
                app_state["manual_observations"][selected_eid] = obs_data
                # Recalculate this entity
                updated_res = measure_entity(
                    context, selected_eid, doc, camera, manual_observations=obs_data
                )
                # Update in full list
                for idx, e in enumerate(app_state["measurements"]["entities"]):
                    if e.get("entity_id") == selected_eid:
                        app_state["measurements"]["entities"][idx] = updated_res
                        break
                save_measurements(context, app_state["measurements"])
                st.success("Updated observations!")
                st.rerun()

        with btn2:
            if st.button("Reset to Automatic", use_container_width=True):
                if selected_eid in app_state["manual_observations"]:
                    del app_state["manual_observations"][selected_eid]
                updated_res = measure_entity(context, selected_eid, doc, camera)
                for idx, e in enumerate(app_state["measurements"]["entities"]):
                    if e.get("entity_id") == selected_eid:
                        app_state["measurements"]["entities"][idx] = updated_res
                        break
                save_measurements(context, app_state["measurements"])
                st.rerun()

    with col_view:
        overlay_img = generate_debug_overlay(
            context["image"],
            gt_mask,
            gen_mask,
            ground_contact_pixel=g_contact,
            top_pixel=top_pt,
        )
        save_entity_overlay(context, selected_eid, overlay_img)
        st.image(
            overlay_img,
            caption="Diagnostic Overlay: Cyan = CAD Contour | Green = SAM Mask | Yellow = Ground Contact | Magenta = Top Point",
            use_container_width=True,
        )

    # Section: 2D Planar Solution (Bird's-Eye Top-Down Projection)
    st.write("---")
    st.markdown("### 🗺️ Solution: 2D Planar Top-Down Projection (Bird's Eye View)")
    st.caption(
        "Direct visual verification on the 2D CAD floor plane: Visualizes the true physical positions on the floor, "
        "proving the Reverse Homography ($H_{image \\to cad}$) transformation is genuine and verifiable in metric 2D space."
    )

    try:
        topdown_solutions = render_topdown_planar_solution(
            context,
            selected_eid,
            camera,
            selected_ent_data,
            width=800,
            height=520,
        )
        tab_diag, tab_tex = st.tabs([
            "📐 2D Metric Floor Plan (CAD Blueprint)",
            "🖼️ Unprojected Floor Texture (Bird's Eye Render)",
        ])
        with tab_diag:
            st.image(
                topdown_solutions["diagram"],
                caption="Orthographic 2D Blueprint: Camera FOV Cone · Cyan Crosshair = CAD Ground Truth · Yellow Dot = AI Predicted Location · Red Arrow = Error Vector",
                use_container_width=True,
            )
        with tab_tex:
            st.image(
                topdown_solutions["warped_texture"],
                caption="Unprojected 2D Aerial View: AI render perspective floor texture warped onto 2D CAD plane and aligned with CAD geometry",
                use_container_width=True,
            )

        # Metric displacement details callout
        pl_data = selected_ent_data.get("planar") if isinstance(selected_ent_data.get("planar"), dict) else {}
        gt_xy = pl_data.get("ground_truth_xy")
        gen_xy = pl_data.get("generated_xy")
        if gt_xy and gen_xy:
            dx = float(gen_xy[0]) - float(gt_xy[0])
            dy = float(gen_xy[1]) - float(gt_xy[1])
            err_m = float(pl_data.get("position_error_m", 0.0))

            angle_deg = (math.degrees(math.atan2(dy, dx))) % 360
            compass_sectors = [
                "East (+X)", "North-East", "North (+Y)", "North-West",
                "West (-X)", "South-West", "South (-Y)", "South-East"
            ]
            compass_dir = compass_sectors[int((angle_deg + 22.5) // 45) % 8]

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("CAD Ground Truth", f"X={gt_xy[0]:.2f}m, Y={gt_xy[1]:.2f}m")
            m_c2.metric("AI Prediction", f"X={gen_xy[0]:.2f}m, Y={gen_xy[1]:.2f}m")
            m_c3.metric("Displacement Vector", f"ΔX={dx:+.2f}m, ΔY={dy:+.2f}m", delta=f"{compass_dir}")
            m_c4.metric("Total Position Error", f"{err_m:.3f} m", delta=f"{err_m*100:.1f} cm", delta_color="inverse")
    except Exception as exc:
        st.warning(f"Could not construct 2D top-down projection: {exc}")

    # Download measurements JSON
    with st.expander("📄 Raw measurements.json Data"):
        st.download_button(
            "Download measurements.json",
            json.dumps(current_data, indent=2),
            file_name="measurements.json",
            mime="application/json",
        )
        st.json(current_data)
