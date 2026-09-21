"""UI for Segment to Reference: Extract and manage furniture reference crops from renders."""
import json
from pathlib import Path
from PIL import Image
import streamlit as st

from app.layout.persistence import resolve
from services.segment_extractor import (
    extract_instance_crops,
    save_extracted_reference,
)


def render(root: Path, state: dict):
    st.header("✂️ Segment to Reference")
    st.caption(
        "Extract individual furniture and object crops from photorealistic renders using instance segmentation masks. "
        "Save them as authoritative reference images to guarantee 100% design and material consistency across other camera views."
    )

    cameras = state.get("cameras", {})
    if not cameras:
        st.warning("No cameras configured for this layout. Please configure at least one camera first.")
        return

    cam_options = list(cameras.keys())
    active_cam = state.get("active_camera")
    default_idx = cam_options.index(active_cam) if active_cam in cam_options else 0

    col_cam, col_render = st.columns([1, 2])
    with col_cam:
        selected_cam_id = st.selectbox(
            "Select Camera",
            cam_options,
            index=default_idx,
            format_func=lambda cid: f"{cid} · {cameras[cid].get('name', cid)} (HFOV: {cameras[cid].get('fov_deg', cameras[cid].get('fov', 60))}°)",
            key=f"segref_cam_{root.name}",
        )

    renders_dir = root / "generated" / selected_cam_id / "renders"
    past_renders = sorted(renders_dir.glob("render_*.png"), reverse=True) if renders_dir.is_dir() else []

    if not past_renders:
        st.info(
            f"No renders found for camera '{selected_cam_id}'. "
            "Please go to the **Generate Image** tab to generate an architectural render first."
        )
        return

    render_options = [r.name for r in past_renders]
    with col_render:
        selected_render_name = st.selectbox(
            "Select Render to Extract From",
            render_options,
            index=0,
            key=f"segref_sel_render_{selected_cam_id}_{root.name}",
        )

    selected_render_path = renders_dir / selected_render_name
    selected_json_path = selected_render_path.with_suffix(".json")
    meta = {}
    if selected_json_path.is_file():
        try:
            with open(selected_json_path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
        except Exception:
            pass

    # Main layout: Left = Full Render, Right = Settings & Extractor
    st.divider()
    r_col1, r_col2 = st.columns([3, 2])
    with r_col1:
        st.markdown(f"**Render Image:** `{selected_render_name}`")
        st.image(str(selected_render_path), use_container_width=True)

    with r_col2:
        st.markdown("**Extraction Settings**")
        if meta:
            st.caption(
                f"Model: `{meta.get('config', {}).get('model', 'N/A')}` | "
                f"Timestamp: `{meta.get('timestamp', 'N/A')}` | "
                f"Latency: `{meta.get('generation_metadata', {}).get('latency_seconds', 'N/A')}s`"
            )

        filter_furniture = st.checkbox(
            "Show only furniture & custom objects",
            value=True,
            key=f"segref_filt_{selected_cam_id}_{root.name}",
        )
        pad_slider = st.slider(
            "Margin Padding (%)",
            min_value=0,
            max_value=25,
            value=8,
            step=1,
            key=f"segref_pad_{selected_cam_id}_{root.name}",
        )

    # Check instance maps
    cam_dir = root / "generated" / selected_cam_id
    inst_png = cam_dir / "instance.png"
    inst_npy = cam_dir / "instance.npy"
    inst_map = state.get("instance_mapping", {}).get(selected_cam_id, {})
    entities = state.get("entities", {})

    if not (inst_png.is_file() or inst_npy.is_file()):
        st.warning(f"Missing instance segmentation map (`instance.png`/`instance.npy`) for camera '{selected_cam_id}'.")
        return

    if not inst_map:
        st.warning(f"No instance mapping found for camera '{selected_cam_id}' in layout.json.")
        return

    inst_source = inst_npy if inst_npy.is_file() else inst_png
    with st.spinner("Extracting instance crops..."):
        try:
            crops = extract_instance_crops(
                render_image=selected_render_path,
                instance_image=inst_source,
                instance_mapping=inst_map,
                entities=entities,
                state=state,
                padding_ratio=pad_slider / 100.0,
            )
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            return

    display_crops = [c for c in crops if c.get("semantic") in ["furniture", "ceiling"]] if filter_furniture else crops

    st.divider()
    st.subheader(f"Extracted Object Crops ({len(display_crops)})")
    st.caption(
        "Click **💾 Save as Reference** to store any crop in `references/furniture/` and attach it to that entity. "
        "It will be immediately available to maintain cross-view consistency in all other camera angles."
    )

    if not display_crops:
        st.info("No matching instances found with sufficient pixel area.")
        return

    grid_cols = st.columns(3)
    for c_idx, c_item in enumerate(display_crops):
        with grid_cols[c_idx % 3]:
            st.image(c_item["crop_image"], use_container_width=True)
            st.markdown(f"**{c_item['display_label']}** (Instance {c_item['instance_id']})")
            st.caption(f"Entity ID: `{c_item['entity_id']}` | Resolution: {c_item['crop_size'][0]}×{c_item['crop_size'][1]} px")

            if c_item.get("has_existing_reference"):
                st.success(f"✓ Saved Reference: `{Path(c_item['existing_reference']).name}`")
            else:
                st.caption("No reference image currently assigned.")

            if st.button(
                "💾 Save as Reference",
                key=f"btn_segref_save_{c_item['instance_id']}_{selected_render_name}_{root.name}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    res = save_extracted_reference(
                        root=root,
                        state=state,
                        entity_id=c_item["entity_id"],
                        crop_image=c_item["crop_image"],
                        camera_id=selected_cam_id,
                    )
                    st.success(f"Successfully saved reference for {c_item['display_label']} to `{res['relative_path']}`!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to save reference: {exc}")
