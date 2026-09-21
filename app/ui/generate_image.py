from datetime import datetime
import json
from pathlib import Path
import streamlit as st
from PIL import Image

from services.api_manager import (
    list_configs,
    get_config,
    save_config,
    delete_config,
    mask_api_key,
    test_connection,
)
from services.prompt_builder import (
    get_available_conditions,
    compile_prompt,
    validate_prompt_request,
    DEFAULT_ROLES,
)
from services.image_generation.gemini_provider import GeminiImageGenerator
from app.ui.prompt_panel import render_prompt_panel
from services.prompt_resolution import SAME_ROOM_ROLE
from services.prepared_request import prepare_request, request_fingerprint
from app.layout.persistence import resolve
from services.segment_extractor import extract_instance_crops, save_extracted_reference


def render(root: Path, state: dict):
    st.header("Multimodal AI Image Generation")
    st.caption("Generate photorealistic architectural renderings conditioned on CAD-derived geometry and calibrated cameras.")

    cameras = state.get("cameras", {})
    if not cameras:
        st.warning("No cameras configured for this layout. Please switch to **Camera Mode** to create at least one calibrated camera.")
        return

    # -------------------------------------------------------------------------
    # 1. CAMERA SELECTION
    # -------------------------------------------------------------------------
    st.subheader("1. Camera Selection")
    cam_options = list(cameras.keys())
    active_cam = state.get("active_camera")
    default_idx = cam_options.index(active_cam) if active_cam in cam_options else 0

    selected_cam_id = st.selectbox(
        "Select Camera",
        cam_options,
        index=default_idx,
        format_func=lambda cid: f"{cid} · {cameras[cid].get('name', cid)} (HFOV: {cameras[cid].get('fov_deg', cameras[cid].get('fov', 60))}°)",
        key=f"gen_cam_{root.name}",
    )

    cam_data = cameras[selected_cam_id]
    pos = cam_data.get("position", [0, 0, 0])
    tgt = cam_data.get("target", [0, 0, 0])
    units = state.get("source", {}).get("units", "m")

    st.caption(
        f"Camera Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] {units} | "
        f"Target: [{tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}] {units} | "
        f"Horizontal FOV: {cam_data.get('fov_deg', cam_data.get('fov', 60))}°"
    )

    # -------------------------------------------------------------------------
    # 2. CONDITION IMAGE SELECTION
    # -------------------------------------------------------------------------
    st.divider()
    c2_col1, c2_col2 = st.columns([3, 1])
    with c2_col1:
        st.subheader("2. Spatial Conditioning Inputs")
        st.caption("Discovered conditioning passes from CAD geometry and calibrated camera.")
    with c2_col2:
        if st.button("🎯 Depth + Instance Only", key=f"btn_quick_cond_{root.name}_{selected_cam_id}", help="Quick select only Depth and Instance conditions"):
            available_conds_temp = get_available_conditions(root, selected_cam_id)
            for c_item in available_conds_temp:
                st.session_state[f"inc_{selected_cam_id}_{c_item['type']}_{root.name}"] = c_item['type'] in ["depth", "instance"]
            st.rerun()

    available_conditions = get_available_conditions(root, selected_cam_id)
    active_condition_inputs = []

    cols = st.columns(len(available_conditions))
    for idx, cond in enumerate(available_conditions):
        ctype = cond["type"]
        with cols[idx]:
            st.markdown(f"**{ctype.upper()}**")
            if cond["exists"]:
                try:
                    st.image(cond["path"], use_container_width=True)
                    st.caption(f"✓ {cond['resolution'][0]}x{cond['resolution'][1]} px")
                except Exception:
                    st.caption("✓ File exists")
                
                default_include = ctype in ["depth", "instance"]
                include = st.checkbox(
                    "Include",
                    value=default_include,
                    key=f"inc_{selected_cam_id}_{ctype}_{root.name}",
                )
                role = st.text_input(
                    "Role",
                    value=cond["role"],
                    key=f"role_v2_{selected_cam_id}_{ctype}_{root.name}",
                    disabled=True,
                    help=cond.get("role_description", ""),
                )
                if include:
                    cond_copy = dict(cond)
                    cond_copy["role"] = role
                    cond_copy["enabled"] = True
                    active_condition_inputs.append(cond_copy)
            else:
                st.info("Not Generated")
                st.caption(f"Missing: `{cond['filename']}`")
                st.checkbox("Include", value=False, disabled=True, key=f"inc_dis_{selected_cam_id}_{ctype}_{root.name}")

    if not any(c["exists"] for c in available_conditions):
        st.warning(
            f"No conditioning maps found for camera '{selected_cam_id}'. "
            "Switch to **Camera Mode** and click **Export Conditioning (PNG & NPY)** to generate them."
        )

    # -------------------------------------------------------------------------
    # 3. ADDITIONAL REFERENCE INPUTS
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("3. Additional Reference Inputs")
    st.caption("Upload reference images or use saved furniture crops from previous renders to retain an existing room design across camera views.")

    reference_inputs = []

    # Check for saved furniture references in the layout
    saved_furniture_refs = []
    saved_material_swatches = []
    for ent_id, ent in state.get("entities", {}).items():
        ref_rel = ent.get("reference_image")
        if ref_rel:
            try:
                ref_abs = resolve(root, ref_rel)
                if ref_abs.is_file():
                    sem = ent.get("semantic", "furniture")
                    label = ent.get("notes") or ent.get("description") or ent.get("category") or ent_id
                    item = {
                        "entity_id": ent_id,
                        "label": label,
                        "rel_path": ref_rel,
                        "abs_path": str(ref_abs),
                        "name": ref_abs.name,
                        "semantic": sem,
                        "role": ent.get("reference_role", "Material Reference" if sem in ("wall", "floor") else SAME_ROOM_ROLE),
                    }
                    if sem in ("wall", "floor"):
                        saved_material_swatches.append(item)
                    else:
                        saved_furniture_refs.append(item)
            except Exception:
                pass

    ceil_ref_rel = state.get("ceiling", {}).get("reference_image")
    if ceil_ref_rel:
        try:
            ceil_ref_abs = resolve(root, ceil_ref_rel)
            if ceil_ref_abs.is_file():
                saved_furniture_refs.append({
                    "entity_id": "auto_ceiling",
                    "label": "Ceiling & Cove Lighting",
                    "rel_path": ceil_ref_rel,
                    "abs_path": str(ceil_ref_abs),
                    "name": ceil_ref_abs.name,
                    "semantic": "ceiling",
                    "role": SAME_ROOM_ROLE,
                })
        except Exception:
            pass

    if saved_furniture_refs:
        with st.expander(f"📦 Saved Furniture References from Layout ({len(saved_furniture_refs)})", expanded=True):
            st.caption("These references preserve 3D object styling and identity (e.g. bed, table, cove lighting) and will be re-projected to fit the target camera angle.")
            include_all_saved = st.checkbox(
                "Enable Saved Furniture References",
                value=True,
                key=f"enable_saved_refs_master_{selected_cam_id}_{root.name}",
                help="Uncheck if you want to use only your uploaded reference images below without sending individual saved crops.",
            )
            if include_all_saved:
                s_cols = st.columns(min(len(saved_furniture_refs), 4))
                for s_idx, s_ref in enumerate(saved_furniture_refs):
                    with s_cols[s_idx % len(s_cols)]:
                        st.image(s_ref["abs_path"], use_container_width=True)
                        use_ref = st.checkbox(
                            f"Include {s_ref['label']}",
                            value=True,
                            key=f"use_saved_ref_{s_ref['entity_id']}_{selected_cam_id}_{root.name}",
                        )
                        if use_ref:
                            try:
                                reference_inputs.append({
                                    "name": s_ref["name"],
                                    "bytes": Path(s_ref["abs_path"]).read_bytes(),
                                    "role": SAME_ROOM_ROLE,
                                    "description": f"Authoritative reference crop for {s_ref['label']} ({s_ref['entity_id']}). Maintain exact design, materials, and styling in this view.",
                                })
                            except Exception:
                                pass
            else:
                st.caption("Saved layout references are disabled. Only references uploaded below will be sent.")

    if saved_material_swatches:
        with st.expander(f"🎨 Saved Material Swatches from Layout ({len(saved_material_swatches)})", expanded=True):
            st.caption("These texture swatches preserve surface finishes (paint plaster, wood grain) without affecting the camera perspective or room layout.")
            m_cols = st.columns(min(len(saved_material_swatches), 4))
            for m_idx, m_ref in enumerate(saved_material_swatches):
                with m_cols[m_idx % len(m_cols)]:
                    st.image(m_ref["abs_path"], use_container_width=True)
                    use_mat = st.checkbox(
                        f"Include {m_ref['label']}",
                        value=True,
                        key=f"use_saved_mat_{m_ref['entity_id']}_{selected_cam_id}_{root.name}",
                    )
                    if use_mat:
                        try:
                            reference_inputs.append({
                                "name": m_ref["name"],
                                "bytes": Path(m_ref["abs_path"]).read_bytes(),
                                "role": "Material Reference",
                                "description": f"Authoritative material texture swatch for {m_ref['label']} ({m_ref['entity_id']}). Apply this surface finish without copying camera perspective.",
                            })
                        except Exception:
                            pass

    uploaded_files = st.file_uploader(
        "Upload Additional Reference Images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"ref_uploads_{root.name}",
    )
    if uploaded_files:
        st.info(
            "💡 **Tip**: You have uploaded reference image(s). "
            "If you want to use ONLY your uploaded image (like on Gemini Web) without combining it with the individual saved crops above, "
            "make sure **Enable Saved Furniture References** above is unchecked."
        )
        role_options = [
            "Appearance / Style Reference",
            SAME_ROOM_ROLE,
            "Material Reference",
            "Furniture Reference",
            "Lighting Reference",
            "Other",
        ]
        for idx, ufile in enumerate(uploaded_files):
            r_col1, r_col2 = st.columns([1, 3])
            with r_col1:
                st.image(ufile, use_container_width=True)
                st.caption(f"{ufile.name} ({round(len(ufile.getvalue()) / 1024, 1)} KB)")
            with r_col2:
                ref_role = st.selectbox(
                    f"Reference #{idx + 1} Role",
                    role_options,
                    key=f"ref_role_{idx}_{ufile.name}_{root.name}",
                )
                default_desc = ""
                same_room = ref_role == SAME_ROOM_ROLE
                if same_room:
                    st.caption("Keep the same furniture design, materials and lighting character. The selected camera's instance map controls the new view; include Depth for distance and occlusion guidance.")
                    st.caption("Use Instructions to name details to retain, such as window-frame finish or wall-mounted accessories. Objects absent from CAD have no calibrated position; annotate them in the layout when precise placement matters.")
                ref_desc = st.text_area(
                    f"Reference #{idx + 1} Instructions",
                    value=st.session_state.get(f"ref_desc_{idx}_{ufile.name}_{root.name}", default_desc),
                    placeholder=("e.g. Retain the reference window-frame finish, dining chairs and wall-mounted TV on the same wall wherever visible in the new view."
                                 if same_room else "e.g. Decorate using materials, lighting, and textures from this image. Do not copy the room layout."),
                    key=f"ref_desc_{idx}_{ufile.name}_{root.name}",
                    help="Describe appearance details for the selected reference role. CAD condition maps remain the primary spatial reference.",
                )
                reference_inputs.append({
                    "name": ufile.name,
                    "bytes": ufile.getvalue(),
                    "role": ref_role,
                    "description": ref_desc,
                })

    # -------------------------------------------------------------------------
    # 4. SEMANTIC INFO & PROMPT BUILDER
    # -------------------------------------------------------------------------
    st.divider()
    p_hdr_col1, p_hdr_col2 = st.columns([3, 1])
    with p_hdr_col1:
        st.subheader("4. Semantic Instructions & Prompt Builder")
    with p_hdr_col2:
        if st.button("🔄 Reset to Auto Prompt", key=f"btn_reset_auto_prompt_{root.name}", help="Clear appearance prompt and reset to CAD-derived auto prompt"):
            st.session_state[f"design_prompt_input_{root.name}"] = ""
            st.session_state[f"design_prompt_{root.name}"] = ""
            st.session_state.pop(f"custom_final_prompt_{root.name}", None)
            st.rerun()

    c_eff1, c_eff2 = st.columns([2, 2])
    with c_eff1:
        strictness_level = st.select_slider(
            "🎯 Camera Adherence Effort",
            options=["Balanced", "Strict", "Very Strict (nanobanana)"],
            value=st.session_state.get(f"strictness_{root.name}", "Very Strict (nanobanana)"),
            key=f"strictness_slider_{root.name}",
            help="Controls how strongly Gemini is instructed to prioritize the target camera condition maps over the reference images.",
        )
        st.session_state[f"strictness_{root.name}"] = strictness_level

    user_design_prompt = st.text_area(
        "Appearance Prompt",
        value=st.session_state.get(f"design_prompt_{root.name}", ""),
        placeholder="e.g. warm afternoon sunlight streaming in, Scandinavian minimalist interior, natural light oak wood finishes, warm ambient glow, photorealistic 8k architectural render",
        height=100,
        key=f"design_prompt_input_{root.name}",
    )
    st.session_state[f"design_prompt_{root.name}"] = user_design_prompt

    prompt_build = render_prompt_panel(root, state, selected_cam_id, active_condition_inputs,
                                       reference_inputs, user_design_prompt,
                                       strictness=strictness_level)
    final_prompt = prompt_build.final_prompt

    # -------------------------------------------------------------------------
    # 5. API & MODEL CONFIGURATION
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("5. Local API Configuration")

    saved_configs = list_configs()
    config_dict = {cfg["id"]: cfg for cfg in saved_configs}
    config_labels = {
        cfg["id"]: f"{cfg.get('name', cfg['id'])} ({cfg.get('model', 'gemini-2.5-flash')} · {cfg.get('masked_key', '')})"
        for cfg in saved_configs
    }

    c_col1, c_col2 = st.columns([3, 1])
    selected_config_id = None
    if saved_configs:
        selected_config_id = c_col1.selectbox(
            "Select API Configuration",
            options=list(config_dict.keys()),
            format_func=lambda cid: config_labels.get(cid, cid),
            key=f"sel_config_{root.name}",
        )
    else:
        c_col1.info("No saved API configurations found. Please create one below.")

    with c_col2:
        st.write("")
        st.write("")
        if selected_config_id:
            if st.button("🔌 Test Connection", key=f"test_conn_{root.name}"):
                cfg_to_test = get_config(selected_config_id)
                if cfg_to_test:
                    with st.spinner("Testing API connection..."):
                        ok, msg = test_connection(cfg_to_test)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    # API Configuration Manager
    with st.expander("⚙️ Manage Local API Configurations", expanded=not bool(saved_configs)):
        st.caption("Configurations are stored strictly on your local disk in `local_api_configs/` and never committed.")

        manage_tab_new, manage_tab_edit = st.tabs(["Add New Configuration", "Edit / Delete Existing"])

        with manage_tab_new:
            with st.form(f"form_add_config_{root.name}"):
                new_name = st.text_input("Friendly Label", placeholder="e.g. Gemini 2.5 Flash / Imagen 3")
                new_provider = st.selectbox("Provider", ["google-genai"])
                model_presets = [
                    "gemini-2.5-flash-image",
                    "gemini-3.1-flash-image",
                    "gemini-3-pro-image",
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                ]
                new_model = st.selectbox("Model", model_presets)
                new_key = st.text_input("API Key", type="password", placeholder="Enter your Google AI API key...")
                new_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

                if st.form_submit_button("Save Configuration", type="primary"):
                    if not new_key.strip():
                        st.error("API Key is required.")
                    else:
                        new_id = save_config({
                            "name": new_name.strip() or f"{new_model} Config",
                            "provider": new_provider,
                            "model": new_model,
                            "api_key": new_key.strip(),
                            "temperature": new_temp,
                        })
                        st.success(f"Configuration saved! (ID: {new_id})")
                        st.rerun()

        with manage_tab_edit:
            if saved_configs:
                edit_id = st.selectbox(
                    "Select Configuration to Manage",
                    options=list(config_dict.keys()),
                    format_func=lambda cid: config_labels.get(cid, cid),
                    key=f"edit_sel_{root.name}",
                )
                curr_cfg = get_config(edit_id)
                if curr_cfg:
                    with st.form(f"form_edit_config_{edit_id}_{root.name}"):
                        edit_name = st.text_input("Friendly Label", value=curr_cfg.get("name", ""))
                        edit_model = st.text_input("Model", value=curr_cfg.get("model", "gemini-2.5-flash"))
                        st.caption(f"Current Key: `{curr_cfg.get('masked_key', '')}`")
                        edit_key = st.text_input("Update API Key (Leave blank to keep existing)", type="password")
                        edit_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(curr_cfg.get("temperature", 0.7)), step=0.05)

                        col_save, col_del = st.columns([1, 1])
                        if col_save.form_submit_button("Update Configuration"):
                            updated_data = {
                                "id": edit_id,
                                "name": edit_name.strip(),
                                "provider": curr_cfg.get("provider", "google-genai"),
                                "model": edit_model.strip(),
                                "api_key": edit_key.strip() if edit_key.strip() else curr_cfg.get("api_key", ""),
                                "temperature": edit_temp,
                            }
                            save_config(updated_data)
                            st.success("Configuration updated!")
                            st.rerun()

                    if st.button("🗑️ Delete Configuration", key=f"del_cfg_{edit_id}_{root.name}"):
                        delete_config(edit_id)
                        st.success("Configuration deleted.")
                        st.rerun()
            else:
                st.info("No configurations available to edit.")

    # -------------------------------------------------------------------------
    # 6. GENERATION ACTION
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("6. Generate Image")

    st.caption("Submit prepares a local snapshot at no API cost. Review it, then Confirm & Generate sends that exact snapshot.")
    prepared_key = f"prepared_request_{root.name}"
    active_cfg = get_config(selected_config_id) if selected_config_id else None
    current_fingerprint = request_fingerprint(prompt_build, state, active_cfg or {})
    prepared = st.session_state.get(prepared_key)
    if prepared and prepared['fingerprint'] != current_fingerprint:
        st.session_state.pop(prepared_key, None)
        prepared = None
        st.info("Inputs changed since Submit. Submit again to review the new request.")
    if st.button("Submit — Prepare Summary", key=f"submit_gen_{root.name}",
                 disabled=bool(prompt_build.errors) or not active_cfg):
        try:
            prepared = prepare_request(prompt_build, state, active_condition_inputs, reference_inputs, active_cfg)
            st.session_state[prepared_key] = prepared
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            prepared = None
            st.session_state.pop(prepared_key, None)
    if prepared:
        st.markdown("**Ready for confirmation**")
        summary_camera = prepared['build'].debug['camera_id']
        st.write(f"Camera: {summary_camera} · {prepared['state']['cameras'][summary_camera].get('name', '')}")
        st.write(f"Model: {prepared['config']['model']} · Temperature: {prepared['config']['temperature']}")
        st.caption(f"Request: {prepared['id']} · Images: {len(prepared['conditions']) + len(prepared['references'])}")
        for item, record in zip([*prepared['conditions'], *prepared['references']], prepared['build'].debug['image_order']):
            st.image(item['bytes'], width=220, caption=f"IMAGE {record['image']} · {record['type']} · {record.get('name') or ''} · {record['resolution'][0]}×{record['resolution'][1]}")
        with st.expander("Exact submitted prompt and camera"):
            st.code(prepared['build'].final_prompt, language=None)
            st.json(prepared['state']['cameras'][summary_camera])
        st.caption("Changing camera, inputs, instructions or model settings requires another Submit. Confirmation uses the image copies shown above.")
    gen_btn = st.button("Confirm & Generate", type="primary", key=f"btn_gen_{root.name}", disabled=prepared is None)

    if gen_btn:
        if not selected_config_id:
            st.error("Please select or configure an API configuration before generating.")
            return

        active_cfg = get_config(selected_config_id)
        if not active_cfg or not active_cfg.get("api_key"):
            st.error("Invalid API configuration or missing API key.")
            return

        try:
            if prepared is None or prepared['fingerprint'] != request_fingerprint(prompt_build, state, active_cfg):
                raise ValueError('Inputs changed. Submit again before confirming.')
            # Consume confirmation before dispatch: reruns and failures require a new Submit.
            st.session_state.pop(prepared_key, None)
            prompt_build = prepared['build']
            state = prepared['state']
            active_condition_inputs = prepared['conditions']
            reference_inputs = prepared['references']
            selected_cam_id = prompt_build.debug['camera_id']
            cam_data = state['cameras'][selected_cam_id]
            units = state.get('source', {}).get('units', 'm')
            user_design_prompt = prompt_build.appearance_prompt
            validate_prompt_request(prompt_build, secrets=[active_cfg.get("api_key", "")])
            final_prompt = prompt_build.final_prompt
            with st.spinner("Executing multimodal image generation..."):
                generator = GeminiImageGenerator()
                res = generator.generate(
                    prompt=final_prompt,
                    condition_images=active_condition_inputs,
                    reference_images=reference_inputs,
                    config=active_cfg,
                )

            # Persist the generated image & metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            renders_dir = root / "generated" / selected_cam_id / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)

            out_png_path = renders_dir / f"render_{timestamp}.png"
            out_json_path = renders_dir / f"render_{timestamp}.json"

            with open(out_png_path, "wb") as f:
                f.write(res.image_bytes)

            with Image.open(out_png_path) as generated_image:
                output_resolution = {"width": generated_image.width, "height": generated_image.height}

            meta_record = {
                "confirmed_request_id": prepared['id'],
                "timestamp": timestamp,
                "camera_id": selected_cam_id,
                "camera": dict(cam_data),
                "drawing_units": units,
                "output_resolution": output_resolution,
                "instance_mapping": state.get("instance_mapping", {}).get(selected_cam_id, {}),
                "ceiling": state.get("ceiling", {}),
                "prompt_builder": prompt_build.debug,
                "prompt_warnings": prompt_build.warnings,
                "spatial_prompt": prompt_build.spatial_prompt,
                "appearance_prompt": prompt_build.appearance_prompt,
                "config": {
                    "id": active_cfg.get("id"),
                    "name": active_cfg.get("name"),
                    "provider": active_cfg.get("provider"),
                    "model": active_cfg.get("model"),
                    "temperature": active_cfg.get("temperature"),
                },
                "conditions": [
                    {
                        "type": c.get("type"),
                        "role": c.get("role"),
                        "resolution": c.get("resolution"),
                        "path": c.get("path"),
                    }
                    for c in active_condition_inputs
                ],
                "references": [
                    {
                        "name": r.get("name"),
                        "role": r.get("role"),
                        "description": r.get("description"),
                    }
                    for r in reference_inputs
                ],
                "user_design_prompt": user_design_prompt,
                "final_prompt": final_prompt,
                "generation_metadata": res.metadata,
                "finish_reason": res.finish_reason,
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(meta_record, f, indent=2, ensure_ascii=False)

            st.session_state[f"latest_render_{root.name}"] = {
                "png_path": str(out_png_path),
                "json_path": str(out_json_path),
                "bytes": res.image_bytes,
                "meta": meta_record,
            }
            st.success(f"Generation successful! Saved to `{out_png_path.name}`.")
        except Exception as exc:
            st.error(f"Generation failed: {str(exc)}")

    # -------------------------------------------------------------------------
    # 7. GENERATED RESULT & HISTORY
    # -------------------------------------------------------------------------
    latest = st.session_state.get(f"latest_render_{root.name}")
    if not latest:
        renders_dir_check = root / "generated" / selected_cam_id / "renders"
        if renders_dir_check.is_dir():
            past = sorted(renders_dir_check.glob("render_*.png"), reverse=True)
            if past:
                most_recent = past[0]
                most_recent_json = most_recent.with_suffix(".json")
                meta_data = {}
                if most_recent_json.is_file():
                    try:
                        with open(most_recent_json, "r", encoding="utf-8") as jf:
                            meta_data = json.load(jf)
                    except Exception:
                        pass
                try:
                    latest = {
                        "png_path": str(most_recent),
                        "bytes": most_recent.read_bytes(),
                        "meta": meta_data,
                    }
                except Exception:
                    latest = None

    if latest:
        st.divider()
        st.subheader("7. Generated Architectural Result")

        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            st.image(latest["bytes"], use_container_width=True)
            st.download_button(
                "💾 Download Render PNG",
                data=latest["bytes"],
                file_name=Path(latest["png_path"]).name,
                mime="image/png",
                key=f"dl_render_{root.name}",
            )

        with res_col2:
            st.markdown("**Generation Metadata**")
            meta = latest["meta"]
            st.write(f"- **Timestamp**: {meta.get('timestamp')}")
            st.write(f"- **Camera**: {meta.get('camera_id')}")
            st.write(f"- **Model**: {meta.get('config', {}).get('model')}")
            st.write(f"- **Latency**: {meta.get('generation_metadata', {}).get('latency_seconds', 'N/A')}s")
            st.write(f"- **Finish Reason**: {meta.get('finish_reason')}")
            st.caption(f"Saved to: `{latest['png_path']}`")
            json_bytes = json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                "📄 Download Metadata JSON",
                data=json_bytes,
                file_name=Path(latest["png_path"]).with_suffix(".json").name,
                mime="application/json",
                key=f"dl_meta_{root.name}",
            )

        with st.expander("🔍 View Generation Metadata Summary", expanded=False):
            st.json({
                "timestamp": meta.get("timestamp"),
                "camera_id": meta.get("camera_id"),
                "model": meta.get("config", {}).get("model"),
                "latency_seconds": meta.get("generation_metadata", {}).get("latency_seconds"),
                "finish_reason": meta.get("finish_reason"),
                "conditions": meta.get("conditions"),
                "references_count": len(meta.get("references", [])),
                "user_design_prompt": meta.get("user_design_prompt"),
            })

        # ---------------------------------------------------------------------
        # 8. SEGMENT TO REFERENCE (EXTRACT FURNITURE CROPS)
        # ---------------------------------------------------------------------
        st.divider()
        with st.container():
            with st.expander("✂️ Segment to Reference: Extract Furniture Crops from this Render", expanded=True):
                render_segment_to_reference(
                    root=root,
                    state=state,
                    camera_id=meta.get("camera_id", selected_cam_id),
                    render_bytes=latest["bytes"],
                    render_name=Path(latest["png_path"]).name,
                    key_prefix="latest",
                )

    # Previous renders gallery
    renders_dir = root / "generated" / selected_cam_id / "renders"
    if renders_dir.is_dir():
        past_renders = sorted(renders_dir.glob("render_*.png"), reverse=True)
        if past_renders:
            with st.expander(f"🖼️ Previous Renders for Camera '{selected_cam_id}' ({len(past_renders)})", expanded=False):
                grid_cols = st.columns(3)
                for idx, r_png in enumerate(past_renders):
                    with grid_cols[idx % 3]:
                        st.image(str(r_png), use_container_width=True)
                        st.caption(r_png.name)
                        r_json = r_png.with_suffix(".json")
                        if r_json.is_file():
                            try:
                                with open(r_json, "r", encoding="utf-8") as jf:
                                    j_data = json.load(jf)
                                    st.caption(f"{j_data.get('config', {}).get('model')} · {j_data.get('timestamp')}")
                            except Exception:
                                pass
                        if st.button("✂️ Extract Crops", key=f"btn_ext_past_{r_png.stem}_{root.name}"):
                            st.session_state[f"extract_target_{root.name}"] = str(r_png)
                            st.rerun()

            # Extraction panel for a selected past render
            extract_target = st.session_state.get(f"extract_target_{root.name}")
            if extract_target and Path(extract_target).is_file():
                st.markdown("---")
                with st.expander(f"✂️ Segment to Reference: Extracting from `{Path(extract_target).name}`", expanded=True):
                    render_segment_to_reference(
                        root=root,
                        state=state,
                        camera_id=selected_cam_id,
                        render_bytes=Path(extract_target).read_bytes(),
                        render_name=Path(extract_target).name,
                        key_prefix=f"past_{Path(extract_target).stem}",
                    )
                    if st.button("Close Extractor", key=f"close_past_ext_{root.name}"):
                        st.session_state.pop(f"extract_target_{root.name}", None)
                        st.rerun()


def render_segment_to_reference(
    root: Path,
    state: dict,
    camera_id: str,
    render_bytes: bytes,
    render_name: str,
    key_prefix: str,
):
    cam_dir = root / "generated" / camera_id
    inst_png = cam_dir / "instance.png"
    inst_npy = cam_dir / "instance.npy"
    inst_map = state.get("instance_mapping", {}).get(camera_id, {})
    entities = state.get("entities", {})

    if not (inst_png.is_file() or inst_npy.is_file()):
        st.info(f"No instance segmentation map found for camera '{camera_id}'. Cannot extract crops.")
        return
    if not inst_map:
        st.info(f"No instance mapping found for camera '{camera_id}'. Cannot map crops to entities.")
        return

    st.caption(
        "Crop and save individual furniture pieces from this render as authoritative reference images. "
        "These saved references will ensure 100% design and material consistency when generating other camera views of this room."
    )

    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        filter_furniture = st.checkbox(
            "Show only furniture & custom objects",
            value=True,
            key=f"filt_furn_{key_prefix}_{root.name}",
        )
    with f_col2:
        pad_slider = st.slider(
            "Margin Padding (%)",
            min_value=0,
            max_value=25,
            value=8,
            step=1,
            key=f"pad_{key_prefix}_{root.name}",
        )

    inst_source = inst_npy if inst_npy.is_file() else inst_png
    try:
        crops = extract_instance_crops(
            render_image=render_bytes,
            instance_image=inst_source,
            instance_mapping=inst_map,
            entities=entities,
            state=state,
            padding_ratio=pad_slider / 100.0,
        )
    except Exception as exc:
        st.error(f"Failed to extract instance crops: {exc}")
        return

    if filter_furniture:
        display_crops = [c for c in crops if c.get("semantic") in ["furniture", "ceiling"]]
    else:
        display_crops = crops

    if not display_crops:
        st.info("No matching instances found in this view (or all below visibility threshold).")
        return

    st.write(f"Found **{len(display_crops)}** extractable objects:")
    c_cols = st.columns(min(len(display_crops), 3))
    for c_idx, c_item in enumerate(display_crops):
        with c_cols[c_idx % len(c_cols)]:
            st.image(c_item["crop_image"], use_container_width=True)
            st.markdown(f"**{c_item['display_label']}** (Instance {c_item['instance_id']})")
            st.caption(f"Entity: `{c_item['entity_id']}` · {c_item['crop_size'][0]}×{c_item['crop_size'][1]} px")

            if c_item.get("has_existing_reference"):
                st.caption(f"✓ Existing Reference: `{Path(c_item['existing_reference']).name}`")

            if st.button(
                "💾 Save as Reference",
                key=f"btn_save_ref_{c_item['instance_id']}_{key_prefix}_{root.name}",
                type="secondary",
                help=f"Save crop as reference for {c_item['display_label']}",
            ):
                try:
                    res = save_extracted_reference(
                        root=root,
                        state=state,
                        entity_id=c_item["entity_id"],
                        crop_image=c_item["crop_image"],
                        camera_id=camera_id,
                        role=c_item.get("default_role"),
                    )
                    st.success(f"Saved reference for {c_item['display_label']} to `{res['relative_path']}`!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to save reference: {exc}")

