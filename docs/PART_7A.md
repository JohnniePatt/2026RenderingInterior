# Part 7A — CAD-guided automatic correspondence with SAM 3

CAD defines the concept to search; SAM 3 supplies observations from the generated image. This replaces the human-first SAM ViT-B page. No SAS, final IoU, metric errors, training, fine-tuning or generation API calls are performed.

## Workflow

1. Check **Validation Concept** in entity Properties and Apply. Explicit values survive other property edits. Legacy values resolve category → semantic → description; conflicting metadata requires an explicit concept.
2. Open **Spatial Validation**, choose camera and render. Only visible nonzero instance IDs are evaluated. GT is `instance.npy == instance_id`.
3. **Run Automatic Correspondence** queries each unique concept once. A cached local SAM 3 model supplies native-resolution candidates.
4. One strong candidate is `auto_matched`, with `human_verified=false`. Zero is `missing_candidate`, not proven absence. Multiple candidates are ambiguous; repeated CAD concepts use one-to-one centroid assignment with confidence/separation checks. Displacement alone does not reject semantic correspondence.
5. Review defaults to cases needing attention. Uncheck the filter to inspect automatic matches. Compare GT and generated masks, choose another candidate, Confirm or Reject.
6. **Manual SAM Prompt** supports positive/negative points and boxes through SAM 3 tracker. **Manual Polygon** works without weights. Other decisions include Cannot Evaluate, Confirm object is missing, and Mark segmentation failure.
7. **Hallucination Review** is separate and human initiated, with type and entity_id=null. CAD queries cannot discover every invented object.

## Local setup

Uses Transformers Sam3Model/Sam3Processor for concepts and Sam3TrackerModel/Sam3TrackerProcessor for points/boxes. SAM 1 is never substituted under a SAM 3 label.

```sh
# Use the app Python environment; torch and torchvision are required.
python -m pip install -r requirements-sam3.txt
hf auth login
python scripts/setup_sam3.py
```

First obtain access to [facebook/sam3](https://huggingface.co/facebook/sam3). Accept its terms yourself and authenticate locally; never paste tokens into chat or research prompts. The setup script downloads an explicit revision to ~/.cache/cad-rendering/sam3 and records source_revision.json. UI Run/Retry only read local files. Select an existing compatible Transformers checkpoint folder in **SAM 3 settings**.

Configuration is in gitignored local_segmentation/sam3.json, separately from layouts. Defaults: CPU, detection threshold 0.3, strong-match threshold 0.8, normalized-centroid separation 0.05. CUDA requires CUDA-enabled PyTorch. Proposal thresholds are not validated research metrics.

**Status 2026-09-22:** Transformers 5.17.0 installed; real class/signature imports checked. Existing PyTorch is CPU-only. No SAM 3 checkpoint or local Hugging Face login was present. Real pretrained inference and accuracy remain unverified until access is supplied. Automated adapter tests use identified test doubles.

Official APIs: [SAM 3](https://huggingface.co/docs/transformers/model_doc/sam3), [SAM 3 tracker](https://huggingface.co/docs/transformers/model_doc/sam3_tracker).

## Saved data

```text
generated/<camera>/renders/evaluation/<render_id>/sam3/
  correspondence.json
  ground_truth/instance.npy
  masks/<candidate-mask-sha256>.png
```

Old human-guided evaluations remain outside sam3/ and are not overwritten. Annotation Apply saves Validation Concept to layout.json; evaluation does not modify layout.json.

- Binary 0/255 PNGs retain native render resolution; canvas coordinates are original pixels.
- Size/aspect mismatches warn and block inference/saving. No silent resizing.
- Render/GT hashes, mapping consistency, immutable GT snapshots, atomic writes, revision checks and locks protect data.
- JSON records concepts/source, candidate counts/scores/descriptors, selected candidate, method, human state, model/package/checkpoint hashes, thresholds, resolutions and timestamps. Debug exposes queries and assignments.
- Reruns preserve human decisions for unchanged concepts. Changed concepts require new inference or a manual mask before confirmation. A candidate cannot be accepted for multiple CAD entities or a separate hallucination.
- Methods: sam3_text, sam3_text_spatial_match, sam3_point, sam3_box, manual_polygon.
- States: unreviewed, auto_matched, ambiguous, missing_candidate, human_verified, rejected, cannot_evaluate, confirmed_missing, segmentation_failure.

## Verification

Run `python -m pytest -q`. Tests cover native coordinates, states, duplicate-query suppression, one-to-one assignment, displaced masks, preserved human decisions, stale concepts, resolution rejection, persistence and Streamlit run/review. Adapter tests check local-only loading, cache isolation and prompt tensor structure with test doubles. Real SAM 3 acceptance remains a separate step: use an existing render after checkpoint setup, without generating another image.
