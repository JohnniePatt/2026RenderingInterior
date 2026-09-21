# Part 7A — Human-guided spatial correspondence

Open a layout → **Spatial Validation** → choose camera and generated render.
The operator chooses the CAD entity. SAM only predicts a boundary and never chooses a semantic class.

1. Select a visible CAD entity. Its ground-truth mask is `instance.npy == instance_id`.
2. Click its counterpart in the generated image with **+ Object Point**. Local SAM runs and displays candidates.
3. Select a candidate, add **- Exclude Point**, drag **Use Box**, or retry as needed.
4. Inspect the overlay, **Mask Only**, and **CAD Ground Truth Mask**. Quality shown is SAM's prediction, not an evaluation metric.
5. Press **Confirm Mask** to accept. Missing and Occluded / Cannot Evaluate do not need a mask.
6. If SAM cannot isolate the object, choose **Manual Polygon**, click vertices, then **Build Polygon Mask** and review before confirming.
7. **+ Mark Hallucinated Object** creates a separate review with a human-selected type and `entity_id: null`.

## Local installation

Install CPU dependencies into the app's environment (or install compatible CUDA wheels for your machine):

```sh
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-segmentation.txt
python scripts/setup_sam.py
```

The setup script downloads the official ViT-B checkpoint once, verifies its SHA-256 and reuses the cache. It is never called on UI reruns. No paid generation service is used. The installed SAM source is pinned in `requirements-segmentation.txt`.

Change model, checkpoint and device under **Local SAM settings**. Configuration lives in gitignored `local_segmentation/config.json`, separately from layouts. ViT-B/L/H checkpoints can be selected; missing or unavailable models show an actionable error and Manual Polygon remains available. Model loading is cached by checkpoint path/version; predictor access is serialized so sessions cannot accidentally share another image's embeddings.

Official implementation and checkpoint documentation: [Meta Segment Anything](https://github.com/facebookresearch/segment-anything).

## Coordinate and provenance rules

- The canvas uses original image dimensions internally. Pointer positions are converted through the displayed canvas bounds; no display-space coordinates are stored.
- Generated masks retain the original render size and are binary PNGs (0/255).
- No silent resizing: render/condition size mismatch is warned and confirmation is blocked.
- For renders with an instance-image hash, changed ground truth blocks review. PNG colors, ID array and mapping must agree. Legacy renders without hashes are explicitly marked unverified.
- The first accepted review snapshots ground-truth IDs, entity mapping and metadata. Later camera exports do not overwrite this evaluation's ground truth.
- Every mask stores its method, original-coordinate prompts, SAM version/commit/checkpoint hash and predicted quality where applicable. Manual masks store polygon vertices.
- Render hashes and optimistic document revisions protect against changed inputs and concurrent overwrites.

## Storage

```text
generated/<camera>/renders/evaluation/<render_id>/
  correspondence.json
  ground_truth/instance.npy
  masks/<unique_id>.png
  hallucinations/<unique_id>.png
```

The extra render directory prevents evaluations for different renders from colliding. JSON is replaced atomically after uniquely named masks are saved. Superseded mask files remain for recovery; only masks referenced by the current JSON are active. No evaluation writes to `layout.json`.

Part 7A computes no IoU, SAS, homography error, metrology or automatic correspondence. Confirming a mask records human review; a model confidence value alone never accepts it.
