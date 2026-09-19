# CAD-to-Rendering Research Tool · V0.3

A local Python + Streamlit application for **From Plan to Perspective: A Hybrid Projective-Geometry Pipeline for Spatially-Accurate AI-Generated Interior Renderings**.

The existing V0.1 Layouts, DXF import, selection and Wall / Floor / Furniture / Void annotations are extended with **V0.2 persistent cameras** and **V0.3 CAD-derived metric 3D proxy + live perspective**. Generative AI, conditioning-map export and evaluation remain future work.

## Run

Python 3.11 or newer is required. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For the existing WSL environment in this workspace:

```bash
cd /home/johnfaqpc/programming/2026RenderingInterior/2026RenderingInterior
../2026RenderingInterior-env/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). `requirements-lock.txt` captures the complete tested Python environment, including test dependencies; use `pip install -r requirements-lock.txt` for that exact environment. Three.js 0.170.0 is bundled with its MIT license; no Node build or runtime CDN is needed.

## Workflow

1. Choose **+ Add Layout**, enter a display name, upload a `.dxf`, and select **Create Layout**.
2. Click an entity's edge. Hold Shift (or Ctrl / Command) and click to add or remove selections. Drag to pan, scroll to zoom around the pointer, and use **Fit to view** to reset.
3. Choose Wall, Floor, Furniture or Void. Enter dimensions in the displayed drawing units, then **Apply & autosave**. This validates and writes immediately; there is no subsequent save step. Property edits are drafts until applied. Draft widgets remain mounted across editing modes, so switching to Camera Mode and back retains them within the session.
4. Furniture supports a custom category, description and local PNG / JPEG / WebP reference image.
5. Use **Layout settings** to rename the Layout or declare unknown source units. This changes the interpretation of CAD coordinates, not their numeric scale. Existing dimensions are not converted.
6. Stop and restart Streamlit. Select the Layout on Home and choose **Open / Edit Layout**. Tags and dimensions are restored from disk.

For multi-selection, Apply replaces each selected entity's semantic property set with the visible values. Clear annotation removes semantic fields but preserves CAD metadata. The entity inventory also supports keyboard selection.

**Void** represents a door, window or other opening. Choose **Opening type → Door / Window / Opening**, enter the opening height and base elevation (bottom edge / window sill in CAD Z coordinates), and apply. For example, a window with `height: 1200` and `base_elevation: 900` spans Z=900–2100 in drawing units. Void appears blue in the plan and persists as `semantic: "void"` with `category`, `height` and `base_elevation`. V0.3 retains these annotations but does **not** cut openings out of 3D walls; the proxy diagnostics explicitly report this limitation. Existing Layouts remain readable without migration.

A synthetic example is included at `examples/living_room.dxf`. Rebuild it with `python examples/create_sample.py`. It is not automatically imported into your workspace.

## Camera Mode and live perspective

1. Open an existing annotated Layout and choose **Camera Mode**. The workspace becomes a 50/50 plan / active-camera perspective split.
2. Click **+ Add Camera**, press on a plan location, drag toward the intended view, then release. Preview updates while dragging. Default height is **1500 mm / 1.5 m**, pitch 0°, horizontal FOV 60°.
3. Drag the filled camera dot to move it, or the outlined end handle to aim it. Click an inactive marker to activate it. Drag empty plan space to pan, scroll to zoom, and choose **Fit plan** to include CAD and cameras. Escape cancels an unfinished placement/drag.
4. Edit name, camera height, heading, pitch or **horizontal** FOV below the split. Valid input updates the preview immediately; leaving the input commits it. The component shows Editing / Saving / Saved state. No Generate Preview action is required.
5. Add another camera and switch via **Active Camera**. Delete removes only the selected camera and selects the next available camera. All final edits use atomic JSON autosave.
6. Restart Streamlit and reopen the Layout. Enter Camera Mode to reconstruct the saved active view, using the same geometry and exact persisted camera parameters.

Camera height is absolute world Z in the drawing units, not an automatically measured offset from a floor. Unknown units must be declared before camera editing. The right viewport is always the research camera; there is no orbit controller that could silently change it.

During a drag, camera transforms and both views update entirely in JavaScript. Python receives the final snapshot on release. A one-in-flight bridge coalesces subsequent edits until acknowledgment, preventing older reruns from overwriting newer local camera changes. Revision checks reject stale camera saves from another session. If a save fails, the component reports **Not saved** and asks you to reopen the Layout.

## Metric proxy rules

- **Closed Wall footprints:** extrude the actual polygon from `base_elevation` to `base_elevation + height`.
- **Open Wall segments / polylines:** zero-thickness, double-sided vertical surfaces by default. Optional **Line wall proxy thickness** in proxy settings produces a centered, flat-capped, mitred buffer. It is explicitly recorded in `proxy_settings.line_wall_thickness` as a proxy assumption, never presented as CAD-derived thickness.
- **Floors:** use a closed footprint with its top at `elevation` and bottom at `elevation - thickness`. A zero thickness makes a planar surface.
- **Furniture:** extrude its actual closed footprint from Z=0 to the tagged height. This is a CAD-derived metric 3D proxy, not detailed or fully accurate furniture morphology.
- Closed nested contours use even/odd containment to represent holes and islands, independent of ring winding. Overlapping regions belonging to the same entity are unioned. Complex furniture block drafting contours can be ambiguous: review the resulting footprint. Extra open furniture strokes are omitted with a diagnostic; missing closed footprints are not replaced by boxes.
- Invalid/self-intersecting rings are skipped with a diagnostic rather than silently repaired. Curves use the existing parser's display tessellation. Orphaned and untagged entities do not create proxy meshes. Existing CAD Z is projected to XY; tagged elevation determines the extrusion Z.
- Void remains annotation-only in this stage. No inferred wall cuts, ceilings, decorative objects, or hidden architectural dimensions are added.

Both camera viewports share one Three.js scene: layer 0 contains CAD lines for the orthographic plan and layer 1 contains the metric meshes for the perspective camera. The same proxy descriptors and camera projection modules can be used by future off-screen renderers. Rebuilding occurs only when geometry/annotations change; a drag does not regenerate proxy geometry.

## Camera conventions and research matrices

World axes stay **DXF X/Y/Z, with Z up**. Heading is counterclockwise from +X toward +Y. Positive pitch looks upward. No visual scale factor changes the metric geometry; a shared XY origin subtraction only improves GPU precision for drawings far from the origin.

`fov_deg` means horizontal FOV. For actual preview aspect `a = width / height`, Three.js receives vertical FOV `2 atan(tan(horizontal_fov / 2) / a)`. Resizing changes aspect and projection without changing stored camera state. Near/far clipping is derived from proxy bounds, camera distance and drawing units.

`app.geometry.camera.camera_matrices(camera, width, height)` returns NumPy `K`, `R`, `t`, and `P = K [R | t]`. Extrinsics map world coordinates into OpenCV camera axes **right / down / forward**; pixel coordinates are measured from the top-left. Focal lengths are equal, `fx = fy = width / (2 tan(horizontal_fov / 2))`, with principal point `(width/2, height/2)`. `project_points` returns pixel coordinates and signed camera depth; points behind the camera have NaN pixel coordinates.

The renderer's projection is tested against this independent Python calculation. Future planar-floor homography and non-planar camera projection remain complementary parts of the hybrid research pipeline; depth/instance/semantic output and SAS/metrology are not implemented here.

## Persistent data

```text
DXF upload → validation → canonical copy → ezdxf → viewport
                                             ↓
                                handle ↔ application entity ID
                                             ↓
                                annotation → atomic JSON
```

Each Layout lives in `workspace/Layout_YYYYMMDD_HHMMSS/`. ID and directory remain unchanged when its display name changes. Same-second ID collisions use the next unused second; actual creation time is stored separately. Duplicate names in Home are disambiguated with their IDs.

```text
Layout_YYYYMMDD_HHMMSS/
├── layout.json
├── source/<sanitized-original-name>.dxf
├── references/{furniture,materials}/
├── generated/{proxy,depth,instance,semantic,renders}/
├── evaluation/sas/
└── cache/
```

`layout.json` is authoritative. Session state only carries navigation, selection and temporary UI values. JSON paths are Layout-relative. Back up or move the **whole Layout directory**, including the source and references. `CAD_WORKSPACE` can override the workspace root; the default is resolved relative to `app.py`, independent of the launch directory.

The storage schema remains the backward-compatible `schema_version: "0.1"`; V0.3 is the application version. Its fields include `layout`, `source`, `entities`, `cameras`, `active_camera`, `outputs`, `evaluation`, and optional `proxy_settings`. Entity records retain DXF handle, type, layer, INSERT transform metadata, semantic properties and `orphaned`. Geometry is re-read from canonical DXF instead of duplicated into annotation JSON. Source SHA-256 and original DXF units preserve provenance. Future output lists remain empty.

Camera state uses the existing JSON fields, for example:

```json
{
  "cameras": {
    "camera_001": {
      "name": "Main view",
      "position": [3000, 800, 1500],
      "heading_deg": 90,
      "pitch_deg": -8,
      "fov_deg": 75
    }
  },
  "active_camera": "camera_001",
  "proxy_settings": {"line_wall_thickness": 0}
}
```

Saving validates a candidate, writes and fsyncs a temporary JSON file beside `layout.json`, then atomically replaces it. An exclusive lock and revision check reject conflicting writes. On a crash during save, old or new complete JSON remains; if `.save.lock` remains, stop all app instances and remove only that stale lock before reopening. A corrupt JSON is reported, not overwritten or silently reset.

## CAD scope and limitations

- Modelspace LINE, LWPOLYLINE, POLYLINE, INSERT, ARC and CIRCLE are supported. Block children are decomposed with transforms and selected/tagged as their owning INSERT. Curves and polyline bulges are tessellated for display with roughly 1 mm flattening tolerance in known units. Exact original curves remain in the DXF.
- DXF handles determine identity, never list order or layer naming. Missing entities remain as orphaned annotations in the inventory. A changed source hash produces a review warning. Reusing a handle for a different object cannot be reliably detected; treat canonical DXFs as immutable.
- Unsupported types (including text, hatch, dimensions and unsupported block children) are reported and omitted from the viewport. Modelspace Z is projected to XY with a warning. This is not a full CAD editor or paperspace renderer.
- mm, cm, m, inches and feet are supported. Unitless and other units require a manual declaration before tagging. Default dimensions are converted into declared units; CAD coordinates themselves remain unchanged.
- Missing reference images retain their JSON paths and display a warning when selected. Replaced or removed reference files remain in the Layout for traceability.
- Invalid DXFs and DXFs needing audit repairs are rejected rather than silently repaired. Repair and re-export using a CAD application before import. DWG is unsupported.
- WebGL is needed for the viewport. If unavailable, an error is shown and the entity inventory remains usable. Picking is edge-based; clicking empty polygon interiors does not select them. Large drawings are not yet performance-benchmarked.
- Status changes from `new` to `annotating` after successful parsing. Viewing a proxy does not automatically declare completeness or mark a Layout `ready` or `generated`.

## Code map

```text
app.py                      Streamlit entry point
app/layout/                 schema, import, discovery, atomic storage
app/cad/                    parser, units, handle reconciliation
app/annotation/             semantic validation and reference copying
app/geometry/               proxy descriptors, cameras, K/R/t/P
app/ui/                     home, creation, editor and properties
components/cad_viewer/       annotation viewer, shared camera scene, proxy and projection
examples/                   synthetic DXF and generator
tests/                      persistence, parser, errors, browser acceptance
workspace/                  local research data (gitignored)
```

Shapely validates polygon topology and buffers explicitly configured line-wall thickness. Three.js triangulates/extrudes these descriptors for rendering. Future homography and evaluation modules can consume source geometry, annotations and camera matrices without changing ownership of the source. No placeholder AI or evaluation implementations are included.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m playwright install chromium
# Linux only, if browser system libraries are missing:
python -m playwright install-deps chromium
RUN_BROWSER_TESTS=1 python -m pytest -q
```

PowerShell equivalent for the last command: `$env:RUN_BROWSER_TESTS='1'; python -m pytest -q`.

The opt-in browser test starts real Streamlit with an isolated workspace, uploads a DXF through Chromium, clicks geometry, checks multi-selection and pan/zoom/fit, saves Wall 2800 / Floor 150 / Sofa 820 with a reference image, completely stops and restarts the server, and checks restoration on disk and in the property panel. Test data never populates the user's workspace. Default tests exercise atomic failure, stale writes, invalid imports, duplicate names, missing files, references, unit validation, transformed blocks, curved polylines and orphan preservation.

Additional V0.2/V0.3 tests cover multiple cameras, updates/deletion/active-state restoration, mm/m/cm/in/ft heights, projection orientation, concave footprints, holes, metric proxy bounds and explicit line-wall thickness. Real Chromium tests verify a changing perspective **before pointer release while JSON stays unchanged**, live property edits, viewport resize, exact camera restoration and identical preview pixels after server restart, JS/Python projection agreement, actual mesh bounds, and annotation draft retention across editing modes.

Implementation references: [Streamlit component protocol](https://docs.streamlit.io/develop/concepts/custom-components/components-v1/intro), [ezdxf path conversion](https://ezdxf.readthedocs.io/en/stable/path.html), and [ezdxf block decomposition](https://ezdxf.readthedocs.io/en/stable/disassemble.html).

Projection/geometry references: [Three.js PerspectiveCamera](https://threejs.org/docs/pages/PerspectiveCamera.html), [ExtrudeGeometry](https://threejs.org/docs/pages/ExtrudeGeometry.html), and [Shapely manual](https://shapely.readthedocs.io/en/latest/manual.html).
