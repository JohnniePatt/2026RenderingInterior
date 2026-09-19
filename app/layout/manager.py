import hashlib
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import ezdxf
from app.layout.persistence import read, resolve, save
from app.layout.schema import new_state

DIRECTORIES = ("source", "references/furniture", "references/materials", "generated/proxy", "generated/depth",
               "generated/instance", "generated/semantic", "generated/renders", "evaluation/sas", "cache")


def safe_filename(name):
    name = str(name).replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w.\-]", "_", name, flags=re.UNICODE).strip(". ")
    return name[:180] or "upload"


def create(workspace, name, filename, content):
    if not name.strip():
        raise ValueError("Enter a Layout name")
    filename = safe_filename(filename)
    if Path(filename).suffix.lower() != ".dxf":
        raise ValueError("Only .dxf files are supported")
    if not content:
        raise ValueError("The DXF is empty")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    # Validate before publishing a discoverable Layout. This also supports binary DXF.
    with tempfile.TemporaryDirectory(dir=workspace, prefix=".import-") as staging:
        staged = Path(staging) / filename
        staged.write_bytes(content)
        try:
            doc = ezdxf.readfile(staged)
            audit = doc.audit()
            if audit.has_errors or audit.has_fixes:
                raise ValueError("DXF needs repair; export or repair it in your CAD application first")
        except (ezdxf.DXFError, OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"Invalid or corrupted DXF: {exc}") from exc
        moment = datetime.now().astimezone()
        while True:
            root = workspace / moment.strftime("Layout_%Y%m%d_%H%M%S")
            try:
                root.mkdir()
                break
            except FileExistsError:
                moment += timedelta(seconds=1)
        try:
            for directory in DIRECTORIES:
                (root / directory).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged, root / "source" / filename)
            state = new_state(root.name, name, f"source/{filename}")
            state["source"]["sha256"] = hashlib.sha256(content).hexdigest()
            save(root, state)
        except Exception:
            # Only this newly allocated import directory is rolled back.
            shutil.rmtree(root)
            raise
    return root, state


def open_layout(root):
    state = read(root)
    source = resolve(root, state["source"]["cad_file"])
    if not source.is_file():
        raise ValueError(f"Source DXF is missing: {state['source']['cad_file']}")
    return state, source
