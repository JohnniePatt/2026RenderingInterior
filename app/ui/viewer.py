from pathlib import Path
from streamlit.components.v1 import declare_component

cad_viewer = declare_component("cad_viewer", path=str(Path(__file__).resolve().parents[2] / "components/cad_viewer"))
