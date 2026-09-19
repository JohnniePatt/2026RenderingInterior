from collections import Counter
from pathlib import Path
from app.layout.persistence import read


def discover(workspace):
    layouts, errors = [], []
    for root in sorted(Path(workspace).glob("Layout_*"), reverse=True):
        if not root.is_dir():
            continue
        try:
            state = read(root)
            layouts.append(state)
        except ValueError as exc:
            errors.append(str(exc))
    counts = Counter(item["layout"]["name"] for item in layouts)
    labels = {
        item["layout"]["id"]: item["layout"]["name"] + (
            " — " + item["layout"]["id"].removeprefix("Layout_") if counts[item["layout"]["name"]] > 1 else ""
        ) for item in layouts
    }
    return layouts, labels, errors
