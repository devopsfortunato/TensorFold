"""A checkpoint's safetensors shard files, wherever the export keeps them (beside config.json or in ``tensors/``)."""

from __future__ import annotations

import json
from pathlib import Path

SUBDIRS = ("", "tensors")


def shard_files(model_dir: str | Path, pattern: str = "model*.safetensors") -> list[Path]:
    """Sorted shard paths: every file the index's weight_map names, plus ``pattern`` matches in the model
    directory and its ``tensors/`` subdirectory (checkpoints without an index, or with extra files)."""

    root = Path(model_dir)
    found: set[Path] = set()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            names = set()
        found.update(p for p in (root / n for n in names) if p.is_file())
    for sub in SUBDIRS:
        found.update(p for p in (root / sub).glob(pattern) if p.is_file())
    return sorted(found)
