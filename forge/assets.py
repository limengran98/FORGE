from __future__ import annotations

import zipfile
from pathlib import Path

from .harness_spec import get_archive_input_prefix, get_dataset_files, get_default_dataset_name
from .paths import DATA_DIR, MS_AEDNET_ZIP


def ensure_ms_aednet_data(data_root: str | Path | None = None) -> dict[str, Path]:
    """Extract the Ms-AeDNet PEMFC CSV files if they are not already present."""
    if not MS_AEDNET_ZIP.exists():
        raise FileNotFoundError(f"Missing baseline archive: {MS_AEDNET_ZIP}")

    root = Path(data_root) if data_root is not None else DATA_DIR / "ms_aednet"
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    dataset_files = get_dataset_files()
    archive_prefix = get_archive_input_prefix().rstrip("/")
    with zipfile.ZipFile(MS_AEDNET_ZIP) as zf:
        archive_names = set(zf.namelist())
        for data_name, filename in dataset_files.items():
            target = input_dir / filename
            if not target.exists():
                member = f"{archive_prefix}/{filename}"
                if member not in archive_names:
                    raise FileNotFoundError(f"Missing {member} inside {MS_AEDNET_ZIP}")
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
            paths[data_name] = target
    return paths


def resolve_data_path(data_name: str | None = None, data_path: str | Path | None = None) -> Path:
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file does not exist: {path}")
        return path

    data_name = (data_name or get_default_dataset_name()).upper()
    dataset_files = get_dataset_files()
    if data_name not in dataset_files:
        raise ValueError(f"Unknown PEMFC dataset {data_name!r}; expected one of {sorted(dataset_files)}")
    return ensure_ms_aednet_data()[data_name]
