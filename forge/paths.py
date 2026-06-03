from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
HARNESS_CONFIG_DIR = CONFIG_DIR / "harness"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"

MS_AEDNET_ZIP = PROJECT_ROOT / "Ms-AeDNet-main.zip"
CELLSCIENTIST_ZIP = PROJECT_ROOT / "CellScientist-main.zip"
INITIAL_MODEL_PATH = WORKSPACE_DIR / "initial_model.py"


def ensure_project_dirs() -> None:
    for path in (CONFIG_DIR, HARNESS_CONFIG_DIR, PROMPTS_DIR, SKILLS_DIR, WORKSPACE_DIR, RUNS_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)
