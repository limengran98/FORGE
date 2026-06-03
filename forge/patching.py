from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import load_yaml, save_json
from .harness_spec import (
    get_heuristic_patch_rules,
    get_model_class_name,
    load_pemfc_harness_spec,
    resolve_project_path,
)
from .llm import chat_json
from .model_io import read_model_source, validate_model_source
from .paths import PROMPTS_DIR


@dataclass
class PatchCandidate:
    source: str
    rationale: str
    summary: str
    component: str
    origin: str
    edit_action: str = ""
    raw_response: dict[str, Any] | None = None


def _load_template_source(template_path: str) -> str:
    path = resolve_project_path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing heuristic model template: {path}")
    return path.read_text(encoding="utf-8").strip() + "\n"


def heuristic_patch_source(_previous_source: str, route: dict[str, Any]) -> PatchCandidate:
    component = route.get("primary_component", "factor_fusion")
    selected = None
    fallback = None
    for rule in get_heuristic_patch_rules():
        components = {str(item) for item in rule.get("components", [])}
        if "default" in components:
            fallback = rule
        if component in components:
            selected = rule
            break
    if selected is None:
        selected = fallback or get_heuristic_patch_rules()[0]
    if not selected.get("template"):
        raise ValueError(f"Heuristic patch rule {selected.get('name')} must define template")
    source = _load_template_source(str(selected["template"]))
    return PatchCandidate(
        source=source,
        rationale=str(selected.get("rationale", "")),
        summary=str(selected.get("summary", "")),
        component=component,
        origin="heuristic",
        edit_action=str(selected.get("name", "heuristic_template")),
        raw_response=None,
    )


def _load_patch_prompt() -> dict[str, str]:
    prompt_path = PROMPTS_DIR / "model_patch.yaml"
    prompt = load_yaml(prompt_path)
    return {
        "system": str(prompt.get("system", "")),
        "user_template": str(prompt.get("user_template", "")),
    }


def _format_user_prompt(
    template: str,
    iteration: int,
    feedback: dict[str, Any],
    route: dict[str, Any],
    previous_source: str,
    history: list[dict[str, Any]],
) -> str:
    payload = {
        "iteration": iteration,
        "harness_spec": load_pemfc_harness_spec(),
        "feedback": feedback,
        "routing": route,
        "history": history[-8:],
        "current_model_source": previous_source,
    }
    if not template:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return template.replace("${payload_json}", json.dumps(payload, indent=2, ensure_ascii=False))


def request_llm_patch(
    llm_config: dict[str, Any],
    iteration: int,
    feedback: dict[str, Any],
    route: dict[str, Any],
    previous_source: str,
    history: list[dict[str, Any]],
) -> PatchCandidate:
    prompt = _load_patch_prompt()
    messages = [
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": _format_user_prompt(
                prompt["user_template"], iteration, feedback, route, previous_source, history
            ),
        },
    ]
    response = chat_json(messages, llm_config)
    source = response.get("full_source") or response.get("source") or ""
    class_name = get_model_class_name()
    if not isinstance(source, str) or f"class {class_name}" not in source:
        raise ValueError(f"LLM response must include full_source defining class {class_name}")
    return PatchCandidate(
        source=source.strip() + "\n",
        rationale=str(response.get("rationale", "")),
        summary=str(response.get("summary", "")),
        component=str(response.get("component", route.get("primary_component", ""))),
        origin="llm",
        edit_action=str(response.get("edit_action", "")),
        raw_response=response,
    )


def apply_candidate(
    candidate: PatchCandidate,
    previous_model_path: str | Path,
    output_model_path: str | Path,
    validation_config: SimpleNamespace,
    feature_dim: int,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    previous_model_path = Path(previous_model_path)
    output_model_path = Path(output_model_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    previous_source = read_model_source(previous_model_path)
    validate_model_source(candidate.source, validation_config, feature_dim)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_path.write_text(candidate.source, encoding="utf-8")

    diff = "".join(
        difflib.unified_diff(
            previous_source.splitlines(keepends=True),
            candidate.source.splitlines(keepends=True),
            fromfile=str(previous_model_path),
            tofile=str(output_model_path),
        )
    )
    diff_path = artifact_dir / "patch.diff"
    diff_path.write_text(diff, encoding="utf-8")
    meta = {
        "origin": candidate.origin,
        "component": candidate.component,
        "summary": candidate.summary,
        "rationale": candidate.rationale,
        "edit_action": candidate.edit_action,
        "output_model_path": str(output_model_path),
        "diff_path": str(diff_path),
        "raw_response": candidate.raw_response,
    }
    save_json(meta, artifact_dir / "patch_meta.json")
    return meta
