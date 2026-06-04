from __future__ import annotations

import difflib
import hashlib
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
    selected_edit = route.get("selected_edit") or {}
    selected_template = str(selected_edit.get("template") or "").strip()
    if selected_template:
        source = _load_template_source(selected_template)
        return PatchCandidate(
            source=source,
            rationale=str(selected_edit.get("prompt_guidance") or "Deterministic template selected by relation-level routing."),
            summary=str(selected_edit.get("description") or "Applies selected relation-level structural template."),
            component=str(selected_edit.get("component") or route.get("primary_component", "temporal_memory")),
            origin="heuristic",
            edit_action=str(selected_edit.get("edit_operator") or "selected_template"),
            raw_response={"selected_relation_id": selected_edit.get("relation_id"), "template": selected_template},
        )

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
    return _load_prompt_file("model_patch.yaml")


def _load_repair_prompt() -> dict[str, str]:
    return _load_prompt_file("model_repair.yaml")


def _load_prompt_file(name: str) -> dict[str, str]:
    prompt_path = PROMPTS_DIR / name
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


def _candidate_hash(source: str) -> str:
    return hashlib.md5(source.strip().encode("utf-8")).hexdigest()


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


def request_llm_repair_patch(
    llm_config: dict[str, Any],
    iteration: int,
    feedback: dict[str, Any],
    route: dict[str, Any],
    parent_source: str,
    broken_candidate: PatchCandidate,
    validation_error: str,
    history: list[dict[str, Any]],
    failed_attempts: list[dict[str, Any]],
    validation_config: SimpleNamespace,
    feature_dim: int,
) -> PatchCandidate:
    prompt = _load_repair_prompt()
    payload = {
        "iteration": iteration,
        "harness_spec": load_pemfc_harness_spec(),
        "validation_contract": {
            "input_shape": ["batch", int(validation_config.seq_len), int(feature_dim)],
            "expected_output_shape": ["batch", int(validation_config.pred_len), int(validation_config.enc_in)],
            "seq_len": int(validation_config.seq_len),
            "pred_len": int(validation_config.pred_len),
            "enc_in": int(validation_config.enc_in),
            "feature_dim": int(feature_dim),
        },
        "feedback": feedback,
        "routing": route,
        "history": history[-8:],
        "parent_working_model_source": parent_source,
        "broken_candidate": {
            "component": broken_candidate.component,
            "origin": broken_candidate.origin,
            "edit_action": broken_candidate.edit_action,
            "summary": broken_candidate.summary,
            "rationale": broken_candidate.rationale,
            "source": broken_candidate.source,
        },
        "validation_error": validation_error,
        "failed_attempts": failed_attempts[-3:],
    }
    user_text = prompt["user_template"].replace("${payload_json}", json.dumps(payload, indent=2, ensure_ascii=False))
    response = chat_json(
        [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_text},
        ],
        llm_config,
    )
    source = response.get("full_source") or response.get("source") or ""
    class_name = get_model_class_name()
    if not isinstance(source, str) or f"class {class_name}" not in source:
        raise ValueError(f"LLM repair response must include full_source defining class {class_name}")
    return PatchCandidate(
        source=source.strip() + "\n",
        rationale=str(response.get("rationale", "")),
        summary=str(response.get("summary", "")),
        component=str(response.get("component", broken_candidate.component or route.get("primary_component", ""))),
        origin="llm_repair",
        edit_action=str(response.get("edit_action", "repair_validation_error")),
        raw_response=response,
    )


def safety_fallback_candidate(parent_source: str, route: dict[str, Any], reason: str) -> PatchCandidate:
    return PatchCandidate(
        source=parent_source.strip() + "\n",
        rationale=f"Safety fallback after unresolved patch validation error: {reason}",
        summary="Reuses the parent working model to keep the fixed harness iteration alive.",
        component=str(route.get("primary_component", "interface")),
        origin="safety_fallback",
        edit_action="reuse_parent_after_failed_repair",
        raw_response={"reason": reason},
    )


def save_failed_candidate_attempt(
    candidate: PatchCandidate,
    artifact_dir: str | Path,
    attempt_index: int,
    validation_error: str,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_path = artifact_dir / f"failed_candidate_{attempt_index:02d}.py"
    source_path.write_text(candidate.source, encoding="utf-8")
    row = {
        "attempt": attempt_index,
        "origin": candidate.origin,
        "component": candidate.component,
        "edit_action": candidate.edit_action,
        "summary": candidate.summary,
        "source_path": str(source_path),
        "source_hash": _candidate_hash(candidate.source),
        "validation_error": validation_error,
    }
    return row


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
