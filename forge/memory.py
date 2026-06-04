from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from .diagnostics import diagnostics_by_name
from .harness_spec import get_edit_operator_spec, get_edit_operators, load_pemfc_harness_spec


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def action_relation_id(diagnostic: str, component: str, edit_operator: str) -> str:
    return f"{diagnostic}->{component}::{edit_operator}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _policy_section(name: str) -> dict[str, Any]:
    section = get_edit_operator_spec().get(name, {})
    return section if isinstance(section, dict) else {}


def _operator_index() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in get_edit_operators()}


def _operators_for_component(component: str) -> list[dict[str, Any]]:
    return [item for item in get_edit_operators() if str(item.get("component")) == component]


def _relation_from_prior(diagnostic: str, component: str, operator: dict[str, Any], prior: float, strength: float) -> dict[str, Any]:
    prior = max(0.01, min(0.99, float(prior)))
    strength = max(1.0, float(strength))
    alpha = prior * strength
    beta = (1.0 - prior) * strength
    op_id = str(operator["id"])
    return {
        "id": action_relation_id(diagnostic, component, op_id),
        "diagnostic": diagnostic,
        "component": component,
        "edit_operator": op_id,
        "risk": str(operator.get("risk", "medium")),
        "alpha": alpha,
        "beta": beta,
        "trust": alpha / (alpha + beta),
        "n": 0,
        "positive_count": 0,
        "negative_count": 0,
        "validation_failures": 0,
        "evidence": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


def ensure_action_memory(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spec = get_edit_operator_spec()
    memory = state.setdefault(
        "action_memory",
        {
            "schema": "forge.action_memory.v1",
            "policy": str(spec.get("name", "edit_operator_library_v1")),
            "created_at": _now(),
            "relations": {},
            "negative_experiences": [],
        },
    )
    memory.setdefault("schema", "forge.action_memory.v1")
    memory.setdefault("policy", str(spec.get("name", "edit_operator_library_v1")))
    memory.setdefault("created_at", _now())
    memory["updated_at"] = _now()
    memory.setdefault("negative_experiences", [])
    relations = memory.setdefault("relations", {})
    strength = float(spec.get("prior_strength", 2.0))
    default_prior = float(spec.get("default_operator_prior", 0.45))
    for operator in get_edit_operators():
        component = str(operator.get("component"))
        diagnostics = operator.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            continue
        for diagnostic, prior in diagnostics.items():
            rid = action_relation_id(str(diagnostic), component, str(operator["id"]))
            relations.setdefault(
                rid,
                _relation_from_prior(
                    str(diagnostic),
                    component,
                    operator,
                    _safe_float(prior, default_prior),
                    strength,
                ),
            )
    return relations


def action_relation_trust(state: dict[str, Any], diagnostic: str, component: str, edit_operator: str) -> float:
    relations = ensure_action_memory(state)
    rid = action_relation_id(diagnostic, component, edit_operator)
    rel = relations.get(rid)
    if rel is None:
        return float(get_edit_operator_spec().get("default_operator_prior", 0.45))
    alpha = _safe_float(rel.get("alpha"), 1.0)
    beta = _safe_float(rel.get("beta"), 1.0)
    trust = alpha / max(alpha + beta, 1e-8)
    rel["trust"] = trust
    return trust


def build_pemfc_context(
    feedback: dict[str, Any],
    result: dict[str, Any] | None = None,
    harness_config: Any | None = None,
) -> dict[str, Any]:
    if is_dataclass(harness_config):
        cfg = asdict(harness_config)
    elif isinstance(harness_config, dict):
        cfg = dict(harness_config)
    elif harness_config is not None and hasattr(harness_config, "__dict__"):
        cfg = dict(vars(harness_config))
    else:
        cfg = {}
    result = result or {}
    harness_spec = load_pemfc_harness_spec()
    data_name = str(cfg.get("data_name") or result.get("data", {}).get("data_name") or "").upper()
    dataset_info = harness_spec.get("datasets", {}).get(data_name, {})
    diagnostics = diagnostics_by_name(feedback)

    def probe(name: str) -> float:
        item = diagnostics.get(name, {})
        return round(_safe_float(item.get("severity")) * _safe_float(item.get("confidence"), 1.0), 6)

    dominant = []
    for item in feedback.get("diagnostics", []):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        dominant.append(
            {
                "name": item.get("name"),
                "probe": round(_safe_float(item.get("severity")) * _safe_float(item.get("confidence"), 1.0), 6),
                "evidence": item.get("evidence", {}),
            }
        )
    dominant = sorted(dominant, key=lambda row: row["probe"], reverse=True)[:6]

    mechanism_hints: list[str] = []
    if probe("dynamic_load_error") > 0:
        mechanism_hints.append("dynamic operating-factor changes are associated with higher error")
    if probe("late_life_error") > 0:
        mechanism_hints.append("chronological test-tail behavior is worse than early test behavior")
    if probe("residual_autocorrelation") > 0:
        mechanism_hints.append("residuals retain temporal correlation after forecasting")
    if probe("residual_drift") > 0:
        mechanism_hints.append("residual mean drifts across the test segment")
    if probe("train_val_gap") > 0:
        mechanism_hints.append("train/validation gap suggests capacity or regularization pressure")

    return {
        "dataset": data_name,
        "dataset_description": dataset_info.get("description"),
        "protocol": {
            "seq_len": cfg.get("seq_len"),
            "pred_len": cfg.get("pred_len"),
            "scaling": cfg.get("scaling"),
            "target_metric": feedback.get("target_metric"),
        },
        "data": result.get("data", {}),
        "dominant_diagnostics": dominant,
        "probe_values": {
            "dynamic_load_error": probe("dynamic_load_error"),
            "late_life_error": probe("late_life_error"),
            "long_horizon_error": probe("long_horizon_error"),
            "residual_autocorrelation": probe("residual_autocorrelation"),
            "residual_drift": probe("residual_drift"),
            "train_val_gap": probe("train_val_gap"),
            "target_degradation": probe("target_degradation"),
        },
        "mechanism_hints": mechanism_hints,
    }


def _operator_prior(operator: dict[str, Any], diagnostic: str) -> float:
    diagnostics = operator.get("diagnostics", {})
    if isinstance(diagnostics, dict) and diagnostic in diagnostics:
        return _safe_float(diagnostics[diagnostic], float(get_edit_operator_spec().get("default_operator_prior", 0.45)))
    return 0.0


def _negative_memory_rows(
    state: dict[str, Any],
    diagnostics: set[str],
    components: set[str],
) -> list[dict[str, Any]]:
    relations = ensure_action_memory(state)
    rows: list[dict[str, Any]] = []
    for rel in relations.values():
        if rel.get("diagnostic") not in diagnostics or rel.get("component") not in components:
            continue
        negative_count = int(rel.get("negative_count", 0))
        validation_failures = int(rel.get("validation_failures", 0))
        if negative_count <= 0 and validation_failures <= 0:
            continue
        last = (rel.get("evidence") or [])[-1] if rel.get("evidence") else {}
        rows.append(
            {
                "relation_id": rel.get("id"),
                "diagnostic": rel.get("diagnostic"),
                "component": rel.get("component"),
                "edit_operator": rel.get("edit_operator"),
                "trust": round(action_relation_trust(state, rel["diagnostic"], rel["component"], rel["edit_operator"]), 6),
                "negative_count": negative_count,
                "validation_failures": validation_failures,
                "last_reason": (last.get("reward") or {}).get("reason"),
                "last_direction": last.get("direction"),
            }
        )
    return sorted(rows, key=lambda row: (row["negative_count"] + row["validation_failures"], -row["trust"]), reverse=True)[:8]


def select_edit_candidates(
    feedback: dict[str, Any],
    graph_state: dict[str, Any] | None,
    active_components: list[str],
    propagations: list[dict[str, Any]],
    mode: str = "trust",
) -> dict[str, Any]:
    diagnostics = diagnostics_by_name(feedback)
    selection = _policy_section("selection")
    max_candidates = int(selection.get("max_candidates", 8))
    negative_penalty = float(selection.get("negative_penalty", 0.10))
    exploration_bonus = float(selection.get("exploration_bonus", 0.02))
    state = graph_state if graph_state is not None and mode == "trust" else {}
    if mode == "trust" and graph_state is not None:
        ensure_action_memory(graph_state)

    candidate_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    active = set(active_components)
    source_props = [item for item in propagations if item.get("diagnostic") and item.get("component")]
    if not source_props:
        for diagnostic, item in diagnostics.items():
            severity = _safe_float(item.get("severity"))
            confidence = _safe_float(item.get("confidence"), 1.0)
            for component in active:
                source_props.append(
                    {
                        "diagnostic": diagnostic,
                        "component": component,
                        "severity": severity,
                        "confidence": confidence,
                        "trust": 1.0,
                        "contribution": severity * confidence,
                        "evidence": item.get("evidence", {}),
                    }
                )

    for prop in source_props:
        diagnostic = str(prop.get("diagnostic"))
        component = str(prop.get("component"))
        if active and component not in active:
            continue
        severity = _safe_float(prop.get("severity"))
        confidence = _safe_float(prop.get("confidence"), 1.0)
        component_trust = _safe_float(prop.get("trust"), 1.0)
        for operator in _operators_for_component(component):
            op_prior = _operator_prior(operator, diagnostic)
            if op_prior <= 0.0:
                continue
            op_id = str(operator["id"])
            rid = action_relation_id(diagnostic, component, op_id)
            if rid in seen:
                continue
            seen.add(rid)
            if mode == "trust" and graph_state is not None:
                op_trust = action_relation_trust(graph_state, diagnostic, component, op_id)
                rel = ensure_action_memory(graph_state).get(rid, {})
                n = int(rel.get("n", 0))
                negative_count = int(rel.get("negative_count", 0))
                validation_failures = int(rel.get("validation_failures", 0))
            else:
                op_trust = op_prior
                n = 0
                negative_count = 0
                validation_failures = 0
            penalty = negative_penalty * (negative_count + validation_failures)
            exploration = exploration_bonus / float(n + 1)
            score = severity * confidence * component_trust * op_trust + exploration - penalty
            candidate_rows.append(
                {
                    "relation_id": rid,
                    "diagnostic": diagnostic,
                    "component": component,
                    "edit_operator": op_id,
                    "score": round(max(0.0, score), 6),
                    "severity": round(severity, 6),
                    "confidence": round(confidence, 6),
                    "component_trust": round(component_trust, 6),
                    "operator_trust": round(op_trust, 6),
                    "operator_prior": round(op_prior, 6),
                    "negative_count": negative_count,
                    "validation_failures": validation_failures,
                    "risk": str(operator.get("risk", "medium")),
                    "description": str(operator.get("description", "")),
                    "prompt_guidance": str(operator.get("prompt_guidance", "")),
                    "evidence": prop.get("evidence", {}),
                }
            )

    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (row["score"], row["operator_trust"], row["component_trust"]),
        reverse=True,
    )
    selected = candidate_rows[0] if candidate_rows else None
    negative_memory: list[dict[str, Any]] = []
    if graph_state is not None:
        negative_memory = _negative_memory_rows(graph_state, set(diagnostics), active)

    return {
        "selected_edit": selected,
        "edit_candidates": candidate_rows[:max_candidates],
        "negative_memory": negative_memory,
        "memory_context": feedback.get("pemfc_context") or build_pemfc_context(feedback),
    }


def _target(result: dict[str, Any], target_metric: str) -> float | None:
    if not result or not result.get("success"):
        return None
    value = result.get("metrics", {}).get("target", {}).get(target_metric)
    return None if value is None else _safe_float(value)


def _diagnostic_probe(feedback: dict[str, Any], diagnostic: str) -> float:
    item = diagnostics_by_name(feedback).get(diagnostic)
    if not item:
        return 0.0
    return _safe_float(item.get("severity")) * _safe_float(item.get("confidence"), 1.0)


def _append_negative_experience(memory: dict[str, Any], evidence: dict[str, Any]) -> None:
    rows = memory.setdefault("negative_experiences", [])
    rows.append(
        {
            "ts": evidence.get("ts"),
            "relation_id": evidence.get("relation_id"),
            "diagnostic": evidence.get("diagnostic"),
            "component": evidence.get("component"),
            "edit_operator": evidence.get("edit_operator"),
            "reward": evidence.get("reward", {}).get("reward"),
            "reason": evidence.get("reward", {}).get("reason"),
            "previous_target": evidence.get("previous_target"),
            "next_target": evidence.get("next_target"),
        }
    )
    del rows[:-80]


def update_action_memory_from_outcome(
    state: dict[str, Any],
    patch_record: dict[str, Any],
    previous_result: dict[str, Any],
    next_result: dict[str, Any],
    previous_feedback: dict[str, Any],
    next_feedback: dict[str, Any],
    target_metric: str,
) -> list[dict[str, Any]]:
    memory = state.setdefault("action_memory", {})
    relations = ensure_action_memory(state)
    selected = patch_record.get("selected_edit") or {}
    diagnostic = str(selected.get("diagnostic") or "")
    component = str(selected.get("component") or patch_record.get("component") or patch_record.get("routed_component") or "")
    edit_operator = str(selected.get("edit_operator") or "")
    if not diagnostic or not component or not edit_operator:
        return []
    rid = action_relation_id(diagnostic, component, edit_operator)
    if rid not in relations:
        return []

    update_policy = _policy_section("update")
    target_weight = float(update_policy.get("target_weight", 0.25))
    diagnostic_weight = float(update_policy.get("diagnostic_weight", 0.65))
    overfit_penalty_weight = float(update_policy.get("overfit_penalty_weight", 0.10))
    positive_threshold = float(update_policy.get("positive_threshold", 0.002))
    negative_threshold = float(update_policy.get("negative_threshold", -0.002))
    update_scale = float(update_policy.get("update_scale", 2.0))
    max_update = float(update_policy.get("max_update", 1.0))
    validation_failure_penalty = float(update_policy.get("validation_failure_penalty", 0.35))

    rel = relations[rid]
    trust_before = action_relation_trust(state, diagnostic, component, edit_operator)
    previous_target = _target(previous_result, target_metric)
    next_target = _target(next_result, target_metric)
    target_delta = 0.0
    if previous_target is not None and next_target is not None:
        target_delta = (previous_target - next_target) / (abs(previous_target) + 1e-8)

    previous_probe = _diagnostic_probe(previous_feedback, diagnostic)
    next_probe = _diagnostic_probe(next_feedback, diagnostic)
    diagnostic_delta = previous_probe - next_probe
    previous_overfit = _safe_float(previous_feedback.get("features", {}).get("overfit_score"))
    next_overfit = _safe_float(next_feedback.get("features", {}).get("overfit_score"))
    overfit_delta = max(0.0, next_overfit - previous_overfit)
    operator_mismatch = bool(patch_record.get("edit_operator_mismatch", False))

    if patch_record.get("validation_fallback"):
        reward = -abs(validation_failure_penalty)
        reason = "patch_validation_fallback"
    elif not next_result.get("success"):
        reward = -1.0
        reason = "harness_failure"
    else:
        reward = (
            diagnostic_weight * diagnostic_delta
            + target_weight * target_delta
            - overfit_penalty_weight * overfit_delta
        )
        reason = "probe_aligned_target_delta"
        if operator_mismatch and reward > 0:
            reward *= 0.25
            reason = "probe_aligned_delta_with_operator_mismatch"
    reward = _clip(reward)

    update_amount = min(max_update, max(0.04, abs(reward) * update_scale))
    if reward > positive_threshold:
        rel["alpha"] = _safe_float(rel.get("alpha"), 1.0) + update_amount
        rel["positive_count"] = int(rel.get("positive_count", 0)) + 1
        direction = "increase"
        candidate_status = "accepted_by_probe_reward"
    elif reward < negative_threshold:
        rel["beta"] = _safe_float(rel.get("beta"), 1.0) + update_amount
        rel["negative_count"] = int(rel.get("negative_count", 0)) + 1
        direction = "decrease"
        candidate_status = "rejected_by_probe_reward"
    else:
        rel["alpha"] = _safe_float(rel.get("alpha"), 1.0) + 0.02
        rel["beta"] = _safe_float(rel.get("beta"), 1.0) + 0.02
        direction = "neutral"
        candidate_status = "ambiguous_probe_reward"
    if patch_record.get("validation_fallback"):
        rel["validation_failures"] = int(rel.get("validation_failures", 0)) + 1

    rel["n"] = int(rel.get("n", 0)) + 1
    rel["trust"] = action_relation_trust(state, diagnostic, component, edit_operator)
    rel["updated_at"] = _now()
    evidence = {
        "ts": _now(),
        "relation_id": rid,
        "diagnostic": diagnostic,
        "component": component,
        "edit_operator": edit_operator,
        "trust_before": round(trust_before, 6),
        "trust_after": round(rel["trust"], 6),
        "direction": direction,
        "candidate_status": candidate_status,
        "update_amount": round(update_amount, 6),
        "reward": {
            "reward": round(reward, 6),
            "reason": reason,
            "target_delta": target_delta,
            "diagnostic_delta": diagnostic_delta,
            "previous_probe": previous_probe,
            "next_probe": next_probe,
            "overfit_delta": overfit_delta,
        },
        "previous_target": previous_target,
        "next_target": next_target,
        "previous_result_path": previous_result.get("paths", {}).get("result"),
        "next_result_path": next_result.get("paths", {}).get("result"),
        "operator_mismatch": operator_mismatch,
    }
    rel.setdefault("evidence", []).append(evidence)
    if direction == "decrease":
        _append_negative_experience(memory, evidence)
    memory["updated_at"] = _now()
    return [evidence]
