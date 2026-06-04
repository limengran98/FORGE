from __future__ import annotations

import math
import hashlib
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


def _risk_value(risk: str, rel: dict[str, Any]) -> float:
    base = {"low": 0.15, "medium": 0.35, "high": 0.70}.get(str(risk or "medium"), 0.35)
    validation = min(0.30, 0.10 * int(rel.get("validation_failures", 0)))
    off_policy = min(0.20, 0.05 * int(rel.get("off_policy_count", 0)))
    return _clip01(base + validation + off_policy)


def _expected_correction(rel: dict[str, Any]) -> float:
    values: list[float] = []
    for item in (rel.get("evidence") or [])[-5:]:
        if item.get("evidence_policy") == "off_policy":
            continue
        reward = item.get("reward") or {}
        values.append(_safe_float(reward.get("reward")))
    if not values:
        return 0.0
    return _clip(sum(values) / len(values))


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    tau = max(1e-6, float(temperature))
    scaled = [value / tau for value in values]
    m = max(scaled)
    exps = [math.exp(max(-60.0, min(60.0, value - m))) for value in scaled]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def _normalized_entropy(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 0.0
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in weights)
    return _clip01(entropy / math.log(len(weights)))


def _stable_unit_interval(payload: str) -> float:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _recent_route_relation_ids(state: dict[str, Any], window: int) -> list[str]:
    rows: list[tuple[int, str]] = []
    for key, record in (state.get("iterations") or {}).items():
        try:
            iteration = int(str(key).split("_")[-1])
        except Exception:
            iteration = int(record.get("iteration", 0))
        selected = (record.get("route") or {}).get("selected_edit") or {}
        rid = selected.get("relation_id")
        if rid:
            rows.append((iteration, str(rid)))
    return [rid for _, rid in sorted(rows)[-max(0, int(window)) :]]


def _negative_streak(memory: dict[str, Any], dataset: str, window: int) -> int:
    rows = memory.get("negative_experiences") or []
    if dataset:
        rows = [row for row in rows if str(row.get("dataset") or "").upper() == dataset]
    streak = 0
    for _row in reversed(rows[-max(1, int(window)) :]):
        streak += 1
    return streak


def _recent_execution_risk(memory: dict[str, Any], window: int = 3) -> float:
    risk = 0.0
    evidence_rows: list[dict[str, Any]] = []
    for rel in (memory.get("relations") or {}).values():
        evidence_rows.extend(rel.get("evidence") or [])
    evidence_rows = sorted(evidence_rows, key=lambda row: str(row.get("ts") or ""))[-window:]
    for row in evidence_rows:
        if row.get("evidence_policy") == "off_policy":
            risk += 0.35
        if row.get("candidate_status") == "off_policy_mismatch_not_rewarded":
            risk += 0.35
        if (row.get("reward") or {}).get("reason") in {"patch_validation_fallback", "harness_failure"}:
            risk += 0.45
    return _clip01(risk)


def _attention_config(selection: dict[str, Any]) -> dict[str, Any]:
    config = selection.get("relation_attention", {})
    return config if isinstance(config, dict) else {}


def _candidate_attention_score(row: dict[str, Any], weights: dict[str, Any]) -> float:
    probe_match = _safe_float(row.get("severity")) * _safe_float(row.get("confidence"), 1.0)
    expected = _safe_float(row.get("expected_correction"))
    uncertainty = _safe_float(row.get("uncertainty"))
    suppression = row.get("suppression") or {}
    negative = _safe_float(suppression.get("total_penalty"))
    risk = _safe_float(row.get("risk_score"))
    score = (
        float(weights.get("trust", 0.45)) * _safe_float(row.get("operator_trust"))
        + float(weights.get("component_trust", 0.15)) * _safe_float(row.get("component_trust"))
        + float(weights.get("probe_match", 0.25)) * probe_match
        + float(weights.get("expected_correction", 0.20)) * expected
        + float(weights.get("uncertainty", 0.08)) * uncertainty
        - float(weights.get("negative_memory", 0.35)) * negative
        - float(weights.get("risk", 0.20)) * risk
    )
    return score


def _adaptive_temperature(
    feedback: dict[str, Any],
    state: dict[str, Any] | None,
    candidate_rows: list[dict[str, Any]],
    config: dict[str, Any],
    dataset: str,
) -> dict[str, Any]:
    base = float(config.get("base_temperature", 0.30))
    min_tau = float(config.get("min_temperature", 0.10))
    max_tau = float(config.get("max_temperature", 0.60))
    adapt = config.get("adapt", {})
    if not isinstance(adapt, dict):
        adapt = {}
    tau = base
    factors: dict[str, float | int | bool] = {}

    current = _safe_float(feedback.get("current_target"), None)
    best = _safe_float(feedback.get("best_target"), None)
    stagnation_threshold = float(adapt.get("stagnation_threshold", 0.002))
    stagnant = False
    if current is not None and best is not None:
        stagnant = current >= best * (1.0 - stagnation_threshold)
    elif _safe_float(feedback.get("features", {}).get("improved_vs_best")) <= 0.0 and feedback.get("previous_target") is not None:
        stagnant = True
    if stagnant:
        bonus = float(adapt.get("stagnation_bonus", 0.08))
        tau += bonus
        factors["stagnation_bonus"] = bonus

    state = state or {}
    memory = state.get("action_memory") or {}
    repeated_window = int(adapt.get("repeated_relation_window", 3))
    recent_relations = _recent_route_relation_ids(state, repeated_window)
    repeated = bool(recent_relations and len(set(recent_relations)) == 1 and len(recent_relations) >= repeated_window)
    if repeated:
        bonus = float(adapt.get("repeated_relation_bonus", 0.07))
        tau += bonus
        factors["repeated_relation_bonus"] = bonus

    raw_scores = [float(row.get("score", 0.0)) for row in candidate_rows if not row.get("blocked")]
    raw_weights = _softmax(raw_scores, base) if raw_scores else []
    entropy = _normalized_entropy(raw_weights)
    factors["candidate_entropy"] = round(entropy, 6)
    if raw_scores and entropy < float(adapt.get("entropy_low_threshold", 0.45)):
        bonus = float(adapt.get("entropy_collapse_bonus", 0.05))
        tau += bonus
        factors["entropy_collapse_bonus"] = bonus

    risk = _recent_execution_risk(memory)
    if risk > 0:
        penalty = float(adapt.get("risk_penalty", 0.10)) * risk
        tau -= penalty
        factors["risk_penalty"] = round(penalty, 6)

    streak = _negative_streak(memory, dataset, int(adapt.get("negative_streak_window", 2)))
    factors["negative_streak"] = streak
    if streak >= int(adapt.get("negative_streak_window", 2)):
        penalty = float(adapt.get("negative_streak_penalty", 0.06))
        tau -= penalty
        factors["negative_streak_penalty"] = penalty

    clipped = max(min_tau, min(max_tau, tau))
    return {
        "enabled": bool(config.get("enabled", True)),
        "base_temperature": base,
        "temperature": round(clipped, 6),
        "min_temperature": min_tau,
        "max_temperature": max_tau,
        "factors": factors,
    }


def _apply_relation_attention(
    feedback: dict[str, Any],
    state: dict[str, Any] | None,
    candidate_rows: list[dict[str, Any]],
    selection: dict[str, Any],
    dataset: str,
) -> dict[str, Any]:
    config = _attention_config(selection)
    if not candidate_rows or not bool(config.get("enabled", True)):
        return {
            "enabled": False,
            "temperature": None,
            "selected_by": "score_argmax",
            "weights_entropy": None,
        }
    weights = config.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}
    for row in candidate_rows:
        score = _candidate_attention_score(row, weights)
        row["attention_score"] = round(score, 6)
    meta = _adaptive_temperature(feedback, state, candidate_rows, config, dataset)
    selectable = [row for row in candidate_rows if not row.get("blocked")]
    attention_weights = _softmax([float(row.get("attention_score", 0.0)) for row in selectable], meta["temperature"])
    for row, weight in zip(selectable, attention_weights):
        row["attention_weight"] = round(weight, 8)
        row["relation_temperature"] = meta["temperature"]
    for row in candidate_rows:
        row.setdefault("attention_weight", 0.0)
        row.setdefault("relation_temperature", meta["temperature"])
    meta["weights_entropy"] = round(_normalized_entropy(attention_weights), 6)
    meta["sample_top_k"] = int(config.get("sample_top_k", 4))
    meta["sampling"] = str(config.get("sampling", "deterministic"))
    meta["selected_by"] = "trust_calibrated_attention"
    return meta


def _select_attention_candidate(
    selectable: list[dict[str, Any]],
    attention_meta: dict[str, Any],
    dataset: str,
    update_count: int,
) -> tuple[dict[str, Any] | None, bool]:
    if not selectable:
        return None, False
    if not attention_meta.get("enabled"):
        return selectable[0], False
    sample_top_k = max(1, int(attention_meta.get("sample_top_k", 4)))
    pool = sorted(selectable, key=lambda row: row.get("attention_weight", 0.0), reverse=True)[:sample_top_k]
    if float(attention_meta.get("temperature") or 0.0) <= float(attention_meta.get("base_temperature") or 0.0) + 1e-8:
        return pool[0], False
    if str(attention_meta.get("sampling", "deterministic")) != "deterministic" or len(pool) == 1:
        return pool[0], False
    weights = [float(row.get("attention_weight", 0.0)) for row in pool]
    total = sum(weights)
    if total <= 0:
        return pool[0], False
    weights = [value / total for value in weights]
    key = "|".join([dataset, str(update_count), ",".join(row.get("relation_id", "") for row in pool)])
    draw = _stable_unit_interval(key)
    acc = 0.0
    for idx, (row, weight) in enumerate(zip(pool, weights)):
        acc += weight
        if draw <= acc:
            return row, idx != 0
    return pool[-1], len(pool) > 1


def _negative_memory_rows(
    state: dict[str, Any],
    diagnostics: set[str],
    components: set[str],
    dataset: str = "",
) -> list[dict[str, Any]]:
    relations = ensure_action_memory(state)
    selection = _policy_section("selection")
    cooldown_updates = int(selection.get("cooldown_updates", 2))
    block_negative_count = int(selection.get("block_negative_count", 4))
    block_dataset_negative_count = int(selection.get("block_dataset_negative_count", 3))
    block_validation_failures = int(selection.get("block_validation_failures", 1))
    memory = state.get("action_memory", {})
    update_count = int(memory.get("update_count", 0))
    rows: list[dict[str, Any]] = []
    for rel in relations.values():
        if rel.get("diagnostic") not in diagnostics or rel.get("component") not in components:
            continue
        negative_count = int(rel.get("negative_count", 0))
        validation_failures = int(rel.get("validation_failures", 0))
        dataset_stats = (rel.get("dataset_stats") or {}).get(dataset, {}) if dataset else {}
        dataset_negative_count = int(dataset_stats.get("negative_count", 0))
        last_negative_update = int(dataset_stats.get("last_negative_update", rel.get("last_negative_update", -10**9)))
        cooldown_remaining = max(0, cooldown_updates - max(0, update_count - last_negative_update))
        blocked = (
            negative_count >= block_negative_count
            or dataset_negative_count >= block_dataset_negative_count
            or validation_failures >= block_validation_failures
        )
        if negative_count <= 0 and validation_failures <= 0 and dataset_negative_count <= 0:
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
                "dataset": dataset,
                "dataset_negative_count": dataset_negative_count,
                "validation_failures": validation_failures,
                "cooldown_remaining": cooldown_remaining,
                "blocked": blocked,
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
    dataset_negative_penalty = float(selection.get("dataset_negative_penalty", 0.08))
    cooldown_penalty = float(selection.get("cooldown_penalty", 0.20))
    cooldown_updates = int(selection.get("cooldown_updates", 2))
    block_negative_count = int(selection.get("block_negative_count", 4))
    block_dataset_negative_count = int(selection.get("block_dataset_negative_count", 3))
    block_validation_failures = int(selection.get("block_validation_failures", 1))
    exploration_bonus = float(selection.get("exploration_bonus", 0.02))
    exploration_cfg = selection.get("controlled_exploration", {})
    if not isinstance(exploration_cfg, dict):
        exploration_cfg = {}
    state = graph_state if graph_state is not None and mode == "trust" else {}
    if mode == "trust" and graph_state is not None:
        ensure_action_memory(graph_state)
    context = feedback.get("pemfc_context") or build_pemfc_context(feedback)
    dataset = str(context.get("dataset") or "").upper()
    update_count = int((graph_state or {}).get("action_memory", {}).get("update_count", 0))

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
                dataset_stats = (rel.get("dataset_stats") or {}).get(dataset, {}) if dataset else {}
                dataset_negative_count = int(dataset_stats.get("negative_count", 0))
                last_negative_update = int(dataset_stats.get("last_negative_update", rel.get("last_negative_update", -10**9)))
            else:
                op_trust = op_prior
                n = 0
                negative_count = 0
                validation_failures = 0
                dataset_negative_count = 0
                last_negative_update = -10**9
                rel = {}
            cooldown_remaining = max(0, cooldown_updates - max(0, update_count - last_negative_update))
            blocked = (
                negative_count >= block_negative_count
                or dataset_negative_count >= block_dataset_negative_count
                or validation_failures >= block_validation_failures
            )
            penalty = (
                negative_penalty * (negative_count + validation_failures)
                + dataset_negative_penalty * dataset_negative_count
                + (cooldown_penalty if cooldown_remaining > 0 else 0.0)
            )
            exploration = exploration_bonus / float(n + 1)
            uncertainty = 1.0 / math.sqrt(float(n + 1))
            expected_correction = _expected_correction(rel)
            risk = _risk_value(str(operator.get("risk", "medium")), rel)
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
                    "n": n,
                    "negative_count": negative_count,
                    "dataset": dataset,
                    "dataset_negative_count": dataset_negative_count,
                    "validation_failures": validation_failures,
                    "cooldown_remaining": cooldown_remaining,
                    "blocked": blocked,
                    "expected_correction": round(expected_correction, 6),
                    "uncertainty": round(uncertainty, 6),
                    "risk_score": round(risk, 6),
                    "suppression": {
                        "negative_penalty": round(negative_penalty * negative_count, 6),
                        "dataset_negative_penalty": round(dataset_negative_penalty * dataset_negative_count, 6),
                        "validation_penalty": round(negative_penalty * validation_failures, 6),
                        "cooldown_penalty": round(cooldown_penalty if cooldown_remaining > 0 else 0.0, 6),
                        "total_penalty": round(penalty, 6),
                        "cooldown_remaining": cooldown_remaining,
                        "blocked": blocked,
                    },
                    "exploration_bonus": round(exploration, 6),
                    "risk": str(operator.get("risk", "medium")),
                    "description": str(operator.get("description", "")),
                    "prompt_guidance": str(operator.get("prompt_guidance", "")),
                    "evidence": prop.get("evidence", {}),
                }
            )

    attention_meta = _apply_relation_attention(feedback, graph_state, candidate_rows, selection, dataset)
    sort_key = "attention_weight" if attention_meta.get("enabled") else "score"
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (row.get(sort_key, 0.0), row.get("score", 0.0), row["operator_trust"], row["component_trust"]),
        reverse=True,
    )
    selectable = [row for row in candidate_rows if not row.get("blocked")]
    selected, attention_sampled = _select_attention_candidate(selectable, attention_meta, dataset, update_count)
    if selected is None:
        selected = candidate_rows[0] if candidate_rows else None
    if selected is not None:
        selected = dict(selected)
        if attention_sampled:
            selected["attention_sampled"] = True
            selected["attention_sample_reason"] = "deterministic_softmax_relation_temperature"
    exploration_selected = False
    if selected and selectable and bool(exploration_cfg.get("enabled", True)):
        margin = float(exploration_cfg.get("score_margin", 0.04))
        min_trust = float(exploration_cfg.get("min_operator_trust", 0.50))
        max_neg = int(exploration_cfg.get("max_negative_count", 1))
        prefer_untried = bool(exploration_cfg.get("prefer_untried", True))
        rank_key = "attention_weight" if attention_meta.get("enabled") else "score"
        alternatives = [
            row
            for row in selectable
            if row.get("relation_id") != selected.get("relation_id")
            and float(row.get("operator_trust", 0.0)) >= min_trust
            and int(row.get("negative_count", 0)) <= max_neg
            and int(row.get("dataset_negative_count", 0)) <= max_neg
            and float(row.get(rank_key, 0.0)) >= float(selected.get(rank_key, 0.0)) - margin
        ]
        if prefer_untried:
            alternatives = sorted(
                alternatives,
                key=lambda row: (int(row.get("n", 0)) == 0, row.get("score", 0.0), row.get("operator_trust", 0.0)),
                reverse=True,
            )
        if alternatives and (
            selected.get("suppression", {}).get("total_penalty", 0.0) > 0
            or int(selected.get("negative_count", 0)) > 0
            or int(selected.get("dataset_negative_count", 0)) > 0
        ):
            selected = dict(alternatives[0])
            selected["exploration_selected"] = True
            selected["exploration_reason"] = "near_top_clean_candidate_after_negative_suppression"
            exploration_selected = True
    negative_memory: list[dict[str, Any]] = []
    if graph_state is not None:
        negative_memory = _negative_memory_rows(graph_state, set(diagnostics), active, dataset=dataset)
    suppression_rows = [
        {
            "relation_id": row.get("relation_id"),
            "diagnostic": row.get("diagnostic"),
            "component": row.get("component"),
            "edit_operator": row.get("edit_operator"),
            "score": row.get("score"),
            "suppression": row.get("suppression"),
            "negative_count": row.get("negative_count"),
            "dataset_negative_count": row.get("dataset_negative_count"),
            "blocked": row.get("blocked"),
        }
        for row in candidate_rows
        if row.get("suppression", {}).get("total_penalty", 0.0) > 0 or row.get("blocked")
    ][:8]

    return {
        "selected_edit": selected,
        "edit_candidates": candidate_rows[:max_candidates],
        "negative_memory": negative_memory,
        "negative_reuse_suppression": suppression_rows,
        "controlled_exploration": {
            "enabled": bool(exploration_cfg.get("enabled", True)),
            "selected": exploration_selected,
            "policy": exploration_cfg,
        },
        "relation_attention": attention_meta,
        "memory_context": context,
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
            "dataset": evidence.get("dataset"),
            "diagnostic": evidence.get("diagnostic"),
            "component": evidence.get("component"),
            "edit_operator": evidence.get("edit_operator"),
            "evidence_policy": evidence.get("evidence_policy"),
            "reward": evidence.get("reward", {}).get("reward"),
            "reason": evidence.get("reward", {}).get("reason"),
            "previous_target": evidence.get("previous_target"),
            "next_target": evidence.get("next_target"),
        }
    )
    del rows[:-80]


def _feedback_dataset(feedback: dict[str, Any]) -> str:
    context = feedback.get("pemfc_context") or {}
    return str(context.get("dataset") or "").upper()


def _trust_outcome_agreement(trust_before: float, direction: str, threshold: float) -> dict[str, Any]:
    expected = "improve" if trust_before >= threshold else "uncertain"
    if direction == "increase":
        outcome = "improved"
    elif direction == "decrease":
        outcome = "degraded"
    elif direction == "neutral":
        outcome = "ambiguous"
    else:
        outcome = "off_policy"
    if outcome == "off_policy":
        label = "off_policy_not_scored"
        agreement = None
    elif outcome == "ambiguous":
        label = "ambiguous_outcome"
        agreement = None
    elif expected == "improve" and outcome == "improved":
        label = "confirmed_high_trust"
        agreement = True
    elif expected == "improve" and outcome == "degraded":
        label = "contradicted_high_trust"
        agreement = False
    elif expected == "uncertain" and outcome == "improved":
        label = "surprising_improvement"
        agreement = False
    else:
        label = "confirmed_low_trust"
        agreement = True
    return {
        "trust_threshold": threshold,
        "expected": expected,
        "outcome": outcome,
        "agreement": agreement,
        "label": label,
    }


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
    agreement_trust_threshold = float(update_policy.get("agreement_trust_threshold", 0.65))

    rel = relations[rid]
    trust_before = action_relation_trust(state, diagnostic, component, edit_operator)
    update_index = int(memory.get("update_count", 0)) + 1
    dataset = _feedback_dataset(previous_feedback) or _feedback_dataset(next_feedback)
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
    component_mismatch = bool(patch_record.get("component_mismatch", False))
    on_policy = not operator_mismatch and not component_mismatch

    if patch_record.get("validation_fallback"):
        raw_reward = -abs(validation_failure_penalty)
        reason = "patch_validation_fallback"
    elif not next_result.get("success"):
        raw_reward = -1.0
        reason = "harness_failure"
    else:
        raw_reward = (
            diagnostic_weight * diagnostic_delta
            + target_weight * target_delta
            - overfit_penalty_weight * overfit_delta
        )
        reason = "probe_aligned_target_delta"
    raw_reward = _clip(raw_reward)

    if not on_policy:
        reward = 0.0
        update_amount = 0.0
        direction = "off_policy"
        candidate_status = "off_policy_mismatch_not_rewarded"
        rel["off_policy_count"] = int(rel.get("off_policy_count", 0)) + 1
    else:
        reward = raw_reward
        update_amount = min(max_update, max(0.04, abs(reward) * update_scale))
    if on_policy and reward > positive_threshold:
        rel["alpha"] = _safe_float(rel.get("alpha"), 1.0) + update_amount
        rel["positive_count"] = int(rel.get("positive_count", 0)) + 1
        direction = "increase"
        candidate_status = "accepted_by_probe_reward"
    elif on_policy and reward < negative_threshold:
        rel["beta"] = _safe_float(rel.get("beta"), 1.0) + update_amount
        rel["negative_count"] = int(rel.get("negative_count", 0)) + 1
        rel["last_negative_update"] = update_index
        direction = "decrease"
        candidate_status = "rejected_by_probe_reward"
    elif on_policy:
        rel["alpha"] = _safe_float(rel.get("alpha"), 1.0) + 0.02
        rel["beta"] = _safe_float(rel.get("beta"), 1.0) + 0.02
        direction = "neutral"
        candidate_status = "ambiguous_probe_reward"
    if on_policy and patch_record.get("validation_fallback"):
        rel["validation_failures"] = int(rel.get("validation_failures", 0)) + 1

    if on_policy:
        rel["n"] = int(rel.get("n", 0)) + 1
    if dataset and on_policy:
        stats = rel.setdefault("dataset_stats", {}).setdefault(
            dataset,
            {"positive_count": 0, "negative_count": 0, "neutral_count": 0, "validation_failures": 0},
        )
        if direction == "increase":
            stats["positive_count"] = int(stats.get("positive_count", 0)) + 1
        elif direction == "decrease":
            stats["negative_count"] = int(stats.get("negative_count", 0)) + 1
            stats["last_negative_update"] = update_index
        elif direction == "neutral":
            stats["neutral_count"] = int(stats.get("neutral_count", 0)) + 1
        if patch_record.get("validation_fallback"):
            stats["validation_failures"] = int(stats.get("validation_failures", 0)) + 1
    rel["trust"] = action_relation_trust(state, diagnostic, component, edit_operator)
    rel["updated_at"] = _now()
    agreement = _trust_outcome_agreement(trust_before, direction, agreement_trust_threshold)
    evidence = {
        "ts": _now(),
        "relation_id": rid,
        "dataset": dataset,
        "diagnostic": diagnostic,
        "component": component,
        "edit_operator": edit_operator,
        "evidence_policy": "on_policy" if on_policy else "off_policy",
        "trust_before": round(trust_before, 6),
        "trust_after": round(rel["trust"], 6),
        "direction": direction,
        "candidate_status": candidate_status,
        "update_amount": round(update_amount, 6),
        "reward": {
            "reward": round(reward, 6),
            "raw_reward": round(raw_reward, 6),
            "reason": reason,
            "target_delta": target_delta,
            "diagnostic_delta": diagnostic_delta,
            "previous_probe": previous_probe,
            "next_probe": next_probe,
            "overfit_delta": overfit_delta,
        },
        "trust_outcome_agreement": agreement,
        "previous_target": previous_target,
        "next_target": next_target,
        "previous_result_path": previous_result.get("paths", {}).get("result"),
        "next_result_path": next_result.get("paths", {}).get("result"),
        "operator_mismatch": operator_mismatch,
        "component_mismatch": component_mismatch,
    }
    rel.setdefault("evidence", []).append(evidence)
    if direction == "decrease":
        _append_negative_experience(memory, evidence)
    memory["update_count"] = update_index
    memory["updated_at"] = _now()
    return [evidence]
