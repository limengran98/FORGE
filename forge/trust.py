from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from .diagnostics import diagnostics_by_name
from .harness_spec import get_component_graph, get_trust_policy


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def relation_id(diagnostic: str, component: str) -> str:
    return f"{diagnostic}->{component}"


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


def _relation_from_prior(diagnostic: str, component: str, prior: float, strength: float) -> dict[str, Any]:
    prior = max(0.01, min(0.99, float(prior)))
    strength = max(1.0, float(strength))
    alpha = prior * strength
    beta = (1.0 - prior) * strength
    return {
        "id": relation_id(diagnostic, component),
        "from": diagnostic,
        "to": component,
        "alpha": alpha,
        "beta": beta,
        "trust": alpha / (alpha + beta),
        "n": 0,
        "evidence": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


def ensure_trust_relations(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policy = get_trust_policy()
    priors = policy.get("diagnostic_component_priors", {})
    strength = float(policy.get("prior_strength", 3.0))
    default_prior = float(policy.get("default_prior", 0.35))
    known_components = set(get_component_graph()["nodes"])
    relations = state.setdefault("relations", {})

    for diagnostic, component_priors in priors.items():
        for component, prior in component_priors.items():
            if component not in known_components:
                continue
            rid = relation_id(str(diagnostic), str(component))
            relations.setdefault(rid, _relation_from_prior(str(diagnostic), str(component), float(prior), strength))

    state.setdefault(
        "trust",
        {
            "policy": str(policy.get("name", "trust_graph_v1")),
            "default_prior": default_prior,
            "prior_strength": strength,
            "created_at": _now(),
        },
    )
    return relations


def relation_trust(state: dict[str, Any], diagnostic: str, component: str) -> float:
    relations = ensure_trust_relations(state)
    rid = relation_id(diagnostic, component)
    rel = relations.get(rid)
    if rel is None:
        return float(get_trust_policy().get("default_prior", 0.35))
    alpha = _safe_float(rel.get("alpha"), 1.0)
    beta = _safe_float(rel.get("beta"), 1.0)
    trust = alpha / max(alpha + beta, 1e-8)
    rel["trust"] = trust
    return trust


def candidate_components_for_diagnostic(diagnostic: str) -> dict[str, float]:
    priors = get_trust_policy().get("diagnostic_component_priors", {})
    return {str(component): float(prior) for component, prior in priors.get(diagnostic, {}).items()}


def diagnostic_message(diagnostic: str) -> str:
    return str(get_trust_policy().get("diagnostic_messages", {}).get(diagnostic, diagnostic))


def _target(result: dict[str, Any], target_metric: str) -> float | None:
    if not result or not result.get("success"):
        return None
    value = result.get("metrics", {}).get("target", {}).get(target_metric)
    return None if value is None else _safe_float(value)


def _diagnostic_severity(feedback: dict[str, Any], diagnostic: str) -> float:
    item = diagnostics_by_name(feedback).get(diagnostic)
    if not item:
        return 0.0
    return _safe_float(item.get("severity")) * _safe_float(item.get("confidence"), 1.0)


def outcome_reward(
    previous_result: dict[str, Any],
    next_result: dict[str, Any],
    previous_feedback: dict[str, Any],
    next_feedback: dict[str, Any],
    diagnostic: str,
    target_metric: str,
) -> dict[str, Any]:
    policy = get_trust_policy().get("reward", {})
    target_weight = float(policy.get("target_weight", 0.7))
    diagnostic_weight = float(policy.get("diagnostic_weight", 0.3))
    overfit_penalty_weight = float(policy.get("overfit_penalty_weight", 0.15))

    if not next_result.get("success"):
        return {
            "reward": -1.0,
            "target_delta": None,
            "diagnostic_delta": None,
            "overfit_delta": None,
            "reason": "harness_failure",
        }

    prev_target = _target(previous_result, target_metric)
    next_target = _target(next_result, target_metric)
    target_delta = 0.0
    if prev_target is not None and next_target is not None:
        target_delta = (prev_target - next_target) / (abs(prev_target) + 1e-8)

    prev_diag = _diagnostic_severity(previous_feedback, diagnostic)
    next_diag = _diagnostic_severity(next_feedback, diagnostic)
    diagnostic_delta = prev_diag - next_diag

    prev_overfit = _safe_float(previous_feedback.get("features", {}).get("overfit_score"))
    next_overfit = _safe_float(next_feedback.get("features", {}).get("overfit_score"))
    overfit_delta = max(0.0, next_overfit - prev_overfit)

    reward = target_weight * target_delta + diagnostic_weight * diagnostic_delta - overfit_penalty_weight * overfit_delta
    return {
        "reward": _clip(reward),
        "target_delta": target_delta,
        "diagnostic_delta": diagnostic_delta,
        "overfit_delta": overfit_delta,
        "reason": "metric_and_diagnostic_delta",
    }


def update_relations_from_outcome(
    state: dict[str, Any],
    patch_record: dict[str, Any],
    previous_result: dict[str, Any],
    next_result: dict[str, Any],
    previous_feedback: dict[str, Any],
    next_feedback: dict[str, Any],
    target_metric: str,
) -> list[dict[str, Any]]:
    relations = ensure_trust_relations(state)
    policy = get_trust_policy().get("reward", {})
    positive_threshold = float(policy.get("positive_threshold", 0.002))
    negative_threshold = float(policy.get("negative_threshold", -0.002))
    update_scale = float(policy.get("update_scale", 2.0))
    max_update = float(policy.get("max_update", 1.0))

    component = str(patch_record.get("component") or patch_record.get("routed_component") or "")
    propagations = patch_record.get("route_propagations") or []
    selected = [
        item for item in propagations if item.get("component") == component and item.get("diagnostic")
    ]
    if not selected and component:
        selected = [
            {"diagnostic": item.get("diagnostic"), "component": component}
            for item in propagations
            if item.get("diagnostic")
        ]

    seen: set[str] = set()
    updates: list[dict[str, Any]] = []
    for item in selected:
        diagnostic = str(item.get("diagnostic"))
        target_component = str(item.get("component") or component)
        rid = relation_id(diagnostic, target_component)
        if rid in seen or rid not in relations:
            continue
        seen.add(rid)
        rel = relations[rid]
        trust_before = relation_trust(state, diagnostic, target_component)
        reward = outcome_reward(
            previous_result,
            next_result,
            previous_feedback,
            next_feedback,
            diagnostic,
            target_metric,
        )
        value = float(reward["reward"])
        update_amount = min(max_update, max(0.05, abs(value) * update_scale))
        if value > positive_threshold:
            rel["alpha"] = _safe_float(rel.get("alpha"), 1.0) + update_amount
            direction = "increase"
        elif value < negative_threshold:
            rel["beta"] = _safe_float(rel.get("beta"), 1.0) + update_amount
            direction = "decrease"
        else:
            rel["alpha"] = _safe_float(rel.get("alpha"), 1.0) + 0.02
            rel["beta"] = _safe_float(rel.get("beta"), 1.0) + 0.02
            direction = "neutral"
        rel["n"] = int(rel.get("n", 0)) + 1
        rel["trust"] = relation_trust(state, diagnostic, target_component)
        rel["updated_at"] = _now()

        evidence = {
            "ts": _now(),
            "diagnostic": diagnostic,
            "component": target_component,
            "trust_before": trust_before,
            "trust_after": rel["trust"],
            "direction": direction,
            "update_amount": update_amount,
            "reward": reward,
            "previous_target": _target(previous_result, target_metric),
            "next_target": _target(next_result, target_metric),
            "previous_result_path": previous_result.get("paths", {}).get("result"),
            "next_result_path": next_result.get("paths", {}).get("result"),
        }
        rel.setdefault("evidence", []).append(evidence)
        updates.append(evidence)
    return updates
