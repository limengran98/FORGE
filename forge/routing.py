from __future__ import annotations

from typing import Any

from .harness_spec import get_component_graph, get_routing_policy
from .memory import build_pemfc_context, select_edit_candidates
from .trust import (
    candidate_components_for_diagnostic,
    diagnostic_message,
    ensure_trust_relations,
    relation_id,
    relation_trust,
)


def _policy_value(policy: dict[str, Any], group: str, key: str, default: float) -> float:
    try:
        return float(policy.get(group, {}).get(key, default))
    except Exception:
        return default


def _policy_text(policy: dict[str, Any], key: str) -> str:
    return str(policy.get("messages", {}).get(key, key))


def route_feedback(
    feedback: dict[str, Any],
    graph_state: dict[str, Any] | None = None,
    mode: str = "trust",
) -> dict[str, Any]:
    component_graph = get_component_graph()
    policy = get_routing_policy()
    mode = str(mode or "trust")
    if mode not in {"rule", "prior", "trust"}:
        raise ValueError("routing mode must be one of: rule, prior, trust")
    if graph_state is not None and mode == "trust":
        ensure_trust_relations(graph_state)
    f = feedback.get("features", {})
    initial_score = float(policy.get("initial_score", 0.05))
    scores = {node: initial_score for node in component_graph["nodes"]}
    reasons: dict[str, list[str]] = {node: [] for node in component_graph["nodes"]}
    propagations: list[dict[str, Any]] = []

    def add(node: str, rule_key: str, amount: float | None = None) -> None:
        if amount is None:
            amount = _policy_value(policy, "rule_weights", rule_key, 0.0)
        scores[node] = scores.get(node, 0.0) + amount
        reasons.setdefault(node, []).append(_policy_text(policy, rule_key))

    def propagate(diagnostic: dict[str, Any]) -> None:
        name = str(diagnostic.get("name") or "")
        severity = float(diagnostic.get("severity") or 0.0)
        confidence = float(diagnostic.get("confidence") or 0.0)
        if not name or severity <= 0.0 or confidence <= 0.0:
            return
        for component, prior in candidate_components_for_diagnostic(name).items():
            trust = relation_trust(graph_state or {}, name, component) if graph_state is not None and mode == "trust" else float(prior)
            contribution = severity * confidence * trust
            if contribution <= 0:
                continue
            scores[component] = scores.get(component, 0.0) + contribution
            message = diagnostic_message(name)
            reasons.setdefault(component, []).append(message)
            propagations.append(
                {
                    "relation_id": relation_id(name, component),
                    "diagnostic": name,
                    "component": component,
                    "severity": round(severity, 6),
                    "confidence": round(confidence, 6),
                    "trust": round(trust, 6),
                    "contribution": round(contribution, 6),
                    "message": message,
                    "evidence": diagnostic.get("evidence", {}),
                }
            )

    if f.get("has_exception", 0.0) > 0:
        if f.get("syntax_error", 0.0) > 0 or f.get("import_error", 0.0) > 0:
            add("interface", "syntax_interface")
        elif f.get("shape_error", 0.0) > 0:
            add("prediction_head", "shape_prediction_head")
            add("interface", "shape_interface")
        elif f.get("oom_error", 0.0) > 0:
            add("temporal_memory", "oom_temporal_memory")
            add("regularization", "oom_regularization")
        elif f.get("nan_or_inf", 0.0) > 0:
            add("normalization", "nan_normalization")
            add("optimization", "nan_optimization")
        else:
            add("interface", "runtime_interface")

    if f.get("cold_start", 0.0) > 0 and f.get("run_success", 0.0) > 0:
        add("factor_fusion", "cold_start_factor_fusion")
        add("temporal_memory", "cold_start_temporal_memory")

    if f.get("degraded_vs_previous", 0.0) > _policy_value(policy, "thresholds", "degraded_vs_previous", 0.02):
        add("optimization", "degraded_optimization")
        add("regularization", "degraded_regularization")

    if f.get("overfit_score", 0.0) > _policy_value(policy, "thresholds", "overfit_score", 0.15):
        add("regularization", "overfit_regularization", min(0.9, f["overfit_score"]))
        add("prediction_head", "overfit_prediction_head")

    if (
        f.get("underfit_score", 0.0) > _policy_value(policy, "thresholds", "underfit_score", 0.35)
        and f.get("val_slope", 0.0) >= _policy_value(policy, "thresholds", "underfit_val_slope", -0.002)
    ):
        add("temporal_memory", "underfit_temporal_memory")
        add("input_embedding", "underfit_input_embedding")

    if f.get("val_volatility", 0.0) > _policy_value(policy, "thresholds", "val_volatility", 0.05):
        add("optimization", "volatile_optimization")
        add("normalization", "volatile_normalization")

    if f.get("mape_inverse_log", 0.0) > _policy_value(policy, "thresholds", "mape_inverse_log", 2.0):
        add("normalization", "mape_normalization")
        add("prediction_head", "mape_prediction_head")

    if mode in {"prior", "trust"}:
        for diagnostic in feedback.get("diagnostics", []):
            if isinstance(diagnostic, dict):
                propagate(diagnostic)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0]
    top_k = int(policy.get("top_k", 3))
    active_threshold = float(policy.get("active_threshold", 0.2))
    active_nodes = [node for node, score in ranked[:top_k] if score >= active_threshold]
    if primary not in active_nodes:
        active_nodes.insert(0, primary)
    active_edges = [
        edge
        for edge in component_graph["edges"]
        if edge["from"] in active_nodes or edge["to"] in active_nodes
    ]
    action_selection: dict[str, Any] = {
        "selected_edit": None,
        "edit_candidates": [],
        "negative_memory": [],
        "negative_reuse_suppression": [],
        "controlled_exploration": {},
        "relation_attention": {},
        "memory_context": feedback.get("pemfc_context") or build_pemfc_context(feedback),
    }
    if mode in {"prior", "trust"}:
        action_selection = select_edit_candidates(
            feedback,
            graph_state,
            active_nodes,
            propagations,
            mode=mode,
        )

    trust_policy = {"rule": "rule_only", "prior": "trust_prior_only", "trust": "trust_graph_v1"}[mode]
    if mode == "trust" and graph_state is None:
        trust_policy = "trust_prior_only"

    return {
        "primary_component": primary,
        "active_components": active_nodes,
        "scores": {node: round(score, 4) for node, score in ranked},
        "reasons": {node: vals for node, vals in reasons.items() if vals},
        "propagations": sorted(propagations, key=lambda item: item["contribution"], reverse=True),
        "selected_edit": action_selection.get("selected_edit"),
        "edit_candidates": action_selection.get("edit_candidates") or [],
        "negative_memory": action_selection.get("negative_memory") or [],
        "negative_reuse_suppression": action_selection.get("negative_reuse_suppression") or [],
        "controlled_exploration": action_selection.get("controlled_exploration") or {},
        "relation_attention": action_selection.get("relation_attention") or {},
        "memory_context": action_selection.get("memory_context") or {},
        "trust_policy": trust_policy,
        "component_graph": component_graph,
        "active_subgraph": {
            "nodes": active_nodes,
            "edges": active_edges,
        },
        "routing_policy": str(policy.get("name", "rule_graph_v1")),
    }
