from __future__ import annotations

from typing import Any

from .harness_spec import get_component_graph, get_routing_policy

def _policy_value(policy: dict[str, Any], group: str, key: str, default: float) -> float:
    try:
        return float(policy.get(group, {}).get(key, default))
    except Exception:
        return default


def _policy_text(policy: dict[str, Any], key: str) -> str:
    return str(policy.get("messages", {}).get(key, key))


def route_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    component_graph = get_component_graph()
    policy = get_routing_policy()
    f = feedback.get("features", {})
    initial_score = float(policy.get("initial_score", 0.05))
    scores = {node: initial_score for node in component_graph["nodes"]}
    reasons: dict[str, list[str]] = {node: [] for node in component_graph["nodes"]}

    def add(node: str, rule_key: str, amount: float | None = None) -> None:
        if amount is None:
            amount = _policy_value(policy, "rule_weights", rule_key, 0.0)
        scores[node] = scores.get(node, 0.0) + amount
        reasons.setdefault(node, []).append(_policy_text(policy, rule_key))

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

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0]
    top_k = int(policy.get("top_k", 3))
    active_threshold = float(policy.get("active_threshold", 0.2))
    active_nodes = [node for node, score in ranked[:top_k] if score >= active_threshold]
    active_edges = [
        edge
        for edge in component_graph["edges"]
        if edge["from"] in active_nodes or edge["to"] in active_nodes
    ]

    return {
        "primary_component": primary,
        "active_components": active_nodes,
        "scores": {node: round(score, 4) for node, score in ranked},
        "reasons": {node: vals for node, vals in reasons.items() if vals},
        "component_graph": component_graph,
        "active_subgraph": {
            "nodes": active_nodes,
            "edges": active_edges,
        },
        "routing_policy": str(policy.get("name", "rule_graph_v1")),
    }
