from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def method_framework(candidate_tournament_k: int = 1) -> dict[str, Any]:
    """Compact, auditable method map for FORGE summaries and reports."""

    return {
        "schema": "forge.method_framework.v1",
        "name": "FORGE: Feedback Observability-guided Routing for Graph-based Evolution",
        "modules": [
            {
                "id": "pemfc_native_diagnostic_harness",
                "name": "PEMFC-native diagnostic harness",
                "role": "fixed executable evidence source",
                "tools": [
                    "residual_by_stage",
                    "residual_by_operating_region",
                    "degradation_slope_error",
                    "early_vs_late_error",
                    "train_val_gap",
                    "stability_across_seeds",
                    "invalid_patch_detector",
                    "repeated_edit_detector",
                    "component_change_summary",
                ],
                "principle": (
                    "The LLM is not trusted to infer temporal evidence from raw PEMFC sequences; "
                    "it can only propose edits grounded in executable diagnostic observations."
                ),
            },
            {
                "id": "evidence_reconstruction_graph",
                "name": "Evidence reconstruction graph",
                "role": "execution-calibrated feedback routing",
                "nodes": ["feedback", "component", "edit", "outcome"],
                "edge_evidence": (
                    "Edges are updated by historical execution outcomes, not semantic similarity "
                    "or LLM confidence."
                ),
                "principle": (
                    "Feedback routing is formulated as active evidence reconstruction over an "
                    "execution-calibrated graph."
                ),
            },
            {
                "id": "test_time_adaptive_strategy_memory",
                "name": "Test-time adaptive strategy memory",
                "role": "short-horizon policy state refreshed after each harness outcome",
                "state_fields": [
                    "current_failure_hypotheses",
                    "proven_ineffective_edits",
                    "trusted_components",
                    "next_candidate_scope",
                    "forbidden_repeats",
                    "expected_improvement_metric",
                ],
                "refresh_triggers": [
                    "stagnation",
                    "route_entropy_high",
                    "same_component_without_gain",
                    "runtime_or_shape_failure",
                    "cross_cell_tradeoff",
                    "diagnostic_signature_change",
                ],
            },
            {
                "id": "k_candidate_evidence_tournament",
                "name": "K-candidate evidence tournament",
                "role": "bounded candidate competition under the fixed harness",
                "configured_k": int(max(1, candidate_tournament_k)),
                "candidate_contract": [
                    "evidence",
                    "target_component",
                    "edit_type",
                    "expected_effect",
                    "risk",
                    "interface_constraints",
                    "stop_or_reject_condition",
                ],
                "default_behavior": (
                    "K=1 preserves the validated trust-routing main line; K>1 should be treated "
                    "as an explicit budgeted tournament experiment."
                ),
            },
        ],
    }


def _safe_float(value: Any, default: float = math.inf) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _iter_key(iteration: int) -> str:
    return f"iter_{int(iteration):03d}"


def _target_value(result: dict[str, Any], target_metric: str) -> float:
    if not isinstance(result, dict) or not result.get("success"):
        return math.inf
    return _safe_float(result.get("metrics", {}).get("target", {}).get(target_metric))


def _row_target(row: dict[str, Any], target_metric: str) -> float:
    return _target_value(row.get("result") or {}, target_metric)


def _parent_target(
    patch: dict[str, Any],
    rows_by_iter: dict[int, dict[str, Any]],
    patch_iteration: int,
    target_metric: str,
) -> float:
    stored = _safe_float(patch.get("parent_target"))
    if math.isfinite(stored):
        return stored
    parent_iteration = patch.get("parent_iteration")
    try:
        parent_key = int(parent_iteration) if parent_iteration is not None else patch_iteration
    except Exception:
        parent_key = patch_iteration
    parent = rows_by_iter.get(parent_key)
    return _row_target(parent or {}, target_metric)


def _dominant_diagnostic(patch: dict[str, Any]) -> str | None:
    propagations = patch.get("route_propagations") or []
    if not propagations:
        return None
    first = propagations[0]
    if not isinstance(first, dict):
        return None
    return str(first.get("diagnostic") or "") or None


def _aligned_with_evidence(patch: dict[str, Any], routed_component: str | None) -> bool | None:
    propagations = [row for row in patch.get("route_propagations") or [] if isinstance(row, dict)]
    if not propagations or not routed_component:
        return None
    components = {str(row.get("component") or "") for row in propagations if row.get("component")}
    if not components:
        return None
    return routed_component in components and not bool(patch.get("component_mismatch"))


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def build_run_evidence_audit(
    graph_state: dict[str, Any],
    history: list[dict[str, Any]],
    target_metric: str,
    candidate_tournament_k: int = 1,
) -> dict[str, Any]:
    """Build method-level evidence metrics from one completed FORGE run."""

    rows_by_iter = {int(row["iteration"]): row for row in history if "iteration" in row}
    if not rows_by_iter:
        return {
            "schema": "forge.evidence_audit.v1",
            "method_framework": method_framework(candidate_tournament_k),
            "metrics": {},
            "strategy_memory": {},
            "attempts": [],
        }

    initial_target = _row_target(rows_by_iter[min(rows_by_iter)], target_metric)
    best_iteration = min(rows_by_iter, key=lambda idx: _row_target(rows_by_iter[idx], target_metric))
    best_target = _row_target(rows_by_iter[best_iteration], target_metric)

    attempts: list[dict[str, Any]] = []
    seen_negative: set[tuple[str, str]] = set()
    negative_counts: Counter[tuple[str, str]] = Counter()
    improvement_counts: Counter[str] = Counter()
    component_attempts: Counter[str] = Counter()
    route_by_diagnostic: dict[str, Counter[str]] = defaultdict(Counter)
    aligned_count = 0
    alignable_count = 0
    repair_attempt_count = 0

    for outcome_iteration in sorted(rows_by_iter):
        if outcome_iteration <= 0:
            continue
        patch_iteration = outcome_iteration - 1
        patch = (
            graph_state.get("iterations", {})
            .get(_iter_key(patch_iteration), {})
            .get("patch", {})
        )
        if not patch:
            continue

        outcome_row = rows_by_iter[outcome_iteration]
        outcome_result = outcome_row.get("result") or {}
        parent_target = _parent_target(patch, rows_by_iter, patch_iteration, target_metric)
        outcome_target = _target_value(outcome_result, target_metric)
        success = bool(outcome_result.get("success"))
        improved = bool(success and math.isfinite(parent_target) and outcome_target < parent_target - 1e-12)
        repair_attempts = patch.get("repair_attempts") or []
        repair_attempt_count += len(repair_attempts)

        component = str(patch.get("component") or patch.get("routed_component") or "unknown")
        edit_action = str(patch.get("edit_action") or "unknown")
        edit_key = (component, edit_action)
        invalid = bool(
            patch.get("validation_fallback")
            or patch.get("edit_operator_mismatch")
            or patch.get("component_mismatch")
            or repair_attempts
            or not success
        )
        repeated_useless = edit_key in seen_negative
        if not improved:
            seen_negative.add(edit_key)
            negative_counts[edit_key] += 1
        else:
            improvement_counts[component] += 1
        component_attempts[component] += 1

        routed_component = str(patch.get("routed_component") or outcome_row.get("primary_component") or "")
        dominant_diagnostic = _dominant_diagnostic(patch)
        if dominant_diagnostic and routed_component:
            route_by_diagnostic[dominant_diagnostic][routed_component] += 1

        aligned = _aligned_with_evidence(patch, routed_component)
        if aligned is not None:
            alignable_count += 1
            aligned_count += 1 if aligned else 0

        attempts.append(
            {
                "patch_iteration": patch_iteration,
                "outcome_iteration": outcome_iteration,
                "success": success,
                "component": component,
                "routed_component": routed_component or None,
                "edit_action": edit_action,
                "dominant_diagnostic": dominant_diagnostic,
                "parent_target": parent_target,
                "outcome_target": outcome_target,
                "target_delta": parent_target - outcome_target
                if math.isfinite(parent_target) and math.isfinite(outcome_target)
                else 0.0,
                "improved": improved,
                "invalid_edit": invalid,
                "repeated_useless_edit": repeated_useless,
                "evidence_aligned": aligned,
            }
        )

    total_attempts = len(attempts)
    improved_attempts = [row for row in attempts if row["improved"]]
    invalid_attempts = [row for row in attempts if row["invalid_edit"]]
    repeated_useless = [row for row in attempts if row["repeated_useless_edit"]]

    stability_rows: list[dict[str, Any]] = []
    for diagnostic, counter in sorted(route_by_diagnostic.items()):
        total = sum(counter.values())
        top_component, top_count = counter.most_common(1)[0]
        stability_rows.append(
            {
                "diagnostic": diagnostic,
                "top_component": top_component,
                "top_component_count": top_count,
                "total": total,
                "stability": _rate(top_count, total),
                "component_counts": dict(counter),
            }
        )
    routing_stability = (
        sum(row["stability"] for row in stability_rows) / len(stability_rows)
        if stability_rows
        else 0.0
    )

    best_gain = initial_target - best_target if math.isfinite(initial_target) and math.isfinite(best_target) else 0.0
    attempts_to_best = max(0, int(best_iteration))
    ineffective = [
        {
            "component": component,
            "edit_action": edit_action,
            "negative_count": count,
        }
        for (component, edit_action), count in negative_counts.most_common(8)
    ]
    trusted_components = [
        {
            "component": component,
            "success_count": count,
            "attempt_count": component_attempts.get(component, 0),
        }
        for component, count in improvement_counts.most_common(8)
    ]

    last_row = rows_by_iter[max(rows_by_iter)]
    last_route_components = last_row.get("active_components") or []
    stagnation_window = attempts[-5:]
    stagnated = bool(stagnation_window and not any(row["improved"] for row in stagnation_window))

    metrics = {
        "best_mae_rmse": {
            "best_iteration": int(best_iteration),
            "best_target": best_target,
            "target_metric": target_metric,
        },
        "improvement_rate": _rate(len(improved_attempts), total_attempts),
        "invalid_edit_rate": _rate(len(invalid_attempts), total_attempts),
        "repeated_useless_edit_rate": _rate(len(repeated_useless), total_attempts),
        "routing_stability": routing_stability,
        "evidence_alignment": _rate(aligned_count, alignable_count),
        "budget_efficiency": {
            "attempts": total_attempts,
            "candidate_tournament_k": int(max(1, candidate_tournament_k)),
            "executed_candidates_per_round": 1,
            "estimated_llm_patch_calls": total_attempts + repair_attempt_count,
            "best_iteration": int(best_iteration),
            "attempts_to_best": attempts_to_best,
            "target_gain_vs_initial": best_gain,
            "target_gain_per_attempt": best_gain / total_attempts if total_attempts else 0.0,
        },
    }

    strategy_memory = {
        "current_failure_hypotheses": last_row.get("primary_component"),
        "proven_ineffective_edits": ineffective,
        "trusted_components": trusted_components,
        "next_candidate_scope": last_route_components,
        "forbidden_repeats": [row for row in ineffective if row["negative_count"] >= 2],
        "expected_improvement_metric": target_metric,
        "refresh_triggers": {
            "stagnation": stagnated,
            "same_component_without_gain": stagnated and bool(last_row.get("primary_component")),
            "runtime_or_shape_failure": bool(invalid_attempts[-1:]),
            "diagnostic_signature_change": False,
            "cross_cell_tradeoff": "requires_sweep_or_cross_run_context",
        },
    }

    return {
        "schema": "forge.evidence_audit.v1",
        "method_framework": method_framework(candidate_tournament_k),
        "metrics": metrics,
        "routing_stability_by_diagnostic": stability_rows,
        "strategy_memory": strategy_memory,
        "attempts_tail": attempts[-12:],
    }
