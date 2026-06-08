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
                "id": "evidence_calibrated_routing_graph",
                "name": "Evidence-calibrated routing graph",
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
                "id": "accepted_parent_negative_suppression",
                "name": "Accepted-parent selection and negative evidence suppression",
                "role": "stable model inheritance and failure reuse",
                "selection_rules": [
                    "successful candidates may update evidence memory",
                    "only accepted or protected best models can become the next parent",
                    "failed, mismatched, or regressive edits are retained as negative evidence",
                    "repeated ineffective feedback-component routes are suppressed",
                ],
                "principle": (
                    "Bad patches are useful evidence, but they must not contaminate the inherited "
                    "model trajectory."
                ),
            },
            {
                "id": "auditable_trajectory",
                "name": "Auditable trajectory",
                "role": "complete evidence chain for every refinement attempt",
                "trace_fields": [
                    "diagnostic_feedback",
                    "routed_relation",
                    "trust_before",
                    "llm_patch_or_fallback",
                    "harness_outcome",
                    "metric_delta",
                    "trust_update",
                    "accept_or_reject_decision",
                ],
                "principle": (
                    "FORGE reports the model it found and the executable evidence explaining how "
                    "each routing decision was supported, corrected, or suppressed."
                ),
            },
        ],
        "experimental_extensions": {
            "candidate_tournament_k": int(max(1, candidate_tournament_k)),
            "not_mainline": [
                "adaptive_relation_temperature",
                "relation_level_action_memory",
                "llm_motif_dispatch",
                "k_candidate_tournament",
            ],
            "note": "These remain available for ablation or future work, but the stable method reports the four modules above.",
        },
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


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else 0.0


def _metric_value(result: dict[str, Any], section: str, name: str) -> float:
    if not isinstance(result, dict) or not result.get("success"):
        return math.inf
    return _safe_float(result.get("metrics", {}).get(section, {}).get(name))


def _paper_mae(result: dict[str, Any]) -> float:
    return _metric_value(result, "paper_scaled", "mae")


def _paper_mse(result: dict[str, Any]) -> float:
    return _metric_value(result, "paper_scaled", "mse")


def _selected_edit_field(patch: dict[str, Any], key: str) -> str | None:
    selected = patch.get("selected_edit") or {}
    if not isinstance(selected, dict):
        return None
    value = selected.get(key)
    return str(value) if value is not None else None


def _route_relation_ids(patch: dict[str, Any]) -> list[str]:
    relation_ids: list[str] = []
    for row in patch.get("route_propagations") or []:
        if isinstance(row, dict) and row.get("relation_id"):
            relation_ids.append(str(row["relation_id"]))
    return relation_ids


def _trust_before_mean(patch: dict[str, Any]) -> float:
    trust_before = patch.get("trust_before") or {}
    if not isinstance(trust_before, dict):
        return 0.0
    relation_ids = _route_relation_ids(patch)
    if relation_ids:
        return _safe_mean([_safe_float(trust_before.get(relation_id), math.nan) for relation_id in relation_ids])
    return _safe_mean([_safe_float(value, math.nan) for value in trust_before.values()])


def _matching_trust_updates(graph_state: dict[str, Any], outcome_iteration: int, patch: dict[str, Any]) -> list[dict[str, Any]]:
    updates = (
        graph_state.get("iterations", {})
        .get(_iter_key(outcome_iteration), {})
        .get("trust_updates", [])
    )
    if not isinstance(updates, list):
        return []
    relation_ids = set(_route_relation_ids(patch))
    components = {
        str(row.get("component"))
        for row in patch.get("route_propagations") or []
        if isinstance(row, dict) and row.get("component")
    }
    diagnostics = {
        str(row.get("diagnostic"))
        for row in patch.get("route_propagations") or []
        if isinstance(row, dict) and row.get("diagnostic")
    }
    matched = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        update_relation = str(update.get("relation_id") or "")
        if update_relation and update_relation in relation_ids:
            matched.append(update)
            continue
        update_component = str(update.get("component") or "")
        update_diagnostic = str(update.get("diagnostic") or "")
        if update_component in components and update_diagnostic in diagnostics:
            matched.append(update)
    return matched


def _trust_after_mean(updates: list[dict[str, Any]]) -> float:
    return _safe_mean([_safe_float(row.get("trust_after"), math.nan) for row in updates])


def _trust_reward_mean(updates: list[dict[str, Any]]) -> float:
    values = []
    for row in updates:
        reward = row.get("reward") or {}
        if isinstance(reward, dict):
            values.append(_safe_float(reward.get("reward"), math.nan))
    return _safe_mean(values)


def _trace_paths(graph_state: dict[str, Any], outcome_iteration: int, patch_iteration: int) -> dict[str, Any]:
    patch_record = graph_state.get("iterations", {}).get(_iter_key(patch_iteration), {})
    outcome_record = graph_state.get("iterations", {}).get(_iter_key(outcome_iteration), {})
    artifacts = outcome_record.get("artifacts", {}) if isinstance(outcome_record, dict) else {}
    patch_artifacts = patch_record.get("artifacts", {}) if isinstance(patch_record, dict) else {}

    def artifact_path(rows: dict[str, Any], name: str) -> str | None:
        row = rows.get(name) if isinstance(rows, dict) else None
        if isinstance(row, dict):
            return row.get("path")
        return None

    patch = patch_record.get("patch", {}) if isinstance(patch_record, dict) else {}
    return {
        "model_path": outcome_record.get("model_path") if isinstance(outcome_record, dict) else None,
        "result_path": artifact_path(artifacts, "result"),
        "metrics_path": artifact_path(artifacts, "metrics"),
        "feedback_path": artifact_path(artifacts, "feedback_vector"),
        "routing_path": artifact_path(artifacts, "routing"),
        "report_path": artifact_path(artifacts, "report"),
        "diff_path": patch.get("diff_path") or artifact_path(patch_artifacts, "diff_path"),
        "output_model_path": patch.get("output_model_path") or artifact_path(patch_artifacts, "output_model_path"),
    }


def _compact_strategy_snapshot(outcome_row: dict[str, Any], patch: dict[str, Any], stagnated: bool) -> dict[str, Any]:
    return {
        "primary_component": outcome_row.get("primary_component"),
        "active_components": outcome_row.get("active_components") or [],
        "selected_component": _selected_edit_field(patch, "component"),
        "selected_edit_operator": _selected_edit_field(patch, "edit_operator"),
        "negative_memory_count": len(patch.get("negative_memory") or []),
        "negative_suppression_count": len(patch.get("negative_reuse_suppression") or []),
        "controlled_exploration": patch.get("controlled_exploration") or {},
        "relation_attention": patch.get("relation_attention") or {},
        "stagnated_window": stagnated,
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _signature(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dominant_diagnostic") or "unknown"),
        str(row.get("component") or "unknown"),
        str(row.get("edit_action") or "unknown"),
    )


def _sum_float(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_safe_float(row.get(key), 0.0) for row in rows)


def _trace_region_row(
    region_type: str,
    signature: tuple[str, str, str],
    rows: list[dict[str, Any]],
    score: float,
    support_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    diagnostic, component, edit_action = signature
    support = support_rows or rows
    mae_gain = _sum_float(support, "paper_mae_delta")
    mse_gain = _sum_float(support, "paper_mse_delta")
    target_gain = _sum_float(support, "target_delta")
    improved_count = sum(1 for row in rows if row.get("improved"))
    invalid_count = sum(1 for row in rows if row.get("invalid_edit"))
    repeated_count = sum(1 for row in rows if row.get("repeated_useless_edit"))
    repair_count = sum(int(row.get("repair_attempt_count") or 0) for row in rows)
    if mae_gain > 0 and mse_gain > 0:
        metric_scope = "joint"
    elif mae_gain > 0:
        metric_scope = "mae_only"
    elif mse_gain > 0:
        metric_scope = "mse_only"
    else:
        metric_scope = "non_improving"
    iterations = sorted({int(row.get("outcome_iteration", 0)) for row in support})
    return {
        "region_type": region_type,
        "signature_id": " | ".join(signature),
        "diagnostic": diagnostic,
        "component": component,
        "edit_action": edit_action,
        "attempt_count": len(rows),
        "support_count": len(support),
        "improved_count": improved_count,
        "best_improvement_count": sum(1 for row in rows if row.get("improved_vs_best_so_far")),
        "invalid_count": invalid_count,
        "repeated_useless_count": repeated_count,
        "repair_attempt_count": repair_count,
        "total_target_delta": target_gain,
        "total_paper_mae_delta": mae_gain,
        "total_paper_mse_delta": mse_gain,
        "mean_target_delta": _safe_mean([_safe_float(row.get("target_delta"), math.nan) for row in support]),
        "mean_paper_mae_delta": _safe_mean([_safe_float(row.get("paper_mae_delta"), math.nan) for row in support]),
        "mean_paper_mse_delta": _safe_mean([_safe_float(row.get("paper_mse_delta"), math.nan) for row in support]),
        "metric_scope": metric_scope,
        "score": score,
        "first_outcome_iteration": min(iterations) if iterations else None,
        "last_outcome_iteration": max(iterations) if iterations else None,
        "evidence_iterations": iterations[:12],
    }


def _build_trace_consolidation(attempts: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
    """Compress execution attempts into core/trap/repair evidence regions."""

    total = len(attempts)
    if not total:
        return {
            "schema": "forge.trace_consolidation.v1",
            "metrics": {},
            "productive_core": [],
            "trap_regions": [],
            "repair_paths": [],
            "adaptive_task_card": {},
            "trace_region_table": [],
        }

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        groups[_signature(row)].append(row)

    productive_rows: list[dict[str, Any]] = []
    trap_rows: list[dict[str, Any]] = []
    for signature, rows in groups.items():
        support = [
            row
            for row in rows
            if row.get("improved") and not row.get("invalid_edit")
        ]
        invalid_count = sum(1 for row in rows if row.get("invalid_edit"))
        repeated_count = sum(1 for row in rows if row.get("repeated_useless_edit"))
        non_improved_count = sum(1 for row in rows if not row.get("improved"))
        negative_mae = sum(max(0.0, -_safe_float(row.get("paper_mae_delta"), 0.0)) for row in rows)
        negative_mse = sum(max(0.0, -_safe_float(row.get("paper_mse_delta"), 0.0)) for row in rows)

        if support:
            mae_gain = _sum_float(support, "paper_mae_delta")
            mse_gain = _sum_float(support, "paper_mse_delta")
            best_gain_count = sum(1 for row in support if row.get("improved_vs_best_so_far"))
            risk_penalty = 0.25 * invalid_count + 0.10 * repeated_count
            joint_bonus = min(max(mae_gain, 0.0), max(mse_gain, 0.0))
            score = mae_gain + 0.25 * mse_gain + 0.50 * joint_bonus + best_gain_count - risk_penalty
            productive_rows.append(_trace_region_row("productive_core", signature, rows, score, support))

        trap_signal = invalid_count + repeated_count + non_improved_count
        if trap_signal:
            score = (
                2.0 * invalid_count
                + repeated_count
                + 0.25 * non_improved_count
                + negative_mae
                + 0.05 * negative_mse
            )
            trap_rows.append(_trace_region_row("trap_region", signature, rows, score, rows))

    repair_paths = []
    for row in attempts:
        repair_attempt_count = int(row.get("repair_attempt_count") or 0)
        if repair_attempt_count <= 0 and not row.get("validation_fallback"):
            continue
        target_delta = _safe_float(row.get("target_delta"), 0.0)
        status = "stabilized" if row.get("success") and target_delta >= 0.0 else "failed_or_regressed"
        repair_paths.append(
            {
                "region_type": "repair_path",
                "status": status,
                "outcome_iteration": row.get("outcome_iteration"),
                "parent_iteration": row.get("parent_iteration"),
                "diagnostic": row.get("dominant_diagnostic"),
                "component": row.get("component"),
                "edit_action": row.get("edit_action"),
                "repair_attempt_count": repair_attempt_count,
                "validation_fallback": bool(row.get("validation_fallback")),
                "target_delta": target_delta,
                "paper_mae_delta": _safe_float(row.get("paper_mae_delta"), 0.0),
                "paper_mse_delta": _safe_float(row.get("paper_mse_delta"), 0.0),
            }
        )

    productive_rows.sort(key=lambda row: row["score"], reverse=True)
    trap_rows.sort(key=lambda row: row["score"], reverse=True)
    repair_paths.sort(
        key=lambda row: (
            row["status"] != "stabilized",
            -int(row.get("repair_attempt_count") or 0),
            -int(row.get("outcome_iteration") or 0),
        )
    )

    improved_count = sum(1 for row in attempts if row.get("improved"))
    invalid_count = sum(1 for row in attempts if row.get("invalid_edit"))
    repeated_count = sum(1 for row in attempts if row.get("repeated_useless_edit"))
    repair_event_count = len(repair_paths)
    joint_core_count = sum(1 for row in productive_rows if row.get("metric_scope") == "joint")

    preferred_components = []
    for row in productive_rows[:limit]:
        item = {
            "component": row["component"],
            "edit_action": row["edit_action"],
            "metric_scope": row["metric_scope"],
            "score": row["score"],
        }
        if item not in preferred_components:
            preferred_components.append(item)
    cooldown_edits = [
        {
            "component": row["component"],
            "edit_action": row["edit_action"],
            "reason": (
                f"invalid={row['invalid_count']}, repeated={row['repeated_useless_count']}, "
                f"non-improving attempts={row['attempt_count'] - row['improved_count']}"
            ),
        }
        for row in trap_rows[:limit]
    ]
    risk_flags = []
    if _rate(invalid_count, total) >= 0.10:
        risk_flags.append("high_invalid_edit_rate")
    if _rate(repeated_count, total) >= 0.25:
        risk_flags.append("high_repeated_useless_edit_rate")
    if _rate(improved_count, total) <= 0.10:
        risk_flags.append("low_core_access_rate")
    if joint_core_count < max(1, len(productive_rows[:limit]) // 3):
        risk_flags.append("mae_mse_tradeoff_requires_joint_selection")

    adaptive_task_card = {
        "preferred_components": preferred_components[:5],
        "cooldown_edits": cooldown_edits[:5],
        "risk_flags": risk_flags,
        "selection_hint": (
            "Use non-destructive final selection; prefer a joint MAE/MSE core when target-only "
            "improvements trade off against MSE."
        ),
        "next_round_prior": (
            "Prioritize productive_core signatures, suppress trap_region repeats, and use repair_path "
            "evidence only for implementation repair rather than component credit."
        ),
    }

    productive_top = productive_rows[:limit]
    trap_top = trap_rows[:limit]
    repair_top = repair_paths[:limit]
    trace_region_table = productive_top + trap_top + repair_top
    return {
        "schema": "forge.trace_consolidation.v1",
        "metrics": {
            "attempt_count": total,
            "productive_core_count": len(productive_rows),
            "trap_region_count": len(trap_rows),
            "repair_path_count": repair_event_count,
            "core_access_rate": _rate(improved_count, total),
            "trap_exposure_rate": _rate(
                sum(1 for row in attempts if row.get("invalid_edit") or row.get("repeated_useless_edit") or not row.get("improved")),
                total,
            ),
            "repair_event_rate": _rate(repair_event_count, total),
            "joint_core_count": joint_core_count,
        },
        "productive_core": productive_top,
        "trap_regions": trap_top,
        "repair_paths": repair_top,
        "adaptive_task_card": adaptive_task_card,
        "trace_region_table": trace_region_table,
    }


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
    best_so_far = initial_target

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
        improved_vs_best_so_far = bool(success and math.isfinite(best_so_far) and outcome_target < best_so_far - 1e-12)
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
        trust_updates = _matching_trust_updates(graph_state, outcome_iteration, patch)
        relation_ids = _route_relation_ids(patch)
        trace_paths = _trace_paths(graph_state, outcome_iteration, patch_iteration)
        parent_iteration = patch.get("parent_iteration")
        try:
            parent_iteration_int = int(parent_iteration) if parent_iteration is not None else patch_iteration
        except Exception:
            parent_iteration_int = patch_iteration
        branch_mode = "best_so_far_parent" if parent_iteration_int != patch_iteration else "last_parent"
        parent_result = rows_by_iter.get(parent_iteration_int, {}).get("result") or {}
        parent_paper_mae = _paper_mae(parent_result)
        parent_paper_mse = _paper_mse(parent_result)
        outcome_paper_mae = _paper_mae(outcome_result)
        outcome_paper_mse = _paper_mse(outcome_result)
        target_delta = (
            parent_target - outcome_target
            if math.isfinite(parent_target) and math.isfinite(outcome_target)
            else 0.0
        )
        paper_mae_delta = (
            parent_paper_mae - outcome_paper_mae
            if math.isfinite(parent_paper_mae) and math.isfinite(outcome_paper_mae)
            else 0.0
        )
        paper_mse_delta = (
            parent_paper_mse - outcome_paper_mse
            if math.isfinite(parent_paper_mse) and math.isfinite(outcome_paper_mse)
            else 0.0
        )
        active_memory_state = _compact_strategy_snapshot(outcome_row, patch, False)

        attempts.append(
            {
                "patch_iteration": patch_iteration,
                "outcome_iteration": outcome_iteration,
                "parent_iteration": parent_iteration_int,
                "branch_mode": branch_mode,
                "success": success,
                "component": component,
                "routed_component": routed_component or None,
                "selected_component": _selected_edit_field(patch, "component"),
                "selected_edit_operator": _selected_edit_field(patch, "edit_operator"),
                "edit_action": edit_action,
                "dominant_diagnostic": dominant_diagnostic,
                "relation_ids": relation_ids,
                "route_relation_count": len(relation_ids),
                "trust_before_mean": _trust_before_mean(patch),
                "trust_after_mean": _trust_after_mean(trust_updates),
                "trust_reward_mean": _trust_reward_mean(trust_updates),
                "parent_target": parent_target,
                "outcome_target": outcome_target,
                "target_delta": target_delta,
                "parent_paper_mae": parent_paper_mae,
                "outcome_paper_mae": outcome_paper_mae,
                "paper_mae_delta": paper_mae_delta,
                "parent_paper_mse": parent_paper_mse,
                "outcome_paper_mse": outcome_paper_mse,
                "paper_mse_delta": paper_mse_delta,
                "improved": improved,
                "improved_vs_best_so_far": improved_vs_best_so_far,
                "invalid_edit": invalid,
                "repeated_useless_edit": repeated_useless,
                "negative_memory_count": len(patch.get("negative_memory") or []),
                "negative_suppression_count": len(patch.get("negative_reuse_suppression") or []),
                "repair_attempt_count": len(repair_attempts),
                "edit_operator_mismatch": bool(patch.get("edit_operator_mismatch")),
                "component_mismatch": bool(patch.get("component_mismatch")),
                "validation_fallback": bool(patch.get("validation_fallback")),
                "evidence_aligned": aligned,
                "active_memory_state": active_memory_state,
                "trace_paths": trace_paths,
            }
        )
        if success and math.isfinite(outcome_target):
            best_so_far = min(best_so_far, outcome_target)

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

    relation_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    component_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        relation_ids = row.get("relation_ids") or []
        if relation_ids:
            for relation_id in relation_ids:
                parts = str(relation_id).split("->", 1)
                diagnostic = parts[0] if parts else str(row.get("dominant_diagnostic") or "unknown")
                component = parts[1] if len(parts) > 1 else str(row.get("routed_component") or row.get("component") or "unknown")
                key = (diagnostic, component, str(row.get("edit_action") or "unknown"), str(relation_id))
                relation_groups[key].append(row)
        else:
            relation_id = (
                f"{row.get('dominant_diagnostic') or 'unknown'}"
                f"->{row.get('routed_component') or row.get('component') or 'unknown'}"
            )
            key = (
                str(row.get("dominant_diagnostic") or "unknown"),
                str(row.get("routed_component") or row.get("component") or "unknown"),
                str(row.get("edit_action") or "unknown"),
                relation_id,
            )
            relation_groups[key].append(row)
        component_groups[str(row.get("component") or "unknown")].append(row)

    relation_table = []
    for (diagnostic, component, edit_action, relation_id), rows in sorted(relation_groups.items()):
        relation_table.append(
            {
                "relation_id": relation_id,
                "diagnostic": diagnostic,
                "component": component,
                "edit_action": edit_action,
                "attempt_count": len(rows),
                "success_count": sum(1 for row in rows if row.get("success")),
                "improved_count": sum(1 for row in rows if row.get("improved")),
                "best_improvement_count": sum(1 for row in rows if row.get("improved_vs_best_so_far")),
                "invalid_count": sum(1 for row in rows if row.get("invalid_edit")),
                "repeated_useless_count": sum(1 for row in rows if row.get("repeated_useless_edit")),
                "mean_target_delta": _safe_mean([_safe_float(row.get("target_delta"), math.nan) for row in rows]),
                "mean_paper_mae_delta": _safe_mean([_safe_float(row.get("paper_mae_delta"), math.nan) for row in rows]),
                "mean_paper_mse_delta": _safe_mean([_safe_float(row.get("paper_mse_delta"), math.nan) for row in rows]),
                "mean_trust_before": _safe_mean([_safe_float(row.get("trust_before_mean"), math.nan) for row in rows]),
                "mean_trust_after": _safe_mean([_safe_float(row.get("trust_after_mean"), math.nan) for row in rows]),
                "mean_trust_reward": _safe_mean([_safe_float(row.get("trust_reward_mean"), math.nan) for row in rows]),
                "last_outcome_iteration": max(int(row.get("outcome_iteration", 0)) for row in rows),
            }
        )

    component_table = []
    for component, rows in sorted(component_groups.items()):
        component_table.append(
            {
                "component": component,
                "attempt_count": len(rows),
                "success_count": sum(1 for row in rows if row.get("success")),
                "improved_count": sum(1 for row in rows if row.get("improved")),
                "best_improvement_count": sum(1 for row in rows if row.get("improved_vs_best_so_far")),
                "invalid_count": sum(1 for row in rows if row.get("invalid_edit")),
                "repeated_useless_count": sum(1 for row in rows if row.get("repeated_useless_edit")),
                "mean_target_delta": _safe_mean([_safe_float(row.get("target_delta"), math.nan) for row in rows]),
                "mean_paper_mae_delta": _safe_mean([_safe_float(row.get("paper_mae_delta"), math.nan) for row in rows]),
                "mean_paper_mse_delta": _safe_mean([_safe_float(row.get("paper_mse_delta"), math.nan) for row in rows]),
                "dominant_diagnostics": sorted(
                    {
                        str(row.get("dominant_diagnostic"))
                        for row in rows
                        if row.get("dominant_diagnostic")
                    }
                ),
                "last_outcome_iteration": max(int(row.get("outcome_iteration", 0)) for row in rows),
            }
        )

    strategy_timeline = [
        {
            "outcome_iteration": row["outcome_iteration"],
            "parent_iteration": row["parent_iteration"],
            "branch_mode": row["branch_mode"],
            "dominant_diagnostic": row.get("dominant_diagnostic"),
            "routed_component": row.get("routed_component"),
            "selected_component": row.get("selected_component"),
            "edit_action": row.get("edit_action"),
            "improved": row.get("improved"),
            "improved_vs_best_so_far": row.get("improved_vs_best_so_far"),
            "invalid_edit": row.get("invalid_edit"),
            "negative_memory_count": row.get("negative_memory_count"),
            "negative_suppression_count": row.get("negative_suppression_count"),
            "trust_before_mean": row.get("trust_before_mean"),
            "trust_after_mean": row.get("trust_after_mean"),
            "target_delta": row.get("target_delta"),
        }
        for row in attempts
    ]
    trace_consolidation = _build_trace_consolidation(attempts)

    method_evidence_table = [
        {
            "claim": "active_memory_reconstruction",
            "saved_evidence": "strategy_timeline, relation_table, trust_before/after, diagnostic route history",
            "primary_metrics": "routing_stability, evidence_alignment, mean_trust_reward",
            "artifact": "evidence/evidence_strategy_timeline.csv; evidence/evidence_relations.csv",
        },
        {
            "claim": "test_time_adaptation",
            "saved_evidence": "per-iteration strategy state and refresh triggers",
            "primary_metrics": "improvement_rate, best_improvement_count, stagnation trigger",
            "artifact": "summary.json/evidence_audit.strategy_memory; evidence/evidence_strategy_timeline.csv",
        },
        {
            "claim": "experience_reuse",
            "saved_evidence": "negative memory, suppression count, repeated useless edit count, trusted components",
            "primary_metrics": "repeated_useless_edit_rate, invalid_edit_rate, trusted component success counts",
            "artifact": "evidence/evidence_attempts.csv; evidence/evidence_components.csv",
        },
        {
            "claim": "outcome_calibrated_trace_consolidation",
            "saved_evidence": "productive core, trap region, repair path, adaptive task card",
            "primary_metrics": "core_access_rate, trap_exposure_rate, repair_event_rate, joint_core_count",
            "artifact": "evidence/evidence_trace_regions.csv; summary.json/evidence_audit.trace_consolidation",
        },
        {
            "claim": "graph_branch_level_search",
            "saved_evidence": "parent_iteration, branch_mode, protected best parent selection",
            "primary_metrics": "best_so_far_parent count, attempts_to_best, budget_efficiency",
            "artifact": "evidence/evidence_attempts.csv",
        },
        {
            "claim": "domain_native_harness",
            "saved_evidence": "same harness metrics, result paths, feedback vectors, validation fallback flags",
            "primary_metrics": "invalid_edit_rate, success_count, benchmark-scaled MAE/MSE deltas",
            "artifact": "iter_*/metrics.json; evidence/evidence_attempts.csv",
        },
        {
            "claim": "auditable_trajectories",
            "saved_evidence": "model path, diff path, feedback path, routing path, report path",
            "primary_metrics": "complete trace availability per attempt",
            "artifact": "evidence/evidence_attempts.csv; task_graph.json; graph_events.jsonl",
        },
    ]

    return {
        "schema": "forge.evidence_audit.v1",
        "method_framework": method_framework(candidate_tournament_k),
        "metrics": metrics,
        "routing_stability_by_diagnostic": stability_rows,
        "strategy_memory": strategy_memory,
        "trace_consolidation": trace_consolidation,
        "tables": {
            "attempts": attempts,
            "relations": relation_table,
            "components": component_table,
            "strategy_timeline": strategy_timeline,
            "trace_regions": trace_consolidation.get("trace_region_table", []),
            "method_evidence": method_evidence_table,
        },
        "attempts_tail": attempts[-12:],
    }
