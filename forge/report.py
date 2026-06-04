from __future__ import annotations

from pathlib import Path
from typing import Any


def write_iteration_report(
    path: str | Path,
    iteration: int,
    result: dict[str, Any],
    feedback: dict[str, Any],
    route: dict[str, Any],
    patch_meta: dict[str, Any] | None = None,
    trust_updates: list[dict[str, Any]] | None = None,
    action_memory_updates: list[dict[str, Any]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# FORGE Iteration {iteration:03d}",
        "",
        f"- Success: {result.get('success')}",
        f"- Run dir: `{result.get('run_dir')}`",
        f"- Target metric: `{feedback.get('target_metric')}`",
        f"- Current target: `{feedback.get('current_target')}`",
        f"- Previous target: `{feedback.get('previous_target')}`",
        f"- Best target: `{feedback.get('best_target')}`",
        f"- Primary routed component: `{route.get('primary_component')}`",
        f"- Active components: `{', '.join(route.get('active_components') or [])}`",
        "",
        "## Metrics",
        "",
    ]
    metrics = result.get("metrics", {})
    for scope in ("normalized", "inverse"):
        if scope in metrics:
            lines.append(f"### {scope}")
            for key, value in metrics[scope].items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    if not result.get("success"):
        lines.extend(
            [
                "## Error",
                "",
                f"- Type: `{result.get('error_type')}`",
                f"- Message: `{result.get('error_message')}`",
                "",
            ]
        )
    if route.get("reasons"):
        lines.extend(["## Routing Reasons", ""])
        for component, reasons in route["reasons"].items():
            lines.append(f"### {component}")
            for reason in reasons:
                lines.append(f"- {reason}")
            lines.append("")
    diagnostics = feedback.get("diagnostics") or []
    if diagnostics:
        lines.extend(["## Diagnostic Feedback", ""])
        for item in sorted(diagnostics, key=lambda row: row.get("severity", 0.0) * row.get("confidence", 0.0), reverse=True):
            lines.append(
                f"- `{item.get('name')}` severity={item.get('severity')} "
                f"confidence={item.get('confidence')}"
            )
        lines.append("")
    propagations = route.get("propagations") or []
    if propagations:
        lines.extend(["## Trust-Guided Propagation", ""])
        for item in propagations[:8]:
            lines.append(
                f"- `{item.get('diagnostic')}` -> `{item.get('component')}` "
                f"trust={item.get('trust')} contribution={item.get('contribution')}"
            )
        lines.append("")
    selected_edit = route.get("selected_edit")
    if selected_edit:
        lines.extend(["## Relation-Level Edit Selection", ""])
        lines.append(
            f"- Selected: `{selected_edit.get('diagnostic')}` -> `{selected_edit.get('component')}` "
            f":: `{selected_edit.get('edit_operator')}` score={selected_edit.get('score')} "
            f"operator_trust={selected_edit.get('operator_trust')}"
        )
        if selected_edit.get("attention_weight") is not None:
            lines.append(
                f"- Attention: weight={selected_edit.get('attention_weight')} "
                f"score={selected_edit.get('attention_score')} tau={selected_edit.get('relation_temperature')}"
            )
        if selected_edit.get("attention_sampled"):
            lines.append(f"- Attention sample: `{selected_edit.get('attention_sample_reason')}`")
        if selected_edit.get("exploration_selected"):
            lines.append(f"- Controlled exploration: `{selected_edit.get('exploration_reason')}`")
        if selected_edit.get("prompt_guidance"):
            lines.append(f"- Guidance: {selected_edit.get('prompt_guidance')}")
        lines.append("")
    attention = route.get("relation_attention") or {}
    if attention:
        lines.extend(["### Trust-Calibrated Relation Attention", ""])
        lines.append(
            f"- Enabled: `{bool(attention.get('enabled'))}` temperature=`{attention.get('temperature')}` "
            f"entropy=`{attention.get('weights_entropy')}` selected_by=`{attention.get('selected_by')}`"
        )
        factors = attention.get("factors") or {}
        if factors:
            factor_text = ", ".join(f"{key}={value}" for key, value in factors.items())
            lines.append(f"- Temperature factors: {factor_text}")
        lines.append("")
    candidates = route.get("edit_candidates") or []
    if candidates:
        lines.extend(["### Top Edit Candidates", ""])
        for item in candidates[:5]:
            lines.append(
                f"- `{item.get('diagnostic')}` -> `{item.get('component')}` :: `{item.get('edit_operator')}` "
                f"score={item.get('score')} attention={item.get('attention_weight')} "
                f"trust={item.get('operator_trust')} negatives={item.get('negative_count')}"
            )
        lines.append("")
    negative_memory = route.get("negative_memory") or []
    if negative_memory:
        lines.extend(["### Negative Memory Reused", ""])
        for item in negative_memory[:5]:
            lines.append(
                f"- `{item.get('relation_id')}` trust={item.get('trust')} "
                f"negative_count={item.get('negative_count')} dataset_negatives={item.get('dataset_negative_count')} "
                f"cooldown={item.get('cooldown_remaining')} blocked={item.get('blocked')} "
                f"validation_failures={item.get('validation_failures')}"
            )
        lines.append("")
    suppressions = route.get("negative_reuse_suppression") or []
    if suppressions:
        lines.extend(["### Negative Reuse Suppression", ""])
        for item in suppressions[:5]:
            suppression = item.get("suppression", {})
            lines.append(
                f"- `{item.get('relation_id')}` score={item.get('score')} "
                f"penalty={suppression.get('total_penalty')} cooldown={suppression.get('cooldown_remaining')} "
                f"blocked={suppression.get('blocked')}"
            )
        lines.append("")
    exploration = route.get("controlled_exploration") or {}
    if exploration:
        lines.extend(["### Controlled Exploration", ""])
        lines.append(f"- Enabled: `{bool(exploration.get('enabled'))}`")
        lines.append(f"- Selected exploration candidate: `{bool(exploration.get('selected'))}`")
        lines.append("")
    if trust_updates:
        lines.extend(["## Trust Updates From Executable Outcome", ""])
        for item in trust_updates:
            reward = item.get("reward", {})
            lines.append(
                f"- `{item.get('diagnostic')}` -> `{item.get('component')}` "
                f"{item.get('direction')}: {item.get('trust_before')} -> {item.get('trust_after')} "
                f"reward={reward.get('reward')}"
            )
        lines.append("")
    if action_memory_updates:
        lines.extend(["## Action Memory Updates From Probe-Aligned Outcome", ""])
        for item in action_memory_updates:
            reward = item.get("reward", {})
            agreement = item.get("trust_outcome_agreement", {})
            lines.append(
                f"- `{item.get('diagnostic')}` -> `{item.get('component')}` :: `{item.get('edit_operator')}` "
                f"{item.get('direction')}: {item.get('trust_before')} -> {item.get('trust_after')} "
                f"policy={item.get('evidence_policy')} reward={reward.get('reward')} raw={reward.get('raw_reward')} "
                f"probe_delta={reward.get('diagnostic_delta')} status={item.get('candidate_status')} "
                f"agreement={agreement.get('label')}"
            )
        lines.append("")
    if patch_meta:
        lines.extend(
            [
                "## Patch For Next Iteration",
                "",
                f"- Origin: `{patch_meta.get('origin')}`",
                f"- Component: `{patch_meta.get('component')}`",
                f"- Edit action: `{patch_meta.get('edit_action')}`",
                f"- Summary: {patch_meta.get('summary')}",
                f"- Parent model: `{patch_meta.get('parent_model_path')}`",
                f"- Diff: `{patch_meta.get('diff_path')}`",
                f"- Validation fallback: `{bool(patch_meta.get('validation_fallback', False))}`",
                f"- Edit operator mismatch: `{bool(patch_meta.get('edit_operator_mismatch', False))}`",
                "",
            ]
        )
        if patch_meta.get("selected_edit"):
            selected = patch_meta["selected_edit"]
            lines.extend(
                [
                    "### Selected Edit Contract",
                    "",
                    f"- Relation: `{selected.get('diagnostic')}` -> `{selected.get('component')}` :: `{selected.get('edit_operator')}`",
                    f"- Score: `{selected.get('score')}`",
                    f"- Operator trust before edit: `{selected.get('operator_trust')}`",
                    "",
                ]
            )
        attempts = patch_meta.get("repair_attempts") or []
        if attempts:
            lines.extend(["### Patch Repair Attempts", ""])
            for item in attempts:
                lines.append(
                    f"- attempt={item.get('attempt')} origin=`{item.get('origin')}` "
                    f"source=`{item.get('source_path')}` error={item.get('validation_error')}"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
