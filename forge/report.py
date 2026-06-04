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
        if selected_edit.get("prompt_guidance"):
            lines.append(f"- Guidance: {selected_edit.get('prompt_guidance')}")
        lines.append("")
    candidates = route.get("edit_candidates") or []
    if candidates:
        lines.extend(["### Top Edit Candidates", ""])
        for item in candidates[:5]:
            lines.append(
                f"- `{item.get('diagnostic')}` -> `{item.get('component')}` :: `{item.get('edit_operator')}` "
                f"score={item.get('score')} trust={item.get('operator_trust')} negatives={item.get('negative_count')}"
            )
        lines.append("")
    negative_memory = route.get("negative_memory") or []
    if negative_memory:
        lines.extend(["### Negative Memory Reused", ""])
        for item in negative_memory[:5]:
            lines.append(
                f"- `{item.get('relation_id')}` trust={item.get('trust')} "
                f"negative_count={item.get('negative_count')} validation_failures={item.get('validation_failures')}"
            )
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
            lines.append(
                f"- `{item.get('diagnostic')}` -> `{item.get('component')}` :: `{item.get('edit_operator')}` "
                f"{item.get('direction')}: {item.get('trust_before')} -> {item.get('trust_after')} "
                f"reward={reward.get('reward')} probe_delta={reward.get('diagnostic_delta')} "
                f"status={item.get('candidate_status')}"
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
