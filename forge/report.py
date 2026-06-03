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
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
