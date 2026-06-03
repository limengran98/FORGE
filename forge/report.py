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
    if patch_meta:
        lines.extend(
            [
                "## Patch For Next Iteration",
                "",
                f"- Origin: `{patch_meta.get('origin')}`",
                f"- Component: `{patch_meta.get('component')}`",
                f"- Summary: {patch_meta.get('summary')}",
                f"- Diff: `{patch_meta.get('diff_path')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")

