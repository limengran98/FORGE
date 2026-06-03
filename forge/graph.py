from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_json, save_json
from .harness_spec import get_component_graph


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def initial_task_graph() -> dict[str, Any]:
    component_graph = get_component_graph()
    tasks = {}
    for idx, node in enumerate(component_graph["nodes"]):
        tasks[node] = {
            "id": node,
            "name": node,
            "status": "active",
            "priority": 0.5 if idx else 0.3,
            "uncertainty": 0.6,
            "evidence": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
    return {
        "version": 1,
        "tasks": tasks,
        "edges": component_graph["edges"],
        "iteration_map": {},
        "created_at": _now(),
    }


def load_or_init_graph(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return load_json(path)
    graph = initial_task_graph()
    save_json(graph, path)
    return graph


def update_graph(
    graph: dict[str, Any],
    iteration: int,
    route: dict[str, Any],
    result: dict[str, Any],
    feedback: dict[str, Any],
) -> None:
    active = route.get("active_components") or [route.get("primary_component")]
    active = [node for node in active if node]
    graph.setdefault("iteration_map", {})[f"iter_{iteration:03d}"] = active
    current_target = feedback.get("current_target")
    improved = bool(feedback.get("features", {}).get("improved_vs_best", 0.0) > 0)
    success = bool(result.get("success"))
    evidence = {
        "iteration": iteration,
        "success": success,
        "target_metric": feedback.get("target_metric"),
        "target_value": current_target,
        "primary_component": route.get("primary_component"),
        "improved_vs_best": improved,
        "result_path": result.get("paths", {}).get("result"),
        "ts": _now(),
    }
    for node in active:
        task = graph.setdefault("tasks", {}).setdefault(node, {"id": node, "name": node})
        task.setdefault("evidence", []).append(evidence)
        priority = float(task.get("priority", 0.5))
        uncertainty = float(task.get("uncertainty", 0.6))
        if success and improved:
            task["priority"] = max(0.0, priority - 0.04)
            task["uncertainty"] = max(0.0, uncertainty - 0.06)
        elif not success:
            task["priority"] = min(1.0, priority + 0.08)
            task["uncertainty"] = min(1.0, uncertainty + 0.08)
        else:
            task["priority"] = min(1.0, priority + 0.02)
            task["uncertainty"] = min(1.0, uncertainty + 0.02)
        task["updated_at"] = _now()
