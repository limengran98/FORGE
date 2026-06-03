from __future__ import annotations

from datetime import datetime
from typing import Any

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
