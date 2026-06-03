from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .config import load_json, save_json
from .graph import initial_task_graph
from .harness_spec import get_component_graph, get_iteration_stages, load_orchestration_spec


def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _iter_key(iteration: int) -> str:
    return f"iter_{iteration:03d}"


class GraphStateError(RuntimeError):
    pass


class GraphOrchestrator:
    """Small durable orchestration layer for FORGE iteration graphs.

    It owns the run-level graph state and event log. Model training failures are
    recorded as feedback, while orchestration failures mean the control flow
    itself broke.
    """

    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        spec = load_orchestration_spec()
        self.stages = get_iteration_stages()
        self.graph_path = self.run_root / str(spec.get("graph_state", "task_graph.json"))
        self.events_path = self.run_root / str(spec.get("event_log", "graph_events.jsonl"))
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_init()
        self.validate()

    @classmethod
    def open(cls, run_root: str | Path) -> "GraphOrchestrator":
        return cls(run_root)

    def _load_or_init(self) -> dict[str, Any]:
        if self.graph_path.exists():
            state = load_json(self.graph_path)
        else:
            state = initial_task_graph()
        state.setdefault("schema", "forge.graph.v1")
        state.setdefault("orchestration", {"version": 1, "stages": self.stages})
        state.setdefault("iterations", {})
        state.setdefault("iteration_map", {})
        state.setdefault("events_count", 0)
        state.setdefault("created_at", _now())
        state["updated_at"] = _now()
        self._ensure_component_graph(state)
        save_json(state, self.graph_path)
        return state

    def _ensure_component_graph(self, state: dict[str, Any]) -> None:
        component_graph = get_component_graph()
        tasks = state.setdefault("tasks", {})
        for node in component_graph["nodes"]:
            tasks.setdefault(
                node,
                {
                    "id": node,
                    "name": node,
                    "status": "active",
                    "priority": 0.5,
                    "uncertainty": 0.6,
                    "evidence": [],
                    "created_at": _now(),
                    "updated_at": _now(),
                },
            )
        state["edges"] = component_graph["edges"]

    def validate(self) -> None:
        required = ["tasks", "edges", "iterations", "orchestration"]
        missing = [key for key in required if key not in self.state]
        if missing:
            raise GraphStateError(f"Graph state missing keys: {missing}")
        known = set(self.state["tasks"])
        for edge in self.state.get("edges", []):
            if edge.get("from") not in known or edge.get("to") not in known:
                raise GraphStateError(f"Graph edge references unknown node: {edge}")
        for stage in self.stages:
            if not isinstance(stage, str) or not stage:
                raise GraphStateError("Iteration stages must be non-empty strings")

    def _require_stage(self, stage_name: str) -> None:
        if stage_name not in self.stages:
            raise GraphStateError(f"Unknown orchestration stage: {stage_name}")

    def _iteration_record(self, iteration: int) -> dict[str, Any]:
        key = _iter_key(iteration)
        try:
            return self.state["iterations"][key]
        except KeyError as exc:
            raise GraphStateError(f"Iteration {key} has not been initialized") from exc

    def save(self) -> None:
        self.state["updated_at"] = _now()
        save_json(self.state, self.graph_path)

    def _resolve_artifact_path(self, path: str | Path) -> Path:
        artifact_path = Path(path).expanduser()
        if artifact_path.is_absolute():
            return artifact_path

        candidates: list[Path] = []
        parts = artifact_path.parts
        if self.run_root.name in parts:
            index = parts.index(self.run_root.name)
            suffix = Path(*parts[index + 1 :]) if index + 1 < len(parts) else Path()
            candidates.append(self.run_root / suffix)
        candidates.extend([self.run_root / artifact_path, Path.cwd() / artifact_path])

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        row = {
            "ts": _now(),
            "event_type": event_type,
            "payload": payload or {},
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.state["events_count"] = int(self.state.get("events_count", 0)) + 1

    def ensure_iteration(self, iteration: int, iter_dir: str | Path, model_path: str | Path | None = None) -> dict[str, Any]:
        key = _iter_key(iteration)
        iterations = self.state.setdefault("iterations", {})
        record = iterations.setdefault(
            key,
            {
                "iteration": iteration,
                "status": "pending",
                "stages": {},
                "artifacts": {},
                "created_at": _now(),
            },
        )
        record["iter_dir"] = str(iter_dir)
        if model_path is not None:
            record["model_path"] = str(model_path)
        for stage in self.stages:
            record.setdefault("stages", {}).setdefault(stage, {"status": "pending"})
        record["updated_at"] = _now()
        self.save()
        return record

    @contextmanager
    def stage(self, iteration: int, stage_name: str, meta: dict[str, Any] | None = None) -> Iterator[None]:
        self._require_stage(stage_name)
        key = _iter_key(iteration)
        record = self.state.setdefault("iterations", {}).setdefault(
            key,
            {"iteration": iteration, "status": "pending", "stages": {}, "artifacts": {}, "created_at": _now()},
        )
        stage = record.setdefault("stages", {}).setdefault(stage_name, {})
        record["status"] = "running"
        stage.update({"status": "running", "started_at": _now(), "meta": meta or {}})
        self.event("stage_started", {"iteration": iteration, "stage": stage_name, "meta": meta or {}})
        self.save()
        try:
            yield
        except Exception as exc:
            stage.update(
                {
                    "status": "failed",
                    "ended_at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc()[-4000:],
                }
            )
            record["status"] = "failed"
            record["updated_at"] = _now()
            self.event("stage_failed", {"iteration": iteration, "stage": stage_name, "error": stage["error"]})
            self.save()
            raise
        else:
            stage.update({"status": "succeeded", "ended_at": _now()})
            record["updated_at"] = _now()
            self.event("stage_succeeded", {"iteration": iteration, "stage": stage_name})
            self.save()

    def skip_stage(self, iteration: int, stage_name: str, reason: str) -> None:
        self._require_stage(stage_name)
        self.ensure_iteration(iteration, self.run_root / _iter_key(iteration))
        record = self._iteration_record(iteration)
        stage = record.setdefault("stages", {}).setdefault(stage_name, {})
        stage.update({"status": "skipped", "reason": reason, "ended_at": _now()})
        record["updated_at"] = _now()
        self.event("stage_skipped", {"iteration": iteration, "stage": stage_name, "reason": reason})
        self.save()

    def record_artifact(self, iteration: int, name: str, path: str | Path, kind: str = "file") -> None:
        record = self._iteration_record(iteration)
        record.setdefault("artifacts", {})[name] = {"path": str(path), "kind": kind, "recorded_at": _now()}
        record["updated_at"] = _now()
        self.save()

    def record_result(self, iteration: int, result: dict[str, Any]) -> None:
        record = self._iteration_record(iteration)
        target = result.get("metrics", {}).get("target", {}) if result.get("success") else {}
        record["harness"] = {
            "success": bool(result.get("success")),
            "target": target,
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
            "duration_sec": result.get("duration_sec"),
            "device": result.get("device"),
        }
        for key, path in (result.get("paths") or {}).items():
            if path:
                self.record_artifact(iteration, key, path)
        self.event("harness_recorded", {"iteration": iteration, "success": bool(result.get("success")), "target": target})
        self.save()

    def record_feedback_and_route(
        self,
        iteration: int,
        feedback: dict[str, Any],
        route: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        record = self._iteration_record(iteration)
        record["feedback"] = {
            "target_metric": feedback.get("target_metric"),
            "current_target": feedback.get("current_target"),
            "previous_target": feedback.get("previous_target"),
            "best_target": feedback.get("best_target"),
            "features": feedback.get("features", {}),
        }
        record["route"] = {
            "primary_component": route.get("primary_component"),
            "active_components": route.get("active_components") or [],
            "routing_policy": route.get("routing_policy"),
        }
        self._update_component_evidence(iteration, route, result, feedback)
        self.event(
            "route_recorded",
            {
                "iteration": iteration,
                "primary_component": route.get("primary_component"),
                "active_components": route.get("active_components") or [],
            },
        )
        self.save()

    def _update_component_evidence(
        self,
        iteration: int,
        route: dict[str, Any],
        result: dict[str, Any],
        feedback: dict[str, Any],
    ) -> None:
        active = route.get("active_components") or [route.get("primary_component")]
        active = [node for node in active if node]
        self.state.setdefault("iteration_map", {})[_iter_key(iteration)] = active
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
            task = self.state.setdefault("tasks", {}).setdefault(
                node,
                {
                    "id": node,
                    "name": node,
                    "status": "active",
                    "priority": 0.5,
                    "uncertainty": 0.7,
                    "evidence": [],
                    "created_at": _now(),
                },
            )
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

    def record_patch(self, iteration: int, patch_meta: dict[str, Any] | None) -> None:
        if not patch_meta:
            return
        record = self._iteration_record(iteration)
        record["patch"] = {
            "origin": patch_meta.get("origin"),
            "component": patch_meta.get("component"),
            "summary": patch_meta.get("summary"),
            "output_model_path": patch_meta.get("output_model_path"),
            "diff_path": patch_meta.get("diff_path"),
        }
        for key in ("output_model_path", "diff_path"):
            if patch_meta.get(key):
                self.record_artifact(iteration, key, patch_meta[key])
        self.event("patch_recorded", {"iteration": iteration, "origin": patch_meta.get("origin")})
        self.save()

    def finish_iteration(self, iteration: int) -> None:
        record = self._iteration_record(iteration)
        if record.get("status") != "failed":
            record["status"] = "completed"
        record["updated_at"] = _now()
        self.event("iteration_finished", {"iteration": iteration, "status": record["status"]})
        self.save()

    def history_rows(self, target_metric: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(self.state.get("iterations", {})):
            record = self.state["iterations"][key]
            result_path = record.get("artifacts", {}).get("result", {}).get("path")
            if not result_path:
                continue
            resolved_result_path = self._resolve_artifact_path(result_path)
            if not resolved_result_path.exists():
                continue
            result = load_json(resolved_result_path)
            route = record.get("route", {})
            target = result.get("metrics", {}).get("target", {}) if result.get("success") else {}
            rows.append(
                {
                    "iteration": int(record.get("iteration", key.split("_")[-1])),
                    "success": result.get("success"),
                    "target": target,
                    "target_metric_value": target.get(target_metric),
                    "primary_component": route.get("primary_component"),
                    "active_components": route.get("active_components"),
                    "run_dir": result.get("run_dir"),
                    "result": result,
                }
            )
        return rows
