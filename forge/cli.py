from __future__ import annotations

import argparse
import csv
import difflib
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .assets import ensure_ms_aednet_data
from .config import load_experiment_config, load_json, save_json
from .evidence import build_run_evidence_audit, method_framework
from .feedback import encode_feedback
from .harness import HarnessConfig, run_harness
from .harness_spec import (
    get_benchmark_grid,
    get_dataset_files,
    get_default_dataset_name,
    get_dataset_metric_scales,
    get_enc_in,
    get_feature_dim,
    load_pemfc_harness_spec,
    validate_harness_specs,
)
from .llm import load_llm_config
from .memory import build_pemfc_context
from .model_io import read_model_source
from .orchestrator import GraphOrchestrator
from .patching import (
    PatchCandidate,
    apply_candidate,
    heuristic_patch_source,
    request_llm_dispatch_patch,
    request_llm_dispatch_summary,
    request_llm_patch,
    request_llm_repair_patch,
    safety_fallback_candidate,
    save_failed_candidate_attempt,
)
from .paths import CONFIG_DIR, INITIAL_MODEL_PATH, PROJECT_ROOT, RUNS_DIR, ensure_project_dirs
from .report import write_iteration_report
from .routing import TRUST_ROUTING_MODES, route_feedback

SWEEP_COMBO_RE = re.compile(r"^(?P<dataset>[^_]+)_L(?P<seq_len>\d+)_P(?P<pred_len>\d+)$")


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _device_label(cfg: HarnessConfig) -> str:
    if cfg.device == "cuda":
        return f"cuda:{cfg.cuda_id}"
    return str(cfg.device)


def _warn_if_heuristic_only(llm_mode: str, rounds: int, scope: str = "run") -> None:
    if llm_mode == "off" and rounds > 0:
        print(
            f"[FORGE] WARNING: {scope} is running with --llm-mode off and --rounds {rounds}. "
            "Model patches will come from deterministic heuristic templates, not from an LLM. "
            "Use --llm-mode required for official LLM-agent experiments."
        )


def _harness_config_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> HarnessConfig:
    data_cfg = cfg["data"]
    harness_cfg = cfg["harness"]
    model_cfg = cfg["model"]

    def choose(cli_value: Any, cfg_value: Any) -> Any:
        return cli_value if cli_value is not None else cfg_value

    return HarnessConfig(
        data_name=args.data or data_cfg.get("name") or get_default_dataset_name(),
        data_path=args.data_path,
        seq_len=int(choose(args.seq_len, data_cfg["seq_len"])),
        pred_len=int(choose(args.pred_len, data_cfg["pred_len"])),
        scaling=str(args.scaling or data_cfg["scaling"]),
        limit_rows=args.limit_rows if args.limit_rows is not None else data_cfg.get("limit_rows"),
        enc_in=int(model_cfg.get("enc_in") or get_enc_in()),
        hidden_dim=int(choose(args.hidden_dim, model_cfg["hidden_dim"])),
        layer=int(choose(args.layer, model_cfg["layer"])),
        dropout=float(args.dropout if args.dropout is not None else model_cfg["dropout"]),
        batch_size=int(choose(args.batch_size, harness_cfg["batch_size"])),
        lr=float(choose(args.lr, harness_cfg["lr"])),
        epochs=int(choose(args.epochs, harness_cfg["epochs"])),
        patience=int(choose(args.patience, harness_cfg["patience"])),
        seed=int(choose(args.seed, harness_cfg["seed"])),
        device=str(args.device or harness_cfg["device"]),
        cuda_id=int(args.cuda_id if args.cuda_id is not None else harness_cfg.get("cuda_id", 0)),
        num_workers=int(harness_cfg.get("num_workers", 0)),
    )


def cmd_init(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    validate_harness_specs()
    paths = ensure_ms_aednet_data()
    print("[FORGE] Initialized project directories.")
    for name, path in paths.items():
        print(f"[FORGE] Data {name}: {path}")
    print(f"[FORGE] Initial model: {INITIAL_MODEL_PATH}")


def _best_result(history: list[dict[str, Any]], target_metric: str) -> dict[str, Any] | None:
    successes = [row["result"] for row in history if row.get("result", {}).get("success")]
    if not successes:
        return None
    return min(
        successes,
        key=lambda result: result.get("metrics", {}).get("target", {}).get(target_metric, float("inf")),
    )


def _history_row(
    iteration: int,
    result: dict[str, Any],
    route: dict[str, Any],
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = result.get("metrics", {}).get("target", {}) if result.get("success") else {}
    return {
        "iteration": iteration,
        "success": result.get("success"),
        "target": target,
        "primary_component": route.get("primary_component"),
        "active_components": route.get("active_components"),
        "run_dir": result.get("run_dir"),
        "result": result,
        "feedback": feedback,
        "route": route,
    }


def _history_feedback(history: list[dict[str, Any]], iteration: int) -> dict[str, Any] | None:
    for row in reversed(history):
        if row.get("iteration") == iteration:
            feedback = row.get("feedback")
            if isinstance(feedback, dict):
                return feedback
    return None


def _history_route(history: list[dict[str, Any]], iteration: int) -> dict[str, Any] | None:
    for row in reversed(history):
        if row.get("iteration") == iteration:
            route = row.get("route")
            if isinstance(route, dict):
                return route
    return None


def _history_row_by_iteration(history: list[dict[str, Any]], iteration: int | None) -> dict[str, Any] | None:
    if iteration is None:
        return None
    for row in reversed(history):
        if row.get("iteration") == iteration:
            return row
    return None


def _attach_saved_feedback_and_routes(orchestrator: GraphOrchestrator, history: list[dict[str, Any]]) -> None:
    for row in history:
        iteration = int(row["iteration"])
        try:
            row["feedback"] = _load_iteration_json(orchestrator, iteration, "feedback_vector")
        except Exception:
            pass
        try:
            row["route"] = _load_iteration_json(orchestrator, iteration, "routing")
        except Exception:
            pass


def _parent_baseline_for_patch(
    orchestrator: GraphOrchestrator,
    history: list[dict[str, Any]],
    patch_iteration: int,
    default_result: dict[str, Any],
    default_feedback: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    patch_record = orchestrator.state.get("iterations", {}).get(f"iter_{patch_iteration:03d}", {}).get("patch", {})
    parent_iteration = patch_record.get("parent_iteration")
    try:
        parent_iteration = int(parent_iteration) if parent_iteration is not None else None
    except Exception:
        parent_iteration = None
    parent_row = _history_row_by_iteration(history, parent_iteration)
    if not parent_row:
        return default_result, default_feedback
    parent_result = parent_row.get("result")
    parent_feedback = parent_row.get("feedback")
    if not isinstance(parent_result, dict) or not isinstance(parent_feedback, dict):
        return default_result, default_feedback
    return parent_result, parent_feedback


def _select_parent_model(
    history: list[dict[str, Any]],
    target_metric: str,
    fallback: Path,
    parent_policy: str,
    run_root: Path,
) -> dict[str, Any]:
    fallback_info = {
        "path": fallback,
        "iteration": history[-1].get("iteration") if history else None,
        "result": history[-1].get("result") if history else None,
        "feedback": history[-1].get("feedback") if history else None,
    }
    if parent_policy != "best":
        return fallback_info
    success_rows = [row for row in history if row.get("result", {}).get("success")]
    if not success_rows:
        return fallback_info
    best_row = min(
        success_rows,
        key=lambda row: row.get("result", {}).get("metrics", {}).get("target", {}).get(target_metric, float("inf")),
    )
    best = best_row["result"]
    path = best.get("model_path") or best.get("paths", {}).get("model") or Path(best.get("run_dir", "")) / "model.py"
    try:
        resolved = _resolve_stored_path(path, run_root)
    except Exception:
        return fallback_info
    if not resolved.exists():
        return fallback_info
    return {
        "path": resolved,
        "iteration": best_row.get("iteration"),
        "result": best,
        "feedback": best_row.get("feedback"),
    }


def _maybe_update_trust_for_iteration(
    orchestrator: GraphOrchestrator,
    history: list[dict[str, Any]],
    outcome_iteration: int,
    target_metric: str,
    update_action_memory: bool = False,
) -> None:
    if outcome_iteration <= 0:
        return
    previous_row = next((row for row in history if row.get("iteration") == outcome_iteration - 1), None)
    outcome_row = next((row for row in history if row.get("iteration") == outcome_iteration), None)
    if not previous_row or not outcome_row:
        return
    previous_feedback = previous_row.get("feedback")
    outcome_feedback = outcome_row.get("feedback")
    if not isinstance(previous_feedback, dict) or not isinstance(outcome_feedback, dict):
        return
    baseline_result, baseline_feedback = _parent_baseline_for_patch(
        orchestrator,
        history,
        outcome_iteration - 1,
        previous_row["result"],
        previous_feedback,
    )
    orchestrator.update_trust_from_outcome(
        outcome_iteration - 1,
        outcome_iteration,
        baseline_result,
        outcome_row["result"],
        baseline_feedback,
        outcome_feedback,
        target_metric,
        update_action_memory=update_action_memory,
    )


def _validation_config(hcfg: HarnessConfig) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=hcfg.seq_len,
        pred_len=hcfg.pred_len,
        enc_in=hcfg.enc_in,
        hidden_dim=hcfg.hidden_dim,
        layer=hcfg.layer,
        dropout=hcfg.dropout,
        feature_dim=get_feature_dim(),
    )


def _resolve_stored_path(path: str | Path, run_root: Path | None = None) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    candidates = [PROJECT_ROOT / resolved]
    if run_root is not None:
        candidates.append(run_root / resolved)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _generate_patch_for_next_iteration(
    orchestrator: GraphOrchestrator,
    current_iteration: int,
    next_iteration: int,
    current_model_path: Path,
    parent_info: dict[str, Any],
    next_dir: Path,
    hcfg: HarnessConfig,
    llm_mode: str,
    feedback: dict[str, Any],
    route: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    next_model_path = next_dir / "model.py"
    next_dir.mkdir(parents=True, exist_ok=True)
    previous_source = read_model_source(current_model_path)
    candidate = None
    llm_cfg = None
    if llm_mode in {"auto", "required"}:
        try:
            llm_cfg = load_llm_config(str(CONFIG_DIR / "forge_llm.yaml"))
            candidate = request_llm_patch(
                llm_cfg,
                next_iteration,
                feedback,
                route,
                previous_source,
                history,
            )
        except Exception as exc:
            if llm_mode == "required":
                raise
            print(f"[FORGE] LLM patch unavailable, using heuristic fallback: {exc}")
    if candidate is None:
        candidate = heuristic_patch_source(previous_source, route)

    feature_dim = get_feature_dim()
    validation_cfg = _validation_config(hcfg)
    repair_attempts: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    max_repair_rounds = int((llm_cfg or {}).get("max_repair_rounds", 2)) if llm_cfg else 0
    attempt = 0
    while True:
        try:
            patch_meta = apply_candidate(
                candidate,
                current_model_path,
                next_model_path,
                validation_cfg,
                feature_dim=feature_dim,
                artifact_dir=next_dir,
            )
            patch_meta["repair_attempts"] = repair_attempts
            break
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            attempt_row = save_failed_candidate_attempt(candidate, next_dir, attempt, validation_error)
            repair_attempts.append(attempt_row)
            orchestrator.event(
                "patch_validation_failed",
                {
                    "iteration": current_iteration,
                    "next_iteration": next_iteration,
                    "attempt": attempt,
                    "origin": candidate.origin,
                    "component": candidate.component,
                    "error": validation_error,
                    "source_path": attempt_row["source_path"],
                },
            )
            seen_hashes.add(attempt_row["source_hash"])

            can_repair = llm_cfg is not None and candidate.origin in {"llm", "llm_repair"} and attempt < max_repair_rounds
            if can_repair:
                try:
                    repaired = request_llm_repair_patch(
                        llm_cfg,
                        next_iteration,
                        feedback,
                        route,
                        previous_source,
                        candidate,
                        validation_error,
                        history,
                        repair_attempts,
                        validation_cfg,
                        feature_dim,
                    )
                    repaired_hash = save_failed_candidate_attempt(repaired, next_dir, attempt + 100, "pre_validation_snapshot")[
                        "source_hash"
                    ]
                    if repaired_hash in seen_hashes:
                        orchestrator.event(
                            "patch_repair_duplicate",
                            {
                                "iteration": current_iteration,
                                "next_iteration": next_iteration,
                                "attempt": attempt,
                            },
                        )
                    else:
                        candidate = repaired
                        attempt += 1
                        continue
                except Exception as repair_exc:
                    repair_attempts.append(
                        {
                            "attempt": attempt,
                            "origin": "llm_repair_call",
                            "validation_error": f"{type(repair_exc).__name__}: {repair_exc}",
                        }
                    )

            if llm_mode == "auto" and candidate.origin in {"llm", "llm_repair"}:
                print(f"[FORGE] LLM patch failed validation, using heuristic fallback: {validation_error}")
                candidate = heuristic_patch_source(previous_source, route)
                attempt += 1
                continue

            fallback_reason = validation_error
            print(f"[FORGE] Patch validation unresolved; using safety fallback: {fallback_reason}")
            candidate = safety_fallback_candidate(previous_source, route, fallback_reason)
            patch_meta = apply_candidate(
                candidate,
                current_model_path,
                next_model_path,
                validation_cfg,
                feature_dim=feature_dim,
                artifact_dir=next_dir,
            )
            patch_meta["repair_attempts"] = repair_attempts
            patch_meta["validation_fallback"] = True
            break
    selected_edit = route.get("selected_edit") or {}
    selected_operator = str(selected_edit.get("edit_operator") or "")
    selected_component = str(selected_edit.get("component") or "")
    actual_operator = str(patch_meta.get("edit_action") or "")
    actual_component = str(patch_meta.get("component") or "")
    patch_meta["routed_component"] = route.get("primary_component")
    patch_meta["route_propagations"] = route.get("propagations") or []
    patch_meta["trust_before"] = {
        item.get("relation_id"): item.get("trust")
        for item in route.get("propagations", [])
        if item.get("relation_id")
    }
    patch_meta["selected_edit"] = selected_edit
    patch_meta["edit_candidates"] = route.get("edit_candidates") or []
    patch_meta["negative_memory"] = route.get("negative_memory") or []
    patch_meta["negative_reuse_suppression"] = route.get("negative_reuse_suppression") or []
    patch_meta["controlled_exploration"] = route.get("controlled_exploration") or {}
    patch_meta["structural_exploration"] = route.get("structural_exploration") or {}
    patch_meta["relation_attention"] = route.get("relation_attention") or {}
    patch_meta["memory_context"] = route.get("memory_context") or feedback.get("pemfc_context") or {}
    patch_meta["edit_operator_mismatch"] = bool(
        selected_operator
        and (not actual_operator or actual_operator != selected_operator)
    )
    patch_meta["component_mismatch"] = bool(
        selected_component
        and (not actual_component or actual_component != selected_component)
    )
    patch_meta["parent_model_path"] = str(current_model_path)
    parent_result = parent_info.get("result") if isinstance(parent_info, dict) else None
    patch_meta["parent_iteration"] = parent_info.get("iteration") if isinstance(parent_info, dict) else None
    patch_meta["parent_result_path"] = (
        (parent_result or {}).get("paths", {}).get("result") if isinstance(parent_result, dict) else None
    )
    target_metric = str(feedback.get("target_metric") or "mae_inverse")
    patch_meta["parent_target"] = (
        (parent_result or {}).get("metrics", {}).get("target", {}).get(target_metric)
        if isinstance(parent_result, dict)
        else None
    )
    save_json(patch_meta, next_dir / "patch_meta.json")
    orchestrator.record_patch(current_iteration, patch_meta)
    return next_model_path, patch_meta


def _load_iteration_json(orchestrator: GraphOrchestrator, iteration: int, artifact_name: str) -> dict[str, Any]:
    key = f"iter_{iteration:03d}"
    path = orchestrator.state["iterations"][key].get("artifacts", {}).get(artifact_name, {}).get("path")
    if not path:
        raise FileNotFoundError(f"Iteration {key} has no artifact named {artifact_name}")
    return load_json(_resolve_stored_path(path, orchestrator.run_root))


def _write_run_summary(
    run_root: Path,
    rounds: int,
    target_metric: str,
    history: list[dict[str, Any]],
    graph_state: dict[str, Any] | None = None,
    hcfg: HarnessConfig | None = None,
    candidate_tournament_k: int = 1,
) -> dict[str, Any]:
    summary = {
        "run_root": str(run_root),
        "rounds": rounds,
        "target_metric": target_metric,
        "method_framework": method_framework(candidate_tournament_k),
        "history": [
            {
                "iteration": row["iteration"],
                "success": row["success"],
                "target": row["target"],
                "primary_component": row["primary_component"],
                "run_dir": row["run_dir"],
            }
            for row in history
        ],
    }
    if hcfg is not None:
        summary["paper_baseline"] = _load_paper_baseline(hcfg)
    best = _best_result(history, target_metric)
    if best:
        best_row = next(
            (row for row in history if row.get("result") is best),
            None,
        )
        if best_row is None:
            best_row = _best_history_row(history, target_metric)
        if best_row is not None:
            summary["best_iteration"] = int(best_row["iteration"])
        summary["best_target"] = best.get("metrics", {}).get("target", {}).get(target_metric)
        summary["best_run_dir"] = best.get("run_dir")
        summary["best_metrics"] = _paper_metric_summary(best)
        summary["paper_gap"] = _paper_positive_gap(best, summary.get("paper_baseline"))
        summary["paper_delta"] = _paper_target_delta(best, summary.get("paper_baseline"))
    successful_iterations = [
        int(row["iteration"])
        for row in history
        if row.get("result", {}).get("success") and row.get("iteration") is not None
    ]
    all_iterations = [int(row["iteration"]) for row in history if row.get("iteration") is not None]
    if all_iterations:
        summary["best_selection"] = {
            "policy": "global_min_target_metric",
            "target_metric": target_metric,
            "search_start_iteration": min(all_iterations),
            "search_end_iteration": max(all_iterations),
            "successful_candidate_count": len(successful_iterations),
            "note": "Best model is selected once from the full iteration history, including resumed continuation rounds.",
        }
    summary["evidence_audit"] = build_run_evidence_audit(
        graph_state or {},
        history,
        target_metric,
        candidate_tournament_k=candidate_tournament_k,
    )
    save_json(summary, run_root / "summary.json")
    return summary


def _format_metric(value: Any) -> str:
    numeric = _safe_metric(value, float("inf"))
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _print_forge_best_summary(summary: dict[str, Any], label: str = "FORGE best") -> None:
    metrics = summary.get("best_metrics") or {}
    paper_mae = metrics.get("paper_mae")
    paper_mse = metrics.get("paper_mse")
    iteration = summary.get("best_iteration")
    if paper_mae is None or paper_mse is None:
        return
    iter_text = f"iter_{int(iteration):03d}" if iteration is not None else "unknown_iter"
    print(
        f"[FORGE] {label}: {iter_text} "
        f"MAE={_format_metric(paper_mae)} MSE={_format_metric(paper_mse)}"
    )
    paper_delta = summary.get("paper_delta") or {}
    if paper_delta:
        print(_format_paper_delta_line("FORGE vs reference target", paper_delta))


def _safe_metric(value: Any, default: float = float("inf")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _target_metric_value(result: dict[str, Any], target_metric: str) -> float:
    if not result.get("success"):
        return float("inf")
    return _safe_metric(result.get("metrics", {}).get("target", {}).get(target_metric))


def _inverse_mse_value(result: dict[str, Any]) -> float:
    if not result.get("success"):
        return float("inf")
    return _safe_metric(result.get("metrics", {}).get("inverse", {}).get("mse"))


def _paper_metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    paper = metrics.get("paper_scaled") or {}
    inverse = metrics.get("inverse") or {}
    return {
        "paper_mae": _paper_mae_value(result),
        "paper_mse": _paper_mse_value(result),
        "mae_inverse": inverse.get("mae"),
        "mse_inverse": inverse.get("mse"),
        "target": metrics.get("target", {}),
    }


def _result_dataset_name(result: dict[str, Any]) -> str | None:
    data_name = (result.get("data") or {}).get("data_name")
    if data_name:
        return str(data_name)
    data_name = (result.get("config") or {}).get("data_name")
    if data_name:
        return str(data_name)
    return None


def _paper_mae_value(result: dict[str, Any]) -> float:
    if not result.get("success"):
        return float("inf")
    metrics = result.get("metrics", {})
    value = metrics.get("paper_scaled", {}).get("mae")
    if value is not None:
        return _safe_metric(value)
    inverse = metrics.get("inverse", {}).get("mae")
    if inverse is None:
        inverse = metrics.get("target", {}).get("mae_inverse")
    dataset = _result_dataset_name(result)
    if inverse is not None and dataset:
        return _safe_metric(inverse) * get_dataset_metric_scales(dataset)["mae"]
    return float("inf")


def _paper_mse_value(result: dict[str, Any]) -> float:
    if not result.get("success"):
        return float("inf")
    metrics = result.get("metrics", {})
    value = metrics.get("paper_scaled", {}).get("mse")
    if value is not None:
        return _safe_metric(value)
    inverse = metrics.get("inverse", {}).get("mse")
    dataset = _result_dataset_name(result)
    if inverse is not None and dataset:
        return _safe_metric(inverse) * get_dataset_metric_scales(dataset)["mse"]
    return float("inf")


def _paper_positive_gap(result: dict[str, Any], paper_baseline: dict[str, float] | None) -> dict[str, float]:
    if not paper_baseline:
        return {}
    mae_gap = max(0.0, _paper_mae_value(result) - float(paper_baseline["mae"]))
    mse_gap = max(0.0, _paper_mse_value(result) - float(paper_baseline["mse"]))
    return {"mae": mae_gap, "mse": mse_gap, "total": mae_gap + mse_gap}


def _paper_target_delta(result: dict[str, Any], paper_baseline: dict[str, float] | None) -> dict[str, Any]:
    if not paper_baseline:
        return {}
    baseline_mae = float(paper_baseline["mae"])
    baseline_mse = float(paper_baseline["mse"])
    forge_mae = _paper_mae_value(result)
    forge_mse = _paper_mse_value(result)
    mae_delta = forge_mae - baseline_mae
    mse_delta = forge_mse - baseline_mse
    mae_improvement_pct = ((baseline_mae - forge_mae) / baseline_mae * 100.0) if baseline_mae else 0.0
    mse_improvement_pct = ((baseline_mse - forge_mse) / baseline_mse * 100.0) if baseline_mse else 0.0
    return {
        "mae": mae_delta,
        "mse": mse_delta,
        "total": mae_delta + mse_delta,
        "mae_improvement_pct": mae_improvement_pct,
        "mse_improvement_pct": mse_improvement_pct,
        "mean_improvement_pct": (mae_improvement_pct + mse_improvement_pct) / 2.0,
        "beats_mae": mae_delta < 0,
        "beats_mse": mse_delta < 0,
        "beats_both": mae_delta < 0 and mse_delta < 0,
        "note": (
            "Signed delta is FORGE benchmark-scaled metric minus reference target; negative means FORGE is better. "
            "Improvement percent is (reference_target - FORGE) / reference_target * 100; positive means FORGE is better."
        ),
    }


def _format_paper_delta_line(label: str, delta: dict[str, Any]) -> str:
    mae_delta = _safe_metric(delta.get("mae"), 0.0)
    mse_delta = _safe_metric(delta.get("mse"), 0.0)
    mae_pct = _safe_metric(delta.get("mae_improvement_pct"), 0.0)
    mse_pct = _safe_metric(delta.get("mse_improvement_pct"), 0.0)
    if bool(delta.get("beats_both")):
        return (
            f"[FORGE] {label}: improvement over reference target "
            f"MAE={mae_pct:.2f}% MSE={mse_pct:.2f}% "
            f"(absolute better by MAE={_format_metric(abs(mae_delta))} MSE={_format_metric(abs(mse_delta))})"
        )
    return (
        f"[FORGE] {label}: improvement over reference target "
        f"MAE={mae_pct:.2f}% MSE={mse_pct:.2f}% "
        f"(signed_delta MAE={_format_metric(mae_delta)} MSE={_format_metric(mse_delta)}; "
        "positive improvement means FORGE is better)"
    )


def _load_paper_baseline(
    hcfg: HarnessConfig,
    method: str = "Ms-AeDNet",
    override_mae: float | None = None,
    override_mse: float | None = None,
) -> dict[str, Any] | None:
    if override_mae is not None and override_mse is not None:
        return {
            "method": method,
            "dataset": hcfg.data_name,
            "seq_len": hcfg.seq_len,
            "pred_len": hcfg.pred_len,
            "mae": float(override_mae),
            "mse": float(override_mse),
            "source": "cli_override",
        }

    baselines = load_pemfc_harness_spec().get("paper_baselines", {})
    method_map = baselines.get(method) or baselines.get(str(method)) or {}
    dataset_map = method_map.get(str(hcfg.data_name).upper(), {}) if isinstance(method_map, dict) else {}
    key = f"L{int(hcfg.seq_len)}_P{int(hcfg.pred_len)}"
    row = dataset_map.get(key, {}) if isinstance(dataset_map, dict) else {}
    if not isinstance(row, dict) or row.get("mae") is None or row.get("mse") is None:
        return None
    return {
        "method": method,
        "dataset": hcfg.data_name,
        "seq_len": hcfg.seq_len,
        "pred_len": hcfg.pred_len,
        "mae": float(row["mae"]),
        "mse": float(row["mse"]),
        "source": "configs/harness/pemfc_harness.yaml",
    }


def _diagnostic_probe_map(feedback: dict[str, Any]) -> dict[str, float]:
    probes: dict[str, float] = {}
    for item in feedback.get("diagnostics") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        probes[str(item["name"])] = _safe_metric(item.get("severity"), 0.0) * _safe_metric(item.get("confidence"), 1.0)
    return probes


def _dominant_diagnostics(feedback: dict[str, Any], max_items: int = 5) -> list[str]:
    probes = _diagnostic_probe_map(feedback)
    return [name for name, _value in sorted(probes.items(), key=lambda item: item[1], reverse=True)[:max_items]]


def _relative_improvement(previous: float, current: float) -> float:
    if previous == float("inf") or current == float("inf"):
        return 0.0
    return (previous - current) / (abs(previous) + 1e-12)


def _finite_delta(previous: float, current: float) -> float:
    if not math.isfinite(previous) or not math.isfinite(current):
        return 0.0
    return previous - current


def _accept_dispatch_candidate(
    protected_result: dict[str, Any],
    candidate_result: dict[str, Any],
    protected_feedback: dict[str, Any],
    candidate_feedback: dict[str, Any],
    target_metric: str,
    target_diagnostics: list[str] | None = None,
    min_relative_improvement: float = 0.0,
    paper_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_diagnostics = target_diagnostics or _dominant_diagnostics(protected_feedback)
    protected_target = _target_metric_value(protected_result, target_metric)
    candidate_target = _target_metric_value(candidate_result, target_metric)
    protected_mse = _inverse_mse_value(protected_result)
    candidate_mse = _inverse_mse_value(candidate_result)

    protected_probes = _diagnostic_probe_map(protected_feedback)
    candidate_probes = _diagnostic_probe_map(candidate_feedback)
    probe_deltas = {
        name: protected_probes.get(name, 0.0) - candidate_probes.get(name, 0.0)
        for name in target_diagnostics
    }
    improved_probes = {name: delta for name, delta in probe_deltas.items() if delta > 1e-9}
    worsened_probes = {name: delta for name, delta in probe_deltas.items() if delta < -1e-9}

    target_rel_improvement = _relative_improvement(protected_target, candidate_target)
    mse_rel_improvement = _relative_improvement(protected_mse, candidate_mse)
    target_non_regression = candidate_target <= protected_target + 1e-12
    mse_non_regression = candidate_mse <= protected_mse + 1e-12
    target_improved = target_rel_improvement > min_relative_improvement
    mse_improved = mse_rel_improvement > min_relative_improvement
    diagnostic_improved = bool(improved_probes) and sum(probe_deltas.values()) > 1e-9

    protected_paper_mae = _paper_mae_value(protected_result)
    protected_paper_mse = _paper_mse_value(protected_result)
    candidate_paper_mae = _paper_mae_value(candidate_result)
    candidate_paper_mse = _paper_mse_value(candidate_result)
    paper_gap_decision: dict[str, Any] = {}
    if paper_baseline:
        protected_gap = _paper_positive_gap(protected_result, paper_baseline)
        candidate_gap = _paper_positive_gap(candidate_result, paper_baseline)
        paper_mae_non_regression = candidate_paper_mae <= protected_paper_mae + 1e-12
        paper_mse_non_regression = candidate_paper_mse <= protected_paper_mse + 1e-12
        gap_mae_non_increase = candidate_gap["mae"] <= protected_gap["mae"] + 1e-12
        gap_mse_non_increase = candidate_gap["mse"] <= protected_gap["mse"] + 1e-12
        protected_has_gap = protected_gap["total"] > 1e-12
        if protected_has_gap:
            gap_shrunk = (
                candidate_gap["total"]
                < protected_gap["total"] * (1.0 - max(0.0, float(min_relative_improvement))) - 1e-12
            )
        else:
            gap_shrunk = (
                candidate_paper_mae < protected_paper_mae - 1e-12
                or candidate_paper_mse < protected_paper_mse - 1e-12
            )
        paper_gap_decision = {
            "baseline": paper_baseline,
            "protected_paper_mae": protected_paper_mae,
            "protected_paper_mse": protected_paper_mse,
            "candidate_paper_mae": candidate_paper_mae,
            "candidate_paper_mse": candidate_paper_mse,
            "protected_gap": protected_gap,
            "candidate_gap": candidate_gap,
            "gap_delta": {
                "mae": protected_gap["mae"] - candidate_gap["mae"],
                "mse": protected_gap["mse"] - candidate_gap["mse"],
                "total": protected_gap["total"] - candidate_gap["total"],
            },
            "paper_mae_non_regression": paper_mae_non_regression,
            "paper_mse_non_regression": paper_mse_non_regression,
            "gap_mae_non_increase": gap_mae_non_increase,
            "gap_mse_non_increase": gap_mse_non_increase,
            "gap_shrunk": gap_shrunk,
            "protected_already_clears_baseline": not protected_has_gap,
        }
    else:
        paper_gap_decision = {
            "baseline": None,
            "protected_paper_mae": protected_paper_mae,
            "protected_paper_mse": protected_paper_mse,
            "candidate_paper_mae": candidate_paper_mae,
            "candidate_paper_mse": candidate_paper_mse,
        }

    if not candidate_result.get("success"):
        reason = "candidate_harness_failed"
    elif not target_non_regression:
        reason = "target_metric_regressed"
    elif not mse_non_regression:
        reason = "mse_regressed"
    elif paper_baseline and not paper_gap_decision["paper_mae_non_regression"]:
        reason = "paper_mae_regressed"
    elif paper_baseline and not paper_gap_decision["paper_mse_non_regression"]:
        reason = "paper_mse_regressed"
    elif paper_baseline and not paper_gap_decision["gap_mae_non_increase"]:
        reason = "paper_mae_gap_increased"
    elif paper_baseline and not paper_gap_decision["gap_mse_non_increase"]:
        reason = "paper_mse_gap_increased"
    elif paper_baseline and not paper_gap_decision["gap_shrunk"]:
        reason = "ms_aednet_gap_not_shrunk"
    elif not paper_baseline and not (target_improved or mse_improved or diagnostic_improved):
        reason = "no_executable_improvement"
    else:
        reason = "accepted_by_counterfactual_gap_harness" if paper_baseline else "accepted_by_non_regression_harness"

    accepted = reason.startswith("accepted_")

    return {
        "accepted": accepted,
        "reason": reason,
        "target_metric": target_metric,
        "protected_target": protected_target,
        "candidate_target": candidate_target,
        "target_relative_improvement": target_rel_improvement,
        "protected_inverse_mse": protected_mse,
        "candidate_inverse_mse": candidate_mse,
        "mse_relative_improvement": mse_rel_improvement,
        "target_non_regression": target_non_regression,
        "mse_non_regression": mse_non_regression,
        "target_diagnostics": target_diagnostics,
        "probe_deltas": probe_deltas,
        "improved_probes": improved_probes,
        "worsened_probes": worsened_probes,
        "paper_gap_decision": paper_gap_decision,
    }


def _best_history_row(history: list[dict[str, Any]], target_metric: str) -> dict[str, Any] | None:
    successes = [row for row in history if row.get("result", {}).get("success")]
    if not successes:
        return None
    return min(successes, key=lambda row: _target_metric_value(row.get("result", {}), target_metric))


def _iteration_source_path(row: dict[str, Any], run_root: Path) -> Path:
    result = row.get("result") or {}
    path = result.get("model_path") or result.get("paths", {}).get("model") or Path(result.get("run_dir", "")) / "model.py"
    return _resolve_stored_path(path, run_root)


def _trajectory_evidence(
    orchestrator: GraphOrchestrator,
    history: list[dict[str, Any]],
    target_metric: str,
    protected_iteration: int,
    limit: int = 16,
) -> dict[str, Any]:
    rows_by_iter = {int(row["iteration"]): row for row in history}
    attempts: list[dict[str, Any]] = []
    for iteration, row in sorted(rows_by_iter.items()):
        if iteration <= 0:
            continue
        patch_record = orchestrator.state.get("iterations", {}).get(f"iter_{iteration - 1:03d}", {}).get("patch", {})
        if not patch_record:
            continue
        result = row.get("result", {})
        parent_iteration = patch_record.get("parent_iteration")
        parent_row = rows_by_iter.get(int(parent_iteration)) if parent_iteration is not None else rows_by_iter.get(iteration - 1)
        parent_result = parent_row.get("result", {}) if parent_row else {}
        parent_target = _target_metric_value(parent_result, target_metric)
        current_target = _target_metric_value(result, target_metric)
        target_delta = _relative_improvement(parent_target, current_target)
        attempts.append(
            {
                "outcome_iteration": iteration,
                "parent_iteration": parent_iteration,
                "component": patch_record.get("component"),
                "edit_action": patch_record.get("edit_action"),
                "origin": patch_record.get("origin"),
                "summary": patch_record.get("summary"),
                "validation_fallback": bool(patch_record.get("validation_fallback", False)),
                "success": bool(result.get("success")),
                "target_delta_vs_parent": target_delta,
                "target": result.get("metrics", {}).get("target", {}) if result.get("success") else {},
                "paper_scaled": result.get("metrics", {}).get("paper_scaled", {}) if result.get("success") else {},
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
                "is_protected_best": iteration == protected_iteration,
            }
        )
    accepted = [
        row for row in attempts
        if row["success"] and row["target_delta_vs_parent"] > 0 and not row["validation_fallback"]
    ]
    rejected = [
        row for row in attempts
        if (not row["success"]) or row["target_delta_vs_parent"] < 0 or row["validation_fallback"]
    ]
    accepted = sorted(accepted, key=lambda row: row["target_delta_vs_parent"], reverse=True)[:limit]
    rejected = sorted(rejected, key=lambda row: (not row["success"], -row["target_delta_vs_parent"]), reverse=True)[:limit]
    best_timeline = [
        {
            "iteration": row["iteration"],
            "success": row["success"],
            "target": row["target"],
            "primary_component": row.get("primary_component"),
        }
        for row in history
        if int(row["iteration"]) == 0 or int(row["iteration"]) == protected_iteration or row.get("success")
    ][-limit:]
    return {
        "accepted_improvements": accepted,
        "rejected_or_failed_attempts": rejected,
        "best_timeline_tail": best_timeline,
    }


def _read_text_excerpt(path: str | Path | None, run_root: Path, max_chars: int) -> str:
    if not path:
        return ""
    resolved = _resolve_stored_path(path, run_root)
    if not resolved.exists():
        return ""
    text = resolved.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]\n"


def _run_matches_harness(run_root: Path, hcfg: HarnessConfig) -> bool:
    run_config_path = run_root / "run_config.json"
    if not run_config_path.exists():
        return False
    try:
        cfg = load_json(run_config_path).get("harness_config", {})
    except Exception:
        return False
    return (
        str(cfg.get("data_name", "")).upper() == str(hcfg.data_name).upper()
        and int(cfg.get("seq_len", -1)) == int(hcfg.seq_len)
        and int(cfg.get("pred_len", -1)) == int(hcfg.pred_len)
    )


def _expand_motif_source_path(path: Path, hcfg: HarnessConfig) -> list[Path]:
    path = path.expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        return []
    if (path / "run_config.json").exists() and (path / "task_graph.json").exists():
        return [path]
    combo = f"{str(hcfg.data_name).upper()}_L{int(hcfg.seq_len)}_P{int(hcfg.pred_len)}"
    combo_dir = path / combo
    if (combo_dir / "run_config.json").exists() and (combo_dir / "task_graph.json").exists():
        return [combo_dir]
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_dir()
        and child.name == combo
        and (child / "run_config.json").exists()
        and (child / "task_graph.json").exists()
    ]


def _default_motif_sources(run_root: Path, hcfg: HarnessConfig) -> list[Path]:
    return [run_root.resolve()] if _run_matches_harness(run_root, hcfg) else []


def _matching_run_motif_sources(run_root: Path, hcfg: HarnessConfig) -> list[Path]:
    combo = f"{str(hcfg.data_name).upper()}_L{int(hcfg.seq_len)}_P{int(hcfg.pred_len)}"
    sources = [run_root]
    for sweep_root in sorted(RUNS_DIR.glob("pilot_trust*")):
        candidate = sweep_root / combo
        if candidate.exists():
            sources.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        key = str(source.resolve())
        if key not in seen and _run_matches_harness(source, hcfg):
            unique.append(source.resolve())
            seen.add(key)
    return unique


def _motif_sources(
    run_root: Path,
    hcfg: HarnessConfig,
    requested_sources: list[str] | None,
    evidence_scope: str = "current-run",
) -> list[Path]:
    if evidence_scope == "current-run":
        return _default_motif_sources(run_root, hcfg)
    if evidence_scope == "matching-runs" and not requested_sources:
        return _matching_run_motif_sources(run_root, hcfg)
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in requested_sources or []:
        for root in _expand_motif_source_path(Path(raw), hcfg):
            key = str(root.resolve())
            if key not in seen and _run_matches_harness(root, hcfg):
                roots.append(root.resolve())
                seen.add(key)
    if evidence_scope == "matching-runs" and str(run_root.resolve()) not in seen and _run_matches_harness(run_root, hcfg):
        roots.insert(0, run_root.resolve())
    return roots


def _readonly_history_rows(run_root: Path, state: dict[str, Any], target_metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(state.get("iterations", {})):
        record = state["iterations"][key]
        result_path = record.get("artifacts", {}).get("result", {}).get("path")
        if not result_path:
            continue
        resolved_result_path = _resolve_stored_path(result_path, run_root)
        if not resolved_result_path.exists():
            continue
        result = load_json(resolved_result_path)
        route = record.get("route", {})
        route_path = record.get("artifacts", {}).get("routing", {}).get("path")
        if route_path:
            resolved_route_path = _resolve_stored_path(route_path, run_root)
            if resolved_route_path.exists():
                route = load_json(resolved_route_path)
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
                "route": route,
            }
        )
    return rows


def _mine_motifs_from_run(
    source_root: Path,
    hcfg: HarnessConfig,
    target_metric: str,
    max_diff_chars: int,
) -> list[dict[str, Any]]:
    if not _run_matches_harness(source_root, hcfg):
        return []
    graph_path = source_root / "task_graph.json"
    if not graph_path.exists():
        return []
    state = load_json(graph_path)
    history = _readonly_history_rows(source_root, state, target_metric)
    rows_by_iter = {int(row["iteration"]): row for row in history}
    motifs: list[dict[str, Any]] = []
    for outcome_iteration, outcome_row in sorted(rows_by_iter.items()):
        if outcome_iteration <= 0:
            continue
        patch_iteration = outcome_iteration - 1
        patch_record = state.get("iterations", {}).get(f"iter_{patch_iteration:03d}", {}).get("patch", {})
        if not patch_record or patch_record.get("validation_fallback"):
            continue
        parent_iteration = patch_record.get("parent_iteration")
        parent_row = rows_by_iter.get(int(parent_iteration)) if parent_iteration is not None else rows_by_iter.get(patch_iteration)
        if not parent_row:
            continue
        parent_result = parent_row.get("result", {})
        outcome_result = outcome_row.get("result", {})
        if not parent_result.get("success") or not outcome_result.get("success"):
            continue

        parent_target = _target_metric_value(parent_result, target_metric)
        outcome_target = _target_metric_value(outcome_result, target_metric)
        parent_mse = _inverse_mse_value(parent_result)
        outcome_mse = _inverse_mse_value(outcome_result)
        parent_paper_mae = _paper_mae_value(parent_result)
        outcome_paper_mae = _paper_mae_value(outcome_result)
        parent_paper_mse = _paper_mse_value(parent_result)
        outcome_paper_mse = _paper_mse_value(outcome_result)

        target_delta = _relative_improvement(parent_target, outcome_target)
        mse_delta = _relative_improvement(parent_mse, outcome_mse)
        paper_mae_delta = _finite_delta(parent_paper_mae, outcome_paper_mae)
        paper_mse_delta = _finite_delta(parent_paper_mse, outcome_paper_mse)
        improved = target_delta > 0 or mse_delta > 0 or paper_mae_delta > 0 or paper_mse_delta > 0
        catastrophic_tradeoff = (
            math.isfinite(outcome_paper_mae)
            and math.isfinite(parent_paper_mae)
            and outcome_paper_mae > parent_paper_mae * 1.01
        ) or (
            math.isfinite(outcome_paper_mse)
            and math.isfinite(parent_paper_mse)
            and outcome_paper_mse > parent_paper_mse * 1.03
        )
        if not improved or catastrophic_tradeoff:
            continue

        score = (
            max(0.0, target_delta)
            + 0.5 * max(0.0, mse_delta)
            + 0.05 * max(0.0, paper_mae_delta)
            + 0.01 * max(0.0, paper_mse_delta)
        )
        motif_id = (
            f"{source_root.name}:patch_{patch_iteration:03d}_to_iter_{outcome_iteration:03d}:"
            f"{patch_record.get('component') or 'unknown'}:{patch_record.get('edit_action') or 'unknown'}"
        )
        motifs.append(
            {
                "motif_id": motif_id,
                "source_run_root": str(source_root),
                "patch_iteration": patch_iteration,
                "outcome_iteration": outcome_iteration,
                "parent_iteration": parent_iteration,
                "component": patch_record.get("component"),
                "edit_action": patch_record.get("edit_action"),
                "summary": patch_record.get("summary"),
                "origin": patch_record.get("origin"),
                "routed_component": patch_record.get("routed_component"),
                "selected_edit": patch_record.get("selected_edit"),
                "score": score,
                "metric_delta": {
                    "target_relative": target_delta,
                    "mse_relative": mse_delta,
                    "paper_mae": paper_mae_delta,
                    "paper_mse": paper_mse_delta,
                },
                "parent_metrics": _paper_metric_summary(parent_result),
                "outcome_metrics": _paper_metric_summary(outcome_result),
                "diff_excerpt": _read_text_excerpt(patch_record.get("diff_path"), source_root, max_diff_chars),
            }
        )
    return motifs


def _mine_dispatch_motifs(
    run_root: Path,
    hcfg: HarnessConfig,
    target_metric: str,
    requested_sources: list[str] | None,
    evidence_scope: str,
    evidence_limit: int,
    max_diff_chars: int,
) -> dict[str, Any]:
    roots = _motif_sources(run_root, hcfg, requested_sources, evidence_scope=evidence_scope)
    motifs: list[dict[str, Any]] = []
    for root in roots:
        motifs.extend(_mine_motifs_from_run(root, hcfg, target_metric, max_diff_chars))
    motifs = sorted(motifs, key=lambda row: row.get("score", 0.0), reverse=True)
    return {
        "sources": [str(root) for root in roots],
        "motifs": motifs[: max(1, int(evidence_limit))],
        "total_mined": len(motifs),
    }


def _source_hash(source: str) -> str:
    import hashlib

    return hashlib.md5(source.strip().encode("utf-8")).hexdigest()


def _mine_archive_model_candidates(
    run_root: Path,
    hcfg: HarnessConfig,
    target_metric: str,
    requested_sources: list[str] | None,
    evidence_scope: str,
    limit: int,
    protected_source: str,
) -> dict[str, Any]:
    roots = _motif_sources(run_root, hcfg, requested_sources, evidence_scope=evidence_scope)
    protected_hash = _source_hash(protected_source)
    seen_hashes = {protected_hash}
    candidates: list[dict[str, Any]] = []
    for root in roots:
        graph_path = root / "task_graph.json"
        if not graph_path.exists():
            continue
        state = load_json(graph_path)
        history = _readonly_history_rows(root, state, target_metric)
        for row in history:
            result = row.get("result", {})
            if not result.get("success"):
                continue
            try:
                model_path = _iteration_source_path(row, root)
                if not model_path.exists():
                    continue
                source = read_model_source(model_path)
                source_hash = _source_hash(source)
            except Exception:
                continue
            if source_hash in seen_hashes:
                continue
            seen_hashes.add(source_hash)
            candidates.append(
                {
                    "archive_id": f"{root.name}:iter_{int(row['iteration']):03d}",
                    "source_run_root": str(root),
                    "iteration": int(row["iteration"]),
                    "model_path": str(model_path),
                    "source_hash": source_hash,
                    "metrics": _paper_metric_summary(result),
                    "paper_mae": _paper_mae_value(result),
                    "paper_mse": _paper_mse_value(result),
                    "target_metric_value": _target_metric_value(result, target_metric),
                }
            )
    candidates = sorted(candidates, key=lambda row: (row["paper_mae"], row["paper_mse"], row["target_metric_value"]))
    return {
        "sources": [str(root) for root in roots],
        "total_mined": len(candidates),
        "candidates": candidates[: max(0, int(limit))],
    }


def _compact_patch_meta(patch_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(patch_meta, dict):
        return patch_meta
    return {
        key: value
        for key, value in patch_meta.items()
        if key not in {"raw_response"}
    }


def _candidate_selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    decision = row.get("decision", {})
    paper_gap = decision.get("paper_gap_decision", {})
    candidate_gap = paper_gap.get("candidate_gap", {}) if isinstance(paper_gap, dict) else {}
    result = row.get("result", {})
    return (
        _safe_metric(candidate_gap.get("total"), float("inf")),
        _paper_mae_value(result),
        _paper_mse_value(result),
    )


def _meaningful_code_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if text.startswith("#"):
        return False
    if text.startswith(("import ", "from ")):
        return False
    if text in {'"""', "'''"}:
        return False
    return True


def _self_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        names.update(re.findall(r"\bself\.([A-Za-z_]\w*)", line))
    return names


def _dispatch_patch_quality(parent_source: str, candidate_source: str) -> dict[str, Any]:
    diff_lines = list(
        difflib.unified_diff(
            parent_source.splitlines(),
            candidate_source.splitlines(),
            lineterm="",
        )
    )
    added = [
        line[1:]
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line[1:]
        for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    ]
    meaningful_added = [line for line in added if _meaningful_code_line(line)]
    meaningful_removed = [line for line in removed if _meaningful_code_line(line)]
    added_self = _self_names(meaningful_added)
    removed_self = _self_names(meaningful_removed)
    destructively_removed = sorted(removed_self - added_self)
    total_meaningful = len(meaningful_added) + len(meaningful_removed)

    if total_meaningful < 4 or len(meaningful_added) < 2:
        reason = "motif_no_effect"
        passed = False
    elif len(destructively_removed) > 2:
        reason = "destructive_motif_transplant"
        passed = False
    elif len(meaningful_removed) > max(20, len(meaningful_added) * 2):
        reason = "motif_removes_too_much_parent_code"
        passed = False
    else:
        reason = "motif_quality_passed"
        passed = True

    return {
        "passed": passed,
        "reason": reason,
        "meaningful_added": len(meaningful_added),
        "meaningful_removed": len(meaningful_removed),
        "added_self_names": sorted(added_self),
        "removed_self_names": sorted(removed_self),
        "destructively_removed_self_names": destructively_removed,
    }


def _dispatch_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "dispatch_mode", None) or "summary").strip().lower()
    if mode not in {"summary", "candidates"}:
        raise ValueError(f"Unsupported dispatch mode: {mode}")
    return mode


def _dispatch_candidate_limit(args: argparse.Namespace, dispatch_mode: str) -> int:
    value = getattr(args, "dispatch_candidates", None)
    if value is None:
        return 0 if dispatch_mode == "summary" else 4
    return max(0, int(value))


def _archive_candidate_limit(args: argparse.Namespace, dispatch_mode: str) -> int:
    if dispatch_mode == "summary":
        return 0
    return max(0, int(getattr(args, "archive_candidates", 0)))


def _truncate_for_report(value: Any, max_chars: int = 1800) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "\n...[truncated]\n"
    return value


def _compact_motif_for_report(motif: dict[str, Any]) -> dict[str, Any]:
    return {
        "motif_id": motif.get("motif_id"),
        "source_run_root": motif.get("source_run_root"),
        "patch_iteration": motif.get("patch_iteration"),
        "outcome_iteration": motif.get("outcome_iteration"),
        "parent_iteration": motif.get("parent_iteration"),
        "component": motif.get("component"),
        "edit_action": motif.get("edit_action"),
        "summary": motif.get("summary"),
        "origin": motif.get("origin"),
        "routed_component": motif.get("routed_component"),
        "selected_edit": motif.get("selected_edit"),
        "score": motif.get("score"),
        "metric_delta": motif.get("metric_delta"),
        "parent_metrics": motif.get("parent_metrics"),
        "outcome_metrics": motif.get("outcome_metrics"),
        "diff_excerpt": _truncate_for_report(motif.get("diff_excerpt", "")),
    }


def _compact_dispatch_report_payload(payload: dict[str, Any], max_motifs: int = 10) -> dict[str, Any]:
    evidence = payload.get("trajectory_evidence") or {}
    motif_library = payload.get("motif_library") or {}
    motifs = motif_library.get("motifs") or []
    return {
        "dispatch_objective": payload.get("dispatch_objective"),
        "dispatch_mode": "summary",
        "protected_best": payload.get("protected_best"),
        "paper_baseline": payload.get("paper_baseline"),
        "evidence_scope": payload.get("evidence_scope"),
        "evidence_audit": {
            "metrics": (payload.get("evidence_audit") or {}).get("metrics", {}),
            "strategy_memory": (payload.get("evidence_audit") or {}).get("strategy_memory", {}),
        },
        "trajectory_evidence": {
            "accepted_improvements": (evidence.get("accepted_improvements") or [])[:8],
            "rejected_or_failed_attempts": (evidence.get("rejected_or_failed_attempts") or [])[:8],
            "best_timeline_tail": (evidence.get("best_timeline_tail") or [])[-12:],
        },
        "motif_library": {
            "sources": motif_library.get("sources", []),
            "total_mined": motif_library.get("total_mined", 0),
            "motifs": [_compact_motif_for_report(motif) for motif in motifs[:max_motifs]],
        },
        "acceptance_policy": payload.get("acceptance_policy"),
    }


def _deterministic_dispatch_report(payload: dict[str, Any]) -> dict[str, Any]:
    protected = payload.get("protected_best") or {}
    protected_iteration = int(protected.get("iteration") or 0)
    motif_library = payload.get("motif_library") or {}
    evidence = payload.get("trajectory_evidence") or {}
    motifs = motif_library.get("motifs") or []
    supported = [
        {
            "motif_id": motif.get("motif_id"),
            "component": motif.get("component"),
            "edit_action": motif.get("edit_action"),
            "evidence": {
                "score": motif.get("score"),
                "metric_delta": motif.get("metric_delta"),
                "outcome_metrics": motif.get("outcome_metrics"),
            },
        }
        for motif in motifs[:5]
    ]
    rejected = [
        {
            "component": row.get("component"),
            "edit_action": row.get("edit_action"),
            "reason": row.get("error_message") or "no executable improvement or validation fallback",
            "target_delta_vs_parent": row.get("target_delta_vs_parent"),
        }
        for row in (evidence.get("rejected_or_failed_attempts") or [])[:5]
    ]
    return {
        "schema": "forge.evidence_dispatch.report.v1",
        "llm_used": False,
        "summary": (
            "Evidence Dispatch ran in summary-only mode. The final model remains the protected "
            "Diagnostic Prober best; no new motif candidate was generated or evaluated."
        ),
        "protected_best_reason": (
            f"Protected iter_{protected_iteration:03d} is the best executable model "
            "selected by the fixed PEMFC harness."
        ),
        "supported_motifs": supported,
        "negative_or_ambiguous_motifs": rejected,
        "audit_trace": [
            "diagnostic feedback -> routed component -> LLM/heuristic patch -> fixed-harness outcome",
            "only executable improvements enter the motif evidence library",
            "summary-only dispatch does not overwrite protected_best",
        ],
        "limitations": [
            "Motif evidence is observational and run-specific.",
            "No final counterfactual candidate is trained in summary-only mode.",
        ],
        "final_recommendation": "keep protected_best",
    }


def _merge_llm_dispatch_report(fallback: dict[str, Any], llm_report: dict[str, Any]) -> dict[str, Any]:
    report = dict(fallback)
    for key in (
        "summary",
        "protected_best_reason",
        "supported_motifs",
        "negative_or_ambiguous_motifs",
        "audit_trace",
        "limitations",
        "final_recommendation",
    ):
        if key in llm_report:
            report[key] = llm_report[key]
    report["llm_used"] = True
    report["llm_raw_response"] = {key: value for key, value in llm_report.items() if key != "_usage"}
    if llm_report.get("_usage"):
        report["llm_usage"] = llm_report["_usage"]
    return report


def _write_sweep_outputs(rows: list[dict[str, Any]], sweep_root: Path, target_metric: str) -> None:
    summary = {
        "sweep_root": str(sweep_root),
        "target_metric": target_metric,
        "count": len(rows),
        "rows": rows,
        "cross_cell_robustness": _cross_cell_robustness(rows),
    }
    save_json(summary, sweep_root / "sweep_summary.json")

    csv_path = sweep_root / "sweep_summary.csv"
    fieldnames = [
        "dataset",
        "seq_len",
        "pred_len",
        "success",
        "best_target",
        "best_iteration",
        "best_paper_mae",
        "best_paper_mse",
        "improvement_rate",
        "invalid_edit_rate",
        "repeated_useless_edit_rate",
        "routing_stability",
        "evidence_alignment",
        "best_run_dir",
        "run_root",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _compact_evidence_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = (summary.get("evidence_audit") or {}).get("metrics") or {}
    return {
        "improvement_rate": metrics.get("improvement_rate"),
        "invalid_edit_rate": metrics.get("invalid_edit_rate"),
        "repeated_useless_edit_rate": metrics.get("repeated_useless_edit_rate"),
        "routing_stability": metrics.get("routing_stability"),
        "evidence_alignment": metrics.get("evidence_alignment"),
    }


def _attach_summary_metrics_to_sweep_row(row: dict[str, Any], summary: dict[str, Any]) -> None:
    best_metrics = summary.get("best_metrics") or {}
    row["best_iteration"] = summary.get("best_iteration")
    row["best_paper_mae"] = best_metrics.get("paper_mae")
    row["best_paper_mse"] = best_metrics.get("paper_mse")
    row["evidence_metrics"] = _compact_evidence_metrics(summary)
    row.update(row["evidence_metrics"])


def _cross_cell_robustness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success")]
    by_dataset = {
        str(row.get("dataset")): set(
            item.get("component")
            for item in ((row.get("evidence_audit") or {}).get("strategy_memory") or {}).get("trusted_components", [])
            if item.get("component")
        )
        for row in successful
    }
    if len(by_dataset) < 2:
        return {"available": False, "reason": "requires_at_least_two_successful_datasets"}
    datasets = sorted(by_dataset)
    scores = []
    pairs = []
    for left_index, left in enumerate(datasets):
        for right in datasets[left_index + 1 :]:
            a = by_dataset[left]
            b = by_dataset[right]
            union = a | b
            score = len(a & b) / len(union) if union else 0.0
            scores.append(score)
            pairs.append({"left": left, "right": right, "trusted_component_jaccard": score})
    return {
        "available": True,
        "datasets": datasets,
        "trusted_components_by_dataset": {key: sorted(value) for key, value in by_dataset.items()},
        "mean_trusted_component_jaccard": sum(scores) / len(scores) if scores else 0.0,
        "pairs": pairs,
    }


def _collect_sweep_rows(sweep_root: Path, target_metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
        match = SWEEP_COMBO_RE.match(child.name)
        if not match:
            continue
        row = {
            "dataset": match.group("dataset"),
            "seq_len": int(match.group("seq_len")),
            "pred_len": int(match.group("pred_len")),
            "success": False,
            "best_target": None,
            "best_iteration": None,
            "best_paper_mae": None,
            "best_paper_mse": None,
            "best_run_dir": None,
            "run_root": str(child),
            "error": None,
        }
        summary_path = child / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            row["success"] = True
            row["best_target"] = summary.get("best_target")
            row["best_run_dir"] = summary.get("best_run_dir")
            row["evidence_audit"] = summary.get("evidence_audit")
            _attach_summary_metrics_to_sweep_row(row, summary)
        else:
            graph_path = child / "task_graph.json"
            if graph_path.exists():
                state = load_json(graph_path)
                failed = [
                    (key, record)
                    for key, record in state.get("iterations", {}).items()
                    if record.get("status") == "failed"
                ]
                if failed:
                    key, record = failed[-1]
                    errors = [
                        stage.get("error")
                        for stage in record.get("stages", {}).values()
                        if stage.get("status") == "failed" and stage.get("error")
                    ]
                    row["error"] = f"{key}: {errors[-1]}" if errors else f"{key}: failed"
                else:
                    row["error"] = "summary.json missing"
            else:
                row["error"] = "task_graph.json missing"
        rows.append(row)
    return rows


def _refresh_sweep_summary(sweep_root: Path, target_metric: str) -> list[dict[str, Any]]:
    rows = _collect_sweep_rows(sweep_root, target_metric)
    _write_sweep_outputs(rows, sweep_root, target_metric)
    return rows


def _maybe_refresh_parent_sweep_summary(run_root: Path, target_metric: str) -> Path | None:
    if not SWEEP_COMBO_RE.match(run_root.name):
        return None
    sweep_root = run_root.parent
    if not (sweep_root / "sweep_summary.json").exists():
        return None
    _refresh_sweep_summary(sweep_root, target_metric)
    return sweep_root


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_project_dirs()
    validate_harness_specs()
    ensure_ms_aednet_data()
    exp_cfg = load_experiment_config(args.experiment_config)
    target_metric = args.target_metric or exp_cfg["evolution"]["target_metric"]
    rounds = int(args.rounds if args.rounds is not None else exp_cfg["evolution"]["rounds"])
    llm_mode = args.llm_mode or exp_cfg["evolution"]["llm_mode"]
    parent_policy = args.parent_policy or "best"
    routing_mode = args.routing_mode or "trust"
    if int(getattr(args, "candidate_tournament_k", 1)) != 1:
        raise ValueError("Stable FORGE currently executes one candidate per round; keep --candidate-tournament-k 1")
    hcfg = _harness_config_from_args(args, exp_cfg)

    run_name = args.run_name or f"forge_{_timestamp()}"
    run_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (RUNS_DIR / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "experiment_config": exp_cfg,
            "harness_config": hcfg.__dict__,
            "runtime": {
                "parent_policy": parent_policy,
                "routing_mode": routing_mode,
                "candidate_tournament_k": int(getattr(args, "candidate_tournament_k", 1)),
            },
        },
        run_root / "run_config.json",
    )

    orchestrator = GraphOrchestrator.open(run_root)
    history: list[dict[str, Any]] = []

    iter0 = run_root / "iter_000"
    iter0.mkdir(parents=True, exist_ok=True)
    current_model_path = iter0 / "model.py"
    if not current_model_path.exists():
        shutil.copy2(INITIAL_MODEL_PATH, current_model_path)

    print(f"[FORGE] Run root: {run_root}")
    print(f"[FORGE] Dataset: {hcfg.data_name} | seq_len: {hcfg.seq_len} | pred_len: {hcfg.pred_len}")
    print(f"[FORGE] Device: {_device_label(hcfg)}")
    print(f"[FORGE] LLM mode: {llm_mode}")
    print(f"[FORGE] Parent policy: {parent_policy}")
    print(f"[FORGE] Routing mode: {routing_mode}")
    print(f"[FORGE] Rounds: {rounds}")
    if not getattr(args, "_sweep_child", False):
        _warn_if_heuristic_only(llm_mode, rounds)

    for iteration in range(rounds + 1):
        iter_dir = run_root / f"iter_{iteration:03d}"
        model_path_for_iter = iter_dir / "model.py"
        orchestrator.ensure_iteration(iteration, iter_dir, model_path_for_iter)
        with orchestrator.stage(iteration, "prepare", {"iter_dir": str(iter_dir)}):
            iter_dir.mkdir(parents=True, exist_ok=True)
            if not model_path_for_iter.exists():
                shutil.copy2(current_model_path, model_path_for_iter)
            current_model_path = model_path_for_iter
            orchestrator.record_artifact(iteration, "model", current_model_path, kind="python_source")

        print(f"[FORGE] Iteration {iteration:03d}: training and evaluating {current_model_path}")
        with orchestrator.stage(iteration, "evaluate", {"model_path": str(current_model_path)}):
            result = run_harness(current_model_path, iter_dir, hcfg)
            orchestrator.record_result(iteration, result)

        previous_result = history[-1]["result"] if history else None
        best_before = _best_result(history, target_metric)

        with orchestrator.stage(iteration, "feedback", {"target_metric": target_metric}):
            feedback = encode_feedback(result, previous_result, best_before, target_metric=target_metric)
            feedback["pemfc_context"] = build_pemfc_context(feedback, result=result, harness_config=hcfg)
            feedback_path = iter_dir / "feedback_vector.json"
            save_json(feedback, feedback_path)
            orchestrator.record_artifact(iteration, "feedback_vector", feedback_path)

        if routing_mode in TRUST_ROUTING_MODES and iteration > 0 and previous_result is not None:
            previous_feedback = _history_feedback(history, iteration - 1)
            if previous_feedback is not None:
                baseline_result, baseline_feedback = _parent_baseline_for_patch(
                    orchestrator,
                    history,
                    iteration - 1,
                    previous_result,
                    previous_feedback,
                )
                orchestrator.update_trust_from_outcome(
                    iteration - 1,
                    iteration,
                    baseline_result,
                    result,
                    baseline_feedback,
                    feedback,
                    target_metric,
                    update_action_memory=(routing_mode == "trust-action"),
                )

        with orchestrator.stage(iteration, "route"):
            route_state = orchestrator.state if routing_mode in TRUST_ROUTING_MODES else None
            route = route_feedback(feedback, route_state, mode=routing_mode)
            route_path = iter_dir / "routing.json"
            save_json(route, route_path)
            orchestrator.record_artifact(iteration, "routing", route_path)
            orchestrator.record_feedback_and_route(iteration, feedback, route, result)

        patch_meta = None
        history.append(_history_row(iteration, result, route, feedback))
        if iteration < rounds:
            next_dir = run_root / f"iter_{iteration + 1:03d}"
            with orchestrator.stage(iteration, "patch", {"next_iteration": iteration + 1}):
                next_dir.mkdir(parents=True, exist_ok=True)
                parent_info = _select_parent_model(
                    history,
                    target_metric,
                    current_model_path,
                    parent_policy,
                    run_root,
                )
                current_model_path, patch_meta = _generate_patch_for_next_iteration(
                    orchestrator,
                    iteration,
                    iteration + 1,
                    parent_info["path"],
                    parent_info,
                    next_dir,
                    hcfg,
                    llm_mode,
                    feedback,
                    route,
                    history,
                )
        else:
            orchestrator.skip_stage(iteration, "patch", "final_iteration")

        with orchestrator.stage(iteration, "report"):
            report_path = iter_dir / "report.md"
            iteration_record = orchestrator.state["iterations"][f"iter_{iteration:03d}"]
            trust_updates = iteration_record.get("trust_updates", [])
            action_updates = iteration_record.get("action_memory_updates", [])
            write_iteration_report(report_path, iteration, result, feedback, route, patch_meta, trust_updates, action_updates)
            orchestrator.record_artifact(iteration, "report", report_path)
        orchestrator.finish_iteration(iteration)

    summary = _write_run_summary(
        run_root,
        rounds,
        target_metric,
        history,
        graph_state=orchestrator.state,
        hcfg=hcfg,
        candidate_tournament_k=int(getattr(args, "candidate_tournament_k", 1)),
    )
    dispatch_summary = _maybe_run_final_dispatch(args, run_root, target_metric)
    if dispatch_summary:
        summary["final_dispatch"] = _compact_dispatch_summary(dispatch_summary)
        save_json(summary, run_root / "summary.json")
    _print_forge_best_summary(summary)
    print(f"[FORGE] Finished. Summary: {run_root / 'summary.json'}")
    return summary


def _load_continue_run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], HarnessConfig, str, str]:
    run_root = Path(args.run_dir).expanduser().resolve()
    run_config_path = run_root / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Cannot continue; missing {run_config_path}")
    run_config = load_json(run_config_path)
    exp_cfg = run_config.get("experiment_config") or load_experiment_config(args.experiment_config)
    hcfg = HarnessConfig(**run_config["harness_config"])

    for attr in ("epochs", "batch_size", "lr", "patience", "seed", "device", "cuda_id"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(hcfg, attr, value)

    target_metric = args.target_metric or exp_cfg["evolution"]["target_metric"]
    llm_mode = args.llm_mode or exp_cfg["evolution"]["llm_mode"]
    return run_root, exp_cfg, hcfg, target_metric, llm_mode


def _resolve_continue_target(args: argparse.Namespace, last_iteration: int) -> int:
    if args.to_round is not None and args.additional_rounds is not None:
        raise ValueError("Use either --to-round or --additional-rounds, not both")
    if args.to_round is not None:
        target = int(args.to_round)
    elif args.additional_rounds is not None:
        target = last_iteration + int(args.additional_rounds)
    else:
        target = last_iteration + 1
    if target <= last_iteration:
        raise ValueError(f"Continuation target must be greater than last completed iteration {last_iteration}")
    return target


def cmd_continue(args: argparse.Namespace) -> dict[str, Any]:
    ensure_project_dirs()
    validate_harness_specs()
    ensure_ms_aednet_data()

    run_root, _exp_cfg, hcfg, target_metric, llm_mode = _load_continue_run(args)
    parent_policy = args.parent_policy or "best"
    routing_mode = args.routing_mode or "trust"
    if int(getattr(args, "candidate_tournament_k", 1)) != 1:
        raise ValueError("Stable FORGE currently executes one candidate per round; keep --candidate-tournament-k 1")
    orchestrator = GraphOrchestrator.open(run_root)
    history = orchestrator.history_rows(target_metric)
    _attach_saved_feedback_and_routes(orchestrator, history)
    if not history:
        raise RuntimeError(f"No completed iterations found in {run_root}")
    last_iteration = history[-1]["iteration"]
    to_round = _resolve_continue_target(args, last_iteration)

    print(f"[FORGE] Continue run root: {run_root}")
    print(f"[FORGE] Dataset: {hcfg.data_name} | seq_len: {hcfg.seq_len} | pred_len: {hcfg.pred_len}")
    print(f"[FORGE] Device: {_device_label(hcfg)}")
    print(f"[FORGE] LLM mode: {llm_mode}")
    print(f"[FORGE] Parent policy: {parent_policy}")
    print(f"[FORGE] Routing mode: {routing_mode}")
    print(f"[FORGE] Continuing from iter_{last_iteration:03d} to iter_{to_round:03d}")
    _warn_if_heuristic_only(llm_mode, to_round - last_iteration, scope="continue")
    orchestrator.event(
        "continue_started",
        {"from_iteration": last_iteration, "to_round": to_round, "llm_mode": llm_mode},
    )
    orchestrator.save()

    current_iteration = last_iteration
    while current_iteration < to_round:
        current_key = f"iter_{current_iteration:03d}"
        current_record = orchestrator.state["iterations"][current_key]
        current_model_path = _resolve_stored_path(
            current_record.get("artifacts", {}).get("model", {}).get("path")
            or current_record.get("model_path")
            or run_root / current_key / "model.py",
            run_root,
        )
        result = _load_iteration_json(orchestrator, current_iteration, "result")
        feedback = _load_iteration_json(orchestrator, current_iteration, "feedback_vector")
        if not feedback.get("pemfc_context"):
            feedback["pemfc_context"] = build_pemfc_context(feedback, result=result, harness_config=hcfg)
        route = _load_iteration_json(orchestrator, current_iteration, "routing")
        current_row = next((row for row in history if row.get("iteration") == current_iteration), None)
        if current_row is not None:
            current_row["feedback"] = feedback
            current_row["route"] = route
        if routing_mode in TRUST_ROUTING_MODES:
            _maybe_update_trust_for_iteration(
                orchestrator,
                history,
                current_iteration,
                target_metric,
                update_action_memory=(routing_mode == "trust-action"),
            )

        next_iteration = current_iteration + 1
        next_dir = run_root / f"iter_{next_iteration:03d}"
        patch_stage = current_record.get("stages", {}).get("patch", {})
        patch_record = current_record.get("patch", {})
        existing_next_model = patch_record.get("output_model_path")
        patch_meta = None
        existing_next_model_path = _resolve_stored_path(existing_next_model, run_root) if existing_next_model else None
        if patch_stage.get("status") == "succeeded" and existing_next_model_path and existing_next_model_path.exists():
            next_model_path = existing_next_model_path
            print(f"[FORGE] Reusing existing patch for iter_{current_iteration:03d} -> iter_{next_iteration:03d}")
        else:
            with orchestrator.stage(current_iteration, "route", {"resume": True, "trust_refresh": True}):
                route_state = orchestrator.state if routing_mode in TRUST_ROUTING_MODES else None
                route = route_feedback(feedback, route_state, mode=routing_mode)
                route_path = run_root / current_key / "routing.json"
                save_json(route, route_path)
                orchestrator.record_artifact(current_iteration, "routing", route_path)
                orchestrator.record_feedback_and_route(current_iteration, feedback, route, result)
                if current_row is not None:
                    current_row["route"] = route
            with orchestrator.stage(current_iteration, "patch", {"next_iteration": next_iteration, "resume": True}):
                parent_info = _select_parent_model(
                    history,
                    target_metric,
                    current_model_path,
                    parent_policy,
                    run_root,
                )
                next_model_path, patch_meta = _generate_patch_for_next_iteration(
                    orchestrator,
                    current_iteration,
                    next_iteration,
                    parent_info["path"],
                    parent_info,
                    next_dir,
                    hcfg,
                    llm_mode,
                    feedback,
                    route,
                    history,
                )
            with orchestrator.stage(current_iteration, "report", {"resume": True, "patch_refresh": True}):
                report_path = run_root / current_key / "report.md"
                trust_updates = current_record.get("trust_updates", [])
                action_updates = current_record.get("action_memory_updates", [])
                write_iteration_report(
                    report_path,
                    current_iteration,
                    result,
                    feedback,
                    route,
                    patch_meta,
                    trust_updates,
                    action_updates,
                )
                orchestrator.record_artifact(current_iteration, "report", report_path)
            orchestrator.finish_iteration(current_iteration)

        orchestrator.ensure_iteration(next_iteration, next_dir, next_model_path)
        with orchestrator.stage(next_iteration, "prepare", {"iter_dir": str(next_dir), "resume": True}):
            next_dir.mkdir(parents=True, exist_ok=True)
            if not (next_dir / "model.py").exists():
                shutil.copy2(next_model_path, next_dir / "model.py")
                next_model_path = next_dir / "model.py"
            orchestrator.record_artifact(next_iteration, "model", next_model_path, kind="python_source")

        print(f"[FORGE] Iteration {next_iteration:03d}: training and evaluating {next_model_path}")
        with orchestrator.stage(next_iteration, "evaluate", {"model_path": str(next_model_path), "resume": True}):
            next_result = run_harness(next_model_path, next_dir, hcfg)
            orchestrator.record_result(next_iteration, next_result)

        previous_result = history[-1]["result"]
        best_before = _best_result(history, target_metric)
        with orchestrator.stage(next_iteration, "feedback", {"target_metric": target_metric, "resume": True}):
            next_feedback = encode_feedback(next_result, previous_result, best_before, target_metric=target_metric)
            next_feedback["pemfc_context"] = build_pemfc_context(next_feedback, result=next_result, harness_config=hcfg)
            feedback_path = next_dir / "feedback_vector.json"
            save_json(next_feedback, feedback_path)
            orchestrator.record_artifact(next_iteration, "feedback_vector", feedback_path)

        if routing_mode in TRUST_ROUTING_MODES:
            baseline_result, baseline_feedback = _parent_baseline_for_patch(
                orchestrator,
                history,
                current_iteration,
                result,
                feedback,
            )
            orchestrator.update_trust_from_outcome(
                current_iteration,
                next_iteration,
                baseline_result,
                next_result,
                baseline_feedback,
                next_feedback,
                target_metric,
                update_action_memory=(routing_mode == "trust-action"),
            )

        with orchestrator.stage(next_iteration, "route", {"resume": True}):
            route_state = orchestrator.state if routing_mode in TRUST_ROUTING_MODES else None
            next_route = route_feedback(next_feedback, route_state, mode=routing_mode)
            route_path = next_dir / "routing.json"
            save_json(next_route, route_path)
            orchestrator.record_artifact(next_iteration, "routing", route_path)
            orchestrator.record_feedback_and_route(next_iteration, next_feedback, next_route, next_result)

        history.append(_history_row(next_iteration, next_result, next_route, next_feedback))
        if next_iteration == to_round:
            orchestrator.skip_stage(next_iteration, "patch", "final_iteration")

        with orchestrator.stage(next_iteration, "report", {"resume": True}):
            report_path = next_dir / "report.md"
            next_record = orchestrator.state["iterations"][f"iter_{next_iteration:03d}"]
            trust_updates = next_record.get("trust_updates", [])
            action_updates = next_record.get("action_memory_updates", [])
            write_iteration_report(
                report_path,
                next_iteration,
                next_result,
                next_feedback,
                next_route,
                None,
                trust_updates,
                action_updates,
            )
            orchestrator.record_artifact(next_iteration, "report", report_path)
        orchestrator.finish_iteration(next_iteration)
        current_iteration = next_iteration

    summary = _write_run_summary(
        run_root,
        to_round,
        target_metric,
        history,
        graph_state=orchestrator.state,
        hcfg=hcfg,
        candidate_tournament_k=int(getattr(args, "candidate_tournament_k", 1)),
    )
    refreshed_sweep = _maybe_refresh_parent_sweep_summary(run_root, target_metric)
    dispatch_summary = _maybe_run_final_dispatch(args, run_root, target_metric)
    if dispatch_summary:
        summary["final_dispatch"] = _compact_dispatch_summary(dispatch_summary)
        save_json(summary, run_root / "summary.json")
    orchestrator.event("continue_finished", {"to_round": to_round})
    orchestrator.save()
    _print_forge_best_summary(summary)
    print(f"[FORGE] Continue finished. Summary: {run_root / 'summary.json'}")
    if refreshed_sweep is not None:
        print(f"[FORGE] Parent sweep summary refreshed: {refreshed_sweep / 'sweep_summary.json'}")
    return summary


def _unique_child_dir(parent: Path, name: str) -> Path:
    base = parent / name
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = parent / f"{name}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique dispatch directory under {parent}")


def _evaluate_dispatch_candidate(
    *,
    candidate_index: int,
    candidate: PatchCandidate,
    candidate_dir: Path,
    protected_copy: Path,
    protected_source: str,
    protected_iteration: int,
    protected_result: dict[str, Any],
    protected_feedback: dict[str, Any],
    protected_route: dict[str, Any],
    history: list[dict[str, Any]],
    hcfg: HarnessConfig,
    target_metric: str,
    target_diagnostics: list[str],
    paper_baseline: dict[str, Any] | None,
    llm_cfg: dict[str, Any] | None,
    max_repair_rounds: int,
    motif: dict[str, Any] | None,
    min_relative_improvement: float,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    validation_cfg = _validation_config(hcfg)
    feature_dim = get_feature_dim()
    candidate_model_path = candidate_dir / "model.py"
    repair_attempts: list[dict[str, Any]] = []
    patch_meta: dict[str, Any] | None = None
    validation_error: str | None = None
    quality_error: str | None = None
    attempt = 0
    while True:
        try:
            patch_meta = apply_candidate(
                candidate,
                protected_copy,
                candidate_model_path,
                validation_cfg,
                feature_dim=feature_dim,
                artifact_dir=candidate_dir,
            )
            validation_error = None
            patch_meta["repair_attempts"] = repair_attempts
            patch_meta["protected_iteration"] = protected_iteration
            patch_meta["selected_motif_id"] = (motif or {}).get("motif_id")
            if candidate.origin == "archive_model":
                quality = {
                    "passed": True,
                    "reason": "archive_model_promotion",
                    "meaningful_added": None,
                    "meaningful_removed": None,
                    "added_self_names": [],
                    "removed_self_names": [],
                    "destructively_removed_self_names": [],
                }
            else:
                quality = _dispatch_patch_quality(protected_source, candidate.source)
            patch_meta["motif_quality"] = quality
            if not quality["passed"]:
                quality_error = str(quality["reason"])
            save_json(patch_meta, candidate_dir / "patch_meta.json")
            break
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            repair_attempts.append(save_failed_candidate_attempt(candidate, candidate_dir, attempt, validation_error))
            can_repair = llm_cfg is not None and candidate.origin in {"llm_dispatch", "llm_motif_dispatch", "llm_repair"} and attempt < max_repair_rounds
            if can_repair:
                try:
                    candidate = request_llm_repair_patch(
                        llm_cfg,
                        protected_iteration,
                        protected_feedback,
                        protected_route,
                        protected_source,
                        candidate,
                        validation_error,
                        history,
                        repair_attempts,
                        validation_cfg,
                        feature_dim,
                    )
                    attempt += 1
                    continue
                except Exception as repair_exc:
                    repair_attempts.append(
                        {
                            "attempt": attempt,
                            "origin": "llm_dispatch_repair_call",
                            "validation_error": f"{type(repair_exc).__name__}: {repair_exc}",
                        }
                    )
            patch_meta = {
                "origin": candidate.origin,
                "component": candidate.component,
                "summary": candidate.summary,
                "rationale": candidate.rationale,
                "edit_action": candidate.edit_action,
                "validation_error": validation_error,
                "repair_attempts": repair_attempts,
                "selected_motif_id": (motif or {}).get("motif_id"),
            }
            save_json(patch_meta, candidate_dir / "patch_meta.json")
            break

    if candidate_model_path.exists() and validation_error is None and quality_error is None:
        print(f"[FORGE] Evidence Dispatch: evaluating candidate {candidate_index:02d} {candidate_model_path}")
        result = run_harness(candidate_model_path, candidate_dir, hcfg)
        save_json(result, candidate_dir / "result.json")
        feedback = encode_feedback(
            result,
            protected_result,
            protected_result,
            target_metric=target_metric,
        )
        feedback["pemfc_context"] = build_pemfc_context(
            feedback,
            result=result,
            harness_config=hcfg,
        )
        save_json(feedback, candidate_dir / "feedback_vector.json")
    else:
        result = {
            "success": False,
            "run_dir": str(candidate_dir),
            "model_path": str(candidate_model_path),
            "error_type": "motif_quality" if quality_error else "validation",
            "error_message": quality_error or validation_error or "candidate model was not produced",
        }
        feedback = {
            "diagnostics": [],
            "features": {"has_exception": 1.0},
            "target_metric": target_metric,
        }

    decision = _accept_dispatch_candidate(
        protected_result,
        result,
        protected_feedback,
        feedback,
        target_metric,
        target_diagnostics=target_diagnostics,
        min_relative_improvement=min_relative_improvement,
        paper_baseline=paper_baseline,
    )
    row = {
        "candidate_index": candidate_index,
        "candidate_dir": str(candidate_dir),
        "model_path": str(candidate_model_path),
        "selected_motif": motif,
        "patch_meta": _compact_patch_meta(patch_meta),
        "success": bool(result.get("success")),
        "result": result,
        "feedback": feedback,
        "decision": decision,
    }
    save_json(
        {
            "candidate_index": candidate_index,
            "selected_motif": motif,
            "patch_meta": _compact_patch_meta(patch_meta),
            "success": bool(result.get("success")),
            "metrics": _paper_metric_summary(result) if result.get("success") else {},
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
            "decision": decision,
        },
        candidate_dir / "candidate_summary.json",
    )
    return row


def cmd_dispatch(args: argparse.Namespace) -> dict[str, Any]:
    ensure_project_dirs()
    validate_harness_specs()
    ensure_ms_aednet_data()

    run_root, _exp_cfg, hcfg, target_metric, llm_mode = _load_continue_run(args)
    orchestrator = GraphOrchestrator.open(run_root)
    history = orchestrator.history_rows(target_metric)
    _attach_saved_feedback_and_routes(orchestrator, history)
    dispatch_mode = _dispatch_mode(args)
    dispatch_candidate_limit = _dispatch_candidate_limit(args, dispatch_mode)
    archive_candidate_limit = _archive_candidate_limit(args, dispatch_mode)
    if not history:
        raise RuntimeError(f"No completed iterations found in {run_root}")
    protected_row = _best_history_row(history, target_metric)
    if protected_row is None:
        raise RuntimeError(f"No successful protected best model found in {run_root}")

    protected_iteration = int(protected_row["iteration"])
    protected_result = protected_row["result"]
    protected_feedback = protected_row.get("feedback")
    if not isinstance(protected_feedback, dict):
        protected_feedback = _load_iteration_json(orchestrator, protected_iteration, "feedback_vector")
    if not protected_feedback.get("pemfc_context"):
        protected_feedback["pemfc_context"] = build_pemfc_context(
            protected_feedback,
            result=protected_result,
            harness_config=hcfg,
        )
    protected_model_path = _iteration_source_path(protected_row, run_root)
    protected_source = read_model_source(protected_model_path)

    dispatch_dir = _unique_child_dir(run_root, args.dispatch_name or "evidence_dispatch")
    protected_dir = dispatch_dir / "protected_best"
    candidates_dir = dispatch_dir / "candidates"
    final_dir = dispatch_dir / "final"
    protected_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    protected_copy = protected_dir / "model.py"
    protected_copy.write_text(protected_source, encoding="utf-8")
    save_json(protected_result, protected_dir / "result.json")
    save_json(protected_feedback, protected_dir / "feedback_vector.json")

    evidence = _trajectory_evidence(
        orchestrator,
        history,
        target_metric,
        protected_iteration,
        limit=int(args.evidence_limit),
    )
    target_diagnostics = args.target_diagnostics or _dominant_diagnostics(protected_feedback)
    paper_baseline = _load_paper_baseline(
        hcfg,
        method=str(args.paper_baseline_method),
        override_mae=args.paper_baseline_mae,
        override_mse=args.paper_baseline_mse,
    )
    motif_library = _mine_dispatch_motifs(
        run_root,
        hcfg,
        target_metric,
        args.motif_sources,
        evidence_scope=str(args.evidence_scope),
        evidence_limit=int(args.evidence_limit),
        max_diff_chars=int(args.motif_diff_chars),
    )
    if dispatch_mode == "candidates":
        archive_library = _mine_archive_model_candidates(
            run_root,
            hcfg,
            target_metric,
            args.motif_sources,
            evidence_scope=str(args.evidence_scope),
            limit=archive_candidate_limit,
            protected_source=protected_source,
        )
    else:
        archive_library = {
            "sources": motif_library["sources"],
            "total_mined": 0,
            "candidates": [],
        }
    evidence_audit = build_run_evidence_audit(
        orchestrator.state,
        history,
        target_metric,
        candidate_tournament_k=max(1, dispatch_candidate_limit or 1),
    )
    payload = {
        "dispatch_objective": (
            "Summarize current-run PEMFC refinement evidence and keep the protected best "
            "model as the final artifact."
            if dispatch_mode == "summary"
            else (
                "Counterfactually transplant one historically successful PEMFC patch motif "
                "onto the protected best model. The candidate is accepted only if the fixed "
                "harness shrinks the paper-scale MAE/MSE gap to the Ms-AeDNet target."
            )
        ),
        "dispatch_mode": dispatch_mode,
        "method_framework": evidence_audit.get("method_framework"),
        "evidence_audit": evidence_audit,
        "protected_best": {
            "iteration": protected_iteration,
            "model_path": str(protected_model_path),
            "metrics": _paper_metric_summary(protected_result),
            "dominant_diagnostics": target_diagnostics,
            "diagnostic_probes": _diagnostic_probe_map(protected_feedback),
        },
        "paper_baseline": paper_baseline,
        "evidence_scope": str(args.evidence_scope),
        "trajectory_evidence": evidence,
        "motif_library": motif_library,
        "archive_model_library": archive_library,
        "acceptance_policy": {
            "protected_best_is_default_final": True,
            "summary_only_no_candidate_training": dispatch_mode == "summary",
            "candidate_must_pass_same_harness": True,
            "candidate_must_not_regress_target_metric": target_metric,
            "candidate_must_not_regress_inverse_mse": True,
            "candidate_must_shrink_ms_aednet_paper_mae_mse_gap": bool(paper_baseline),
            "candidate_must_improve_target_or_mse_or_target_diagnostic_without_paper_baseline": not bool(paper_baseline),
            "target_diagnostics": target_diagnostics,
            "min_relative_improvement": float(args.min_relative_improvement),
        },
    }
    save_json(payload, dispatch_dir / "dispatch_payload.json")

    llm_cfg = None
    if llm_mode in {"auto", "required"}:
        try:
            llm_cfg = load_llm_config(str(CONFIG_DIR / "forge_llm.yaml"))
        except Exception as exc:
            if llm_mode == "required":
                raise
            print(f"[FORGE] Evidence dispatcher LLM unavailable; no motif candidates will be generated: {exc}")
    max_repair_rounds = int((llm_cfg or {}).get("max_repair_rounds", 2)) if llm_cfg else 0

    print(f"[FORGE] Evidence Dispatch mode: {dispatch_mode}")
    print(f"[FORGE] Evidence Dispatch motif sources: {len(motif_library['sources'])}")
    print(f"[FORGE] Evidence Dispatch motifs mined: {motif_library['total_mined']}")
    if dispatch_mode == "candidates":
        print(f"[FORGE] Evidence Dispatch archive models mined: {archive_library['total_mined']}")
    if paper_baseline:
        print(
            "[FORGE] Reference target: "
            f"{paper_baseline['dataset']} L{paper_baseline['seq_len']} P{paper_baseline['pred_len']} "
            f"MAE={paper_baseline['mae']} MSE={paper_baseline['mse']}"
        )
    else:
        print("[FORGE] No reference target found; falling back to protected non-regression acceptance.")
    if not bool(getattr(args, "suppress_metric_summary", False)):
        protected_summary = {
            "best_iteration": protected_iteration,
            "best_metrics": _paper_metric_summary(protected_result),
            "paper_delta": _paper_target_delta(protected_result, paper_baseline),
        }
        _print_forge_best_summary(protected_summary, label="FORGE protected best")

    if dispatch_mode == "summary":
        report_payload = _compact_dispatch_report_payload(payload)
        report = _deterministic_dispatch_report(report_payload)
        if llm_cfg is not None:
            try:
                llm_report = request_llm_dispatch_summary(llm_cfg, report_payload)
                report = _merge_llm_dispatch_report(report, llm_report)
            except Exception as exc:
                if llm_mode == "required":
                    raise
                report["llm_error"] = f"{type(exc).__name__}: {exc}"
        save_json(report, dispatch_dir / "dispatch_report.json")

        final_model_path = final_dir / "model.py"
        shutil.copy2(protected_copy, final_model_path)
        save_json(protected_result, final_dir / "selected_result.json")

        motifs_for_report = report_payload["motif_library"]["motifs"]
        summary = {
            "schema": "forge.evidence_dispatch.summary.v1",
            "run_root": str(run_root),
            "dispatch_dir": str(dispatch_dir),
            "dispatch_mode": dispatch_mode,
            "target_metric": target_metric,
            "evidence_scope": str(args.evidence_scope),
            "paper_baseline": paper_baseline,
            "evidence_audit": evidence_audit,
            "protected_best": {
                "iteration": protected_iteration,
                "model_path": str(protected_model_path),
                "metrics": _paper_metric_summary(protected_result),
                "paper_gap": _paper_positive_gap(protected_result, paper_baseline),
                "paper_delta": _paper_target_delta(protected_result, paper_baseline),
            },
            "motif_library": {
                "sources": motif_library["sources"],
                "total_mined": motif_library["total_mined"],
                "used_count": len(motifs_for_report),
                "motifs": motifs_for_report,
            },
            "archive_model_library": {
                "sources": archive_library["sources"],
                "total_mined": archive_library["total_mined"],
                "used_count": 0,
                "candidates": [],
            },
            "candidates": [],
            "accepted_count": 0,
            "selected": "protected_best",
            "selected_candidate_index": None,
            "final_model_path": str(final_model_path),
            "dispatch_report_path": str(dispatch_dir / "dispatch_report.json"),
            "dispatch_report": report,
            "non_regression_guarantee": (
                "summary-only Evidence Dispatch never generates or adopts a new candidate; "
                "final_model is a copy of the protected best Diagnostic Prober model."
            ),
        }
        save_json(summary, dispatch_dir / "dispatch_summary.json")
        print("[FORGE] Evidence Dispatch summary-only: protected_best remains final")
        print(f"[FORGE] Dispatch report: {dispatch_dir / 'dispatch_report.json'}")
        print(f"[FORGE] Dispatch summary: {dispatch_dir / 'dispatch_summary.json'}")
        print(f"[FORGE] Final model: {final_model_path}")
        return summary

    candidate_records: list[dict[str, Any]] = []
    next_candidate_index = 0
    for archive in archive_library["candidates"]:
        archive_source = read_model_source(_resolve_stored_path(archive["model_path"], run_root))
        archive_candidate = PatchCandidate(
            source=archive_source,
            rationale="Historical same-harness model achieved stronger executable metrics; verify as an archive promotion candidate.",
            summary=f"Promote historical model {archive['archive_id']} as a fixed-harness candidate.",
            component="archive_model",
            origin="archive_model",
            edit_action="promote_historical_best_model",
            raw_response={"archive_model": archive},
        )
        archive_motif = {
            "motif_id": archive["archive_id"],
            "type": "archive_model_promotion",
            "source_run_root": archive["source_run_root"],
            "iteration": archive["iteration"],
            "model_path": archive["model_path"],
            "metrics": archive["metrics"],
        }
        candidate_records.append(
            _evaluate_dispatch_candidate(
                candidate_index=next_candidate_index,
                candidate=archive_candidate,
                candidate_dir=candidates_dir / f"candidate_{next_candidate_index:02d}_archive",
                protected_copy=protected_copy,
                protected_source=protected_source,
                protected_iteration=protected_iteration,
                protected_result=protected_result,
                protected_feedback=protected_feedback,
                protected_route=protected_row.get("route") or {},
                history=history,
                hcfg=hcfg,
                target_metric=target_metric,
                target_diagnostics=target_diagnostics,
                paper_baseline=paper_baseline,
                llm_cfg=llm_cfg,
                max_repair_rounds=0,
                motif=archive_motif,
                min_relative_improvement=float(args.min_relative_improvement),
            )
        )
        next_candidate_index += 1

    motifs = motif_library["motifs"][:dispatch_candidate_limit]
    if llm_cfg is None or not motifs:
        if llm_mode == "required" and not motifs and not candidate_records:
            raise RuntimeError("Evidence Dispatch required LLM candidates, but no successful motifs were mined")
    else:
        for motif in motifs:
            index = next_candidate_index
            candidate_payload = {
                **payload,
                "selected_motif": motif,
                "candidate_index": index,
                "candidate_count": len(motifs),
            }
            try:
                candidate = request_llm_dispatch_patch(llm_cfg, protected_source, candidate_payload)
                candidate.origin = "llm_motif_dispatch"
                if candidate.raw_response is None:
                    candidate.raw_response = {}
                candidate.raw_response["selected_motif_id"] = motif.get("motif_id")
            except Exception as exc:
                if llm_mode == "required":
                    raise
                candidate_records.append(
                    {
                        "candidate_index": index,
                        "selected_motif": motif,
                        "success": False,
                        "error_type": "llm_dispatch",
                        "error_message": f"{type(exc).__name__}: {exc}",
                        "decision": {"accepted": False, "reason": "llm_dispatch_failed"},
                    }
                )
                continue

            candidate_records.append(
                _evaluate_dispatch_candidate(
                    candidate_index=index,
                    candidate=candidate,
                    candidate_dir=candidates_dir / f"candidate_{index:02d}",
                    protected_copy=protected_copy,
                    protected_source=protected_source,
                    protected_iteration=protected_iteration,
                    protected_result=protected_result,
                    protected_feedback=protected_feedback,
                    protected_route=protected_row.get("route") or {},
                    history=history,
                    hcfg=hcfg,
                    target_metric=target_metric,
                    target_diagnostics=target_diagnostics,
                    paper_baseline=paper_baseline,
                    llm_cfg=llm_cfg,
                    max_repair_rounds=max_repair_rounds,
                    motif=motif,
                    min_relative_improvement=float(args.min_relative_improvement),
                )
            )
            next_candidate_index += 1

    accepted_records = [row for row in candidate_records if row.get("decision", {}).get("accepted")]
    selected_record = min(accepted_records, key=_candidate_selection_key) if accepted_records else None
    selected = "candidate" if selected_record else "protected_best"
    selected_model_path = Path(selected_record["model_path"]) if selected_record else protected_copy
    final_model_path = final_dir / "model.py"
    shutil.copy2(selected_model_path, final_model_path)
    final_result = selected_record["result"] if selected_record else protected_result
    save_json(final_result, final_dir / "selected_result.json")

    compact_candidates = [
        {
            "candidate_index": row.get("candidate_index"),
            "candidate_dir": row.get("candidate_dir"),
            "model_path": row.get("model_path"),
            "selected_motif_id": (row.get("selected_motif") or {}).get("motif_id"),
            "selected_motif": row.get("selected_motif"),
            "patch_meta": row.get("patch_meta"),
            "success": bool(row.get("success")),
            "metrics": _paper_metric_summary(row.get("result", {})) if row.get("success") else {},
            "error_type": row.get("result", {}).get("error_type") if isinstance(row.get("result"), dict) else row.get("error_type"),
            "error_message": row.get("result", {}).get("error_message") if isinstance(row.get("result"), dict) else row.get("error_message"),
            "decision": row.get("decision"),
        }
        for row in candidate_records
    ]
    summary = {
        "schema": "forge.evidence_dispatch.v2",
        "run_root": str(run_root),
        "dispatch_dir": str(dispatch_dir),
        "dispatch_mode": dispatch_mode,
        "target_metric": target_metric,
        "evidence_scope": str(args.evidence_scope),
        "paper_baseline": paper_baseline,
        "evidence_audit": evidence_audit,
        "protected_best": {
            "iteration": protected_iteration,
            "model_path": str(protected_model_path),
            "metrics": _paper_metric_summary(protected_result),
            "paper_gap": _paper_positive_gap(protected_result, paper_baseline),
            "paper_delta": _paper_target_delta(protected_result, paper_baseline),
        },
        "motif_library": {
            "sources": motif_library["sources"],
            "total_mined": motif_library["total_mined"],
            "used_count": len(motifs),
            "motifs": motifs,
        },
        "archive_model_library": {
            "sources": archive_library["sources"],
            "total_mined": archive_library["total_mined"],
            "used_count": len(archive_library["candidates"]),
            "candidates": archive_library["candidates"],
        },
        "candidates": compact_candidates,
        "accepted_count": len(accepted_records),
        "selected": selected,
        "selected_candidate_index": selected_record.get("candidate_index") if selected_record else None,
        "final_model_path": str(final_model_path),
        "non_regression_guarantee": (
            "final_model is a motif candidate only if it passes the fixed harness and shrinks the "
            "paper-scale MAE/MSE gap to the configured Ms-AeDNet target; otherwise protected_best is copied."
        ),
    }
    save_json(summary, dispatch_dir / "dispatch_summary.json")
    print(f"[FORGE] Evidence Dispatch selected: {selected}")
    print(f"[FORGE] Accepted motif candidates: {len(accepted_records)} / {len(candidate_records)}")
    print(f"[FORGE] Dispatch summary: {dispatch_dir / 'dispatch_summary.json'}")
    print(f"[FORGE] Final model: {final_model_path}")
    return summary


def _compact_dispatch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary.get("schema"),
        "dispatch_dir": summary.get("dispatch_dir"),
        "dispatch_mode": summary.get("dispatch_mode"),
        "evidence_scope": summary.get("evidence_scope"),
        "selected": summary.get("selected"),
        "accepted_count": summary.get("accepted_count"),
        "selected_candidate_index": summary.get("selected_candidate_index"),
        "final_model_path": summary.get("final_model_path"),
        "dispatch_report_path": summary.get("dispatch_report_path"),
        "protected_best": summary.get("protected_best"),
        "evidence_metrics": ((summary.get("evidence_audit") or {}).get("metrics") or {}),
    }


def _maybe_run_final_dispatch(args: argparse.Namespace, run_root: Path, target_metric: str) -> dict[str, Any] | None:
    if not bool(getattr(args, "final_dispatch", False)):
        return None
    dispatch_llm_mode = getattr(args, "dispatch_llm_mode", None) or getattr(args, "llm_mode", None) or "required"
    dispatch_mode = str(getattr(args, "dispatch_mode", None) or "summary")
    dispatch_args = argparse.Namespace(
        experiment_config=getattr(args, "experiment_config", str(CONFIG_DIR / "forge_experiment.yaml")),
        run_dir=str(run_root),
        dispatch_name=getattr(args, "dispatch_name", None) or "evidence_dispatch_summary",
        llm_mode=dispatch_llm_mode,
        dispatch_mode=dispatch_mode,
        target_metric=target_metric,
        target_diagnostics=None,
        evidence_limit=int(getattr(args, "evidence_limit", 16)),
        dispatch_candidates=getattr(args, "dispatch_candidates", None),
        archive_candidates=int(getattr(args, "archive_candidates", 0)),
        evidence_scope="current-run",
        motif_sources=None,
        motif_diff_chars=int(getattr(args, "motif_diff_chars", 12000)),
        paper_baseline_method=getattr(args, "paper_baseline_method", "Ms-AeDNet"),
        paper_baseline_mae=getattr(args, "paper_baseline_mae", None),
        paper_baseline_mse=getattr(args, "paper_baseline_mse", None),
        min_relative_improvement=float(getattr(args, "min_relative_improvement", 0.0)),
        epochs=getattr(args, "epochs", None),
        batch_size=getattr(args, "batch_size", None),
        lr=getattr(args, "lr", None),
        patience=getattr(args, "patience", None),
        seed=getattr(args, "seed", None),
        device=getattr(args, "device", None),
        cuda_id=getattr(args, "cuda_id", None),
        suppress_metric_summary=True,
    )
    print(f"[FORGE] Final Evidence Dispatch: current-run-only {dispatch_mode} evidence")
    return cmd_dispatch(dispatch_args)


def cmd_sweep(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    validate_harness_specs()
    ensure_ms_aednet_data()

    grid = get_benchmark_grid()
    datasets = args.datasets or grid["datasets"]
    seq_lens = args.seq_lens or grid["seq_lens"]
    pred_lens = args.pred_lens or grid["pred_lens"]

    exp_cfg = load_experiment_config(args.experiment_config)
    target_metric = args.target_metric or exp_cfg["evolution"]["target_metric"]
    rounds = int(args.rounds if args.rounds is not None else exp_cfg["evolution"]["rounds"])
    llm_mode = args.llm_mode or exp_cfg["evolution"]["llm_mode"]
    sweep_name = args.run_name or f"forge_sweep_{_timestamp()}"
    sweep_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (RUNS_DIR / sweep_name).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print(f"[FORGE] Sweep root: {sweep_root}")
    print(f"[FORGE] Sweep datasets: {datasets}")
    print(f"[FORGE] Sweep seq_lens: {seq_lens}")
    print(f"[FORGE] Sweep pred_lens: {pred_lens}")
    print(f"[FORGE] Sweep LLM mode: {llm_mode} | rounds: {rounds}")
    _warn_if_heuristic_only(llm_mode, rounds, scope="sweep")

    for dataset in datasets:
        for seq_len in seq_lens:
            for pred_len in pred_lens:
                combo_name = f"{dataset}_L{seq_len}_P{pred_len}"
                combo_args = argparse.Namespace(**vars(args))
                combo_args.data = dataset
                combo_args.seq_len = int(seq_len)
                combo_args.pred_len = int(pred_len)
                combo_args.run_dir = str(sweep_root / combo_name)
                combo_args.run_name = None
                combo_args._sweep_child = True
                print(f"[FORGE] Sweep combo: {combo_name}")
                row = {
                    "dataset": dataset,
                    "seq_len": int(seq_len),
                    "pred_len": int(pred_len),
                    "success": False,
                    "best_target": None,
                    "best_iteration": None,
                    "best_paper_mae": None,
                    "best_paper_mse": None,
                    "best_run_dir": None,
                    "run_root": combo_args.run_dir,
                    "error": None,
                }
                try:
                    summary = cmd_run(combo_args)
                    row["success"] = True
                    row["best_target"] = summary.get("best_target")
                    row["best_run_dir"] = summary.get("best_run_dir")
                    row["evidence_audit"] = summary.get("evidence_audit")
                    _attach_summary_metrics_to_sweep_row(row, summary)
                    if summary.get("final_dispatch"):
                        row["final_dispatch"] = summary.get("final_dispatch")
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[FORGE] Sweep combo failed: {combo_name}: {row['error']}")
                rows.append(row)
                _write_sweep_outputs(rows, sweep_root, target_metric)

    print(f"[FORGE] Sweep finished. Summary: {sweep_root / 'sweep_summary.json'}")


def cmd_summarize_sweep(args: argparse.Namespace) -> None:
    sweep_root = Path(args.sweep_dir).expanduser().resolve()
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_root}")
    existing_summary = sweep_root / "sweep_summary.json"
    if args.target_metric:
        target_metric = args.target_metric
    elif existing_summary.exists():
        target_metric = str(load_json(existing_summary).get("target_metric") or "mae_inverse")
    else:
        target_metric = "mae_inverse"

    _refresh_sweep_summary(sweep_root, target_metric)
    print(f"[FORGE] Sweep summary refreshed: {sweep_root / 'sweep_summary.json'}")


def cmd_summarize_run(args: argparse.Namespace) -> dict[str, Any]:
    run_root, _exp_cfg, hcfg, target_metric, _llm_mode = _load_continue_run(args)
    if int(getattr(args, "candidate_tournament_k", 1)) != 1:
        raise ValueError("Stable FORGE currently executes one candidate per round; keep --candidate-tournament-k 1")
    previous_summary = load_json(run_root / "summary.json") if (run_root / "summary.json").exists() else {}
    orchestrator = GraphOrchestrator.open(run_root)
    history = orchestrator.history_rows(target_metric)
    _attach_saved_feedback_and_routes(orchestrator, history)
    if not history:
        raise RuntimeError(f"No completed iterations found in {run_root}")
    last_iteration = max(int(row["iteration"]) for row in history)
    summary = _write_run_summary(
        run_root,
        last_iteration,
        target_metric,
        history,
        graph_state=orchestrator.state,
        hcfg=hcfg,
        candidate_tournament_k=int(getattr(args, "candidate_tournament_k", 1)),
    )
    if previous_summary.get("final_dispatch"):
        summary["final_dispatch"] = previous_summary["final_dispatch"]
        save_json(summary, run_root / "summary.json")
    _print_forge_best_summary(summary)
    print(f"[FORGE] Run summary refreshed: {run_root / 'summary.json'}")
    return summary


def _add_final_dispatch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-dispatch", action="store_true", help="run current-run-only evidence summary after iterations")
    parser.add_argument("--dispatch-name", default="evidence_dispatch_summary")
    parser.add_argument("--dispatch-llm-mode", choices=["auto", "off", "required"], default=None)
    parser.add_argument("--dispatch-mode", choices=["summary", "candidates"], default="summary")
    parser.add_argument("--dispatch-candidates", type=int, default=None)
    parser.add_argument("--archive-candidates", type=int, default=0)
    parser.add_argument("--evidence-limit", type=int, default=16)
    parser.add_argument("--motif-diff-chars", type=int, default=12000)
    parser.add_argument("--paper-baseline-method", default="Ms-AeDNet")
    parser.add_argument("--paper-baseline-mae", type=float, default=None)
    parser.add_argument("--paper-baseline-mse", type=float, default=None)
    parser.add_argument("--min-relative-improvement", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FORGE PEMFC model evolution system")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="extract assets and verify project layout")
    init_p.set_defaults(func=cmd_init)

    run_p = sub.add_parser("run", help="run fixed harness and feedback-routed evolution")
    run_p.add_argument("--experiment-config", default=str(CONFIG_DIR / "forge_experiment.yaml"))
    run_p.add_argument("--run-dir", default=None)
    run_p.add_argument("--run-name", default=None)
    run_p.add_argument("--rounds", type=int, default=None)
    run_p.add_argument("--llm-mode", choices=["auto", "off", "required"], default=None)
    run_p.add_argument("--parent-policy", choices=["best", "last"], default="best")
    run_p.add_argument("--routing-mode", choices=["trust", "trust-action", "prior", "rule"], default="trust")
    run_p.add_argument("--candidate-tournament-k", type=int, default=1)
    run_p.add_argument("--target-metric", default=None)
    run_p.add_argument("--data", choices=sorted(get_dataset_files()), default=None)
    run_p.add_argument("--data-path", default=None)
    run_p.add_argument("--seq-len", type=int, default=None)
    run_p.add_argument("--pred-len", type=int, default=None)
    run_p.add_argument("--scaling", choices=["baseline", "train"], default=None)
    run_p.add_argument("--limit-rows", type=int, default=None)
    run_p.add_argument("--epochs", type=int, default=None)
    run_p.add_argument("--batch-size", type=int, default=None)
    run_p.add_argument("--lr", type=float, default=None)
    run_p.add_argument("--patience", type=int, default=None)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    run_p.add_argument("--cuda-id", type=int, default=None)
    run_p.add_argument("--hidden-dim", type=int, default=None)
    run_p.add_argument("--layer", type=int, default=None)
    run_p.add_argument("--dropout", type=float, default=None)
    _add_final_dispatch_args(run_p)
    run_p.set_defaults(func=cmd_run)

    continue_p = sub.add_parser("continue", help="continue an existing run from its last completed iteration")
    continue_p.add_argument("--experiment-config", default=str(CONFIG_DIR / "forge_experiment.yaml"))
    continue_p.add_argument("--run-dir", required=True)
    continue_p.add_argument("--to-round", type=int, default=None, help="absolute final iteration index to reach")
    continue_p.add_argument(
        "--additional-rounds",
        type=int,
        default=None,
        help="number of new evolution rounds to add after the last completed iteration",
    )
    continue_p.add_argument("--llm-mode", choices=["auto", "off", "required"], default=None)
    continue_p.add_argument("--parent-policy", choices=["best", "last"], default="best")
    continue_p.add_argument("--routing-mode", choices=["trust", "trust-action", "prior", "rule"], default="trust")
    continue_p.add_argument("--candidate-tournament-k", type=int, default=1)
    continue_p.add_argument("--target-metric", default=None)
    continue_p.add_argument("--epochs", type=int, default=None)
    continue_p.add_argument("--batch-size", type=int, default=None)
    continue_p.add_argument("--lr", type=float, default=None)
    continue_p.add_argument("--patience", type=int, default=None)
    continue_p.add_argument("--seed", type=int, default=None)
    continue_p.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    continue_p.add_argument("--cuda-id", type=int, default=None)
    _add_final_dispatch_args(continue_p)
    continue_p.set_defaults(func=cmd_continue)

    dispatch_p = sub.add_parser("dispatch", help="run protected Evidence Dispatch over a completed run")
    dispatch_p.add_argument("--experiment-config", default=str(CONFIG_DIR / "forge_experiment.yaml"))
    dispatch_p.add_argument("--run-dir", required=True)
    dispatch_p.add_argument("--dispatch-name", default="evidence_dispatch_summary")
    dispatch_p.add_argument("--llm-mode", choices=["auto", "off", "required"], default="required")
    dispatch_p.add_argument("--dispatch-mode", choices=["summary", "candidates"], default="summary")
    dispatch_p.add_argument("--target-metric", default=None)
    dispatch_p.add_argument("--target-diagnostics", nargs="+", default=None)
    dispatch_p.add_argument("--evidence-limit", type=int, default=16)
    dispatch_p.add_argument("--dispatch-candidates", type=int, default=None)
    dispatch_p.add_argument("--archive-candidates", type=int, default=0)
    dispatch_p.add_argument(
        "--evidence-scope",
        choices=["current-run", "matching-runs", "explicit"],
        default="current-run",
    )
    dispatch_p.add_argument("--motif-sources", nargs="*", default=None)
    dispatch_p.add_argument("--motif-diff-chars", type=int, default=12000)
    dispatch_p.add_argument("--paper-baseline-method", default="Ms-AeDNet")
    dispatch_p.add_argument("--paper-baseline-mae", type=float, default=None)
    dispatch_p.add_argument("--paper-baseline-mse", type=float, default=None)
    dispatch_p.add_argument("--min-relative-improvement", type=float, default=0.0)
    dispatch_p.add_argument("--epochs", type=int, default=None)
    dispatch_p.add_argument("--batch-size", type=int, default=None)
    dispatch_p.add_argument("--lr", type=float, default=None)
    dispatch_p.add_argument("--patience", type=int, default=None)
    dispatch_p.add_argument("--seed", type=int, default=None)
    dispatch_p.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    dispatch_p.add_argument("--cuda-id", type=int, default=None)
    dispatch_p.set_defaults(func=cmd_dispatch)

    sweep_p = sub.add_parser("sweep", help="run benchmark grid over datasets, history lengths, and horizons")
    sweep_p.add_argument("--experiment-config", default=str(CONFIG_DIR / "forge_experiment.yaml"))
    sweep_p.add_argument("--run-dir", default=None)
    sweep_p.add_argument("--run-name", default=None)
    sweep_p.add_argument("--rounds", type=int, default=None)
    sweep_p.add_argument("--llm-mode", choices=["auto", "off", "required"], default=None)
    sweep_p.add_argument("--parent-policy", choices=["best", "last"], default="best")
    sweep_p.add_argument("--routing-mode", choices=["trust", "trust-action", "prior", "rule"], default="trust")
    sweep_p.add_argument("--candidate-tournament-k", type=int, default=1)
    sweep_p.add_argument("--target-metric", default=None)
    sweep_p.add_argument("--datasets", nargs="+", choices=sorted(get_dataset_files()), default=None)
    sweep_p.add_argument("--seq-lens", nargs="+", type=int, default=None)
    sweep_p.add_argument("--pred-lens", nargs="+", type=int, default=None)
    sweep_p.add_argument("--data-path", default=None)
    sweep_p.add_argument("--scaling", choices=["baseline", "train"], default=None)
    sweep_p.add_argument("--limit-rows", type=int, default=None)
    sweep_p.add_argument("--epochs", type=int, default=None)
    sweep_p.add_argument("--batch-size", type=int, default=None)
    sweep_p.add_argument("--lr", type=float, default=None)
    sweep_p.add_argument("--patience", type=int, default=None)
    sweep_p.add_argument("--seed", type=int, default=None)
    sweep_p.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    sweep_p.add_argument("--cuda-id", type=int, default=None)
    sweep_p.add_argument("--hidden-dim", type=int, default=None)
    sweep_p.add_argument("--layer", type=int, default=None)
    sweep_p.add_argument("--dropout", type=float, default=None)
    _add_final_dispatch_args(sweep_p)
    sweep_p.set_defaults(func=cmd_sweep)

    summarize_run_p = sub.add_parser("summarize-run", help="refresh one completed run summary without training")
    summarize_run_p.add_argument("--experiment-config", default=str(CONFIG_DIR / "forge_experiment.yaml"))
    summarize_run_p.add_argument("--run-dir", required=True)
    summarize_run_p.add_argument("--target-metric", default=None)
    summarize_run_p.add_argument("--llm-mode", choices=["auto", "off", "required"], default=None)
    summarize_run_p.add_argument("--candidate-tournament-k", type=int, default=1)
    summarize_run_p.set_defaults(func=cmd_summarize_run)

    summarize_p = sub.add_parser("summarize-sweep", help="refresh sweep summary from child run summaries")
    summarize_p.add_argument("--sweep-dir", required=True)
    summarize_p.add_argument("--target-metric", default=None)
    summarize_p.set_defaults(func=cmd_summarize_sweep)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
