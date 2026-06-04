from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .assets import ensure_ms_aednet_data
from .config import load_experiment_config, load_json, save_json
from .feedback import encode_feedback
from .harness import HarnessConfig, run_harness
from .harness_spec import (
    get_benchmark_grid,
    get_dataset_files,
    get_default_dataset_name,
    get_enc_in,
    get_feature_dim,
    validate_harness_specs,
)
from .llm import load_llm_config
from .memory import build_pemfc_context
from .model_io import read_model_source
from .orchestrator import GraphOrchestrator
from .patching import (
    apply_candidate,
    heuristic_patch_source,
    request_llm_patch,
    request_llm_repair_patch,
    safety_fallback_candidate,
    save_failed_candidate_attempt,
)
from .paths import CONFIG_DIR, INITIAL_MODEL_PATH, PROJECT_ROOT, RUNS_DIR, ensure_project_dirs
from .report import write_iteration_report
from .routing import TRUST_ROUTING_MODES, route_feedback


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
) -> dict[str, Any]:
    summary = {
        "run_root": str(run_root),
        "rounds": rounds,
        "target_metric": target_metric,
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
    best = _best_result(history, target_metric)
    if best:
        summary["best_target"] = best.get("metrics", {}).get("target", {}).get(target_metric)
        summary["best_run_dir"] = best.get("run_dir")
    save_json(summary, run_root / "summary.json")
    return summary


def _write_sweep_outputs(rows: list[dict[str, Any]], sweep_root: Path, target_metric: str) -> None:
    summary = {
        "sweep_root": str(sweep_root),
        "target_metric": target_metric,
        "count": len(rows),
        "rows": rows,
    }
    save_json(summary, sweep_root / "sweep_summary.json")

    csv_path = sweep_root / "sweep_summary.csv"
    fieldnames = [
        "dataset",
        "seq_len",
        "pred_len",
        "success",
        "best_target",
        "best_run_dir",
        "run_root",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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
    hcfg = _harness_config_from_args(args, exp_cfg)

    run_name = args.run_name or f"forge_{_timestamp()}"
    run_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (RUNS_DIR / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "experiment_config": exp_cfg,
            "harness_config": hcfg.__dict__,
            "runtime": {"parent_policy": parent_policy, "routing_mode": routing_mode},
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

    summary = _write_run_summary(run_root, rounds, target_metric, history)
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

    summary = _write_run_summary(run_root, to_round, target_metric, history)
    orchestrator.event("continue_finished", {"to_round": to_round})
    orchestrator.save()
    print(f"[FORGE] Continue finished. Summary: {run_root / 'summary.json'}")
    return summary


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
                    "best_run_dir": None,
                    "run_root": combo_args.run_dir,
                    "error": None,
                }
                try:
                    summary = cmd_run(combo_args)
                    row["success"] = True
                    row["best_target"] = summary.get("best_target")
                    row["best_run_dir"] = summary.get("best_run_dir")
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

    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^(?P<dataset>[^_]+)_L(?P<seq_len>\d+)_P(?P<pred_len>\d+)$")
    for child in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
        match = pattern.match(child.name)
        if not match:
            continue
        row = {
            "dataset": match.group("dataset"),
            "seq_len": int(match.group("seq_len")),
            "pred_len": int(match.group("pred_len")),
            "success": False,
            "best_target": None,
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

    _write_sweep_outputs(rows, sweep_root, target_metric)
    print(f"[FORGE] Sweep summary refreshed: {sweep_root / 'sweep_summary.json'}")


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
    continue_p.add_argument("--target-metric", default=None)
    continue_p.add_argument("--epochs", type=int, default=None)
    continue_p.add_argument("--batch-size", type=int, default=None)
    continue_p.add_argument("--lr", type=float, default=None)
    continue_p.add_argument("--patience", type=int, default=None)
    continue_p.add_argument("--seed", type=int, default=None)
    continue_p.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    continue_p.add_argument("--cuda-id", type=int, default=None)
    continue_p.set_defaults(func=cmd_continue)

    sweep_p = sub.add_parser("sweep", help="run benchmark grid over datasets, history lengths, and horizons")
    sweep_p.add_argument("--experiment-config", default=str(CONFIG_DIR / "forge_experiment.yaml"))
    sweep_p.add_argument("--run-dir", default=None)
    sweep_p.add_argument("--run-name", default=None)
    sweep_p.add_argument("--rounds", type=int, default=None)
    sweep_p.add_argument("--llm-mode", choices=["auto", "off", "required"], default=None)
    sweep_p.add_argument("--parent-policy", choices=["best", "last"], default="best")
    sweep_p.add_argument("--routing-mode", choices=["trust", "trust-action", "prior", "rule"], default="trust")
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
    sweep_p.set_defaults(func=cmd_sweep)

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
