from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .assets import ensure_ms_aednet_data
from .config import load_experiment_config, save_json
from .feedback import encode_feedback
from .harness import HarnessConfig, run_harness
from .harness_spec import get_dataset_files, get_default_dataset_name, get_enc_in, get_feature_dim
from .llm import load_llm_config
from .model_io import read_model_source
from .orchestrator import GraphOrchestrator
from .patching import apply_candidate, heuristic_patch_source, request_llm_patch
from .paths import CONFIG_DIR, INITIAL_MODEL_PATH, RUNS_DIR, ensure_project_dirs
from .report import write_iteration_report
from .routing import route_feedback


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _harness_config_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> HarnessConfig:
    data_cfg = cfg["data"]
    harness_cfg = cfg["harness"]
    model_cfg = cfg["model"]
    return HarnessConfig(
        data_name=args.data or data_cfg.get("name") or get_default_dataset_name(),
        data_path=args.data_path,
        seq_len=int(args.seq_len or data_cfg["seq_len"]),
        pred_len=int(args.pred_len or data_cfg["pred_len"]),
        scaling=str(args.scaling or data_cfg["scaling"]),
        limit_rows=args.limit_rows if args.limit_rows is not None else data_cfg.get("limit_rows"),
        enc_in=int(model_cfg.get("enc_in", get_enc_in())),
        hidden_dim=int(args.hidden_dim or model_cfg["hidden_dim"]),
        layer=int(args.layer or model_cfg["layer"]),
        dropout=float(args.dropout if args.dropout is not None else model_cfg["dropout"]),
        batch_size=int(args.batch_size or harness_cfg["batch_size"]),
        lr=float(args.lr or harness_cfg["lr"]),
        epochs=int(args.epochs or harness_cfg["epochs"]),
        patience=int(args.patience or harness_cfg["patience"]),
        seed=int(args.seed or harness_cfg["seed"]),
        device=str(args.device or harness_cfg["device"]),
        cuda_id=int(args.cuda_id if args.cuda_id is not None else harness_cfg.get("cuda_id", 0)),
        num_workers=int(harness_cfg.get("num_workers", 0)),
    )


def cmd_init(args: argparse.Namespace) -> None:
    ensure_project_dirs()
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


def _history_row(iteration: int, result: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    target = result.get("metrics", {}).get("target", {}) if result.get("success") else {}
    return {
        "iteration": iteration,
        "success": result.get("success"),
        "target": target,
        "primary_component": route.get("primary_component"),
        "active_components": route.get("active_components"),
        "run_dir": result.get("run_dir"),
        "result": result,
    }


def cmd_run(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    ensure_ms_aednet_data()
    exp_cfg = load_experiment_config(args.experiment_config)
    target_metric = args.target_metric or exp_cfg["evolution"]["target_metric"]
    rounds = int(args.rounds if args.rounds is not None else exp_cfg["evolution"]["rounds"])
    llm_mode = args.llm_mode or exp_cfg["evolution"]["llm_mode"]
    hcfg = _harness_config_from_args(args, exp_cfg)

    run_name = args.run_name or f"forge_{_timestamp()}"
    run_root = Path(args.run_dir) if args.run_dir else RUNS_DIR / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    save_json({"experiment_config": exp_cfg, "harness_config": hcfg.__dict__}, run_root / "run_config.json")

    orchestrator = GraphOrchestrator.open(run_root)
    history: list[dict[str, Any]] = []

    iter0 = run_root / "iter_000"
    iter0.mkdir(parents=True, exist_ok=True)
    current_model_path = iter0 / "model.py"
    if not current_model_path.exists():
        shutil.copy2(INITIAL_MODEL_PATH, current_model_path)

    print(f"[FORGE] Run root: {run_root}")
    print(f"[FORGE] Device: {hcfg.device}:{hcfg.cuda_id if hcfg.device == 'cuda' else ''}")
    print(f"[FORGE] LLM mode: {llm_mode}")
    print(f"[FORGE] Rounds: {rounds}")

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
            feedback_path = iter_dir / "feedback_vector.json"
            save_json(feedback, feedback_path)
            orchestrator.record_artifact(iteration, "feedback_vector", feedback_path)

        with orchestrator.stage(iteration, "route"):
            route = route_feedback(feedback)
            route_path = iter_dir / "routing.json"
            save_json(route, route_path)
            orchestrator.record_artifact(iteration, "routing", route_path)
            orchestrator.record_feedback_and_route(iteration, feedback, route, result)

        patch_meta = None
        history.append(_history_row(iteration, result, route))
        if iteration < rounds:
            next_dir = run_root / f"iter_{iteration + 1:03d}"
            with orchestrator.stage(iteration, "patch", {"next_iteration": iteration + 1}):
                next_dir.mkdir(parents=True, exist_ok=True)
                previous_source = read_model_source(current_model_path)
                candidate = None
                if llm_mode in {"auto", "required"}:
                    try:
                        llm_cfg = load_llm_config(str(CONFIG_DIR / "forge_llm.yaml"))
                        candidate = request_llm_patch(
                            llm_cfg,
                            iteration + 1,
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
                validation_cfg = SimpleNamespace(
                    seq_len=hcfg.seq_len,
                    pred_len=hcfg.pred_len,
                    enc_in=hcfg.enc_in,
                    hidden_dim=hcfg.hidden_dim,
                    layer=hcfg.layer,
                    dropout=hcfg.dropout,
                    feature_dim=feature_dim,
                )
                try:
                    patch_meta = apply_candidate(
                        candidate,
                        current_model_path,
                        next_dir / "model.py",
                        validation_cfg,
                        feature_dim=feature_dim,
                        artifact_dir=next_dir,
                    )
                except Exception as exc:
                    if llm_mode == "auto" and candidate.origin == "llm":
                        print(f"[FORGE] LLM patch failed validation, using heuristic fallback: {exc}")
                        candidate = heuristic_patch_source(previous_source, route)
                        patch_meta = apply_candidate(
                            candidate,
                            current_model_path,
                            next_dir / "model.py",
                            validation_cfg,
                            feature_dim=feature_dim,
                            artifact_dir=next_dir,
                        )
                    else:
                        raise
                orchestrator.record_patch(iteration, patch_meta)
                current_model_path = next_dir / "model.py"
        else:
            orchestrator.skip_stage(iteration, "patch", "final_iteration")

        with orchestrator.stage(iteration, "report"):
            report_path = iter_dir / "report.md"
            write_iteration_report(report_path, iteration, result, feedback, route, patch_meta)
            orchestrator.record_artifact(iteration, "report", report_path)
        orchestrator.finish_iteration(iteration)

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
    print(f"[FORGE] Finished. Summary: {run_root / 'summary.json'}")


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
