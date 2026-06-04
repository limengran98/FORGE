import json
from argparse import Namespace

import pytest

from forge.cli import _maybe_refresh_parent_sweep_summary, _parent_baseline_for_patch, _resolve_continue_target, build_parser
from forge.orchestrator import GraphOrchestrator


def test_sweep_parser_accepts_subset_grid():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sweep",
            "--datasets",
            "FC1",
            "FC2",
            "--seq-lens",
            "24",
            "48",
            "--pred-lens",
            "6",
            "12",
            "--epochs",
            "1",
            "--llm-mode",
            "off",
        ]
    )
    assert args.datasets == ["FC1", "FC2"]
    assert args.seq_lens == [24, 48]
    assert args.pred_lens == [6, 12]
    assert args.epochs == 1


def test_continue_parser_accepts_resume_target():
    parser = build_parser()
    args = parser.parse_args(
        [
            "continue",
            "--run-dir",
            "runs/demo",
            "--to-round",
            "3",
            "--epochs",
            "10",
            "--llm-mode",
            "required",
            "--parent-policy",
            "best",
            "--routing-mode",
            "trust",
            "--device",
            "cuda",
            "--cuda-id",
            "1",
        ]
    )
    assert args.run_dir == "runs/demo"
    assert args.to_round == 3
    assert args.epochs == 10
    assert args.llm_mode == "required"
    assert args.parent_policy == "best"
    assert args.routing_mode == "trust"
    assert args.device == "cuda"
    assert args.cuda_id == 1


def test_sweep_parser_accepts_trust_action_ablation_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sweep",
            "--datasets",
            "FC1",
            "--seq-lens",
            "24",
            "--pred-lens",
            "12",
            "--llm-mode",
            "off",
            "--routing-mode",
            "trust-action",
        ]
    )
    assert args.routing_mode == "trust-action"


def test_continue_target_defaults_to_one_more_round():
    args = Namespace(to_round=None, additional_rounds=None)
    assert _resolve_continue_target(args, last_iteration=1) == 2


def test_continue_target_accepts_additional_rounds():
    args = Namespace(to_round=None, additional_rounds=2)
    assert _resolve_continue_target(args, last_iteration=1) == 3


def test_continue_target_rejects_ambiguous_target():
    args = Namespace(to_round=3, additional_rounds=1)
    with pytest.raises(ValueError, match="Use either"):
        _resolve_continue_target(args, last_iteration=1)


def test_summarize_sweep_parser_accepts_sweep_dir():
    parser = build_parser()
    args = parser.parse_args(["summarize-sweep", "--sweep-dir", "runs/demo_sweep"])
    assert args.sweep_dir == "runs/demo_sweep"


def test_continue_refreshes_parent_sweep_summary(tmp_path):
    sweep_root = tmp_path / "pilot"
    fc1 = sweep_root / "FC1_L24_P12"
    fc2 = sweep_root / "FC2_L24_P12"
    fc1.mkdir(parents=True)
    fc2.mkdir(parents=True)
    (sweep_root / "sweep_summary.json").write_text(
        json.dumps({"target_metric": "mae_inverse", "rows": []}),
        encoding="utf-8",
    )
    (fc1 / "summary.json").write_text(
        json.dumps({"best_target": 0.11, "best_run_dir": str(fc1 / "iter_020")}),
        encoding="utf-8",
    )
    (fc2 / "summary.json").write_text(
        json.dumps({"best_target": 0.22, "best_run_dir": str(fc2 / "iter_010")}),
        encoding="utf-8",
    )

    refreshed = _maybe_refresh_parent_sweep_summary(fc1, "mae_inverse")

    assert refreshed == sweep_root
    summary = json.loads((sweep_root / "sweep_summary.json").read_text(encoding="utf-8"))
    rows = {row["dataset"]: row for row in summary["rows"]}
    assert rows["FC1"]["best_target"] == 0.11
    assert rows["FC2"]["best_target"] == 0.22
    assert (sweep_root / "sweep_summary.csv").exists()


def test_standalone_combo_name_does_not_create_parent_sweep_summary(tmp_path):
    run_root = tmp_path / "FC1_L24_P12"
    run_root.mkdir()

    refreshed = _maybe_refresh_parent_sweep_summary(run_root, "mae_inverse")

    assert refreshed is None
    assert not (tmp_path / "sweep_summary.json").exists()


def test_parent_baseline_uses_recorded_best_parent(tmp_path):
    orch = GraphOrchestrator.open(tmp_path)
    orch.ensure_iteration(1, tmp_path / "iter_001")
    orch.state["iterations"]["iter_001"]["patch"] = {"parent_iteration": 0}
    parent_result = {"metrics": {"target": {"mae_inverse": 0.1}}}
    parent_feedback = {"current_target": 0.1}
    default_result = {"metrics": {"target": {"mae_inverse": 0.5}}}
    default_feedback = {"current_target": 0.5}
    history = [
        {"iteration": 0, "result": parent_result, "feedback": parent_feedback},
        {"iteration": 1, "result": default_result, "feedback": default_feedback},
    ]

    result, feedback = _parent_baseline_for_patch(
        orch,
        history,
        patch_iteration=1,
        default_result=default_result,
        default_feedback=default_feedback,
    )

    assert result is parent_result
    assert feedback is parent_feedback
