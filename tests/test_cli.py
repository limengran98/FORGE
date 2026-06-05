import json
from argparse import Namespace

import pytest

from forge.cli import (
    _accept_dispatch_candidate,
    _dispatch_patch_quality,
    _format_paper_delta_line,
    _maybe_refresh_parent_sweep_summary,
    _paper_positive_gap,
    _paper_target_delta,
    _parent_baseline_for_patch,
    _print_forge_best_summary,
    _resolve_continue_target,
    _write_run_summary,
    build_parser,
)
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
            "--candidate-tournament-k",
            "1",
        ]
    )
    assert args.routing_mode == "trust-action"
    assert args.candidate_tournament_k == 1


def test_dispatch_parser_accepts_protected_evidence_dispatch():
    parser = build_parser()
    args = parser.parse_args(
        [
            "dispatch",
            "--run-dir",
            "runs/demo/FC1_L24_P12",
            "--llm-mode",
            "required",
            "--target-diagnostics",
            "long_horizon_error",
            "residual_autocorrelation",
            "--device",
            "cuda",
            "--cuda-id",
            "0",
        ]
    )
    assert args.run_dir == "runs/demo/FC1_L24_P12"
    assert args.llm_mode == "required"
    assert args.dispatch_mode == "summary"
    assert args.target_diagnostics == ["long_horizon_error", "residual_autocorrelation"]
    assert args.evidence_scope == "current-run"
    assert args.dispatch_candidates is None
    assert args.archive_candidates == 0


def test_sweep_parser_defaults_final_dispatch_to_summary_only():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sweep",
            "--datasets",
            "FC1",
            "FC2",
            "--seq-lens",
            "24",
            "--pred-lens",
            "12",
            "--rounds",
            "20",
            "--llm-mode",
            "required",
            "--final-dispatch",
            "--archive-candidates",
            "0",
        ]
    )
    assert args.final_dispatch is True
    assert args.dispatch_mode == "summary"
    assert args.dispatch_candidates is None
    assert args.archive_candidates == 0


def test_sweep_parser_accepts_candidate_dispatch_ablation():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sweep",
            "--datasets",
            "FC1",
            "FC2",
            "--seq-lens",
            "24",
            "--pred-lens",
            "12",
            "--rounds",
            "20",
            "--llm-mode",
            "required",
            "--final-dispatch",
            "--dispatch-mode",
            "candidates",
            "--dispatch-candidates",
            "4",
            "--archive-candidates",
            "0",
        ]
    )
    assert args.final_dispatch is True
    assert args.dispatch_mode == "candidates"
    assert args.dispatch_candidates == 4
    assert args.archive_candidates == 0


def test_dispatch_candidate_rejects_metric_regression():
    protected = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.10},
            "inverse": {"mse": 0.20},
        },
    }
    candidate = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.11},
            "inverse": {"mse": 0.19},
        },
    }
    feedback = {"diagnostics": [{"name": "long_horizon_error", "severity": 0.8, "confidence": 1.0}]}
    candidate_feedback = {"diagnostics": [{"name": "long_horizon_error", "severity": 0.1, "confidence": 1.0}]}

    decision = _accept_dispatch_candidate(
        protected,
        candidate,
        feedback,
        candidate_feedback,
        "mae_inverse",
        target_diagnostics=["long_horizon_error"],
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "target_metric_regressed"


def test_dispatch_candidate_accepts_non_regression_with_probe_gain():
    protected = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.10},
            "inverse": {"mse": 0.20},
        },
    }
    candidate = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.10},
            "inverse": {"mse": 0.20},
        },
    }
    feedback = {"diagnostics": [{"name": "long_horizon_error", "severity": 0.8, "confidence": 1.0}]}
    candidate_feedback = {"diagnostics": [{"name": "long_horizon_error", "severity": 0.2, "confidence": 1.0}]}

    decision = _accept_dispatch_candidate(
        protected,
        candidate,
        feedback,
        candidate_feedback,
        "mae_inverse",
        target_diagnostics=["long_horizon_error"],
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "accepted_by_non_regression_harness"


def test_dispatch_candidate_accepts_ms_aednet_gap_shrink():
    protected = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.10},
            "inverse": {"mse": 0.20},
            "paper_scaled": {"mae": 5.00, "mse": 12.00},
        },
    }
    candidate = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.09},
            "inverse": {"mse": 0.18},
            "paper_scaled": {"mae": 4.80, "mse": 11.20},
        },
    }

    decision = _accept_dispatch_candidate(
        protected,
        candidate,
        {"diagnostics": []},
        {"diagnostics": []},
        "mae_inverse",
        paper_baseline={"method": "Ms-AeDNet", "mae": 4.56, "mse": 10.40},
    )

    assert decision["accepted"] is True
    assert decision["reason"] == "accepted_by_counterfactual_gap_harness"
    assert decision["paper_gap_decision"]["gap_delta"]["total"] > 0


def test_paper_gap_zero_when_forge_beats_target_but_signed_delta_is_negative():
    result = {
        "success": True,
        "metrics": {
            "paper_scaled": {"mae": 4.2593, "mse": 8.9407},
            "target": {"mae_inverse": 0.042593},
        },
    }
    baseline = {"method": "Ms-AeDNet", "mae": 4.76, "mse": 9.59}

    gap = _paper_positive_gap(result, baseline)
    delta = _paper_target_delta(result, baseline)

    assert gap == {"mae": 0.0, "mse": 0.0, "total": 0.0}
    assert delta["mae"] < 0
    assert delta["mse"] < 0
    assert delta["beats_both"] is True
    assert delta["mae_improvement_pct"] == pytest.approx(10.5189, rel=1e-4)
    assert delta["mse_improvement_pct"] == pytest.approx(6.7706, rel=1e-4)


def test_paper_delta_line_always_reports_signed_improvement_pct():
    line = _format_paper_delta_line(
        "FORGE vs paper target",
        {
            "mae": 0.50,
            "mse": -0.20,
            "mae_improvement_pct": -10.0,
            "mse_improvement_pct": 2.0,
            "beats_both": False,
        },
    )

    assert "improvement over paper target" in line
    assert "MAE=-10.00%" in line
    assert "MSE=2.00%" in line


def test_print_forge_best_summary_omits_clipped_remaining_gap(capsys):
    _print_forge_best_summary(
        {
            "best_iteration": 16,
            "best_metrics": {"paper_mae": 4.2593, "paper_mse": 8.9407},
            "paper_gap": {"mae": 0.0, "mse": 0.0, "total": 0.0},
            "paper_delta": {
                "mae": -0.5007,
                "mse": -0.6493,
                "mae_improvement_pct": 10.52,
                "mse_improvement_pct": 6.77,
                "beats_both": True,
            },
        }
    )

    output = capsys.readouterr().out
    assert "FORGE best: iter_016 MAE=4.2593 MSE=8.9407" in output
    assert "Remaining gap" not in output
    assert "improvement over paper target" in output


def test_write_run_summary_selects_single_best_from_full_history(tmp_path):
    history = [
        {
            "iteration": 0,
            "success": True,
            "target": {"mae_inverse": 0.50},
            "primary_component": "initial",
            "run_dir": str(tmp_path / "iter_000"),
            "result": {"success": True, "metrics": {"target": {"mae_inverse": 0.50}}},
        },
        {
            "iteration": 10,
            "success": True,
            "target": {"mae_inverse": 0.20},
            "primary_component": "encoder",
            "run_dir": str(tmp_path / "iter_010"),
            "result": {"success": True, "metrics": {"target": {"mae_inverse": 0.20}}},
        },
        {
            "iteration": 30,
            "success": True,
            "target": {"mae_inverse": 0.30},
            "primary_component": "head",
            "run_dir": str(tmp_path / "iter_030"),
            "result": {"success": True, "metrics": {"target": {"mae_inverse": 0.30}}},
        },
    ]

    summary = _write_run_summary(tmp_path, 30, "mae_inverse", history)

    assert summary["best_iteration"] == 10
    assert summary["best_selection"]["search_start_iteration"] == 0
    assert summary["best_selection"]["search_end_iteration"] == 30
    assert summary["best_selection"]["successful_candidate_count"] == 3


def test_dispatch_candidate_rejects_probe_gain_without_ms_aednet_gap_shrink():
    protected = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.10},
            "inverse": {"mse": 0.20},
            "paper_scaled": {"mae": 5.00, "mse": 12.00},
        },
    }
    candidate = {
        "success": True,
        "metrics": {
            "target": {"mae_inverse": 0.10},
            "inverse": {"mse": 0.20},
            "paper_scaled": {"mae": 5.00, "mse": 12.00},
        },
    }
    feedback = {"diagnostics": [{"name": "residual_drift", "severity": 0.8, "confidence": 1.0}]}
    candidate_feedback = {"diagnostics": [{"name": "residual_drift", "severity": 0.1, "confidence": 1.0}]}

    decision = _accept_dispatch_candidate(
        protected,
        candidate,
        feedback,
        candidate_feedback,
        "mae_inverse",
        target_diagnostics=["residual_drift"],
        paper_baseline={"method": "Ms-AeDNet", "mae": 4.56, "mse": 10.40},
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "ms_aednet_gap_not_shrunk"


def test_dispatch_patch_quality_rejects_noop_comment_patch():
    parent = "import torch\nclass ForgeModel:\n    def __init__(self):\n        self.head = 1\n"
    candidate = "import torch\nclass ForgeModel:\n    def __init__(self):\n        # same head\n        self.head = 1\n"

    quality = _dispatch_patch_quality(parent, candidate)

    assert quality["passed"] is False
    assert quality["reason"] == "motif_no_effect"


def test_dispatch_patch_quality_rejects_destructive_transplant():
    parent = """
class ForgeModel:
    def __init__(self):
        self.head = 1
        self.gate = 2
        self.limiter = 3
        self.trend = 4
    def forward(self, x):
        return self.head + self.gate + self.limiter + self.trend
"""
    candidate = """
class ForgeModel:
    def __init__(self):
        self.new_gate = 5
    def forward(self, x):
        return self.new_gate
"""

    quality = _dispatch_patch_quality(parent, candidate)

    assert quality["passed"] is False
    assert quality["reason"] == "destructive_motif_transplant"


def test_dispatch_patch_quality_accepts_additive_transplant():
    parent = """
class ForgeModel:
    def __init__(self):
        self.head = 1
    def forward(self, x):
        return self.head
"""
    candidate = """
class ForgeModel:
    def __init__(self):
        self.head = 1
        self.motif_gate = 2
        self.motif_scale = 3
    def forward(self, x):
        y = self.head
        return y + self.motif_gate * self.motif_scale
"""

    quality = _dispatch_patch_quality(parent, candidate)

    assert quality["passed"] is True
    assert quality["reason"] == "motif_quality_passed"


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


def test_summarize_run_parser_accepts_run_dir():
    parser = build_parser()
    args = parser.parse_args(["summarize-run", "--run-dir", "runs/demo"])
    assert args.run_dir == "runs/demo"
    assert args.candidate_tournament_k == 1


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
