import json
from argparse import Namespace

import pytest

from forge.cli import (
    _accept_dispatch_candidate,
    _build_fragment_cards,
    _build_metric_tradeoff_summary,
    _dispatch_candidate_limit,
    _dispatch_patch_quality,
    _final_dispatch_enabled,
    _format_paper_delta_line,
    _maybe_refresh_parent_sweep_summary,
    _metric_aware_final_row,
    _paper_positive_gap,
    _paper_target_delta,
    _parent_baseline_for_patch,
    _print_evidence_summary_table,
    _print_forge_best_summary,
    _resolve_continue_target,
    _run_dir_name,
    _synthesis_patch_quality,
    _trace_synthesis_variants,
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


def test_run_dir_name_replaces_stale_round_suffix(monkeypatch):
    monkeypatch.setattr("forge.cli._run_timestamp", lambda: "06081432")

    name = _run_dir_name("pilot_short_merge_FC1_L24_P12_R10", 20, "forge")

    assert name == "pilot_short_merge_FC1_L24_P12_R20_06081432"


def test_run_dir_name_adds_round_suffix_when_missing(monkeypatch):
    monkeypatch.setattr("forge.cli._run_timestamp", lambda: "06081433")

    name = _run_dir_name("pilot_llm_FC1_FC2_L24_P12", 10, "forge")

    assert name == "pilot_llm_FC1_FC2_L24_P12_R10_06081433"


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
    assert args.dispatch_mode == "synthesis"
    assert args.target_diagnostics == ["long_horizon_error", "residual_autocorrelation"]
    assert args.evidence_scope == "current-run"
    assert args.dispatch_candidates is None
    assert args.archive_candidates == 0


def test_sweep_parser_defaults_final_dispatch_to_synthesis():
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
    assert _final_dispatch_enabled(args) is True
    assert args.dispatch_mode == "synthesis"
    assert args.dispatch_candidates is None
    assert args.archive_candidates == 0


def test_sweep_parser_accepts_final_summary_true_switch():
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
            "--rounds",
            "20",
            "--llm-mode",
            "required",
            "--final-summary",
            "true",
        ]
    )
    assert args.final_dispatch is False
    assert args.final_summary == "true"
    assert _final_dispatch_enabled(args) is True
    assert _dispatch_candidate_limit(args, args.dispatch_mode) == 5


def test_final_summary_false_overrides_legacy_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "continue",
            "--run-dir",
            "runs/demo",
            "--additional-rounds",
            "10",
            "--final-dispatch",
            "--final-summary",
            "false",
        ]
    )
    assert args.final_dispatch is True
    assert args.final_summary == "false"
    assert _final_dispatch_enabled(args) is False


def test_sweep_parser_accepts_summary_only_dispatch():
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
            "--rounds",
            "20",
            "--llm-mode",
            "required",
            "--final-dispatch",
            "--dispatch-mode",
            "summary",
        ]
    )
    assert args.final_dispatch is True
    assert args.dispatch_mode == "summary"


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


def test_sweep_parser_accepts_synthesis_dispatch():
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
            "--rounds",
            "20",
            "--llm-mode",
            "required",
            "--final-dispatch",
            "--dispatch-mode",
            "synthesis",
            "--dispatch-candidates",
            "2",
        ]
    )
    assert args.final_dispatch is True
    assert args.dispatch_mode == "synthesis"
    assert args.dispatch_candidates == 2


def test_trace_synthesis_variants_support_k5():
    trace = {
        "productive_core": [
            {"metric_scope": "joint", "component": "regularization", "edit_action": "add_gate"},
            {"metric_scope": "joint", "component": "temporal_memory", "edit_action": "add_memory"},
            {"metric_scope": "mae_only", "component": "prediction_head", "edit_action": "add_head"},
            {"metric_scope": "mse_only", "component": "normalization", "edit_action": "add_norm"},
        ],
        "trap_regions": [{"component": "temporal_memory", "edit_action": "repeat_bad"}],
        "repair_paths": [{"component": "prediction_head", "edit_action": "repair_shape"}],
        "adaptive_task_card": {"risk_flags": ["high_repeated_useless_edit_rate"]},
    }

    variants = _trace_synthesis_variants(trace, 5)

    assert len(variants) == 5
    assert {row["variant_id"] for row in variants} == {
        "joint_core_merge",
        "mse_guarded_core_merge",
        "low_risk_trap_aware_merge",
        "dataset_specific_conservative_merge",
        "repair_stabilized_core_merge",
    }


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
        "FORGE vs reference target",
        {
            "mae": 0.50,
            "mse": -0.20,
            "mae_improvement_pct": -10.0,
            "mse_improvement_pct": 2.0,
            "beats_both": False,
        },
    )

    assert "improvement over reference target" in line
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
    assert "improvement over reference target" in output


def test_print_evidence_summary_table_is_readable(capsys):
    _print_evidence_summary_table(
        {
            "best_iteration": 7,
            "best_metrics": {"paper_mae": 4.12, "paper_mse": 8.34},
            "paper_delta": {"mae_improvement_pct": 3.5, "mse_improvement_pct": -1.25},
            "evidence_artifacts": {"table_counts": {"attempts": 10, "relations": 42, "components": 3}},
            "final_dispatch": {
                "dispatch_mode": "synthesis",
                "selected": "candidate",
                "accepted_count": 1,
                "candidate_count": 2,
                "selected_candidate_index": 0,
                "final_metrics": {"paper_mae": 4.10, "paper_mse": 8.12},
            },
            "metric_tradeoff": {
                "verdict": "mae_mse_both_improved",
                "best_by_mae": {
                    "iteration": 7,
                    "metrics": {"paper_mae": 4.12, "paper_mse": 8.34},
                    "reference_delta": {"mae_improvement_pct": 3.5, "mse_improvement_pct": -1.25},
                },
                "best_by_mse": {"iteration": 5, "metrics": {"paper_mae": 4.20, "paper_mse": 8.10}},
                "joint_reference_best": {
                    "iteration": 6,
                    "metrics": {"paper_mae": 4.16, "paper_mse": 8.20},
                    "reference_delta": {"mae_improvement_pct": 2.0, "mse_improvement_pct": 1.0},
                },
            },
            "evidence_audit": {
                "metrics": {
                    "improvement_rate": 0.3,
                    "invalid_edit_rate": 0.1,
                    "repeated_useless_edit_rate": 0.2,
                    "routing_stability": 0.75,
                    "evidence_alignment": 1.0,
                    "budget_efficiency": {"attempts_to_best": 7},
                },
                "strategy_memory": {
                    "trusted_components": [
                        {"component": "temporal_memory", "success_count": 2, "attempt_count": 5}
                    ]
                },
                "trace_consolidation": {
                    "productive_core": [
                        {
                            "component": "temporal_memory",
                            "edit_action": "add_gate",
                            "metric_scope": "joint",
                            "total_paper_mae_delta": 0.20,
                            "total_paper_mse_delta": 0.40,
                        }
                    ],
                    "trap_regions": [
                        {
                            "component": "regularization",
                            "edit_action": "repeat_dropout",
                            "invalid_count": 1,
                            "repeated_useless_count": 2,
                        }
                    ],
                    "repair_paths": [
                        {
                            "outcome_iteration": 4,
                            "component": "prediction_head",
                            "edit_action": "repair_shape",
                            "status": "stabilized",
                            "paper_mae_delta": 0.0,
                            "paper_mse_delta": 0.0,
                        }
                    ],
                    "adaptive_task_card": {"risk_flags": ["high_repeated_useless_edit_rate"]},
                },
            },
        }
    )

    output = capsys.readouterr().out
    assert "Concise evidence summary" in output
    assert "Target best" in output
    assert "iter_007" in output
    assert "MAE-best" in output
    assert "MAE 4.1200 (3.50%), MSE 8.3400 (-1.25%)" in output
    assert "MSE-best" in output
    assert "Joint-best" in output
    assert "MAE 4.1600 (2.00%), MSE 8.2000 (1.00%)" in output
    assert "Protected best model" in output
    assert "Dispatch winner" in output
    assert "synthesis candidate_00 won" in output
    assert "Metric verdict" in output
    assert "MAE and MSE both improved" in output
    assert "temporal_memory(2/5)" in output
    assert "Productive core" in output
    assert "temporal_memory/add_gate(joint" in output
    assert "Trap regions" in output
    assert "Trace risk flags" in output


def test_metric_tradeoff_detects_mae_gain_mse_regression():
    history = [
        {
            "iteration": 0,
            "success": True,
            "run_dir": "iter_000",
            "result": {
                "success": True,
                "metrics": {"paper_scaled": {"mae": 4.56, "mse": 10.40}, "target": {"mae_inverse": 0.000456}},
            },
        },
        {
            "iteration": 1,
            "success": True,
            "run_dir": "iter_001",
            "result": {
                "success": True,
                "metrics": {"paper_scaled": {"mae": 4.35, "mse": 10.44}, "target": {"mae_inverse": 0.000435}},
            },
        },
        {
            "iteration": 2,
            "success": True,
            "run_dir": "iter_002",
            "result": {
                "success": True,
                "metrics": {"paper_scaled": {"mae": 4.45, "mse": 9.78}, "target": {"mae_inverse": 0.000445}},
            },
        },
    ]

    tradeoff = _build_metric_tradeoff_summary(
        history,
        history[1],
        {"mae": 4.56, "mse": 10.40},
    )

    assert tradeoff["verdict"] == "mae_improved_mse_regressed"
    assert tradeoff["selected_beats_mae"] is True
    assert tradeoff["selected_beats_mse"] is False
    assert tradeoff["best_by_mse"]["iteration"] == 2
    assert tradeoff["joint_reference_best"]["iteration"] == 2


def test_metric_aware_final_row_auto_prefers_joint_without_changing_target_best():
    history = [
        {
            "iteration": 1,
            "success": True,
            "run_dir": "iter_001",
            "result": {
                "success": True,
                "metrics": {"paper_scaled": {"mae": 4.35, "mse": 10.44}, "target": {"mae_inverse": 0.000435}},
            },
        },
        {
            "iteration": 2,
            "success": True,
            "run_dir": "iter_002",
            "result": {
                "success": True,
                "metrics": {"paper_scaled": {"mae": 4.45, "mse": 9.78}, "target": {"mae_inverse": 0.000445}},
            },
        },
    ]
    baseline = {"mae": 4.56, "mse": 10.40}

    auto_row, auto_name, auto_reason = _metric_aware_final_row(history, "mae_inverse", baseline, "auto")
    target_row, target_name, _target_reason = _metric_aware_final_row(history, "mae_inverse", baseline, "target")

    assert target_row["iteration"] == 1
    assert target_name == "target_best"
    assert auto_row["iteration"] == 2
    assert auto_name == "joint_reference_best"
    assert auto_reason == "auto_switches_to_joint_best_to_avoid_metric_tradeoff"


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
    assert summary["final_selection"]["selection"] == "target_best"
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


def test_synthesis_patch_quality_rejects_tiny_delta_noop():
    parent = "import torch\nclass ForgeModel:\n    def __init__(self):\n        self.head = 1.0\n"
    candidate = "import torch\nclass ForgeModel:\n    def __init__(self):\n        self.head = 0.95\n"

    motif_quality = _dispatch_patch_quality(parent, candidate)
    synthesis_quality = _synthesis_patch_quality(parent, candidate)

    assert motif_quality["passed"] is False
    assert motif_quality["reason"] == "motif_no_effect"
    assert synthesis_quality["passed"] is False
    assert synthesis_quality["reason"] == "motif_no_effect"


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


def test_fragment_cards_prioritize_dual_metric_safe_motifs():
    protected = {
        "success": True,
        "metrics": {
            "paper_scaled": {"mae": 4.40, "mse": 10.00},
            "target": {"mae_inverse": 0.00044},
            "inverse": {"mse": 0.000001},
        },
    }
    mae_only = {
        "motif_id": "mae_only",
        "component": "prediction_head",
        "edit_action": "sharpen_mae",
        "metric_delta": {
            "target_relative": 0.20,
            "mse_relative": -0.10,
            "paper_mae": 0.30,
            "paper_mse": -0.60,
        },
        "outcome_metrics": {"paper_mae": 4.20, "paper_mse": 10.80},
        "diff_excerpt": "+        self.head_gate = torch.nn.Linear(8, 8)\n",
    }
    joint_safe = {
        "motif_id": "joint_safe",
        "component": "temporal_memory",
        "edit_action": "smooth_joint",
        "metric_delta": {
            "target_relative": 0.08,
            "mse_relative": 0.06,
            "paper_mae": 0.08,
            "paper_mse": 0.20,
        },
        "outcome_metrics": {"paper_mae": 4.36, "paper_mse": 9.90},
        "diff_excerpt": "+        self.temporal_smoother = torch.nn.Linear(8, 8)\n",
    }
    trace = {
        "trap_regions": [
            {
                "component": "prediction_head",
                "edit_action": "sharpen_mae",
                "invalid_count": 0,
                "repeated_useless_count": 4,
                "attempt_count": 5,
            }
        ]
    }

    cards = _build_fragment_cards([mae_only, joint_safe], protected, trace, limit=2)

    assert cards[0]["motif_id"] == "joint_safe"
    assert cards[0]["advantage_breakdown"]["joint_non_regressive_vs_parent"] is True
    assert cards[1]["motif_id"] == "mae_only"


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
