from argparse import Namespace

import pytest

from forge.cli import _resolve_continue_target, build_parser


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
    assert args.device == "cuda"
    assert args.cuda_id == 1


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
