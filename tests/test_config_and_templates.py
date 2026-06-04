from types import SimpleNamespace

from forge.config import load_experiment_config
from forge.harness import _resolve_device, validate_harness_config
from forge.harness_spec import (
    get_benchmark_grid,
    get_dataset_metric_scales,
    get_edit_operators,
    get_enc_in,
    get_feature_dim,
    validate_harness_specs,
)
from forge.model_io import validate_model_source
from forge.patching import apply_candidate, heuristic_patch_source, safety_fallback_candidate, save_failed_candidate_attempt
from forge.paths import INITIAL_MODEL_PATH


def test_default_device_is_cuda_zero():
    cfg = load_experiment_config("configs/forge_experiment.yaml")
    assert cfg["harness"]["device"] == "cuda"
    assert cfg["harness"]["cuda_id"] == 0


def test_harness_specs_are_self_consistent():
    validate_harness_specs()


def test_benchmark_grid_matches_table_protocol():
    grid = get_benchmark_grid()
    assert grid["datasets"] == ["FC1", "FC2"]
    assert grid["seq_lens"] == [12, 24, 48, 96, 192]
    assert grid["pred_lens"] == [1, 3, 6, 12]


def test_dataset_metric_scales_match_paper_tables():
    assert get_dataset_metric_scales("FC1") == {"mae": 10000.0, "mse": 10000000.0}
    assert get_dataset_metric_scales("FC2") == {"mae": 100.0, "mse": 1000.0}


def test_edit_operator_library_is_configured():
    operators = get_edit_operators()
    ids = {item["id"] for item in operators}
    assert "add_temporal_smoothing" in ids
    assert "add_multiscale_temporal_context" in ids
    assert "increase_regularization" in ids
    assert all(item.get("component") for item in operators)


def test_structural_operator_template_is_shape_safe():
    route = {
        "primary_component": "temporal_memory",
        "selected_edit": {
            "relation_id": "long_horizon_error->temporal_memory::add_multiscale_temporal_context",
            "diagnostic": "long_horizon_error",
            "component": "temporal_memory",
            "edit_operator": "add_multiscale_temporal_context",
            "template": "skills/forge_model_templates/multiscale_context_gru.py",
            "prompt_guidance": "Use the configured structural template.",
        },
    }
    candidate = heuristic_patch_source("", route)
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=get_enc_in(),
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=get_feature_dim(),
    )
    validate_model_source(candidate.source, cfg, feature_dim=get_feature_dim())
    assert candidate.edit_action == "add_multiscale_temporal_context"
    assert candidate.component == "temporal_memory"


def test_cpu_device_resolution_is_explicit():
    assert str(_resolve_device("cpu", 0)) == "cpu"


def test_invalid_harness_config_fails_early():
    cfg = load_experiment_config("configs/forge_experiment.yaml")
    from forge.harness import HarnessConfig

    bad = HarnessConfig(
        data_name=cfg["data"].get("name") or "FC2",
        seq_len=0,
        pred_len=12,
        batch_size=32,
        epochs=1,
        patience=1,
        hidden_dim=16,
        layer=1,
        lr=0.001,
        dropout=0.1,
    )
    try:
        validate_harness_config(bad)
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("invalid harness config should fail")


def test_heuristic_template_is_loaded_from_skill_file():
    candidate = heuristic_patch_source("", {"primary_component": "factor_fusion"})
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=get_enc_in(),
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=get_feature_dim(),
    )
    validate_model_source(candidate.source, cfg, feature_dim=get_feature_dim())
    assert candidate.origin == "heuristic"


def test_safety_fallback_candidate_reuses_valid_parent_model(tmp_path):
    parent_source = INITIAL_MODEL_PATH.read_text(encoding="utf-8")
    candidate = safety_fallback_candidate(parent_source, {"primary_component": "temporal_memory"}, "shape error")
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=get_enc_in(),
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=get_feature_dim(),
    )
    meta = apply_candidate(
        candidate,
        INITIAL_MODEL_PATH,
        tmp_path / "model.py",
        cfg,
        feature_dim=get_feature_dim(),
        artifact_dir=tmp_path,
    )
    assert meta["origin"] == "safety_fallback"
    assert (tmp_path / "model.py").exists()


def test_failed_candidate_attempt_is_saved_for_repair_audit(tmp_path):
    candidate = safety_fallback_candidate("class ForgeModel: pass\n", {"primary_component": "interface"}, "boom")
    row = save_failed_candidate_attempt(candidate, tmp_path, 0, "RuntimeError: boom")
    assert row["attempt"] == 0
    assert row["source_path"]
    assert (tmp_path / "failed_candidate_00.py").exists()
