from types import SimpleNamespace

from forge.config import load_experiment_config
from forge.harness import _resolve_device, validate_harness_config
from forge.harness_spec import get_benchmark_grid, get_enc_in, get_feature_dim, validate_harness_specs
from forge.model_io import validate_model_source
from forge.patching import heuristic_patch_source


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
