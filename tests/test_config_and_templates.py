from types import SimpleNamespace

from forge.config import load_experiment_config
from forge.harness import _resolve_device
from forge.model_io import validate_model_source
from forge.patching import heuristic_patch_source


def test_default_device_is_cuda_zero():
    cfg = load_experiment_config("configs/forge_experiment.yaml")
    assert cfg["harness"]["device"] == "cuda"
    assert cfg["harness"]["cuda_id"] == 0


def test_cpu_device_resolution_is_explicit():
    assert str(_resolve_device("cpu", 0)) == "cpu"


def test_heuristic_template_is_loaded_from_skill_file():
    candidate = heuristic_patch_source("", {"primary_component": "factor_fusion"})
    cfg = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=5,
        hidden_dim=16,
        layer=1,
        dropout=0.1,
        feature_dim=23,
    )
    validate_model_source(candidate.source, cfg, feature_dim=23)
    assert candidate.origin == "heuristic"

