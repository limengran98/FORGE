from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import load_yaml
from .paths import HARNESS_CONFIG_DIR, PROJECT_ROOT


def _load_named_yaml(name: str) -> dict[str, Any]:
    path = HARNESS_CONFIG_DIR / name
    data = load_yaml(path)
    if not data:
        raise FileNotFoundError(f"Missing or empty harness config: {path}")
    return data


@lru_cache(maxsize=1)
def load_pemfc_harness_spec() -> dict[str, Any]:
    return _load_named_yaml("pemfc_harness.yaml")


@lru_cache(maxsize=1)
def load_feedback_schema_spec() -> dict[str, Any]:
    return _load_named_yaml("feedback_schema.yaml")


@lru_cache(maxsize=1)
def load_routing_graph_spec() -> dict[str, Any]:
    return _load_named_yaml("routing_graph.yaml")


@lru_cache(maxsize=1)
def load_routing_policy_spec() -> dict[str, Any]:
    return _load_named_yaml("routing_policy.yaml")


@lru_cache(maxsize=1)
def load_trust_policy_spec() -> dict[str, Any]:
    return _load_named_yaml("trust_policy.yaml")


@lru_cache(maxsize=1)
def load_edit_operator_spec() -> dict[str, Any]:
    return _load_named_yaml("edit_operators.yaml")


@lru_cache(maxsize=1)
def load_heuristic_patch_spec() -> dict[str, Any]:
    return _load_named_yaml("heuristic_patches.yaml")


@lru_cache(maxsize=1)
def load_orchestration_spec() -> dict[str, Any]:
    return _load_named_yaml("orchestration.yaml")


@lru_cache(maxsize=1)
def load_benchmark_grid_spec() -> dict[str, Any]:
    return _load_named_yaml("benchmark_grid.yaml")


def get_dataset_files() -> dict[str, str]:
    datasets = load_pemfc_harness_spec().get("datasets", {})
    out = {}
    for name, info in datasets.items():
        if not isinstance(info, dict) or not info.get("filename"):
            raise ValueError(f"Dataset {name!r} must define a filename")
        out[str(name).upper()] = str(info["filename"])
    return out


def get_dataset_metric_scales(data_name: str) -> dict[str, float]:
    datasets = load_pemfc_harness_spec().get("datasets", {})
    info = datasets.get(str(data_name).upper(), {})
    if not isinstance(info, dict):
        return {"mae": 1.0, "mse": 1.0}
    raw_scales = info.get("paper_metric_scale", {})
    if not isinstance(raw_scales, dict):
        raw_scales = {}
    return {
        "mae": float(raw_scales.get("mae", 1.0)),
        "mse": float(raw_scales.get("mse", 1.0)),
    }


def get_default_dataset_name() -> str:
    raw_default = load_pemfc_harness_spec().get("default_dataset")
    dataset_files = get_dataset_files()
    if raw_default is not None and str(raw_default).strip():
        default_name = str(raw_default).upper()
        if default_name not in dataset_files:
            raise ValueError(f"default_dataset {default_name!r} is not defined in datasets")
        return default_name
    if dataset_files:
        return sorted(dataset_files)[0]
    raise ValueError("pemfc_harness.yaml must define at least one dataset")


def get_archive_input_prefix() -> str:
    return str(load_pemfc_harness_spec().get("archive", {}).get("input_prefix", "Ms-AeDNet-main/input"))


def get_feature_groups() -> dict[str, list[str]]:
    features = load_pemfc_harness_spec().get("features", {})
    groups = {
        "voltage_inputs": list(features.get("voltage_inputs", [])),
        "factor_inputs": list(features.get("factor_inputs", [])),
        "targets": list(features.get("targets", features.get("voltage_inputs", []))),
    }
    for name, cols in groups.items():
        if not cols:
            raise ValueError(f"pemfc_harness.yaml features.{name} must not be empty")
    return groups


def get_split_ratios() -> tuple[float, float, float]:
    split = load_pemfc_harness_spec().get("split", {})
    ratios = (
        float(split.get("train", 0.6)),
        float(split.get("val", 0.2)),
        float(split.get("test", 0.2)),
    )
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("Split ratios must be non-negative and sum to a positive value")
    return ratios


def get_feature_dim() -> int:
    features = get_feature_groups()
    return len(features["voltage_inputs"]) + len(features["factor_inputs"])


def get_enc_in() -> int:
    return len(get_feature_groups()["targets"])


def get_model_class_name() -> str:
    interface = load_pemfc_harness_spec().get("model_interface", {})
    return str(interface.get("class_name", "ForgeModel"))


def get_feedback_schema() -> list[str]:
    schema = load_feedback_schema_spec().get("vector_schema", [])
    if not schema:
        raise ValueError("feedback_schema.yaml must define vector_schema")
    return [str(name) for name in schema]


def get_component_graph() -> dict[str, Any]:
    graph = load_routing_graph_spec().get("component_graph", {})
    nodes = [str(node) for node in graph.get("nodes", [])]
    edges = list(graph.get("edges", []))
    if not nodes:
        raise ValueError("routing_graph.yaml must define component_graph.nodes")
    known = set(nodes)
    for edge in edges:
        if edge.get("from") not in known or edge.get("to") not in known:
            raise ValueError(f"Routing edge references unknown component: {edge}")
    return {"nodes": nodes, "edges": edges}


def get_routing_policy() -> dict[str, Any]:
    return load_routing_policy_spec()


def get_trust_policy() -> dict[str, Any]:
    return load_trust_policy_spec()


def get_edit_operator_spec() -> dict[str, Any]:
    return load_edit_operator_spec()


def get_edit_operators() -> list[dict[str, Any]]:
    operators = load_edit_operator_spec().get("operators", [])
    if not operators:
        raise ValueError("edit_operators.yaml must define at least one operator")
    return list(operators)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_heuristic_patch_rules() -> list[dict[str, Any]]:
    rules = load_heuristic_patch_spec().get("rules", [])
    if not rules:
        raise ValueError("heuristic_patches.yaml must define at least one rule")
    return list(rules)


def get_iteration_stages() -> list[str]:
    stages = load_orchestration_spec().get("iteration_stages", [])
    if not stages:
        raise ValueError("orchestration.yaml must define iteration_stages")
    out = [str(stage) for stage in stages]
    if len(out) != len(set(out)):
        raise ValueError("orchestration.yaml iteration_stages must be unique")
    return out


def get_benchmark_grid() -> dict[str, list[int] | list[str]]:
    spec = load_benchmark_grid_spec()
    datasets = [str(name).upper() for name in spec.get("datasets", [])]
    seq_lens = [int(value) for value in spec.get("seq_lens", [])]
    pred_lens = [int(value) for value in spec.get("pred_lens", [])]
    if not datasets or not seq_lens or not pred_lens:
        raise ValueError("benchmark_grid.yaml must define datasets, seq_lens, and pred_lens")
    known = set(get_dataset_files())
    unknown = [name for name in datasets if name not in known]
    if unknown:
        raise ValueError(f"benchmark_grid.yaml references unknown datasets: {unknown}")
    if any(value < 1 for value in seq_lens + pred_lens):
        raise ValueError("benchmark seq_lens and pred_lens must be positive")
    return {"datasets": datasets, "seq_lens": seq_lens, "pred_lens": pred_lens}


def validate_harness_specs() -> None:
    dataset_files = get_dataset_files()
    if get_default_dataset_name() not in dataset_files:
        raise ValueError("Default dataset must exist in datasets")
    for name in dataset_files:
        scales = get_dataset_metric_scales(name)
        if scales["mae"] <= 0 or scales["mse"] <= 0:
            raise ValueError(f"Dataset {name} paper_metric_scale values must be positive")

    features = get_feature_groups()
    interface = load_pemfc_harness_spec().get("model_interface", {})
    expected_feature_dim = get_feature_dim()
    expected_target_dim = get_enc_in()
    if int(interface.get("input_feature_dim", expected_feature_dim)) != expected_feature_dim:
        raise ValueError("model_interface.input_feature_dim does not match configured input features")
    if int(interface.get("target_dim", expected_target_dim)) != expected_target_dim:
        raise ValueError("model_interface.target_dim does not match configured targets")
    if len(features["voltage_inputs"]) < expected_target_dim:
        raise ValueError("Targets cannot exceed configured voltage input count")

    schema = get_feedback_schema()
    if len(schema) != len(set(schema)):
        raise ValueError("feedback_schema.yaml vector_schema contains duplicates")

    policy = get_routing_policy()
    if int(policy.get("top_k", 1)) < 1:
        raise ValueError("routing_policy.yaml top_k must be positive")
    if float(policy.get("active_threshold", 0.0)) < 0:
        raise ValueError("routing_policy.yaml active_threshold must be non-negative")

    trust_policy = get_trust_policy()
    priors = trust_policy.get("diagnostic_component_priors", {})
    if not priors:
        raise ValueError("trust_policy.yaml must define diagnostic_component_priors")
    known_components = set(get_component_graph()["nodes"])
    known_diagnostics = set(priors)
    for diagnostic, component_priors in priors.items():
        if not isinstance(component_priors, dict) or not component_priors:
            raise ValueError(f"trust prior for {diagnostic!r} must map to at least one component")
        unknown = [component for component in component_priors if component not in known_components]
        if unknown:
            raise ValueError(f"trust prior for {diagnostic!r} references unknown components: {unknown}")

    edit_operator_ids: set[str] = set()
    edit_spec = get_edit_operator_spec()
    selection = edit_spec.get("selection", {})
    if isinstance(selection, dict):
        attention = selection.get("relation_attention", {})
        if isinstance(attention, dict) and attention.get("enabled", True):
            min_tau = float(attention.get("min_temperature", 0.10))
            base_tau = float(attention.get("base_temperature", 0.30))
            max_tau = float(attention.get("max_temperature", 0.60))
            if not (0.0 < min_tau <= base_tau <= max_tau):
                raise ValueError("edit_operators.yaml relation_attention temperatures must satisfy 0 < min <= base <= max")
            if int(attention.get("sample_top_k", 1)) < 1:
                raise ValueError("edit_operators.yaml relation_attention.sample_top_k must be positive")
    for operator in get_edit_operators():
        op_id = str(operator.get("id") or "")
        component = str(operator.get("component") or "")
        diagnostics = operator.get("diagnostics", {})
        if not op_id:
            raise ValueError("Each edit operator must define an id")
        if op_id in edit_operator_ids:
            raise ValueError(f"Duplicate edit operator id: {op_id}")
        edit_operator_ids.add(op_id)
        if component not in known_components:
            raise ValueError(f"Edit operator {op_id!r} references unknown component: {component}")
        if not isinstance(diagnostics, dict) or not diagnostics:
            raise ValueError(f"Edit operator {op_id!r} must define diagnostic priors")
        unknown_diagnostics = [name for name in diagnostics if name not in known_diagnostics]
        if unknown_diagnostics:
            raise ValueError(f"Edit operator {op_id!r} references unknown diagnostics: {unknown_diagnostics}")
        for diagnostic, prior in diagnostics.items():
            value = float(prior)
            if not (0.0 < value < 1.0):
                raise ValueError(f"Edit operator prior for {op_id!r}/{diagnostic!r} must be in (0, 1)")
        template = operator.get("template")
        if template and not resolve_project_path(str(template)).exists():
            raise ValueError(f"Edit operator {op_id!r} template does not exist: {template}")

    get_component_graph()
    get_iteration_stages()
    get_benchmark_grid()

    for rule in get_heuristic_patch_rules():
        template = rule.get("template")
        if not template:
            raise ValueError(f"Heuristic patch rule {rule.get('name')} must define template")
        if not resolve_project_path(template).exists():
            raise FileNotFoundError(f"Heuristic patch template does not exist: {template}")
