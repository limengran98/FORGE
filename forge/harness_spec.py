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
def load_heuristic_patch_spec() -> dict[str, Any]:
    return _load_named_yaml("heuristic_patches.yaml")


@lru_cache(maxsize=1)
def load_orchestration_spec() -> dict[str, Any]:
    return _load_named_yaml("orchestration.yaml")


def get_dataset_files() -> dict[str, str]:
    datasets = load_pemfc_harness_spec().get("datasets", {})
    out = {}
    for name, info in datasets.items():
        if not isinstance(info, dict) or not info.get("filename"):
            raise ValueError(f"Dataset {name!r} must define a filename")
        out[str(name).upper()] = str(info["filename"])
    return out


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


def validate_harness_specs() -> None:
    dataset_files = get_dataset_files()
    if get_default_dataset_name() not in dataset_files:
        raise ValueError("Default dataset must exist in datasets")

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

    get_component_graph()
    get_iteration_stages()

    for rule in get_heuristic_patch_rules():
        template = rule.get("template")
        if not template:
            raise ValueError(f"Heuristic patch rule {rule.get('name')} must define template")
        if not resolve_project_path(template).exists():
            raise FileNotFoundError(f"Heuristic patch template does not exist: {template}")
