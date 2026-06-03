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
    return {str(name).upper(): str(info["filename"]) for name, info in datasets.items()}


def get_default_dataset_name() -> str:
    default_name = str(load_pemfc_harness_spec().get("default_dataset", "")).upper()
    dataset_files = get_dataset_files()
    if default_name in dataset_files:
        return default_name
    if dataset_files:
        return sorted(dataset_files)[0]
    raise ValueError("pemfc_harness.yaml must define at least one dataset")


def get_archive_input_prefix() -> str:
    return str(load_pemfc_harness_spec().get("archive", {}).get("input_prefix", "Ms-AeDNet-main/input"))


def get_feature_groups() -> dict[str, list[str]]:
    features = load_pemfc_harness_spec().get("features", {})
    return {
        "voltage_inputs": list(features.get("voltage_inputs", [])),
        "factor_inputs": list(features.get("factor_inputs", [])),
        "targets": list(features.get("targets", features.get("voltage_inputs", []))),
    }


def get_split_ratios() -> tuple[float, float, float]:
    split = load_pemfc_harness_spec().get("split", {})
    ratios = (
        float(split.get("train", 0.6)),
        float(split.get("val", 0.2)),
        float(split.get("test", 0.2)),
    )
    if sum(ratios) <= 0:
        raise ValueError("Split ratios must sum to a positive value")
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
    return [str(stage) for stage in stages]
