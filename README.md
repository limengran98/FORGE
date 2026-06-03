# FORGE

FORGE is a runnable PEMFC model-evolution system:

Feedback Observability-guided Routing for Graph-based Evolution.

It fixes the Ms-AeDNet PEMFC harness and evolves only the model code. Each iteration:

1. Loads the PEMFC data with the Ms-AeDNet chronological 6:2:2 protocol.
2. Trains and evaluates a saved initial GRU model.
3. Encodes metrics, errors, logs, and train/validation curves into a feedback vector.
4. Routes noisy feedback to a model component on a component graph.
5. Requests an LLM patch, or uses a deterministic heuristic fallback.
6. Validates the patched `ForgeModel` interface.
7. Saves metrics, feedback, route, code version, diff, task graph, and report.

## Quick Start

```bash
python -m forge.cli init
python -m forge.cli run --rounds 1 --epochs 5 --batch-size 64 --llm-mode off
```

The default device is `cuda:0`. Select a different GPU or force CPU with:

```bash
python -m forge.cli run --device cuda --cuda-id 1
python -m forge.cli run --device cpu
```

Use the configured LLM:

```bash
python -m forge.cli run --rounds 2 --epochs 200 --llm-mode auto
```

The LLM config is at `configs/forge_llm.yaml`.

## Flexible Harness Files

Most FORGE protocol constants live outside Python:

- `configs/harness/pemfc_harness.yaml`: datasets, feature columns, split ratios, model interface
- `configs/harness/feedback_schema.yaml`: feedback vector schema
- `configs/harness/routing_graph.yaml`: component graph
- `configs/harness/routing_policy.yaml`: routing thresholds, weights, and reason text
- `configs/harness/heuristic_patches.yaml`: component-to-template fallback mapping
- `skills/forge_model_templates/`: complete fallback model templates
- `prompts/model_patch.yaml`: LLM patch prompt

## Outputs

Runs are written under `runs/<run_name>/`:

- `iter_000/model.py`: initial GRU source copied from `workspace/initial_model.py`
- `iter_*/metrics.json`: normalized and inverse-voltage metrics
- `iter_*/train_curve.jsonl`: epoch train/validation losses
- `iter_*/feedback_vector.json`: noisy feedback vector and schema
- `iter_*/routing.json`: graph routing result
- `iter_*/patch.diff`: source diff for the next iteration
- `task_graph.json`: evolving component graph state
- `graph_events.jsonl`: append-only orchestration event log
- `summary.json`: run-level summary

## Graph Orchestration

FORGE uses a deliberately small orchestration lifecycle before adding more
advanced agent behavior. Each iteration is tracked as:

```text
prepare -> evaluate -> feedback -> route -> patch -> report
```

The orchestrator records stage status, artifacts, harness result summaries,
feedback snapshots, routing decisions, patch metadata, component evidence, and
append-only events. Harness/model failures are treated as valid feedback;
orchestration failure means the control flow or artifact handling itself broke.

The lifecycle is configured in `configs/harness/orchestration.yaml`, and the
implementation is in `forge/orchestrator.py`.
