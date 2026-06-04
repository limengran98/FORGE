# FORGE

FORGE is a runnable PEMFC model-evolution system:

Feedback Observability-guided Routing for Graph-based Evolution.

It fixes the Ms-AeDNet PEMFC harness and evolves only the model code. Each iteration:

1. Loads the PEMFC data with the Ms-AeDNet chronological 6:2:2 protocol.
2. Trains and evaluates a saved initial GRU model.
3. Encodes metrics, errors, logs, and train/validation curves into a feedback vector.
4. Converts executable artifacts into PEMFC diagnostic feedback nodes.
5. Routes diagnostics to model components through trust-scored graph relations.
6. Requests an LLM patch, or uses a deterministic heuristic fallback.
7. Updates feedback-to-component trust from the next executable outcome.
8. Saves metrics, feedback, route, trust updates, code version, diff, task graph, and report.

## Quick Start

```bash
python -m forge.cli init
python -m forge.cli run --rounds 1 --epochs 5 --batch-size 64 --llm-mode off
```

`--llm-mode off` is for smoke tests and harness debugging only. In this mode,
FORGE generates the next model from deterministic fallback templates in
`skills/forge_model_templates/`; it does not call an LLM.

By default, `run` uses the harness default dataset `FC2`, `seq_len=24`, and
`pred_len=12`. A single Table-style setting is selected explicitly with:

```bash
python -m forge.cli run --data FC1 --seq-len 96 --pred-len 12
python -m forge.cli run --data FC2 --seq-len 24 --pred-len 6
```

The default device is `cuda:0`. Select a different GPU or force CPU with:

```bash
python -m forge.cli run --device cuda --cuda-id 1
python -m forge.cli run --device cpu
```

Use the configured LLM:

```bash
python -m forge.cli run --rounds 2 --epochs 200 --llm-mode required --routing-mode trust --parent-policy best
```

The LLM config is at `configs/forge_llm.yaml`.

## Continue Iterations

`run --rounds 1` creates and evaluates `iter_000` and `iter_001`. To keep
improving that same run, continue from the last completed iteration instead of
starting over:

```bash
python -m forge.cli continue --run-dir runs/<run_name> --additional-rounds 1 --epochs 200 --llm-mode required --routing-mode trust --parent-policy best
```

You can also target an absolute iteration index:

```bash
python -m forge.cli continue --run-dir runs/<run_name> --to-round 3 --epochs 200 --llm-mode required --routing-mode trust --parent-policy best
```

`--additional-rounds 1` means "add one more patch/evaluation after the current
last iteration." `--to-round 3` means "finish at `iter_003`." The command reuses
the existing `run_config.json`, model code, feedback vectors, routing records,
task graph, and patch artifacts. It only generates the missing next patch and
then evaluates the next model.

`--parent-policy best` uses degraded iterations to update trust, but starts the
next edit from the current best model source. `--parent-policy last` gives a
strict sequential chain.

## Evidence Dispatch

After Diagnostic Probers finish, Evidence Dispatch is summary-only by default.
FORGE mines successful PEMFC patch motifs only from the preceding iterations of
the same run, asks the LLM to summarize the executable evidence, and copies the
protected prober best to `final/model.py`. It does not generate or evaluate a new
model candidate in the main workflow.

```bash
python -m forge.cli dispatch \
  --run-dir runs/<sweep_name>/FC1_L24_P12 \
  --llm-mode required \
  --dispatch-mode summary \
  --evidence-scope current-run \
  --archive-candidates 0 \
  --target-diagnostics long_horizon_error residual_autocorrelation residual_drift \
  --device cuda \
  --cuda-id 0
```

The dispatch artifact is saved under
`runs/<run_name>/evidence_dispatch*/` with `protected_best/`, `final/`,
`dispatch_payload.json`, `dispatch_report.json`, and `dispatch_summary.json`.
The old counterfactual motif tournament is kept as an explicit ablation via
`--dispatch-mode candidates --dispatch-candidates 4`; it is not the recommended
main path because the protected harness usually rejects these late candidates.
For settings not listed in `configs/harness/pemfc_harness.yaml`, pass
`--paper-baseline-mae` and `--paper-baseline-mse`.

Run iterations and current-run-only evidence summary as one integrated workflow:

```bash
python -m forge.cli sweep \
  --datasets FC1 FC2 \
  --seq-lens 24 \
  --pred-lens 12 \
  --rounds 20 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --final-dispatch \
  --dispatch-mode summary \
  --dispatch-llm-mode required \
  --archive-candidates 0
```

## Trust Routing And Ablations

FORGE supports four routing modes:

- `trust`: main FORGE method, matching the `pilot_trust` line. Diagnostic feedback propagates through learned feedback-component relations, and those component relations are updated by executable outcomes. The LLM is constrained by the fixed harness but is not locked to a relation-level edit operator.
- `trust-action`: ablation mode for relation-level action memory, attention gates, negative-memory suppression, and structural operator experiments.
- `prior`: diagnostic feedback uses fixed PEMFC priors without outcome-based trust updates.
- `rule`: legacy rule routing only, without diagnostic trust propagation.

For a small two-dataset, ten-iteration pilot:

```bash
python -m forge.cli sweep \
  --datasets FC1 FC2 \
  --seq-lens 24 \
  --pred-lens 12 \
  --rounds 10 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --run-name pilot_trust_FC1_FC2_L24_P12_R10
```

Run an ablation control by changing only the routing mode:

```bash
python -m forge.cli sweep --datasets FC1 FC2 --seq-lens 24 --pred-lens 12 --rounds 10 --epochs 200 --llm-mode required --routing-mode rule --parent-policy best --run-name ablate_rule_FC1_FC2_L24_P12_R10
python -m forge.cli sweep --datasets FC1 FC2 --seq-lens 24 --pred-lens 12 --rounds 10 --epochs 200 --llm-mode required --routing-mode prior --parent-policy best --run-name ablate_prior_FC1_FC2_L24_P12_R10
python -m forge.cli sweep --datasets FC1 FC2 --seq-lens 24 --pred-lens 12 --rounds 10 --epochs 200 --llm-mode required --routing-mode trust-action --parent-policy best --run-name ablate_trust_action_FC1_FC2_L24_P12_R10
```

## Benchmark Grid

The Ms-AeDNet-style benchmark grid is configured in
`configs/harness/benchmark_grid.yaml`:

- datasets: `FC1`, `FC2`
- historical lengths: `12, 24, 48, 96, 192`
- prediction lengths: `1, 3, 6, 12`

Run the full grid:

```bash
python -m forge.cli sweep --rounds 1 --epochs 200 --llm-mode required
```

Run a small subset:

```bash
python -m forge.cli sweep --datasets FC1 FC2 --seq-lens 24 48 --pred-lens 6 12 --epochs 20 --llm-mode required
```

Sweep outputs are saved under `runs/<sweep_name>/` as `sweep_summary.json` and
`sweep_summary.csv`, with each combination stored in its own run directory.


## Flexible Harness Files

Most FORGE protocol constants live outside Python:

- `configs/harness/pemfc_harness.yaml`: datasets, feature columns, split ratios, model interface
- `configs/harness/benchmark_grid.yaml`: dataset/history/horizon benchmark combinations
- `configs/harness/feedback_schema.yaml`: feedback vector schema
- `configs/harness/routing_graph.yaml`: component graph
- `configs/harness/routing_policy.yaml`: routing thresholds, weights, and reason text
- `configs/harness/trust_policy.yaml`: diagnostic-component priors and executable outcome trust updates
- `configs/harness/heuristic_patches.yaml`: component-to-template fallback mapping
- `skills/forge_model_templates/`: complete fallback model templates
- `prompts/model_patch.yaml`: LLM patch prompt
- `prompts/evidence_summary.yaml`: summary-only evidence dispatch prompt
- `prompts/evidence_dispatch.yaml`: optional final-candidate ablation prompt

## Outputs

Runs are written under `runs/<run_name>/`:

- `iter_000/model.py`: initial GRU source copied from `workspace/initial_model.py`
- `iter_*/metrics.json`: normalized, inverse-voltage, and paper-scaled metrics
- `iter_*/train_curve.jsonl`: epoch train/validation losses
- `iter_*/feedback_vector.json`: noisy feedback vector and schema
- `iter_*/routing.json`: graph routing result with diagnostic propagation evidence
- `iter_*/patch.diff`: source diff for the next iteration
- `task_graph.json`: evolving component graph state and feedback-component trust relations
- `graph_events.jsonl`: append-only orchestration event log
- `summary.json`: run-level summary
- `evidence_dispatch*/dispatch_summary.json`: protected best, mined motifs, summary report, and final model path

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
