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

## Recommended Experiment Workflow

Use this section for normal FORGE experiments. The recommended protocol is:

1. Run each dataset independently for 20 rounds.
2. Inspect `best_iteration`, paper-scaled MAE/MSE, improvement percent, and
   `evidence_audit`.
3. If the best model appears near the last round, continue the same run by 10
   rounds.
4. Use `--parent-policy best` so bad exploratory patches update evidence memory
   but do not become the next parent model.

Start a fresh FC1 20-round run:

```bash
python -m forge.cli run \
  --data FC1 \
  --seq-len 24 \
  --pred-len 12 \
  --rounds 20 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --candidate-tournament-k 1 \
  --final-dispatch \
  --dispatch-mode summary \
  --dispatch-llm-mode required \
  --archive-candidates 0 \
  --device cuda \
  --cuda-id 0 \
  --run-name pilot_trust_summary_FC1_L24_P12_R20
```

Start a fresh FC2 20-round run:

```bash
python -m forge.cli run \
  --data FC2 \
  --seq-len 24 \
  --pred-len 12 \
  --rounds 20 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --candidate-tournament-k 1 \
  --final-dispatch \
  --dispatch-mode summary \
  --dispatch-llm-mode required \
  --archive-candidates 0 \
  --device cuda \
  --cuda-id 0 \
  --run-name pilot_trust_summary_FC2_L24_P12_R20
```

Continue an existing FC1 run by 10 more rounds:

```bash
python -m forge.cli continue \
  --run-dir runs/pilot_trust_summary_FC1_L24_P12_R20 \
  --additional-rounds 10 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --candidate-tournament-k 1 \
  --final-dispatch \
  --dispatch-mode summary \
  --dispatch-llm-mode required \
  --archive-candidates 0 \
  --device cuda \
  --cuda-id 0
```

Continue an existing FC2 run by 10 more rounds:

```bash
python -m forge.cli continue \
  --run-dir runs/pilot_trust_summary_FC2_L24_P12_R20 \
  --additional-rounds 10 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --candidate-tournament-k 1 \
  --final-dispatch \
  --dispatch-mode summary \
  --dispatch-llm-mode required \
  --archive-candidates 0 \
  --device cuda \
  --cuda-id 0
```

Continue to an absolute final iteration index, for example `iter_050`:

```bash
python -m forge.cli continue \
  --run-dir runs/pilot_trust_summary_FC1_L24_P12_R20 \
  --to-round 50 \
  --epochs 200 \
  --llm-mode required \
  --routing-mode trust \
  --parent-policy best \
  --candidate-tournament-k 1 \
  --final-dispatch \
  --dispatch-mode summary \
  --dispatch-llm-mode required \
  --archive-candidates 0 \
  --device cuda \
  --cuda-id 0
```

Refresh one completed run without retraining:

```bash
python -m forge.cli summarize-run --run-dir runs/pilot_trust_summary_FC1_L24_P12_R20
```

The refreshed summary prints both the paper target and the FORGE best result:

```text
[FORGE] FORGE best: iter_016 MAE=4.2593 MSE=8.9407
[FORGE] FORGE vs paper target: improvement over paper target MAE=10.52% MSE=6.77% (absolute better by MAE=0.5007 MSE=0.6493)
```

The improvement percentage is `(paper_target - FORGE) / paper_target * 100`.
Positive means FORGE is better than the paper target; negative means it is
worse.

## Command Modes

Use `run` when you want a new independent experiment. It creates a new run
directory, copies `workspace/initial_model.py` to `iter_000/model.py`, evaluates
`iter_000`, and then creates/evaluates new iterations up to `iter_N`.
For a true fresh rerun, use a new `--run-name` or an empty `--run-dir`; do not
reuse an old run directory.

Use `continue` when you want to extend an existing run. It reuses the existing
`run_config.json`, model versions, feedback vectors, routing records,
`task_graph.json`, and previous patch artifacts. This is the right command when
you say "在这个数据集已有结果基础上继续跑".

Use `sweep` when you want a grid of datasets/history lengths/horizons. It
creates one child run per combination, for example `FC1_L24_P12` and
`FC2_L24_P12`.

Use `summarize-run` or `summarize-sweep` when you only want to refresh reports
and evidence metrics. These commands do not train or call the LLM.

Use `dispatch` when you want to run only the final evidence summary over an
already completed run. In the main protocol, this is normally handled by
`--final-dispatch`.

## Parameter Reference

| Parameter | Values / Range | Recommended | Meaning |
| --- | --- | --- | --- |
| `--data` | `FC1`, `FC2` | run both separately | PEMFC dataset/cell. |
| `--seq-len` | `12`, `24`, `48`, `96`, `192` | `24` for pilot | Historical input length. |
| `--pred-len` | `1`, `3`, `6`, `12` | `12` for L24-P12 | Forecast horizon. |
| `--rounds` | integer `>=0` | `20` first, then maybe `50` | For `run`; final iteration index to create from scratch. `--rounds 20` evaluates `iter_000` through `iter_020`. |
| `--additional-rounds` | integer `>0` | `10` | For `continue`; add this many new iterations after the current last iteration. |
| `--to-round` | integer greater than current last iteration | `50` for full continuation | For `continue`; absolute final iteration index, e.g. `--to-round 50` finishes at `iter_050`. |
| `--epochs` | integer `>0` | `200` official, `5` smoke | Max epochs per harness training run. Early stopping may stop earlier. |
| `--batch-size` | integer `>0` | `128` | Training batch size. |
| `--lr` | float `>0` | `0.001` | Learning rate. |
| `--patience` | integer `>0` | `5` | Early-stopping patience. |
| `--seed` | integer | `2025` | Random seed for harness training. |
| `--device` | `cuda`, `cpu`, `auto` | `cuda` | Runtime device. |
| `--cuda-id` | visible CUDA index | `0` | GPU id when using CUDA. |
| `--llm-mode` | `required`, `auto`, `off` | `required` official | LLM patch mode. `off` is only for smoke tests. |
| `--routing-mode` | `trust`, `prior`, `rule`, `trust-action` | `trust` | Feedback routing policy. |
| `--parent-policy` | `best`, `last` | `best` | Parent model selection. `best` protects best-so-far while still learning from failed attempts. |
| `--candidate-tournament-k` | currently must be `1` | `1` | Stable FORGE executes one candidate per round. Larger K is reserved for a future budgeted tournament implementation. |
| `--final-dispatch` | flag | enabled for official runs | Runs final evidence summary after iterations. |
| `--dispatch-mode` | `summary`, `candidates` | `summary` | Summary-only is the main path. `candidates` is an ablation. |
| `--dispatch-llm-mode` | `required`, `auto`, `off` | `required` | LLM mode for final evidence summary. |
| `--archive-candidates` | integer `>=0` | `0` | Historical model promotion candidates. Keep `0` for the main method. |
| `--run-name` | string | descriptive name | Creates `runs/<run-name>`. Use a new name for a fresh run. |
| `--run-dir` | path | existing path for `continue` | Existing run directory for continuation or report refresh. |
| `--scaling` | `baseline`, `train` | `baseline` | Data scaling protocol. Keep `baseline` for Ms-AeDNet-compatible reporting. |
| `--limit-rows` | integer or omitted | omit | Debug-only row limit. Do not use for official results. |

## When To Continue

Continue by 10 rounds if one of these is true:

- `best_iteration` is equal to the current last iteration.
- `best_iteration` is within the last 3-5 iterations.
- `evidence_audit.metrics.improvement_rate` is still nonzero.
- The latest summary says FORGE still has negative improvement percentage
  against the paper target on the metric you care about.

Stop or switch to another seed/run if:

- the best iteration is far behind the current last iteration,
- repeated useless edit rate keeps rising,
- invalid edit rate is high,
- or several continuations do not improve `best_metrics.paper_mae` or
  `best_metrics.paper_mse`.

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

## Method Framework

FORGE is organized as four explicit modules:

- **PEMFC-native diagnostic harness**: fixed data protocol, train/validation/test
  split, paper-scaled metrics, diagnostic probes, invalid patch detection, and
  component-change summaries. The LLM only sees structured harness evidence.
- **Evidence reconstruction graph**: feedback, component, edit, and outcome nodes
  connected by execution-calibrated evidence rather than semantic similarity.
- **Test-time adaptive strategy memory**: per-run policy state records current
  failure hypotheses, proven ineffective edits, trusted components, forbidden
  repeats, and the expected improvement metric.
- **K-candidate evidence tournament**: a budgeted candidate competition contract.
  The validated main line currently enforces `--candidate-tournament-k 1`; larger
  K should be implemented and reported as a separate budgeted extension rather
  than silently mixed into the main result.

Run summaries include `method_framework` and `evidence_audit` so the final
result is not judged only by MAE/RMSE. The audit reports:

- `improvement_rate`
- `invalid_edit_rate`
- `repeated_useless_edit_rate`
- `routing_stability`
- `evidence_alignment`
- `budget_efficiency`
- sweep-level `cross_cell_robustness`

Refresh these fields for an existing run without retraining:

```bash
python -m forge.cli summarize-run --run-dir runs/<run_name>
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
- `summary.json/evidence_audit`: trustworthy feedback-routing metrics and adaptive strategy memory
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
