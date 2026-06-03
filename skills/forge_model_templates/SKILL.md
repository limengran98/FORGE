# FORGE Model Templates

This project-local skill folder stores complete `ForgeModel` source templates used by
the deterministic FORGE fallback patcher.

Each template must:

- Define `class ForgeModel(nn.Module)`.
- Accept a `configs` object in `__init__`.
- Accept only `x` in `forward`.
- Return a tensor shaped `(batch, pred_len, 5)`.
- Avoid file I/O, subprocesses, network calls, and harness changes.

Routing from model component to template is configured in
`configs/harness/heuristic_patches.yaml`.

