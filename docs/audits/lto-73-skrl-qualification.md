# LTO-73 skrl qualification record

## Scope and provenance

This record qualifies the skrl commit
`77335ef7f39601aa934a3ab319ff1adca5d90773` (LTO-72 merge) for review. It
does not import or modify PathSim, execute an environment rollout, or make a
training, convergence, throughput, or checkpoint-quality claim. The local
CPU environment was Python 3.10.12 with `torch 1.11.0+cu102`, `pytest 9.1.1`,
and `gymnasium 1.3.0`; the GPU checks used GPU 0, an NVIDIA GeForce RTX 3090
Ti (`GPU-6eb235b4-11c6-ea58-e616-57a44fba619b`).

The repository does not contain an `AGENTS.md`; the applicable operational
rules were read from PathSim's checked-out GPU-development guide because it
owns the approved shared admission broker and validation envelope.

## CPU and ownership evidence

The strict ownership command passed:

```sh
python3 -m pytest -q tests/test_lto69_inventory.py
```

It reported `2 passed` and retained the audited 532-module ownership and
PathSim-boundary classification. The documented Torch CPU baseline was run
with `CUDA_VISIBLE_DEVICES=''`:

```sh
python3 -m pytest -q \
  tests/agents/torch tests/memories/torch tests/test_torch_config.py \
  tests/test_lto69_inventory.py
```

It reported `106 passed, 52 skipped, 8 failed`. The eight failures are the
pre-existing `tests/memories/torch/test_base.py` attempts to instantiate the
abstract `Memory` base class; they reproduce the exception explicitly
documented by LTO-69 and are outside this qualification's compatibility
surface. The owned CPU contracts additionally passed with `35 passed` for the
inventory, DiscreteSAC, and factorized DiscreteSAC suites.

## Bounded GPU evidence

All CUDA work was sequential, lease-first, and bounded by the approved
PathSim admission broker: GPU 0, 2,048 MiB PyTorch allocator envelope, and a
300-second timeout. The validation image was `pathsim-dev:humble`
(`sha256:0dd05d3df35d415a74b56bda4934a183f82c1dc17bcb1e32c449da5fd26b0ba1`);
its pinned skrl copy is intentionally independent of the mounted LTO-73 checkout, so
the skrl-side check mounted this checkout read-only at `/opt/lto73-skrl`.
No run received busy exit `75`.

The bounded skrl smoke used `torch.manual_seed(73)` and synthetic tensors
only; the fixed neural parity fixtures use `torch.manual_seed(5300)`. No
dataset, environment, or training configuration was applicable.

| Check | Result | Peak allocated / reserved |
| --- | --- | --- |
| PathSim `env-map` parity | pass, exact | 1,024 B / 2,097,152 B |
| PathSim `agent-belief` parity | pass, exact | 2,560 B / 2,097,152 B |
| PathSim `supported-neural` parity | pass, `rtol=1e-5`, `atol=1e-6`, finite gradients | 17,046,016 B / 23,068,672 B |
| PathSim `compatibility-neural` parity | pass, `rtol=1e-5`, `atol=1e-6`, finite gradients | 17,046,016 B / 23,068,672 B |
| Mounted skrl compatibility smoke | pass: DiscreteSAC action/update, factorized distribution, ReplayBuffer, RolloutBuffer, serialization | 41,984 B / 2,097,152 B |
| CUDA `StepTrainer` suite | pass: 4 passed, 4 skipped, 8 deselected | 175,104 B / 2,097,152 B |

The external broker reports are generated, gitignored runtime evidence under
`runs/pathsim-gpu-contract-validation/` in the PathSim validation worktree;
the local execution logs are retained at `/tmp/lto-73-evidence/` for this
review session. Every completed brokered command released its lease, used an
ephemeral `docker compose run --rm` container, and declared tensor cleanup;
`docker compose ps --all` was empty after the runs.

The fixed validation image does not include `hypothesis`, so the bounded,
ephemeral container installed that test-only dependency before collecting the
existing CUDA `StepTrainer` selection. It passed in 16.08 seconds; its
post-test logger writes encountered pytest's closed capture stream, which was
non-fatal and did not alter the successful test or broker exit status.

## Documentation coverage and exceptions

This change adds only this Markdown qualification record. It adds no
human-authored Python or production code, so no `Purpose:`/`Usage:` header
exception applies; generated broker reports and temporary harnesses are not
committed.

## Review validation

Reviewers can rerun the CPU commands above from this checkout. GPU validation
must use the approved PathSim broker/envelope and the fixed PathSim contract
IDs; do not substitute a training command, unbounded selector, or a direct
unmanaged CUDA invocation.
