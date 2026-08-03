# LTO-69 skrl surface audit

## Scope and reproducibility

This is the ownership and compatibility snapshot for `origin/develop` at
`cf946bc36205f56fa61c6b30f71ff03f8097cb64` (the LTO-27 and LTO-28 merge
base). The current tracked surface contains **532 Python files**: 281 library modules, 99
test modules, 132 runnable examples, and 20 documentation snippets.  The
source-derived guard in `tests/test_lto69_inventory.py` covers every file in
that snapshot; it deliberately excludes itself, because it is audit metadata
added after the snapshot.

The audit is intentionally descriptive.  It changes no production code and
does not import, execute, or require PathSim.

## Classification and contract ownership

Every Python path belongs to one row of this table (the test has the exact
prefix rules). `Abstract` means an import barrel or an explicit `base.py`
contract, rather than an unowned implementation. `CPU-capable` means the
surface can be inspected or unit tested without CUDA; it does not promise that
an optional simulator dependency is installed.

| Paths | Count | Contract owner | Upstream status | Device classification |
| --- | ---: | --- | --- | --- |
| `skrl/**` | 281 | skrl library API; `__init__.py` and `base.py` are abstract/import contracts | Upstream-compatible except the fork manifest below | CPU-capable; Isaac-family adapters are GPU-required at runtime |
| `tests/**` | 99 | Test contract for the matching library API; package initializers are abstract | Upstream-compatible except the two fork contract suites | CPU-capable; Isaac-family wrapper tests require their external GPU simulator |
| `examples/**` | 132 | Runnable consumer examples | Upstream-compatible | CPU-capable unless the path names Isaac Gym/Lab or its documented simulator dependency |
| `docs/source/**` | 20 | Documentation configuration and executable/snippet contract | Upstream-compatible | CPU-capable documentation surface |

The complete path universe is reproducible with:

```bash
rg --files -g '*.py' | sort
```

`tests/test_lto69_inventory.py` enforces the four ownership roots, identifies
abstract module contracts, applies the simulator device exception, and asserts
the tracked 532-file baseline. This makes additions fail review until their
owner/category is explicitly covered by a prefix rule or the audit is updated.

## Fork delta from upstream-compatible skrl

`5a078cff1a611d51eb2164101cdd6fa84e3eec2d` is the common ancestor of this
fork and `Toni-SM/skrl` `develop` at audit time. The focused fork delta is the
following 14 files (12 production/import surfaces and two CPU contract suites):

```text
skrl/agents/torch/ppo/ppo_rnn.py
skrl/agents/torch/sac/__init__.py
skrl/agents/torch/sac/_common.py
skrl/agents/torch/sac/_factorized.py
skrl/agents/torch/sac/discrete_sac.py
skrl/agents/torch/sac/discrete_sac_cfg.py
skrl/agents/torch/sac/discrete_sac_cfg_factorized.py
skrl/agents/torch/sac/discrete_sac_factorized_simple.py
skrl/memories/torch/__init__.py
skrl/memories/torch/replay.py
skrl/memories/torch/rollout.py
skrl/trainers/torch/base.py
tests/agents/torch/test_discrete_sac.py
tests/agents/torch/test_discrete_sac_factorized_simple.py
```

All remaining snapshot files are upstream-compatible relative to that ancestor.
The current upstream head is intentionally not used as the classification base:
it has independently advanced substantially, so comparing it directly would
mislabel unrelated upstream evolution as a PathSim fork change.

## PathSim compatibility boundary

There are **no `PathSim` imports in this repository**. The compatibility delta
is therefore an API/data-contract boundary, not a source dependency. The eight
direct boundary modules are `_common.py`, `_factorized.py`, `discrete_sac.py`,
`discrete_sac_cfg.py`, `discrete_sac_cfg_factorized.py`,
`discrete_sac_factorized_simple.py`, `replay.py`, and `rollout.py`; two package barrels expose imports, and
`ppo_rnn.py` plus `trainers/torch/base.py` are integration adaptations.

```text
PathSim consumer configuration (external; no import edge in skrl)
    -> Torch discrete / factorized action, state, done, and info payloads
    -> DiscreteSAC or FactorizedDiscreteSACSimple
    -> internal SAC setup and factorized-math helpers
    -> ReplayBuffer (off-policy) or RolloutBuffer (sequence/PPO_RNN)
    -> skrl Model / Agent / Trainer public contracts
```

The externally consumable symbols and required payload contracts are:

| Symbol | Import path | Contract |
| --- | --- | --- |
| `DiscreteSAC` | `skrl.agents.torch.sac` | Policy emits either (`probs`, `log_probs`) or `logits`/`net_output`; optional `invalid_action_mask`; critics return action-value vectors. |
| `DISCRETE_SAC_CFG` | `skrl.agents.torch.sac` | SAC configuration plus separated policy/critic gradient clips and target-update interval. |
| `FactorizedDiscreteSACSimple` | `skrl.agents.torch.sac.discrete_sac_factorized_simple` | `MultiDiscrete` `[position, orientation]`; policy emits `logits_pos` and conditional `logits_rot`; critics support all-action and selected-action queries. |
| `FACTORIZED_DISCRETE_SAC_CFG` | `skrl.agents.torch.sac.discrete_sac_cfg_factorized` | Factorized entropy, replay curriculum, conservative-Q, and action validity controls. |
| `ReplayBuffer` | `skrl.memories.torch` | Nested/list tensor replay, logical sampling, and registered tensor graph storage. |
| `RolloutBuffer` | `skrl.memories.torch.rollout` | Per-environment sequence storage for PPO_RNN-compatible multimodal rollouts. |

The two contract suites cover nested replay/device transfer, distribution
equivalence, invalid-action sanitation, factorized shape/mask handling,
curriculum/replay fallback, and synthetic CPU updates. They are the owned
non-regression boundary for the later LTO-70 and LTO-71 characterization work.

## Device matrix

| Surface | CPU | GPU requirement |
| --- | --- | --- |
| Core Torch agents, models, memories, trainers, utilities, and the fork delta | Supported when constructed with `device="cpu"` | Optional; no CUDA execution is needed for this audit |
| JAX and Warp library trees | Import/API classification only in this audit | Optional backend acceleration; not run |
| Isaac Gym / Isaac Lab examples and wrappers | Not a standalone CPU test target | Simulator/runtime GPU required |
| PathSim-facing discrete/factorized contracts | Synthetic CPU coverage | GPU is not required to establish the public contract |

## CPU baseline

The supported baseline is the Torch CI environment described by
`.github/workflows/tests-torch.yml`: Ubuntu 22.04, Python 3.10,
`numpy<2.0`, CPU PyTorch 1.11, and `.[torch,tests]`. LTO-69's planning input
names a historic **67-test baseline**, but no current skrl command reproduces
that number. The audited current-tree command was:

```bash
python3 -m pytest -q \
  tests/agents/torch tests/memories/torch tests/test_torch_config.py
```

The recorded result was `83 passed, 52 skipped, 8 failed` with
`torch 1.11.0+cu102` and `torch.cuda.is_available() == False`. All eight
failures are pre-existing `tests/memories/torch/test_base.py` expectations
against the fork's memory surface; the agent suite itself completed without a
failure. This replaces the stale 67-test claim with reproducible evidence and
is deliberately not repaired by this audit. The baseline excludes JAX, Warp,
Isaac-family wrappers, training runs, and all GPU commands.

## Documentation coverage and exceptions

The LTO-72 helper modules and modified SAC implementations begin with
`Purpose:` and `Usage:` documentation; the existing inventory test already
does too. This Markdown audit is documentation rather than executable code, so
it is not a header exception. No generated or comment-free code exception was
created.

## Known limits

The repository does not contain PathSim's six consumer configuration files, so
their names and runtime instantiation sites cannot be asserted here without
leaving the skrl repository. This audit proves the skrl-side import graph is
empty and records the exact public data contracts those external configurations
must satisfy. It also records CPU contract coverage only; it is not evidence of
simulator, training-quality, JAX, Warp, or CUDA behavior.
