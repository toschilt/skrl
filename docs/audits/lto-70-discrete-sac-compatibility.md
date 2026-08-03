# LTO-70 DiscreteSAC compatibility characterization

## Scope

`tests/agents/torch/test_discrete_sac.py` is the CPU-only contract suite for
the PathSim-facing `DiscreteSAC` boundary. It uses synthetic Torch models and
replay payloads; it neither imports PathSim nor executes an environment rollout
or a training run.

## Characterized public contract

| Surface | Deterministic assertion |
| --- | --- |
| State selection and preprocessing | Explicit states are passed through the configured state preprocessor; absent states fall back to observations in `record_transition`. |
| Nested replay values | Lists, tuples, and dictionaries retain their container types, tensor shapes, and device placement during recursive transfer. |
| Discrete policy and masks | Policies may return `probs` plus `log_probs`, or `logits`/`net_output`; invalid, all-invalid, and out-of-range selected actions safely gather a valid critic entry without mutating the policy mask. |
| Random actions | Before `random_timesteps`, the standard discrete model sampler is used and returns its normal empty auxiliary-output mapping. |
| Memory transitions | The agent records observations, selected/fallback states, actions, rewards, next values, and done signals without adding application-specific payload requirements. |
| Updates | A CPU update changes policy and both critics, updates learned entropy, and invokes policy, online critics, then target critics under their documented roles. |
| Checkpoints | The manifest contains five models, learned-entropy optimizer, three training optimizers, and configured state preprocessor; loading restores model and preprocessor state. |
| Errors | Incomplete/missing policy distributions and invalid sanitizer sizes or mask shapes fail explicitly. |

## Device tolerance

The added tests are parametrized with `device="cpu"` and validated with
`CUDA_VISIBLE_DEVICES=''`. They make exact deterministic assertions for the
small CPU tensors. No GPU test, tolerance relaxation, simulator integration, or
training-quality claim is part of LTO-70; a future bounded GPU validation must
choose its own numerical tolerances rather than treating this CPU suite as GPU
evidence.

## Documentation coverage and exceptions

The modified Python test module begins with accurate `Purpose:` and `Usage:`
headers. This Markdown audit is documentation rather than code, so those code
headers do not apply. No generated or comment-free source exception was added.
