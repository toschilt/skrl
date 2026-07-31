"""Purpose: Define configuration for the simple factorized discrete SAC agent.

Usage: Import ``FACTORIZED_DISCRETE_SAC_CFG`` and subclass or instantiate it to
configure independent position and orientation entropy terms and optional replay
features.
"""

from __future__ import annotations

import dataclasses

from .discrete_sac_cfg import DISCRETE_SAC_CFG


@dataclasses.dataclass(kw_only=True)
class FACTORIZED_DISCRETE_SAC_CFG(DISCRETE_SAC_CFG):
    """Configuration for factorized Discrete SAC with separate entropy terms."""

    initial_position_entropy_value: float = 0.2
    """Initial entropy coefficient for the position policy."""

    initial_orientation_entropy_value: float = 0.2
    """Initial entropy coefficient for the orientation policy."""

    target_position_entropy: float | None = None
    """Target entropy for the position policy."""

    target_orientation_entropy: float | None = None
    """Target entropy for the orientation policy."""

    actor_update_delay: int = 1
    """Number of critic updates per actor/entropy update (1 updates every step)."""

    actor_learning_starts: int = 0
    """Environment timestep before actor/entropy updates start. Critics may learn earlier."""

    conservative_q_regularization_scale: float = 0.0
    """CQL-style critic penalty scale. 0 disables conservative Q regularization."""

    conservative_q_temperature: float = 1.0
    """Temperature for logsumexp over all factorized critic actions."""

    objective_gate_aux_loss_scale: float = 0.1
    """Scale for the auxiliary objective-gate supervised loss (0 disables it)."""

    objective_gate_target_temperature: float = 0.5
    """Temperature used to convert dense objective terms into gate targets."""

    objective_gate_target_floor: float = 1e-4
    """Small positive floor used for log-relative objective-gate targets."""

    objective_gate_target_min_signal: float = 1e-8
    """Minimum dense objective sum required before using non-uniform gate targets."""

    objective_gate_target_uniform_mix: float = 0.0
    """Amount of uniform smoothing mixed into non-uniform objective-gate targets."""

    objective_gate_entropy_regularization: float = 0.01
    """Entropy bonus coefficient for objective-gate weights."""

    objective_gate_batch_balance_regularization: float = 0.0
    """Penalty coefficient that keeps the batch-average gate weights balanced."""

    curriculum_replay_enabled: bool = False
    """Sample replay batches with awareness of current curriculum phase/difficulty."""

    curriculum_replay_current_fraction: float = 0.70
    """Fraction of each batch sampled from exact current phase/difficulty buckets."""

    curriculum_replay_adjacent_fraction: float = 0.20
    """Fraction of each batch sampled from adjacent-phase buckets at the same difficulty."""

    curriculum_replay_uniform_fraction: float = 0.10
    """Fraction of each batch sampled uniformly from all replay entries."""

    curriculum_replay_final_phase_current_fraction: float = 0.55
    """Current-bucket fraction once the reward curriculum reaches its final phase."""

    curriculum_replay_final_phase_adjacent_fraction: float = 0.35
    """Adjacent-phase fraction once the reward curriculum reaches its final phase."""

    curriculum_replay_final_phase_uniform_fraction: float = 0.10
    """Uniform fraction once the reward curriculum reaches its final phase."""

    curriculum_replay_min_bucket_size: int = 96
    """Minimum samples required before using a curriculum replay bucket."""

    curriculum_replay_adjacent_stage_distance: int = 1
    """Maximum phase distance considered adjacent for curriculum replay sampling."""

    curriculum_replay_reward_names: tuple[str, ...] = ()
    """Reward component names to store unweighted for phase-aware reward recomputation."""

    curriculum_replay_reward_phase_weights: tuple[tuple[float, ...], ...] = ()
    """Per-phase reward weights matching curriculum_replay_reward_names."""

    curriculum_replay_recompute_rewards: bool = False
    """Recompute sampled replay rewards with the active reward phase weights."""

    terminal_reward_diagnostic_name: str = ""
    """Optional reward component name used to diagnose reward/done target alignment."""

    terminal_reward_diagnostic_threshold: float = 0.0
    """Reward threshold above which terminal-reward alignment diagnostics are emitted."""

    skip_previous_done_transitions: bool = False
    """Skip replay insertion for vector-env transitions whose observation came from a previous done episode."""

    sequential_critic_update: bool = False
    """Update twin critics one at a time to reduce peak activation memory for large graph batches."""

    invalid_replay_action_fail_threshold: float | None = None
    """Raise when the sampled invalid-position-action fraction exceeds this value. None only logs."""
