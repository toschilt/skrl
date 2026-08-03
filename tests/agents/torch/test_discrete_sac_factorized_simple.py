"""Purpose: Verify the CPU contract of the simple factorized discrete SAC agent.

Usage: Run ``pytest -q tests/agents/torch/test_discrete_sac_factorized_simple.py``
from an isolated skrl checkout with the Torch test dependencies installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import pytest

import torch

from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.sac.discrete_sac_cfg_factorized import FACTORIZED_DISCRETE_SAC_CFG
from skrl.agents.torch.sac.discrete_sac_factorized_simple import FactorizedDiscreteSACSimple
from skrl.memories.torch.replay import ReplayBuffer
from skrl.models.torch import Model

N_POSITION_ACTIONS = 3
N_ORIENTATION_ACTIONS = 2
OBSERVATION_SIZE = 4


class FactorizedPolicy(Model):
    """Small policy exposing the factorized output contract used by the agent."""

    def __init__(self) -> None:
        observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(OBSERVATION_SIZE,),
        )
        action_space = gymnasium.spaces.MultiDiscrete([N_POSITION_ACTIONS, N_ORIENTATION_ACTIONS])
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
        )
        self.position = torch.nn.Linear(OBSERVATION_SIZE, N_POSITION_ACTIONS)
        self.orientation = torch.nn.Linear(
            OBSERVATION_SIZE,
            N_POSITION_ACTIONS * N_ORIENTATION_ACTIONS,
        )

    def compute(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del role
        observations = inputs["observations"]
        logits_position = self.position(observations)
        logits_orientation = self.orientation(observations).reshape(
            -1,
            N_POSITION_ACTIONS,
            N_ORIENTATION_ACTIONS,
        )
        position_actions = logits_position.argmax(dim=-1)
        batch_index = torch.arange(observations.shape[0])
        orientation_actions = logits_orientation[
            batch_index,
            position_actions,
        ].argmax(dim=-1)
        actions = torch.stack((position_actions, orientation_actions), dim=-1)
        return actions, {
            "logits_pos": logits_position,
            "logits_rot": logits_orientation,
        }

    def act(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.compute(inputs, role=role)


class FactorizedCritic(Model):
    """Small critic supporting all-action and selected-action queries."""

    def __init__(self) -> None:
        observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(OBSERVATION_SIZE,),
        )
        action_space = gymnasium.spaces.MultiDiscrete([N_POSITION_ACTIONS, N_ORIENTATION_ACTIONS])
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
        )
        self.values = torch.nn.Linear(
            OBSERVATION_SIZE,
            N_POSITION_ACTIONS * N_ORIENTATION_ACTIONS,
        )
        self.selected_action_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def _all_values(self, observations: torch.Tensor) -> torch.Tensor:
        return self.values(observations).reshape(
            -1,
            N_POSITION_ACTIONS,
            N_ORIENTATION_ACTIONS,
        )

    def compute(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del role
        all_values = self._all_values(inputs["observations"])
        if inputs.get("all_position_actions", False):
            return all_values, {}

        position_actions = inputs["position_actions"].reshape(-1).long()
        orientation_actions = inputs["rotation_actions"].reshape(-1).long()
        batch_index = torch.arange(all_values.shape[0])
        selected_values = all_values[
            batch_index,
            position_actions,
            orientation_actions,
        ].unsqueeze(1)
        self.selected_action_shapes.append(
            (
                position_actions.shape,
                orientation_actions.shape,
                selected_values.shape,
            )
        )
        return selected_values, {}

    def act(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.compute(inputs, role=role)


class CountingReplayBuffer(ReplayBuffer):
    """Baseline replay buffer that records uniform sampling calls."""

    def __init__(self) -> None:
        super().__init__(memory_size=32, num_envs=1, device="cpu")
        self.sample_calls = 0

    def sample(
        self,
        names: list[str],
        *,
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> list[list[torch.Tensor | list[torch.Tensor]]]:
        self.sample_calls += 1
        return super().sample(
            names,
            batch_size=batch_size,
            mini_batches=mini_batches,
            sequence_length=sequence_length,
        )


class MultiplyPreprocessor:
    """Deterministic callable preprocessor used to observe agent input routing."""

    def __init__(self, *, multiplier: float) -> None:
        self.multiplier = multiplier

    def __call__(self, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return value * self.multiplier


def _spaces() -> tuple[gymnasium.Space, gymnasium.Space]:
    observation_space = gymnasium.spaces.Box(
        low=-1,
        high=1,
        shape=(OBSERVATION_SIZE,),
    )
    action_space = gymnasium.spaces.MultiDiscrete([N_POSITION_ACTIONS, N_ORIENTATION_ACTIONS])
    return observation_space, action_space


def _states(batch_size: int, *, pad_last_position: bool = False) -> list[torch.Tensor]:
    state_parts = [torch.zeros(batch_size, 1) for _ in range(8)]
    state_parts[4] = torch.zeros(batch_size, 1, 1, dtype=torch.long)
    state_parts[6] = torch.arange(N_POSITION_ACTIONS).reshape(1, N_POSITION_ACTIONS, 1).repeat(batch_size, 1, 1)
    state_parts[7] = torch.zeros(
        batch_size,
        1,
        N_POSITION_ACTIONS,
        dtype=torch.bool,
    )
    if pad_last_position:
        state_parts[7][:, :, -1] = True
    return state_parts


def _agent(
    memory: ReplayBuffer | None = None,
    **overrides: Any,
) -> FactorizedDiscreteSACSimple:
    observation_space, action_space = _spaces()
    cfg_kwargs = dict(
        gradient_steps=1,
        batch_size=4,
        learning_rate=(1e-3, 1e-3, 1e-3),
        learn_entropy=True,
        target_position_entropy=None,
        target_orientation_entropy=None,
        curriculum_replay_enabled=False,
        mixed_precision=False,
        experiment=ExperimentCfg(write_interval=0, checkpoint_interval=0),
    )
    cfg_kwargs.update(overrides)
    cfg = FACTORIZED_DISCRETE_SAC_CFG(**cfg_kwargs)
    models = {
        "policy": FactorizedPolicy(),
        "critic_1": FactorizedCritic(),
        "critic_2": FactorizedCritic(),
        "target_critic_1": FactorizedCritic(),
        "target_critic_2": FactorizedCritic(),
    }
    return FactorizedDiscreteSACSimple(
        models=models,
        memory=memory,
        observation_space=observation_space,
        state_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )


def _record_batch(agent: FactorizedDiscreteSACSimple, *, position_action: int = 1) -> None:
    """Store one deterministic batch that is valid for the standard state fixture."""
    batch_size = 8
    agent.record_transition(
        observations=torch.arange(batch_size * OBSERVATION_SIZE, dtype=torch.float32).reshape(
            batch_size, OBSERVATION_SIZE
        ),
        states=_states(batch_size),
        actions=torch.tensor(
            [[position_action, index % N_ORIENTATION_ACTIONS] for index in range(batch_size)],
            dtype=torch.float32,
        ),
        rewards=torch.ones(batch_size, 1),
        next_observations=torch.zeros(batch_size, OBSERVATION_SIZE),
        next_states=_states(batch_size),
        terminated=torch.zeros(batch_size, 1, dtype=torch.bool),
        truncated=torch.zeros(batch_size, 1, dtype=torch.bool),
        infos={},
        timestep=0,
        timesteps=1,
    )


def test_factorized_configuration_defaults() -> None:
    cfg = FACTORIZED_DISCRETE_SAC_CFG()

    assert cfg.initial_position_entropy_value == pytest.approx(0.2)
    assert cfg.initial_orientation_entropy_value == pytest.approx(0.2)
    assert cfg.target_position_entropy is None
    assert cfg.target_orientation_entropy is None
    assert cfg.actor_update_delay == 1
    assert cfg.actor_learning_starts == 0
    assert cfg.curriculum_replay_enabled is False
    assert cfg.curriculum_replay_recompute_rewards is False


def test_position_mask_distribution_and_action_sanitization() -> None:
    states = _states(2, pad_last_position=True)
    invalid_mask = FactorizedDiscreteSACSimple._build_position_invalid_mask_from_states(states)

    assert invalid_mask.shape == (2, N_POSITION_ACTIONS)
    assert invalid_mask.tolist() == [[True, False, True], [True, False, True]]

    logits = torch.tensor([[10.0, 2.0, 5.0], [1.0, 3.0, 7.0]])
    log_probabilities, probabilities = FactorizedDiscreteSACSimple._masked_position_distribution(
        logits,
        invalid_mask,
    )

    assert log_probabilities.shape == probabilities.shape == (2, N_POSITION_ACTIONS)
    assert torch.isfinite(log_probabilities).all()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.equal(probabilities[:, 0], torch.zeros(2))
    assert torch.equal(probabilities[:, 1], torch.ones(2))
    assert torch.equal(probabilities[:, 2], torch.zeros(2))

    sanitized = FactorizedDiscreteSACSimple._sanitize_position_actions(
        torch.tensor([0, 99]),
        invalid_mask,
    )
    assert sanitized.tolist() == [1, 1]

    all_invalid = torch.ones(1, N_POSITION_ACTIONS, dtype=torch.bool)
    _, fallback_probabilities = FactorizedDiscreteSACSimple._masked_position_distribution(
        torch.zeros(1, N_POSITION_ACTIONS),
        all_invalid,
    )
    assert fallback_probabilities.tolist() == [[1.0, 0.0, 0.0]]


def test_action_policy_entropy_and_critic_shapes() -> None:
    agent = _agent()
    observations = torch.randn(5, OBSERVATION_SIZE)

    actions, outputs = agent.act(
        observations,
        None,
        timestep=0,
        timesteps=1,
    )
    logits_position, logits_orientation = agent._extract_factorized_policy_outputs(outputs)
    invalid_mask = torch.zeros(5, N_POSITION_ACTIONS, dtype=torch.bool)
    position_log_probabilities, position_probabilities = agent._masked_position_distribution(
        logits_position, invalid_mask
    )
    orientation_log_probabilities = torch.log_softmax(
        logits_orientation,
        dim=-1,
    )
    orientation_probabilities = orientation_log_probabilities.exp()
    agent._resolve_target_entropies(
        logits_position.shape[1],
        logits_orientation.shape[2],
    )

    assert actions.shape == (5, 2)
    assert logits_position.shape == (5, N_POSITION_ACTIONS)
    assert logits_orientation.shape == (
        5,
        N_POSITION_ACTIONS,
        N_ORIENTATION_ACTIONS,
    )
    assert position_log_probabilities.shape == position_probabilities.shape
    assert orientation_log_probabilities.shape == orientation_probabilities.shape
    assert agent._target_position_entropy == pytest.approx(torch.log(torch.tensor(float(N_POSITION_ACTIONS))).item())
    assert agent._target_orientation_entropy == pytest.approx(
        torch.log(torch.tensor(float(N_ORIENTATION_ACTIONS))).item()
    )

    critic = agent.critic_1
    all_values, _ = critic.act(
        {"observations": observations, "all_position_actions": True},
        role="critic_1",
    )
    selected_values, _ = critic.act(
        {
            "observations": observations,
            "position_actions": actions[:, 0],
            "rotation_actions": actions[:, 1],
        },
        role="critic_1",
    )
    assert all_values.shape == (
        5,
        N_POSITION_ACTIONS,
        N_ORIENTATION_ACTIONS,
    )
    assert selected_values.shape == (5, 1)


def test_baseline_replay_fallback_and_synthetic_update() -> None:
    torch.manual_seed(7)
    memory = CountingReplayBuffer()
    assert not hasattr(memory, "valid_indices")

    agent = _agent(memory)
    agent.init()

    batch_size = 8
    observations = torch.randn(batch_size, OBSERVATION_SIZE)
    next_observations = torch.randn(batch_size, OBSERVATION_SIZE)
    states = _states(batch_size)
    next_states = _states(batch_size)
    actions = torch.stack(
        (
            torch.ones(batch_size),
            torch.arange(batch_size) % N_ORIENTATION_ACTIONS,
        ),
        dim=1,
    )
    agent.record_transition(
        observations=observations,
        states=states,
        actions=actions,
        rewards=torch.randn(batch_size, 1),
        next_observations=next_observations,
        next_states=next_states,
        terminated=torch.zeros(batch_size, 1, dtype=torch.bool),
        truncated=torch.zeros(batch_size, 1, dtype=torch.bool),
        infos={},
        timestep=0,
        timesteps=1,
    )

    sampled = agent._sample_training_batch()
    assert memory.sample_calls == 1
    assert len(sampled) == 8
    assert sampled[2].shape == (agent.cfg.batch_size, 2)
    assert isinstance(sampled[1], list)
    assert sampled[1][6].shape == (
        agent.cfg.batch_size,
        N_POSITION_ACTIONS,
        1,
    )

    policy_before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    agent.update(timestep=1, timesteps=1)

    assert memory.sample_calls == 2
    assert any(not torch.equal(before, after) for before, after in zip(policy_before, agent.policy.parameters()))
    assert agent._target_position_entropy == pytest.approx(torch.log(torch.tensor(float(N_POSITION_ACTIONS))).item())
    assert agent._target_orientation_entropy == pytest.approx(
        torch.log(torch.tensor(float(N_ORIENTATION_ACTIONS))).item()
    )
    for critic in (agent.critic_1, agent.critic_2):
        assert critic.selected_action_shapes
        position_shape, orientation_shape, gathered_shape = critic.selected_action_shapes[-1]
        assert position_shape == orientation_shape == torch.Size([agent.cfg.batch_size])
        assert gathered_shape == torch.Size([agent.cfg.batch_size, 1])


@pytest.mark.parametrize(
    ("position_target", "orientation_target"),
    [(0.0, 0.5), (1.25, 2.5)],
)
def test_explicit_entropy_targets_are_preserved(
    position_target: float,
    orientation_target: float,
) -> None:
    agent = _agent(
        target_position_entropy=position_target,
        target_orientation_entropy=orientation_target,
    )

    agent._resolve_target_entropies(N_POSITION_ACTIONS, N_ORIENTATION_ACTIONS)

    assert agent._target_position_entropy == position_target
    assert agent._target_orientation_entropy == orientation_target


@pytest.mark.parametrize("bad_neighbors", [torch.zeros(2, 3, 1, 1), torch.zeros(2)])
def test_position_mask_rejects_unsupported_neighbor_dimensions(
    bad_neighbors: torch.Tensor,
) -> None:
    states = _states(2)
    states[6] = bad_neighbors

    with pytest.raises(ValueError, match="current_neighbors must have dim 2 or 3"):
        FactorizedDiscreteSACSimple._build_position_invalid_mask_from_states(states)


@pytest.mark.parametrize("outputs", [{}, {"logits_pos": torch.zeros(1, 3)}])
def test_policy_output_contract_requires_both_factorized_logits(outputs: dict[str, torch.Tensor]) -> None:
    with pytest.raises(RuntimeError, match="logits_pos.*logits_rot"):
        FactorizedDiscreteSACSimple._extract_factorized_policy_outputs(outputs)


def test_conservative_penalty_and_q_diagnostics_exclude_invalid_actions() -> None:
    q_all = torch.tensor([[[1.0, 100.0], [2.0, 3.0], [4.0, 5.0]]])
    q_taken = torch.tensor([[2.0]])
    invalid_position_mask = torch.tensor([[True, False, False]])
    invalid_orientation_mask = torch.tensor([[[False, False], [False, True], [True, True]]])

    penalty = FactorizedDiscreteSACSimple._conservative_q_penalty(
        q_all,
        q_taken,
        invalid_position_mask,
        temperature=0.0,
        invalid_action_mask=invalid_orientation_mask,
    )
    diagnostics = FactorizedDiscreteSACSimple._factorized_q_policy_diagnostics(
        q_all,
        torch.tensor([[0.8, 0.1, 0.1]]),
        torch.full((1, 3, 2), 0.5),
        invalid_position_mask,
        invalid_orientation_mask,
    )

    assert penalty == pytest.approx(0.0, abs=1e-4)
    assert diagnostics["q_all_max"].item() == 2.0
    assert diagnostics["q_max_action_position_mean"].item() == 1.0
    assert diagnostics["q_max_action_orientation_mean"].item() == 0.0


def test_delayed_actor_updates_and_learning_start_gate_policy_gradients() -> None:
    torch.manual_seed(11)
    memory = CountingReplayBuffer()
    agent = _agent(memory, actor_update_delay=2, actor_learning_starts=2)
    agent.init()
    _record_batch(agent)
    policy_before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    agent.update(timestep=1, timesteps=3)

    assert all(torch.equal(before, after) for before, after in zip(policy_before, agent.policy.parameters()))
    assert agent._gradient_update_counter == 1

    agent.update(timestep=2, timesteps=3)

    assert any(not torch.equal(before, after) for before, after in zip(policy_before, agent.policy.parameters()))
    assert agent._gradient_update_counter == 2


def test_entropy_disabled_keeps_manual_target_unchanged() -> None:
    agent = _agent(learn_entropy=False)

    agent.set_curriculum_entropy_target((0.3, 0.7))

    assert not hasattr(agent, "entropy_coefficients")
    assert agent._target_position_entropy is None
    assert agent._target_orientation_entropy is None


def test_act_routes_observations_and_fallback_states_through_preprocessors() -> None:
    agent = _agent(
        observation_preprocessor=MultiplyPreprocessor,
        observation_preprocessor_kwargs={"multiplier": 2.0},
        state_preprocessor=MultiplyPreprocessor,
        state_preprocessor_kwargs={"multiplier": 3.0},
    )
    captured: dict[str, torch.Tensor] = {}
    original_act = agent.policy.act

    def capture(inputs: dict[str, torch.Tensor], *, role: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        captured.update(inputs)
        return original_act(inputs, role=role)

    agent.policy.act = capture  # type: ignore[method-assign]
    observations = torch.ones(2, OBSERVATION_SIZE)
    agent.act(observations, None, timestep=0, timesteps=1)

    assert torch.equal(captured["observations"], observations * 2)
    assert torch.equal(captured["states"], observations * 3)


def test_invalid_replay_action_threshold_fails_before_critic_update() -> None:
    memory = CountingReplayBuffer()
    agent = _agent(memory, invalid_replay_action_fail_threshold=0.0)
    agent.init()
    _record_batch(agent, position_action=0)

    with pytest.raises(RuntimeError, match="Replay sampled invalid position actions"):
        agent.update(timestep=1, timesteps=1)


@pytest.mark.parametrize("sequential", [False, True])
def test_critic_update_modes_and_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
    sequential: bool,
) -> None:
    memory = CountingReplayBuffer()
    agent = _agent(
        memory,
        sequential_critic_update=sequential,
        conservative_q_regularization_scale=0.25,
        policy_grad_norm_clip=0.5,
        q_network_grad_norm_clip=0.5,
    )
    agent.init()
    _record_batch(agent)
    calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters: Any, max_norm: float, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    agent.update(timestep=1, timesteps=1)

    assert calls.count(0.5) >= 2
    assert agent._gradient_update_counter == 1


def test_save_load_round_trip_preserves_models_and_optimizer_state(tmp_path: Path) -> None:
    memory = CountingReplayBuffer()
    agent = _agent(memory)
    agent.init()
    _record_batch(agent)
    agent.update(timestep=1, timesteps=1)
    checkpoint = tmp_path / "factorized-sac.pt"
    agent.save(str(checkpoint))

    restored = _agent()
    restored.load(str(checkpoint))

    assert checkpoint.exists()
    assert restored.checkpoint_modules.keys() == agent.checkpoint_modules.keys()
    for expected, actual in zip(agent.policy.parameters(), restored.policy.parameters()):
        assert torch.equal(expected, actual)
    assert restored.policy_optimizer.state_dict()["state"]
    assert restored.position_entropy_optimizer.state_dict()["state"]
