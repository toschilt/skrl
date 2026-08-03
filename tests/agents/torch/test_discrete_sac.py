"""Purpose: Verify DiscreteSAC nested replay, distribution, and masked-action compatibility.

Usage: Run ``pytest -q tests/agents/torch/test_discrete_sac.py`` in the supported
CPU test environment; the synthetic models and replay samples require no environment rollout.
"""

from __future__ import annotations

from typing import Any

import pytest

import gymnasium

import torch

from skrl.agents.torch.sac import DISCRETE_SAC_CFG, DiscreteSAC
from skrl.models.torch import Model


def _batch_size(inputs: dict[str, Any]) -> int:
    states = inputs["states"]
    return states[0].shape[0] if isinstance(states, (list, tuple)) else states.shape[0]


class _SyntheticPolicy(Model):
    def __init__(self, contract: str, invalid_action_mask: torch.Tensor | None = None) -> None:
        super().__init__(
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            action_space=gymnasium.spaces.Discrete(3),
            device="cpu",
        )
        self.contract = contract
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.5]))
        self.register_buffer("invalid_action_mask", invalid_action_mask)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls.append((role, inputs))
        logits = self.logits.unsqueeze(0).expand(_batch_size(inputs), -1)
        if self.contract == "distribution":
            outputs = {
                "probs": torch.softmax(logits, dim=-1),
                "log_probs": torch.log_softmax(logits, dim=-1),
            }
        else:
            outputs = {self.contract: logits}
        if self.invalid_action_mask is not None:
            outputs["invalid_action_mask"] = self.invalid_action_mask[: logits.shape[0]]
        return logits.argmax(dim=-1, keepdim=True), outputs


class _SyntheticCritic(Model):
    def __init__(self) -> None:
        super().__init__(
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            action_space=gymnasium.spaces.Discrete(3),
            device="cpu",
        )
        self.q_values = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.7]))
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls.append((role, inputs))
        return self.q_values.unsqueeze(0).expand(_batch_size(inputs), -1), {}


class _SyntheticReplay:
    def __init__(self, sample: tuple[Any, ...]) -> None:
        self.sample_value = sample

    def sample(self, names: list[str], *, batch_size: int) -> list[tuple[Any, ...]]:
        assert batch_size == self.sample_value[2].shape[0]
        return [self.sample_value]


class _RecordingReplay(_SyntheticReplay):
    """Minimal replay double that records the public transition payload."""

    def __init__(self, sample: tuple[Any, ...]) -> None:
        super().__init__(sample)
        self.transitions: list[dict[str, Any]] = []

    def add_samples(self, **samples: Any) -> None:
        self.transitions.append(samples)


class _OffsetPreprocessor(torch.nn.Module):
    """Deterministic state preprocessor that preserves checkpoint state."""

    def __init__(self, *, offset: float) -> None:
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset))
        self.calls: list[tuple[Any, bool]] = []

    def forward(self, value: Any, train: bool = False) -> Any:
        self.calls.append((value, train))
        return value + self.offset if torch.is_tensor(value) else value


def _invalid_action_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, False],
            [True, True, True],
            [False, False, False],
        ]
    )


def _nested_states() -> list[Any]:
    return [
        torch.arange(24, dtype=torch.float32).reshape(3, 4, 2),
        {"context": torch.ones(3, 2)},
        (torch.zeros(3, 1), [torch.full((3, 2), 2.0)]),
    ]


def _nested_replay_sample() -> tuple[Any, ...]:
    observations = {
        "features": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "history": (torch.ones(3, 1), [torch.zeros(3, 2)]),
    }
    return (
        observations,
        _nested_states(),
        torch.tensor([[1.0], [99.0], [-4.0]]),
        torch.tensor([[1.0], [0.5], [-0.5]]),
        {
            "features": observations["features"] + 1,
            "history": (torch.full((3, 1), 2.0), [torch.full((3, 2), 3.0)]),
        },
        _nested_states(),
        torch.zeros(3, 1, dtype=torch.bool),
        torch.zeros(3, 1, dtype=torch.bool),
    )


def _make_agent(
    contract: str,
    sample: tuple[Any, ...],
    invalid_action_mask: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    memory: _SyntheticReplay | None = None,
    learn_entropy: bool = False,
    state_preprocessor: type[torch.nn.Module] | None = None,
    random_timesteps: int = 0,
) -> DiscreteSAC:
    models = {
        "policy": _SyntheticPolicy(contract, invalid_action_mask),
        "critic_1": _SyntheticCritic(),
        "critic_2": _SyntheticCritic(),
        "target_critic_1": _SyntheticCritic(),
        "target_critic_2": _SyntheticCritic(),
    }
    cfg = DISCRETE_SAC_CFG(
        batch_size=3,
        learn_entropy=learn_entropy,
        random_timesteps=random_timesteps,
        state_preprocessor=state_preprocessor,
        state_preprocessor_kwargs={"offset": 2.0} if state_preprocessor is not None else {},
        steps_to_target_net_update=64,
        experiment={
            "directory": "",
            "experiment_name": "test",
            "write_interval": 0,
            "checkpoint_interval": 0,
        },
    )
    agent = DiscreteSAC(
        models=models,
        memory=_SyntheticReplay(sample) if memory is None else memory,
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=gymnasium.spaces.Discrete(3),
        device=device,
        cfg=cfg,
    )
    agent._tensors_names = [
        "observations",
        "states",
        "actions",
        "rewards",
        "next_observations",
        "next_states",
        "terminated",
        "truncated",
    ]
    return agent


def test_recursive_device_transfer_preserves_nested_types_and_shapes() -> None:
    agent = object.__new__(DiscreteSAC)
    agent.device = torch.device("meta")
    nested = {
        "list": [torch.ones(2, 3), (torch.zeros(1), "unchanged")],
        "tuple": ({"tensor": torch.arange(4)}, 7),
    }

    moved = agent._to_device_recursive(nested)

    assert isinstance(moved, dict)
    assert isinstance(moved["list"], list)
    assert isinstance(moved["list"][1], tuple)
    assert isinstance(moved["tuple"], tuple)
    assert moved["list"][0].shape == (2, 3)
    assert moved["list"][1][0].shape == (1,)
    assert moved["tuple"][0]["tensor"].shape == (4,)
    assert moved["list"][0].device.type == "meta"
    assert moved["list"][1][0].device.type == "meta"
    assert moved["tuple"][0]["tensor"].device.type == "meta"
    assert moved["list"][1][1] == "unchanged"
    assert moved["tuple"][1] == 7


@pytest.mark.parametrize("logits_key", ["logits", "net_output"])
def test_distribution_and_logits_contracts_are_equivalent(logits_key: str) -> None:
    logits = torch.tensor([[0.2, -0.4, 1.0], [1.2, 0.1, -0.7]])
    expected_log_probs = torch.log_softmax(logits, dim=-1)
    expected_probs = expected_log_probs.exp()

    explicit = DiscreteSAC._action_distribution_from_outputs({"probs": expected_probs, "log_probs": expected_log_probs})
    from_logits = DiscreteSAC._action_distribution_from_outputs({logits_key: logits})

    torch.testing.assert_close(explicit[0], from_logits[0])
    torch.testing.assert_close(explicit[1], from_logits[1])


def test_incomplete_distribution_contract_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="both 'probs' and 'log_probs'"):
        DiscreteSAC._action_distribution_from_outputs({"probs": torch.tensor([[0.5, 0.5]])})


def test_policy_mask_sanitizes_invalid_and_out_of_range_actions_without_mutation() -> None:
    invalid_mask = _invalid_action_mask()
    original_invalid_mask = invalid_mask.clone()
    sanitized = DiscreteSAC._sanitize_discrete_action_indices(
        torch.tensor([[1], [99], [-4]]), invalid_mask, num_actions=3
    )

    torch.testing.assert_close(sanitized, torch.tensor([[0], [0], [0]]))
    torch.testing.assert_close(invalid_mask, original_invalid_mask)


def test_all_invalid_policy_mask_retains_one_safe_action() -> None:
    invalid_mask = torch.ones(2, 3, dtype=torch.bool)
    original_invalid_mask = invalid_mask.clone()

    sanitized = DiscreteSAC._sanitize_discrete_action_indices(torch.tensor([[2], [0]]), invalid_mask, num_actions=3)

    torch.testing.assert_close(sanitized, torch.tensor([[0], [0]]))
    torch.testing.assert_close(invalid_mask, original_invalid_mask)


def test_out_of_range_actions_are_safe_without_policy_mask() -> None:
    sanitized = DiscreteSAC._sanitize_discrete_action_indices(torch.tensor([[-1], [3], [2]]), None, num_actions=3)
    torch.testing.assert_close(sanitized, torch.tensor([[0], [0], [2]]))


def test_update_accepts_equivalent_contracts_and_all_invalid_policy_masks() -> None:
    distribution_sample = _nested_replay_sample()
    logits_sample = _nested_replay_sample()
    distribution_mask = _invalid_action_mask()
    logits_mask = _invalid_action_mask()
    distribution_agent = _make_agent("distribution", distribution_sample, distribution_mask)
    logits_agent = _make_agent("logits", logits_sample, logits_mask)

    distribution_agent.update(timestep=0, timesteps=1)
    logits_agent.update(timestep=0, timesteps=1)

    for model_name in ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        distribution_parameters = dict(distribution_agent.models[model_name].named_parameters())
        logits_parameters = dict(logits_agent.models[model_name].named_parameters())
        assert distribution_parameters.keys() == logits_parameters.keys()
        for name in distribution_parameters:
            torch.testing.assert_close(distribution_parameters[name], logits_parameters[name])

    # The all-invalid row remains unchanged; the safe fallback exists only in the sanitizer's clone.
    torch.testing.assert_close(distribution_mask[1], torch.tensor([True, True, True]))
    torch.testing.assert_close(logits_mask[1], torch.tensor([True, True, True]))


@pytest.mark.parametrize("device", ["cpu"])
def test_cpu_act_preprocesses_explicit_states_and_random_actions(device: str) -> None:
    agent = _make_agent(
        "logits",
        _nested_replay_sample(),
        _invalid_action_mask(),
        device=device,
        state_preprocessor=_OffsetPreprocessor,
        random_timesteps=1,
    )
    observations = torch.zeros(3, 2, device=device)
    states = torch.ones(3, 2, device=device)

    random_actions, random_outputs = agent.act(observations, states, timestep=0, timesteps=2)
    actions, _ = agent.act(observations, states, timestep=1, timesteps=2)

    assert random_actions.shape == (3, 1)
    assert random_outputs == {}
    torch.testing.assert_close(actions, torch.full((3, 1), 2, device=device))
    preprocessor = agent._state_preprocessor
    assert isinstance(preprocessor, _OffsetPreprocessor)
    assert preprocessor.calls[-1] == (states, False)
    policy_inputs = agent.policy.calls[-1][1]
    torch.testing.assert_close(policy_inputs["states"], states + 2)
    assert policy_inputs["states"].device.type == device


def test_record_transition_uses_explicit_states_and_preserves_nested_payloads() -> None:
    sample = _nested_replay_sample()
    memory = _RecordingReplay(sample)
    agent = _make_agent("logits", sample, _invalid_action_mask(), memory=memory)
    observations = {"nested": [torch.zeros(3, 2)]}
    next_observations = {"nested": [torch.ones(3, 2)]}

    agent.record_transition(
        observations=observations,
        states=None,
        actions=torch.tensor([[0.0], [1.0], [2.0]]),
        rewards=torch.ones(3, 1),
        next_observations=next_observations,
        next_states=None,
        terminated=torch.zeros(3, 1, dtype=torch.bool),
        truncated=torch.zeros(3, 1, dtype=torch.bool),
        infos={"external": "ignored"},
        timestep=0,
        timesteps=1,
    )

    assert len(memory.transitions) == 1
    transition = memory.transitions[0]
    assert transition["observations"] is observations
    assert transition["states"] is observations
    assert transition["next_observations"] is next_observations
    assert transition["next_states"] is next_observations


@pytest.mark.parametrize("device", ["cpu"])
def test_cpu_update_changes_actor_critics_and_entropy_in_contract_order(device: str) -> None:
    torch.manual_seed(7)
    agent = _make_agent("logits", _nested_replay_sample(), _invalid_action_mask(), device=device, learn_entropy=True)
    before = {
        name: next(agent.models[name].parameters()).detach().clone()
        for name in ("policy", "critic_1", "critic_2")
    }
    entropy_before = agent._entropy_coefficient

    agent.update(timestep=0, timesteps=1)

    for name, parameter in before.items():
        assert not torch.equal(parameter, next(agent.models[name].parameters()))
    assert entropy_before != agent._entropy_coefficient.item()
    assert [role for role, _ in agent.policy.calls] == ["policy", "policy"]
    assert [role for role, _ in agent.critic_1.calls] == ["critic_1", "critic_1"]
    assert [role for role, _ in agent.target_critic_1.calls] == ["target_critic_1"]


def test_checkpoint_manifest_round_trips_models_optimizers_entropy_and_preprocessor(tmp_path: Any) -> None:
    agent = _make_agent(
        "logits",
        _nested_replay_sample(),
        _invalid_action_mask(),
        learn_entropy=True,
        state_preprocessor=_OffsetPreprocessor,
    )
    agent.update(timestep=0, timesteps=1)
    path = tmp_path / "discrete-sac.pt"
    agent.save(path)
    modules = torch.load(path)

    assert list(modules) == [
        "policy",
        "critic_1",
        "critic_2",
        "target_critic_1",
        "target_critic_2",
        "entropy_optimizer",
        "policy_optimizer",
        "critic1_optimizer",
        "critic2_optimizer",
        "state_preprocessor",
    ]
    saved_policy = modules["policy"]["logits"].clone()
    with torch.no_grad():
        agent.policy.logits.add_(5)
    agent.load(path)
    torch.testing.assert_close(agent.policy.logits, saved_policy)
    torch.testing.assert_close(agent._state_preprocessor.offset, torch.tensor(2.0))


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ({}, "must contain either"),
        ({"log_probs": torch.tensor([[0.0]])}, "both 'probs' and 'log_probs'"),
    ],
)
def test_invalid_policy_output_contracts_are_rejected(outputs: dict[str, torch.Tensor], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        DiscreteSAC._action_distribution_from_outputs(outputs)


def test_invalid_sanitizer_dimensions_and_action_count_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        DiscreteSAC._sanitize_discrete_action_indices(torch.tensor([[0]]), None, num_actions=0)
    with pytest.raises(ValueError, match="Invalid action mask shape"):
        DiscreteSAC._sanitize_discrete_action_indices(
            torch.tensor([[0], [1]]), torch.zeros(2, 2, dtype=torch.bool), num_actions=3
        )
