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
    def __init__(self, contract: str) -> None:
        super().__init__(
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            action_space=gymnasium.spaces.Discrete(3),
            device="cpu",
        )
        self.contract = contract
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.5]))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        logits = self.logits.unsqueeze(0).expand(_batch_size(inputs), -1)
        if self.contract == "distribution":
            outputs = {
                "probs": torch.softmax(logits, dim=-1),
                "log_probs": torch.log_softmax(logits, dim=-1),
            }
        else:
            outputs = {self.contract: logits}
        return logits.argmax(dim=-1, keepdim=True), outputs


class _SyntheticCritic(Model):
    def __init__(self) -> None:
        super().__init__(
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            action_space=gymnasium.spaces.Discrete(3),
            device="cpu",
        )
        self.q_values = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.7]))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.q_values.unsqueeze(0).expand(_batch_size(inputs), -1), {}


class _SyntheticReplay:
    def __init__(self, sample: tuple[Any, ...]) -> None:
        self.sample_value = sample

    def sample(self, names: list[str], *, batch_size: int) -> list[tuple[Any, ...]]:
        assert batch_size == self.sample_value[2].shape[0]
        return [self.sample_value]


def _ariadne_states(*, new_layout: bool = True) -> list[torch.Tensor]:
    batch_size, num_nodes = 3, 4
    node_inputs = torch.arange(batch_size * num_nodes * 2, dtype=torch.float32).reshape(batch_size, num_nodes, 2)
    current_edge = torch.tensor(
        [
            [[0], [1], [2]],
            [[0], [1], [2]],
            [[0], [1], [2]],
        ]
    )
    edge_padding_mask = torch.tensor(
        [
            [[False, True, False]],
            [[True, True, True]],
            [[False, False, False]],
        ]
    )
    filler = [torch.zeros(batch_size, 1) for _ in range(5)]
    if new_layout:
        return [
            node_inputs,
            node_inputs.clone(),
            *filler,
            current_edge,
            edge_padding_mask,
        ]
    return [node_inputs, *filler, current_edge, edge_padding_mask]


def _nested_replay_sample() -> tuple[Any, ...]:
    observations = {
        "features": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "history": (torch.ones(3, 1), [torch.zeros(3, 2)]),
    }
    states = _ariadne_states()
    return (
        observations,
        states,
        torch.tensor([[1.0], [99.0], [-4.0]]),
        torch.tensor([[1.0], [0.5], [-0.5]]),
        {
            "features": observations["features"] + 1,
            "history": (torch.full((3, 1), 2.0), [torch.full((3, 2), 3.0)]),
        },
        [tensor.clone() for tensor in states],
        torch.zeros(3, 1, dtype=torch.bool),
        torch.zeros(3, 1, dtype=torch.bool),
    )


def _make_agent(contract: str, sample: tuple[Any, ...]) -> DiscreteSAC:
    models = {
        "policy": _SyntheticPolicy(contract),
        "critic_1": _SyntheticCritic(),
        "critic_2": _SyntheticCritic(),
        "target_critic_1": _SyntheticCritic(),
        "target_critic_2": _SyntheticCritic(),
    }
    cfg = DISCRETE_SAC_CFG(
        batch_size=3,
        learn_entropy=False,
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
        memory=_SyntheticReplay(sample),
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=gymnasium.spaces.Discrete(3),
        device="cpu",
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


@pytest.mark.parametrize("new_layout", [False, True])
def test_ariadne_mask_sanitizes_invalid_and_out_of_range_actions(new_layout: bool) -> None:
    states = _ariadne_states(new_layout=new_layout)
    replay_padding_mask = states[8 if new_layout else 7]
    original_padding_mask = replay_padding_mask.clone()

    invalid_mask = DiscreteSAC._build_discrete_invalid_mask_from_states(states)
    sanitized = DiscreteSAC._sanitize_discrete_action_indices(
        torch.tensor([[1], [99], [-4]]), invalid_mask, num_actions=3
    )

    torch.testing.assert_close(
        invalid_mask,
        torch.tensor(
            [
                [False, True, False],
                [False, True, True],
                [False, False, False],
            ]
        ),
    )
    torch.testing.assert_close(sanitized, torch.tensor([[0], [0], [0]]))
    torch.testing.assert_close(replay_padding_mask, original_padding_mask)


def test_out_of_range_actions_are_safe_without_ariadne_mask() -> None:
    sanitized = DiscreteSAC._sanitize_discrete_action_indices(torch.tensor([[-1], [3], [2]]), None, num_actions=3)
    torch.testing.assert_close(sanitized, torch.tensor([[0], [0], [2]]))


def test_update_accepts_equivalent_contracts_and_all_invalid_replay_masks() -> None:
    distribution_sample = _nested_replay_sample()
    logits_sample = _nested_replay_sample()
    distribution_agent = _make_agent("distribution", distribution_sample)
    logits_agent = _make_agent("logits", logits_sample)

    distribution_agent.update(timestep=0, timesteps=1)
    logits_agent.update(timestep=0, timesteps=1)

    for model_name in ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        distribution_parameters = dict(distribution_agent.models[model_name].named_parameters())
        logits_parameters = dict(logits_agent.models[model_name].named_parameters())
        assert distribution_parameters.keys() == logits_parameters.keys()
        for name in distribution_parameters:
            torch.testing.assert_close(distribution_parameters[name], logits_parameters[name])

    # The all-invalid replay row retains its original mask; the safe action exists only in the derived clone.
    torch.testing.assert_close(distribution_sample[1][8][1], torch.tensor([[True, True, True]]))
    torch.testing.assert_close(logits_sample[1][8][1], torch.tensor([[True, True, True]]))
