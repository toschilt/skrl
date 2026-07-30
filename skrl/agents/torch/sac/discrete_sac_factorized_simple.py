"""Purpose: Implement factorized discrete SAC with position and orientation actions.

Usage: Construct ``FactorizedDiscreteSACSimple`` with a policy that emits
``logits_pos`` and ``logits_rot`` and critics that accept either all factorized
actions or selected position/orientation action tensors.
"""

from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .discrete_sac_cfg_factorized import FACTORIZED_DISCRETE_SAC_CFG


class FactorizedDiscreteSACSimple(Agent):
    """Minimal factorized discrete SAC.

    The policy outputs:
    - position logits: [B, N_pos]
    - orientation logits: [B, N_pos, N_rot]

    Replay stores factorized actions as [position_action, rotation_action].
    """

    CURRICULUM_REPLAY_TENSOR_NAMES = (
        "curriculum_reward_phase",
        "curriculum_sampled_difficulty",
        "curriculum_target_difficulty",
        "curriculum_mix_stage",
    )
    CURRICULUM_REWARD_COMPONENTS_TENSOR_NAME = "curriculum_reward_components"

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: FACTORIZED_DISCRETE_SAC_CFG | None = None,
        env=None,
        curriculum_cfg: type[Any] | None = None,
    ) -> None:
        self.cfg: FACTORIZED_DISCRETE_SAC_CFG
        self._latest_infos = None
        self._previous_done_mask = None
        self.curriculum_controller = None
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=FACTORIZED_DISCRETE_SAC_CFG() if cfg is None else cfg,
        )

        if curriculum_cfg is not None:
            self.curriculum_controller = curriculum_cfg.create(env=env)
            self.curriculum_controller.set_agent(self)

        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type,
                enabled=self.cfg.mixed_precision,
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        self._target_update_counter = 1
        self._gradient_update_counter = 0

        self._position_entropy_coefficient = self.cfg.initial_position_entropy_value
        self._orientation_entropy_coefficient = self.cfg.initial_orientation_entropy_value
        self._target_position_entropy = self.cfg.target_position_entropy
        self._target_orientation_entropy = self.cfg.target_orientation_entropy

        if self.cfg.learn_entropy:
            self.entropy_coefficients = nn.Module()
            self.entropy_coefficients.register_parameter(
                "log_position_entropy_coefficient",
                nn.Parameter(
                    torch.log(
                        torch.tensor([self._position_entropy_coefficient], device=self.device)
                    )
                ),
            )
            self.entropy_coefficients.register_parameter(
                "log_orientation_entropy_coefficient",
                nn.Parameter(
                    torch.log(
                        torch.tensor([self._orientation_entropy_coefficient], device=self.device)
                    )
                ),
            )
            self.log_position_entropy_coefficient = (
                self.entropy_coefficients.log_position_entropy_coefficient
            )
            self.log_orientation_entropy_coefficient = (
                self.entropy_coefficients.log_orientation_entropy_coefficient
            )

            entropy_lr = self.cfg.learning_rate[2]
            self.position_entropy_optimizer = torch.optim.Adam(
                [self.log_position_entropy_coefficient],
                lr=entropy_lr,
            )
            self.orientation_entropy_optimizer = torch.optim.Adam(
                [self.log_orientation_entropy_coefficient],
                lr=entropy_lr,
            )
            self.checkpoint_modules["position_entropy_optimizer"] = self.position_entropy_optimizer
            self.checkpoint_modules["orientation_entropy_optimizer"] = (
                self.orientation_entropy_optimizer
            )
            self.checkpoint_modules["entropy_coefficients"] = self.entropy_coefficients

        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.cfg.learning_rate[0],
            )
            self.critic1_optimizer = torch.optim.Adam(
                self.critic_1.parameters(),
                lr=self.cfg.learning_rate[1],
            )
            self.critic2_optimizer = torch.optim.Adam(
                self.critic_2.parameters(),
                lr=self.cfg.learning_rate[1],
            )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic1_optimizer"] = self.critic1_optimizer
            self.checkpoint_modules["critic2_optimizer"] = self.critic2_optimizer

            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic1_scheduler = self.cfg.learning_rate_scheduler[1]
            self.critic2_scheduler = self.cfg.learning_rate_scheduler[1]
            self.position_entropy_scheduler = (
                self.cfg.learning_rate_scheduler[2] if self.cfg.learn_entropy else None
            )
            self.orientation_entropy_scheduler = (
                self.cfg.learning_rate_scheduler[2] if self.cfg.learn_entropy else None
            )

            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer,
                    **self.cfg.learning_rate_scheduler_kwargs[0],
                )
            if self.critic1_scheduler is not None:
                self.critic1_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic1_optimizer,
                    **self.cfg.learning_rate_scheduler_kwargs[1],
                )
                self.critic2_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic2_optimizer,
                    **self.cfg.learning_rate_scheduler_kwargs[1],
                )
            if self.position_entropy_scheduler is not None:
                self.position_entropy_scheduler = self.cfg.learning_rate_scheduler[2](
                    self.position_entropy_optimizer,
                    **self.cfg.learning_rate_scheduler_kwargs[2],
                )
                self.orientation_entropy_scheduler = self.cfg.learning_rate_scheduler[2](
                    self.orientation_entropy_optimizer,
                    **self.cfg.learning_rate_scheduler_kwargs[2],
                )

        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor

        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(
                **self.cfg.state_preprocessor_kwargs
            )
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def _get_states(self, observations: torch.Tensor, states: torch.Tensor | None) -> torch.Tensor:
        return observations if states is None else states

    @staticmethod
    def _scalar_value(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _extract_factorized_policy_outputs(
        outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_pos = outputs.get("logits_pos", None)
        logits_rot = outputs.get("logits_rot", None)
        if logits_pos is None or logits_rot is None:
            raise RuntimeError(
                "Policy outputs must contain 'logits_pos' [B, N_pos] and "
                "'logits_rot' [B, N_pos, N_rot]"
            )
        return logits_pos, logits_rot

    def _to_device_recursive(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, list):
            return [self._to_device_recursive(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_device_recursive(v) for v in value)
        if isinstance(value, dict):
            return {k: self._to_device_recursive(v) for k, v in value.items()}
        return value

    def _filter_transition_batch(self, value, mask: torch.Tensor):
        """Select valid env rows from a tensor/list/dict transition payload."""
        if value is None:
            return None
        if torch.is_tensor(value):
            mask = mask.to(device=value.device, dtype=torch.bool)
            if value.ndim > 0 and value.shape[0] == mask.shape[0]:
                return value[mask]
            return value
        if isinstance(value, list):
            return [self._filter_transition_batch(v, mask) for v in value]
        if isinstance(value, tuple):
            return tuple(self._filter_transition_batch(v, mask) for v in value)
        if isinstance(value, dict):
            return {k: self._filter_transition_batch(v, mask) for k, v in value.items()}
        return value

    def _curriculum_replay_enabled(self) -> bool:
        return (
            bool(getattr(self.cfg, "curriculum_replay_enabled", False))
            and self.curriculum_controller is not None
        )

    def _create_curriculum_replay_tensors(self) -> None:
        if self.memory is None or not hasattr(self.memory, "create_tensor"):
            return
        for name in self.CURRICULUM_REPLAY_TENSOR_NAMES:
            self.memory.create_tensor(name=name, size=1, dtype=torch.float32)
        if self._curriculum_reward_recompute_enabled():
            self.memory.create_tensor(
                name=self.CURRICULUM_REWARD_COMPONENTS_TENSOR_NAME,
                size=len(self._curriculum_reward_names()),
                dtype=torch.float32,
            )

    def _curriculum_reward_names(self) -> tuple[str, ...]:
        names = getattr(self.cfg, "curriculum_replay_reward_names", ())
        return tuple(str(name) for name in names)

    def _curriculum_reward_phase_weights(self) -> tuple[tuple[float, ...], ...]:
        phase_weights = getattr(self.cfg, "curriculum_replay_reward_phase_weights", ())
        return tuple(tuple(float(value) for value in weights) for weights in phase_weights)

    def _curriculum_reward_recompute_enabled(self) -> bool:
        return (
            bool(getattr(self.cfg, "curriculum_replay_recompute_rewards", False))
            and len(self._curriculum_reward_names()) > 0
            and len(self._curriculum_reward_phase_weights()) > 0
        )

    def _active_reward_phase_index(self) -> int:
        controller = self.curriculum_controller
        if controller is None:
            return 0
        return int(getattr(controller, "phase_index", 0))

    def _num_reward_phases(self) -> int:
        if self.curriculum_controller is not None:
            phases_cfg = getattr(self.curriculum_controller.cfg, "phase_cfgs", ())
            if len(phases_cfg) > 0:
                return len(phases_cfg)
        phase_weights = self._curriculum_reward_phase_weights()
        return max(1, len(phase_weights))

    def _is_final_reward_phase(self) -> bool:
        return self._active_reward_phase_index() >= self._num_reward_phases() - 1

    def _weights_for_reward_phase(
        self,
        phase_index: int | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        phase_weights = self._curriculum_reward_phase_weights()
        if not phase_weights:
            return None
        if phase_index is None:
            phase_index = self._active_reward_phase_index()
        phase_index = min(max(int(phase_index), 0), len(phase_weights) - 1)
        weights = torch.tensor(phase_weights[phase_index], dtype=torch.float32, device=device)
        if weights.numel() != len(self._curriculum_reward_names()):
            return None
        return weights

    def _curriculum_reward_components(
        self,
        infos: Any,
        *,
        device: torch.device | str,
        batch_size: int,
    ) -> torch.Tensor | None:
        if not self._curriculum_reward_recompute_enabled():
            return None
        if not isinstance(infos, dict):
            return None
        sep_reward = infos.get("sep_reward", None)
        if not isinstance(sep_reward, dict):
            return None

        names = self._curriculum_reward_names()
        weights = self._weights_for_reward_phase(device=device)
        if weights is None:
            return None

        components = []
        for i, reward_name in enumerate(names):
            term = sep_reward.get(reward_name, None)
            if term is None:
                component = torch.zeros(batch_size, dtype=torch.float32, device=device)
            else:
                if not torch.is_tensor(term):
                    term = torch.as_tensor(term, dtype=torch.float32, device=device)
                component = term.to(device=device, dtype=torch.float32).view(batch_size)
                weight = weights[i]
                if abs(float(weight.item())) > 1e-8:
                    component = component / weight
                else:
                    component = torch.zeros_like(component)
            components.append(component)
        return torch.stack(components, dim=1)

    def _recompute_sampled_rewards(self, reward_components: torch.Tensor) -> torch.Tensor | None:
        if not self._curriculum_reward_recompute_enabled():
            return None
        if not torch.is_tensor(reward_components) or reward_components.numel() == 0:
            return None
        weights = self._weights_for_reward_phase(device=reward_components.device)
        if weights is None:
            return None
        if reward_components.shape[-1] != weights.numel():
            return None
        rewards = (reward_components.float() * weights.view(1, -1)).sum(dim=1, keepdim=True)
        self.track_data(
            "ReplayBatch/curriculum/recomputed_reward_mean",
            float(rewards.mean().item()),
        )
        names = self._curriculum_reward_names()
        object_reward_name = None
        if "ObjectInspectionProgressReward" in names:
            object_reward_name = "ObjectInspectionProgressReward"
        elif "ObjectCoverageReward" in names:
            object_reward_name = "ObjectCoverageReward"

        if object_reward_name is not None:
            object_index = names.index(object_reward_name)
            self.track_data(
                "ReplayBatch/curriculum/object_opportunity_component_mean",
                float(reward_components[:, object_index].float().mean().item()),
            )
        return rewards

    def _curriculum_replay_metadata(self, *, device: torch.device | str) -> dict[str, torch.Tensor]:
        if not self._curriculum_replay_enabled():
            return {}
        replay_metadata = getattr(self.curriculum_controller, "replay_metadata", None)
        if replay_metadata is None:
            return {}
        metadata = replay_metadata(device=device)
        return {
            name: value
            for name, value in metadata.items()
            if name in self.CURRICULUM_REPLAY_TENSOR_NAMES and torch.is_tensor(value)
        }

    def _uniform_training_batch(self, *, track_uniform: bool = False) -> list[Any]:
        if track_uniform:
            self.track_data("ReplayBatch/curriculum/current_exact_fraction", 0.0)
            self.track_data("ReplayBatch/curriculum/adjacent_fraction", 0.0)
            self.track_data("ReplayBatch/curriculum/uniform_fraction", 1.0)
            self.track_data("ReplayBatch/curriculum/final_phase_adjacent_fraction", 0.0)
        return self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

    def _curriculum_replay_values(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        if self.memory is None or not hasattr(self.memory, "get_tensor_by_name"):
            return None
        if not hasattr(self.memory, "valid_indices"):
            return None

        valid_indices = self.memory.valid_indices()
        if valid_indices.numel() == 0:
            return None

        values = {}
        for name in self.CURRICULUM_REPLAY_TENSOR_NAMES:
            tensor = self.memory.get_tensor_by_name(name)
            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                return None
            tensor = tensor.view(-1).to(device=valid_indices.device)
            if tensor.shape[0] != valid_indices.shape[0]:
                return None
            values[name] = tensor
        return valid_indices, values

    @staticmethod
    def _sample_index_pool(pool: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return pool.new_empty((0,), dtype=torch.long)
        choices = torch.randint(0, pool.numel(), (count,), device=pool.device)
        return pool[choices]

    def _curriculum_replay_quotas(self, batch_size: int) -> tuple[int, int, int]:
        if self._is_final_reward_phase():
            current_fraction = max(0.0, float(getattr(
                self.cfg,
                "curriculum_replay_final_phase_current_fraction",
                getattr(self.cfg, "curriculum_replay_current_fraction", 0.70),
            )))
            adjacent_fraction = max(0.0, float(getattr(
                self.cfg,
                "curriculum_replay_final_phase_adjacent_fraction",
                getattr(self.cfg, "curriculum_replay_adjacent_fraction", 0.20),
            )))
            uniform_fraction = max(0.0, float(getattr(
                self.cfg,
                "curriculum_replay_final_phase_uniform_fraction",
                getattr(self.cfg, "curriculum_replay_uniform_fraction", 0.10),
            )))
        else:
            current_fraction = max(0.0, float(getattr(self.cfg, "curriculum_replay_current_fraction", 0.70)))
            adjacent_fraction = max(0.0, float(getattr(self.cfg, "curriculum_replay_adjacent_fraction", 0.20)))
            uniform_fraction = max(0.0, float(getattr(self.cfg, "curriculum_replay_uniform_fraction", 0.10)))
        total = current_fraction + adjacent_fraction + uniform_fraction
        if total <= 0.0:
            return 0, 0, batch_size
        current_count = int(round(batch_size * current_fraction / total))
        adjacent_count = int(round(batch_size * adjacent_fraction / total))
        current_count = min(batch_size, current_count)
        adjacent_count = min(batch_size - current_count, adjacent_count)
        uniform_count = batch_size - current_count - adjacent_count
        return current_count, adjacent_count, uniform_count

    def _track_curriculum_replay_batch(
        self,
        metadata: dict[str, torch.Tensor],
        source_labels: torch.Tensor,
    ) -> None:
        source_labels = source_labels.float()
        batch_size = max(1, int(source_labels.numel()))
        self.track_data(
            "ReplayBatch/curriculum/current_exact_fraction",
            float((source_labels == 1).float().mean().item()),
        )
        self.track_data(
            "ReplayBatch/curriculum/adjacent_fraction",
            float((source_labels == 2).float().mean().item()),
        )
        self.track_data(
            "ReplayBatch/curriculum/uniform_fraction",
            float((source_labels == 0).float().mean().item()),
        )
        self.track_data(
            "ReplayBatch/curriculum/final_phase_adjacent_fraction",
            float((source_labels == 2).float().mean().item()) if self._is_final_reward_phase() else 0.0,
        )

        phase_values = metadata["curriculum_reward_phase"].view(-1).long()
        if self.curriculum_controller is not None:
            phases_cfg = getattr(
                self.curriculum_controller.cfg,
                "phase_cfgs",
                (),
            )
            num_phases = max(1, len(phases_cfg))
            map_controller = getattr(self.curriculum_controller, "map_difficulty_controller", None)
            difficulty_order = tuple(getattr(map_controller, "difficulty_order", ()))
        else:
            num_phases = max(1, int(phase_values.max().item()) + 1)
            difficulty_order = ("easy", "medium", "hard")
        if not difficulty_order:
            difficulty_order = ("easy", "medium", "hard")

        for phase in range(num_phases):
            self.track_data(
                f"ReplayBatch/reward_phase/phase_{phase}_fraction",
                float((phase_values == phase).float().sum().item() / batch_size),
            )

        sampled_difficulty = metadata["curriculum_sampled_difficulty"].view(-1).long()
        for difficulty_index, difficulty_name in enumerate(difficulty_order):
            self.track_data(
                f"ReplayBatch/difficulty/{difficulty_name}_fraction",
                float((sampled_difficulty == difficulty_index).float().sum().item() / batch_size),
            )

        if "curriculum_mix_stage" in metadata:
            mix_stage = metadata["curriculum_mix_stage"].view(-1).float()
            self.track_data("ReplayBatch/curriculum/mix_stage_mean", float(mix_stage.mean().item()))

    def _sample_training_batch(self) -> list[Any]:
        if not self._curriculum_replay_enabled():
            return self._uniform_training_batch()

        replay_values = self._curriculum_replay_values()
        if replay_values is None:
            return self._uniform_training_batch(track_uniform=True)

        valid_indices, replay_metadata = replay_values
        live_metadata = self._curriculum_replay_metadata(device=valid_indices.device)
        if not live_metadata:
            return self._uniform_training_batch(track_uniform=True)

        batch_size = int(self.cfg.batch_size)
        current_count, adjacent_count, uniform_count = self._curriculum_replay_quotas(batch_size)
        min_bucket_size = int(getattr(self.cfg, "curriculum_replay_min_bucket_size", batch_size))
        adjacent_distance = int(getattr(self.cfg, "curriculum_replay_adjacent_stage_distance", 1))

        phase_values = replay_metadata["curriculum_reward_phase"].long()
        sampled_difficulty_values = replay_metadata["curriculum_sampled_difficulty"].long()
        live_phases = live_metadata["curriculum_reward_phase"].view(-1).long()
        live_sampled_difficulties = live_metadata["curriculum_sampled_difficulty"].view(-1).long()

        replay_keys = sampled_difficulty_values * 1000 + phase_values
        live_keys = live_sampled_difficulties * 1000 + live_phases
        exact_mask = (replay_keys[:, None] == live_keys[None, :]).any(dim=1)
        adjacent_mask = (
            (sampled_difficulty_values[:, None] == live_sampled_difficulties[None, :])
            & ((phase_values[:, None] - live_phases[None, :]).abs() <= adjacent_distance)
        ).any(dim=1) & (~exact_mask)

        exact_pool = valid_indices[exact_mask]
        adjacent_pool = valid_indices[adjacent_mask]
        uniform_pool = valid_indices

        if exact_pool.numel() < min_bucket_size:
            uniform_count += current_count
            current_count = 0
        if adjacent_pool.numel() < min_bucket_size:
            uniform_count += adjacent_count
            adjacent_count = 0

        index_parts = []
        source_parts = []
        if current_count > 0:
            index_parts.append(self._sample_index_pool(exact_pool, current_count))
            source_parts.append(torch.ones(current_count, dtype=torch.long, device=valid_indices.device))
        if adjacent_count > 0:
            index_parts.append(self._sample_index_pool(adjacent_pool, adjacent_count))
            source_parts.append(torch.full((adjacent_count,), 2, dtype=torch.long, device=valid_indices.device))
        if uniform_count > 0:
            index_parts.append(self._sample_index_pool(uniform_pool, uniform_count))
            source_parts.append(torch.zeros(uniform_count, dtype=torch.long, device=valid_indices.device))

        if not index_parts:
            return self._uniform_training_batch(track_uniform=True)

        indexes = torch.cat(index_parts, dim=0)
        source_labels = torch.cat(source_parts, dim=0)
        permutation = torch.randperm(indexes.shape[0], device=indexes.device)
        indexes = indexes[permutation]
        source_labels = source_labels[permutation]

        sample_names = self._tensors_names + list(self.CURRICULUM_REPLAY_TENSOR_NAMES)
        include_reward_components = self._curriculum_reward_recompute_enabled()
        if include_reward_components:
            sample_names.append(self.CURRICULUM_REWARD_COMPONENTS_TENSOR_NAME)
        sampled = self.memory.sample_by_index(names=sample_names, indexes=indexes)[0]
        sampled_batch = sampled[: len(self._tensors_names)]
        sampled_metadata = {
            name: sampled[len(self._tensors_names) + i]
            for i, name in enumerate(self.CURRICULUM_REPLAY_TENSOR_NAMES)
        }
        if include_reward_components:
            reward_components = sampled[len(self._tensors_names) + len(self.CURRICULUM_REPLAY_TENSOR_NAMES)]
            recomputed_rewards = self._recompute_sampled_rewards(reward_components)
            if recomputed_rewards is not None:
                sampled_batch[3] = recomputed_rewards
        self._track_curriculum_replay_batch(sampled_metadata, source_labels)
        return sampled_batch

    def _resolve_target_entropies(self, n_pos: int, n_rot: int) -> None:
        if self._target_position_entropy is None:
            self._target_position_entropy = float(torch.log(torch.tensor(float(n_pos))))
        if self._target_orientation_entropy is None:
            self._target_orientation_entropy = float(torch.log(torch.tensor(float(n_rot))))

    @staticmethod
    def _build_position_invalid_mask_from_states(states: Any) -> torch.Tensor:
        current_index = states[4]
        current_neighbors = states[6]
        edge_padding_mask = states[7]

        if current_neighbors.dim() == 2:
            current_neighbors = current_neighbors.unsqueeze(-1).long()
        elif current_neighbors.dim() == 3:
            current_neighbors = current_neighbors.long()
        else:
            raise ValueError(
                f"current_neighbors must have dim 2 or 3, got {current_neighbors.dim()}"
            )

        if current_index.dim() == 3:
            current_node_idx = current_index.squeeze(-1).squeeze(-1).long()
        elif current_index.dim() == 2:
            current_node_idx = current_index.squeeze(-1).long()
        else:
            current_node_idx = current_index.long()

        if edge_padding_mask.dim() == 3:
            invalid_mask = edge_padding_mask.squeeze(1).bool()
        else:
            invalid_mask = edge_padding_mask.bool()

        self_mask = current_neighbors.squeeze(-1) == current_node_idx.view(-1, 1)
        return invalid_mask | self_mask

    @staticmethod
    def _masked_position_distribution(
        logits_pos: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_logits = logits_pos.masked_fill(invalid_mask, -1e9)
        pos_probs = torch.softmax(masked_logits, dim=-1)
        pos_probs = pos_probs * (~invalid_mask).float()

        probs_sum = pos_probs.sum(dim=-1, keepdim=True)
        normalized_probs = pos_probs / probs_sum.clamp_min(1e-12)

        has_valid = (~invalid_mask).any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(normalized_probs)
        fallback[:, 0] = 1.0
        normalized_probs = torch.where(has_valid, normalized_probs, fallback)

        log_probs = torch.log(normalized_probs.clamp_min(1e-8))
        return log_probs, normalized_probs

    @staticmethod
    def _sanitize_position_actions(
        position_actions: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_actions = (
            position_actions.view(-1)
            .long()
            .clamp(min=0, max=invalid_mask.shape[1] - 1)
        )

        valid_mask = ~invalid_mask
        first_valid = valid_mask.float().argmax(dim=1).long()
        has_valid = valid_mask.any(dim=1)
        replacement = torch.where(has_valid, first_valid, torch.zeros_like(first_valid))

        batch_index = torch.arange(position_actions.shape[0], device=position_actions.device)
        chosen_invalid = invalid_mask[batch_index, position_actions]
        return torch.where(chosen_invalid, replacement, position_actions)

    @staticmethod
    def _conservative_q_penalty(
        q_all: torch.Tensor,
        q_taken: torch.Tensor,
        invalid_position_mask: torch.Tensor,
        temperature: float,
        invalid_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CQL-style logsumexp penalty over all valid factorized actions."""
        temperature = max(float(temperature), 1e-6)
        invalid_position_action_mask = invalid_position_mask.unsqueeze(-1).expand_as(q_all)
        if invalid_action_mask is None:
            invalid_action_mask = invalid_position_action_mask
        else:
            invalid_action_mask = invalid_action_mask.bool() | invalid_position_action_mask
        masked_q_all = q_all.masked_fill(invalid_action_mask, -1e9)
        action_logits = masked_q_all.reshape(masked_q_all.shape[0], -1) / temperature
        conservative_value = temperature * torch.logsumexp(
            action_logits,
            dim=1,
            keepdim=True,
        )
        return (conservative_value - q_taken).mean()

    def _critic_loss_for_taken_actions(
        self,
        critic,
        inputs: dict[str, Any],
        position_actions: torch.Tensor,
        rotation_actions: torch.Tensor,
        target_values: torch.Tensor,
        current_invalid_mask: torch.Tensor,
        logits_rot_all: torch.Tensor,
        conservative_q_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        critic_values, _ = critic.act(
            {
                **inputs,
                "position_actions": position_actions,
                "rotation_actions": rotation_actions,
            },
            role="critic",
        )
        td_loss = F.mse_loss(critic_values, target_values)
        conservative_loss = critic_values.new_zeros(())

        if conservative_q_scale > 0.0:
            current_orientation_invalid_mask = logits_rot_all.detach() <= -1e8
            current_action_invalid_mask = (
                current_invalid_mask.unsqueeze(-1) | current_orientation_invalid_mask
            )
            conservative_q_temperature = float(
                getattr(self.cfg, "conservative_q_temperature", 1.0)
            )
            critic_all, _ = critic.act(
                {**inputs, "all_position_actions": True},
                role="critic",
            )
            conservative_loss = self._conservative_q_penalty(
                critic_all,
                critic_values,
                current_invalid_mask,
                conservative_q_temperature,
                current_action_invalid_mask,
            )

        loss = td_loss + conservative_q_scale * conservative_loss
        return loss, td_loss, conservative_loss, critic_values

    @staticmethod
    def _factorized_q_policy_diagnostics(
        q_all: torch.Tensor,
        position_probs: torch.Tensor,
        orientation_probs: torch.Tensor,
        invalid_position_mask: torch.Tensor,
        invalid_orientation_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Summarize how all-action Q compares with policy-weighted Q."""
        q_all = q_all.detach()
        position_probs = position_probs.detach()
        orientation_probs = orientation_probs.detach()
        invalid_action_mask = invalid_position_mask.detach().bool().unsqueeze(-1)
        invalid_action_mask = invalid_action_mask.expand_as(q_all)
        if invalid_orientation_mask is not None:
            invalid_action_mask = invalid_action_mask | invalid_orientation_mask.detach().bool()

        masked_q_all = q_all.masked_fill(invalid_action_mask, -1e9)
        flat_q = masked_q_all.reshape(masked_q_all.shape[0], -1)
        q_max_per_sample, flat_argmax = torch.max(flat_q, dim=1)

        orientation_dim = int(q_all.shape[-1])
        q_max_position = flat_argmax // orientation_dim
        q_max_orientation = flat_argmax % orientation_dim

        action_probs = position_probs.unsqueeze(-1) * orientation_probs
        action_probs = action_probs.masked_fill(invalid_action_mask, 0.0)
        action_probs = action_probs / action_probs.sum(
            dim=(1, 2),
            keepdim=True,
        ).clamp_min(1e-12)

        valid_q_for_expectation = q_all.masked_fill(invalid_action_mask, 0.0)
        q_policy_expectation = torch.sum(action_probs * valid_q_for_expectation, dim=(1, 2))
        q_max_action_probability = torch.gather(
            action_probs.reshape(action_probs.shape[0], -1),
            1,
            flat_argmax.view(-1, 1),
        ).squeeze(1)

        return {
            "q_all_max": q_max_per_sample.max(),
            "q_all_mean": masked_q_all[~invalid_action_mask].mean(),
            "q_policy_expectation_mean": q_policy_expectation.mean(),
            "q_policy_expectation_max": q_policy_expectation.max(),
            "q_max_minus_policy_expectation_mean": (
                q_max_per_sample - q_policy_expectation
            ).mean(),
            "q_max_minus_policy_expectation_max": (
                q_max_per_sample - q_policy_expectation
            ).max(),
            "q_max_action_position_mean": q_max_position.float().mean(),
            "q_max_action_orientation_mean": q_max_orientation.float().mean(),
            "q_max_action_probability_mean": q_max_action_probability.mean(),
            "q_max_action_probability_max": q_max_action_probability.max(),
        }

    @staticmethod
    def _privileged_observation_stats(observations) -> dict[str, torch.Tensor]:
        if not isinstance(observations, (list, tuple)) or len(observations) < 11:
            return {}

        node_padding_mask = observations[1].detach().bool().squeeze(1)
        valid_node_mask = ~node_padding_mask
        scene_priv = observations[8].detach().to(dtype=torch.float32)
        position_priv = observations[9].detach().to(dtype=torch.float32)
        orientation_priv = observations[10].detach().to(dtype=torch.float32)

        def valid_node_values(values: torch.Tensor) -> torch.Tensor:
            mask = valid_node_mask
            while mask.dim() < values.dim():
                mask = mask.unsqueeze(-1)
            return values[mask.expand_as(values)].reshape(-1)

        stats = {}
        scene_names = (
            "remaining_budget_fraction",
            "discovered_free_fraction",
            "camera_revealed_free_coverage",
            "object_detected",
            "object_coverage_fraction",
            "object_remaining_utility_norm",
        )
        for idx, name in enumerate(scene_names):
            if idx < scene_priv.shape[-1]:
                stats[f"scene/{name}_mean"] = scene_priv[:, idx].mean()

        if position_priv.shape[-1] >= 4:
            object_distance = valid_node_values(position_priv[..., 0])
            object_visibility = valid_node_values(position_priv[..., 2])
            object_inspection = valid_node_values(position_priv[..., 3])
            if object_distance.numel() > 0:
                stats["object_distance_mean"] = object_distance.mean()
            if object_visibility.numel() > 0:
                stats["object_visibility_position_mean"] = object_visibility.mean()
                stats["object_visibility_position_max"] = object_visibility.max()
            if object_inspection.numel() > 0:
                stats["object_inspection_gain_position_mean"] = object_inspection.mean()
                stats["object_inspection_gain_position_max"] = object_inspection.max()

        orientation_feature_dim = 4
        if orientation_priv.shape[-1] % orientation_feature_dim == 0:
            orientation_dim = orientation_priv.shape[-1] // orientation_feature_dim
            orientation_priv = orientation_priv.reshape(
                orientation_priv.shape[0],
                orientation_priv.shape[1],
                orientation_dim,
                orientation_feature_dim,
            )
            orientation_valid_mask = valid_node_mask.unsqueeze(-1).expand(-1, -1, orientation_dim)
            object_visibility = orientation_priv[..., 0][orientation_valid_mask]
            object_inspection = orientation_priv[..., 2][orientation_valid_mask]
            if object_visibility.numel() > 0:
                stats["object_visibility_mean"] = object_visibility.mean()
                stats["object_visibility_max"] = object_visibility.max()
            if object_inspection.numel() > 0:
                stats["object_inspection_gain_mean"] = object_inspection.mean()
                stats["object_inspection_gain_max"] = object_inspection.max()

        return stats

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self.memory is not None and hasattr(self.memory, "create_tensor"):
            self.memory.create_tensor(
                name="observations",
                size=self.observation_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="next_observations",
                size=self.observation_space,
                dtype=torch.float32,
            )

            state_space = self.state_space if self.state_space is not None else self.observation_space
            self.memory.create_tensor(name="states", size=state_space, dtype=torch.float32)
            self.memory.create_tensor(
                name="next_states",
                size=state_space,
                dtype=torch.float32,
            )

            self.memory.create_tensor(name="actions", size=2, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            if self._curriculum_replay_enabled():
                self._create_curriculum_replay_tensors()

        self._tensors_names = [
            "observations",
            "states",
            "actions",
            "rewards",
            "next_observations",
            "next_states",
            "terminated",
            "truncated",
        ]

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del timesteps
        states = self._get_states(observations, states)
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }

        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        self._latest_infos = infos
        done_mask = (terminated.detach().bool() | truncated.detach().bool()).view(-1)
        previous_done_mask = self._previous_done_mask
        if (
            previous_done_mask is None
            or previous_done_mask.numel() != done_mask.numel()
            or previous_done_mask.device != done_mask.device
        ):
            previous_done_mask = torch.zeros_like(done_mask)
        else:
            previous_done_mask = previous_done_mask.to(device=done_mask.device, dtype=torch.bool)
        store_mask = torch.ones_like(done_mask, dtype=torch.bool)
        if bool(getattr(self.cfg, "skip_previous_done_transitions", False)):
            store_mask = ~previous_done_mask

        sep_reward = infos.get("sep_reward", None) if isinstance(infos, dict) else None
        if isinstance(sep_reward, dict):
            for reward_name, reward_values in sep_reward.items():
                if torch.is_tensor(reward_values):
                    self.track_data(
                        f"Reward/env_components/{reward_name}_mean",
                        float(reward_values.float().mean().item()),
                    )
            terminal_reward_name = str(
                getattr(self.cfg, "terminal_reward_diagnostic_name", "")
            )
            if terminal_reward_name:
                terminal_reward = sep_reward.get(terminal_reward_name, None)
                if torch.is_tensor(terminal_reward):
                    terminal_reward = terminal_reward.detach().float().view(-1)
                    terminal_threshold = float(
                        getattr(self.cfg, "terminal_reward_diagnostic_threshold", 0.0)
                    )
                    terminal_reward_positive = terminal_reward > terminal_threshold
                    terminated_flat = terminated.detach().bool().view(-1)
                    truncated_flat = truncated.detach().bool().view(-1)
                    done_flat = terminated_flat | truncated_flat
                    positive_count = terminal_reward_positive.float().sum().clamp_min(1.0)
                    positive_nonterminal = terminal_reward_positive & (~done_flat)
                    self.track_data(
                        "TerminalSignal/env_terminal_reward_positive_fraction",
                        float(terminal_reward_positive.float().mean().item()),
                    )
                    self.track_data(
                        "TerminalSignal/env_terminal_reward_terminated_fraction",
                        float((terminal_reward_positive & terminated_flat).float().sum().item() / positive_count.item()),
                    )
                    self.track_data(
                        "TerminalSignal/env_terminal_reward_truncated_fraction",
                        float((terminal_reward_positive & truncated_flat).float().sum().item() / positive_count.item()),
                    )
                    self.track_data(
                        "TerminalSignal/env_terminal_reward_nonterminal_fraction",
                        float(positive_nonterminal.float().sum().item() / positive_count.item()),
                    )
                    self.track_data(
                        "TerminalSignal/env_terminal_reward_max",
                        float(terminal_reward.max().item()),
                    )

        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.memory is None:
            self._previous_done_mask = done_mask.detach().clone()
            return

        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

        states = self._get_states(observations, states)
        next_states = self._get_states(next_observations, next_states)

        if bool(getattr(self.cfg, "skip_previous_done_transitions", False)):
            skipped_fraction = 1.0 - float(store_mask.float().mean().item())
            self.track_data("Replay/stale_done_transition_skipped_fraction", skipped_fraction)
            if not bool(store_mask.any().item()):
                self._previous_done_mask = done_mask.detach().clone()
                return

            observations = self._filter_transition_batch(observations, store_mask)
            states = self._filter_transition_batch(states, store_mask)
            actions = self._filter_transition_batch(actions, store_mask)
            rewards = self._filter_transition_batch(rewards, store_mask)
            next_observations = self._filter_transition_batch(next_observations, store_mask)
            next_states = self._filter_transition_batch(next_states, store_mask)
            terminated = self._filter_transition_batch(terminated, store_mask)
            truncated = self._filter_transition_batch(truncated, store_mask)

        transition_tensors = {
            "observations": observations,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "next_states": next_states,
            "terminated": terminated,
            "truncated": truncated,
        }
        curriculum_metadata = self._curriculum_replay_metadata(device=rewards.device)
        if bool(getattr(self.cfg, "skip_previous_done_transitions", False)):
            curriculum_metadata = {
                key: self._filter_transition_batch(value, store_mask)
                for key, value in curriculum_metadata.items()
            }
        transition_tensors.update(curriculum_metadata)
        reward_components = self._curriculum_reward_components(
            infos,
            device=rewards.device,
            batch_size=int(done_mask.shape[0]),
        )
        if reward_components is not None:
            if bool(getattr(self.cfg, "skip_previous_done_transitions", False)):
                reward_components = self._filter_transition_batch(
                    reward_components,
                    store_mask,
                )
            transition_tensors[self.CURRICULUM_REWARD_COMPONENTS_TENSOR_NAME] = reward_components

        self.memory.add_samples(
            **transition_tensors,
        )
        self._previous_done_mask = done_mask.detach().clone()

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        del timestep, timesteps

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.curriculum_controller is not None:
            self.curriculum_controller.on_step(
                timestep=timestep,
                timesteps=timesteps,
                infos=self._latest_infos,
            )

        if timestep >= self.cfg.learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def get_curriculum_state(self) -> dict | None:
        if self.curriculum_controller is None:
            return None
        return self.curriculum_controller.get_state()

    def set_curriculum_state(self, state: dict | None):
        if self.curriculum_controller is None or state is None:
            return
        self.curriculum_controller.set_state(state)

    def set_curriculum_entropy_target(self, target_entropy: float | tuple[float, float]):
        if not self.cfg.learn_entropy:
            return
        if isinstance(target_entropy, tuple):
            position_entropy, orientation_entropy = target_entropy
        else:
            position_entropy = orientation_entropy = float(target_entropy)
        self._target_position_entropy = float(position_entropy)
        self._target_orientation_entropy = float(orientation_entropy)
        self.cfg.target_position_entropy = float(position_entropy)
        self.cfg.target_orientation_entropy = float(orientation_entropy)
        self.track_data("Curriculum / SAC target position entropy", float(position_entropy))
        self.track_data("Curriculum / SAC target orientation entropy", float(orientation_entropy))

    def update(self, *, timestep: int, timesteps: int) -> None:
        del timesteps
        current_timestep = int(timestep)
        actor_learning_starts = max(
            0,
            int(getattr(self.cfg, "actor_learning_starts", 0)),
        )

        for _ in range(self.cfg.gradient_steps):
            self._gradient_update_counter += 1
            actor_update_delay = max(1, int(getattr(self.cfg, "actor_update_delay", 1)))
            actor_learning_enabled = current_timestep >= actor_learning_starts
            update_actor = (
                actor_learning_enabled
                and (self._gradient_update_counter % actor_update_delay) == 0
            )

            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self._sample_training_batch()

            sampled_observations = self._to_device_recursive(sampled_observations)
            sampled_states = self._to_device_recursive(sampled_states)
            sampled_actions = self._to_device_recursive(sampled_actions)
            sampled_rewards = self._to_device_recursive(sampled_rewards)
            sampled_next_observations = self._to_device_recursive(sampled_next_observations)
            sampled_next_states = self._to_device_recursive(sampled_next_states)
            sampled_terminated = self._to_device_recursive(sampled_terminated)
            sampled_truncated = self._to_device_recursive(sampled_truncated)
            privileged_observation_stats = self._privileged_observation_stats(
                sampled_observations
            )

            current_invalid_mask = self._build_position_invalid_mask_from_states(sampled_states)
            next_invalid_mask = self._build_position_invalid_mask_from_states(sampled_next_states)

            inputs = {
                "observations": self._observation_preprocessor(
                    sampled_observations,
                    train=True,
                ),
                "states": self._state_preprocessor(sampled_states, train=True),
            }
            next_inputs = {
                "observations": self._observation_preprocessor(
                    sampled_next_observations,
                    train=True,
                ),
                "states": self._state_preprocessor(sampled_next_states, train=True),
            }

            alpha_position = self._position_entropy_coefficient
            alpha_orientation = self._orientation_entropy_coefficient
            policy_loss = None
            position_policy_entropy = None
            orientation_policy_entropy = None
            position_entropy_loss = None
            orientation_entropy_loss = None
            current_q_policy_diagnostics = {}
            target_q_policy_diagnostics = {}
            current_q_taken_values = None

            if update_actor:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    _, outputs = self.policy.act(inputs, role="policy")
                    logits_pos, logits_rot_all = self._extract_factorized_policy_outputs(outputs)

                    pos_log_probs, pos_probs = self._masked_position_distribution(
                        logits_pos,
                        current_invalid_mask,
                    )
                    rot_log_probs_all = F.log_softmax(logits_rot_all, dim=-1)
                    rot_probs_all = rot_log_probs_all.exp()

                    n_pos = logits_pos.shape[1]
                    n_rot = logits_rot_all.shape[2]
                    self._resolve_target_entropies(n_pos, n_rot)

                    position_policy_entropy = -(pos_probs * pos_log_probs).sum(dim=-1)
                    orientation_policy_entropy = -(
                        pos_probs * torch.sum(rot_probs_all * rot_log_probs_all, dim=-1)
                    ).sum(dim=-1)

                    with torch.no_grad():
                        q1_all, _ = self.critic_1.act(
                            {**inputs, "all_position_actions": True},
                            role="critic_1",
                        )
                        q2_all, _ = self.critic_2.act(
                            {**inputs, "all_position_actions": True},
                            role="critic_2",
                        )
                        q_all = torch.min(q1_all, q2_all)
                        current_orientation_invalid_mask = logits_rot_all.detach() <= -1e8
                        current_q_policy_diagnostics = self._factorized_q_policy_diagnostics(
                            q_all,
                            pos_probs,
                            rot_probs_all,
                            current_invalid_mask,
                            current_orientation_invalid_mask,
                        )

                    policy_terms = (
                        q_all
                        - alpha_position * pos_log_probs.unsqueeze(-1)
                        - alpha_orientation * rot_log_probs_all
                    )
                    inner_expectation = torch.sum(rot_probs_all * policy_terms, dim=-1)
                    policy_objective = torch.sum(pos_probs * inner_expectation, dim=-1)
                    policy_loss = -policy_objective.mean()

                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()

                if self.cfg.policy_grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.policy_grad_norm_clip)

                self.scaler.step(self.policy_optimizer)
            else:
                with torch.no_grad():
                    with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                        _, outputs = self.policy.act(inputs, role="policy")
                        logits_pos, logits_rot_all = self._extract_factorized_policy_outputs(outputs)

                        pos_log_probs, pos_probs = self._masked_position_distribution(
                            logits_pos,
                            current_invalid_mask,
                        )
                        rot_log_probs_all = F.log_softmax(logits_rot_all, dim=-1)
                        rot_probs_all = rot_log_probs_all.exp()

                        n_pos = logits_pos.shape[1]
                        n_rot = logits_rot_all.shape[2]
                        self._resolve_target_entropies(n_pos, n_rot)

                        position_policy_entropy = -(pos_probs * pos_log_probs).sum(dim=-1)
                        orientation_policy_entropy = -(
                            pos_probs * torch.sum(rot_probs_all * rot_log_probs_all, dim=-1)
                        ).sum(dim=-1)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                with torch.no_grad():
                    _, next_outputs = self.policy.act(next_inputs, role="policy")
                    next_logits_pos, next_logits_rot_all = self._extract_factorized_policy_outputs(
                        next_outputs
                    )
                    next_pos_log_probs, next_pos_probs = self._masked_position_distribution(
                        next_logits_pos,
                        next_invalid_mask,
                    )
                    next_rot_log_probs_all = F.log_softmax(next_logits_rot_all, dim=-1)
                    next_rot_probs_all = next_rot_log_probs_all.exp()

                    target_q1_all, _ = self.target_critic_1.act(
                        {**next_inputs, "all_position_actions": True},
                        role="target_critic_1",
                    )
                    target_q2_all, _ = self.target_critic_2.act(
                        {**next_inputs, "all_position_actions": True},
                        role="target_critic_2",
                    )
                    target_q_all = torch.min(target_q1_all, target_q2_all)
                    next_orientation_invalid_mask = next_logits_rot_all.detach() <= -1e8
                    target_q_policy_diagnostics = self._factorized_q_policy_diagnostics(
                        target_q_all,
                        next_pos_probs,
                        next_rot_probs_all,
                        next_invalid_mask,
                        next_orientation_invalid_mask,
                    )

                    target_terms = (
                        target_q_all
                        - alpha_position * next_pos_log_probs.unsqueeze(-1)
                        - alpha_orientation * next_rot_log_probs_all
                    )
                    target_inner_expectation = torch.sum(
                        next_rot_probs_all * target_terms,
                        dim=-1,
                    )
                    target_q_values = torch.sum(
                        next_pos_probs * target_inner_expectation,
                        dim=-1,
                    ).unsqueeze(1)

                    target_done_mask = sampled_terminated | sampled_truncated
                    target_nonterminal_mask = target_done_mask.logical_not()
                    target_bootstrap_values = (
                        self.cfg.discount_factor
                        * target_nonterminal_mask
                        * target_q_values
                    )
                    target_values = (
                        sampled_rewards
                        + target_bootstrap_values
                    ).detach()

                if sampled_actions.dim() == 1:
                    raise RuntimeError(
                        "Expected factorized actions [B, 2], got flat actions [B]."
                    )

                position_actions = sampled_actions[:, 0].long()
                rotation_actions = sampled_actions[:, 1].long()
                raw_position_actions = position_actions.view(-1)
                out_of_range = (
                    (raw_position_actions < 0)
                    | (raw_position_actions >= current_invalid_mask.shape[1])
                )
                clamped_position_actions = raw_position_actions.clamp(
                    min=0,
                    max=current_invalid_mask.shape[1] - 1,
                )
                batch_index = torch.arange(
                    clamped_position_actions.shape[0],
                    device=clamped_position_actions.device,
                )
                sampled_invalid_position_actions = (
                    out_of_range
                    | current_invalid_mask[batch_index, clamped_position_actions]
                )
                invalid_replay_action_fraction = (
                    sampled_invalid_position_actions.float().mean()
                )
                self.track_data(
                    "Replay/invalid_sampled_position_action_fraction",
                    invalid_replay_action_fraction.item(),
                )
                fail_threshold = getattr(
                    self.cfg,
                    "invalid_replay_action_fail_threshold",
                    None,
                )
                if (
                    fail_threshold is not None
                    and invalid_replay_action_fraction.item() > float(fail_threshold)
                ):
                    raise RuntimeError(
                        "Replay sampled invalid position actions: "
                        f"fraction={invalid_replay_action_fraction.item():.6f}, "
                        f"threshold={float(fail_threshold):.6f}"
                    )
                position_actions = self._sanitize_position_actions(
                    position_actions,
                    current_invalid_mask,
                )

                conservative_q_scale = float(
                    getattr(self.cfg, "conservative_q_regularization_scale", 0.0)
                )
                sequential_critic_update = bool(
                    getattr(self.cfg, "sequential_critic_update", False)
                )

                if sequential_critic_update:
                    (
                        critic_1_loss,
                        critic_1_td_loss,
                        critic_1_conservative_loss,
                        critic_1_values,
                    ) = self._critic_loss_for_taken_actions(
                        self.critic_1,
                        inputs,
                        position_actions,
                        rotation_actions,
                        target_values,
                        current_invalid_mask,
                        logits_rot_all,
                        conservative_q_scale,
                    )

                    self.critic1_optimizer.zero_grad()
                    self.scaler.scale(critic_1_loss).backward()

                    if config.torch.is_distributed:
                        self.critic_1.reduce_parameters()

                    if self.cfg.q_network_grad_norm_clip > 0:
                        self.scaler.unscale_(self.critic1_optimizer)
                        nn.utils.clip_grad_norm_(
                            self.critic_1.parameters(),
                            self.cfg.q_network_grad_norm_clip,
                        )

                    self.scaler.step(self.critic1_optimizer)
                    critic_1_values_detached = critic_1_values.detach()
                    critic_1_loss = critic_1_loss.detach()
                    critic_1_td_loss = critic_1_td_loss.detach()
                    critic_1_conservative_loss = critic_1_conservative_loss.detach()
                    del critic_1_values

                    (
                        critic_2_loss,
                        critic_2_td_loss,
                        critic_2_conservative_loss,
                        critic_2_values,
                    ) = self._critic_loss_for_taken_actions(
                        self.critic_2,
                        inputs,
                        position_actions,
                        rotation_actions,
                        target_values,
                        current_invalid_mask,
                        logits_rot_all,
                        conservative_q_scale,
                    )

                    self.critic2_optimizer.zero_grad()
                    self.scaler.scale(critic_2_loss).backward()

                    if config.torch.is_distributed:
                        self.critic_2.reduce_parameters()

                    if self.cfg.q_network_grad_norm_clip > 0:
                        self.scaler.unscale_(self.critic2_optimizer)
                        nn.utils.clip_grad_norm_(
                            self.critic_2.parameters(),
                            self.cfg.q_network_grad_norm_clip,
                        )

                    self.scaler.step(self.critic2_optimizer)
                    critic_2_values_detached = critic_2_values.detach()
                    critic_2_loss = critic_2_loss.detach()
                    critic_2_td_loss = critic_2_td_loss.detach()
                    critic_2_conservative_loss = critic_2_conservative_loss.detach()
                    del critic_2_values

                    current_q_taken_values = torch.min(
                        critic_1_values_detached,
                        critic_2_values_detached,
                    )
                    critic_1_values = critic_1_values_detached
                    critic_2_values = critic_2_values_detached
                else:
                    (
                        critic_1_loss,
                        critic_1_td_loss,
                        critic_1_conservative_loss,
                        critic_1_values,
                    ) = self._critic_loss_for_taken_actions(
                        self.critic_1,
                        inputs,
                        position_actions,
                        rotation_actions,
                        target_values,
                        current_invalid_mask,
                        logits_rot_all,
                        conservative_q_scale,
                    )
                    (
                        critic_2_loss,
                        critic_2_td_loss,
                        critic_2_conservative_loss,
                        critic_2_values,
                    ) = self._critic_loss_for_taken_actions(
                        self.critic_2,
                        inputs,
                        position_actions,
                        rotation_actions,
                        target_values,
                        current_invalid_mask,
                        logits_rot_all,
                        conservative_q_scale,
                    )

                    current_q_taken_values = torch.min(
                        critic_1_values.detach(),
                        critic_2_values.detach(),
                    )

            if not bool(getattr(self.cfg, "sequential_critic_update", False)):
                self.critic1_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()
                self.scaler.scale(critic_1_loss).backward()
                self.scaler.scale(critic_2_loss).backward()

                if config.torch.is_distributed:
                    self.critic_1.reduce_parameters()
                    self.critic_2.reduce_parameters()

                if self.cfg.q_network_grad_norm_clip > 0:
                    self.scaler.unscale_(self.critic1_optimizer)
                    self.scaler.unscale_(self.critic2_optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                        self.cfg.q_network_grad_norm_clip,
                    )

                self.scaler.step(self.critic1_optimizer)
                self.scaler.step(self.critic2_optimizer)
            else:
                self.track_data(
                    "Update/sequential_critic_update",
                    1.0,
                )

            if update_actor and self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    position_log_prob = -position_policy_entropy
                    orientation_log_prob = -orientation_policy_entropy

                    position_entropy_loss = -(
                        self.log_position_entropy_coefficient
                        * (position_log_prob.detach() + self._target_position_entropy)
                    ).mean()
                    orientation_entropy_loss = -(
                        self.log_orientation_entropy_coefficient
                        * (orientation_log_prob.detach() + self._target_orientation_entropy)
                    ).mean()

                    entropy_loss = position_entropy_loss + orientation_entropy_loss

                self.position_entropy_optimizer.zero_grad()
                self.orientation_entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.position_entropy_optimizer)
                self.scaler.step(self.orientation_entropy_optimizer)

                self._position_entropy_coefficient = torch.exp(
                    self.log_position_entropy_coefficient.detach()
                )
                self._orientation_entropy_coefficient = torch.exp(
                    self.log_orientation_entropy_coefficient.detach()
                )

            self.scaler.update()

            self._target_update_counter += 1
            if self._target_update_counter > self.cfg.steps_to_target_net_update:
                self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)
                self._target_update_counter = 1

            if self.policy_scheduler:
                if update_actor:
                    self.policy_scheduler.step()
            if self.critic1_scheduler:
                self.critic1_scheduler.step()
            if self.critic2_scheduler:
                self.critic2_scheduler.step()
            if update_actor:
                if self.position_entropy_scheduler:
                    self.position_entropy_scheduler.step()
                if self.orientation_entropy_scheduler:
                    self.orientation_entropy_scheduler.step()

            if self.write_interval > 0:
                self.track_data("Update/actor_update", 1.0 if update_actor else 0.0)
                self.track_data(
                    "Update/actor_learning_enabled",
                    1.0 if actor_learning_enabled else 0.0,
                )
                self.track_data(
                    "Update/actor_learning_starts",
                    float(actor_learning_starts),
                )
                self.track_data("Update/actor_update_delay", float(actor_update_delay))
                if policy_loss is not None:
                    self.track_data("Loss/policy_total", policy_loss.item())
                    self.track_data("Loss/Policy/Base", policy_loss.item())
                self.track_data("Loss/critic_1", critic_1_loss.item())
                self.track_data("Loss/critic_2", critic_2_loss.item())
                self.track_data("Loss/critic_1_td", critic_1_td_loss.item())
                self.track_data("Loss/critic_2_td", critic_2_td_loss.item())
                self.track_data(
                    "Loss/critic_1_conservative_q",
                    critic_1_conservative_loss.item(),
                )
                self.track_data(
                    "Loss/critic_2_conservative_q",
                    critic_2_conservative_loss.item(),
                )

                self.track_data("Critic/q1_value(max)", torch.max(critic_1_values).item())
                self.track_data("Critic/q1_value(min)", torch.min(critic_1_values).item())
                self.track_data("Critic/q1_value_mean", torch.mean(critic_1_values).item())

                self.track_data("Critic/q2_value(max)", torch.max(critic_2_values).item())
                self.track_data("Critic/q2_value(min)", torch.min(critic_2_values).item())
                self.track_data("Critic/q2_value_mean", torch.mean(critic_2_values).item())

                self.track_data("Critic/target_q(max)", torch.max(target_values).item())
                self.track_data("Critic/target_q(min)", torch.min(target_values).item())
                self.track_data("Critic/target_q_mean", torch.mean(target_values).item())
                self.track_data("TargetDiagnostics/reward_max", sampled_rewards.max().item())
                self.track_data("TargetDiagnostics/reward_min", sampled_rewards.min().item())
                self.track_data("TargetDiagnostics/reward_mean", sampled_rewards.mean().item())
                self.track_data(
                    "TargetDiagnostics/bootstrap_max",
                    target_bootstrap_values.max().item(),
                )
                self.track_data(
                    "TargetDiagnostics/bootstrap_mean",
                    target_bootstrap_values.mean().item(),
                )
                self.track_data(
                    "TargetDiagnostics/terminated_fraction",
                    sampled_terminated.float().mean().item(),
                )
                self.track_data(
                    "TargetDiagnostics/truncated_fraction",
                    sampled_truncated.float().mean().item(),
                )
                self.track_data(
                    "TargetDiagnostics/nonterminal_fraction",
                    target_nonterminal_mask.float().mean().item(),
                )
                terminal_threshold = float(
                    getattr(self.cfg, "terminal_reward_diagnostic_threshold", 0.0)
                )
                if terminal_threshold > 0.0:
                    high_reward_mask = sampled_rewards > terminal_threshold
                    high_reward_fraction = high_reward_mask.float().mean()
                    self.track_data(
                        "TargetDiagnostics/high_reward_fraction",
                        high_reward_fraction.item(),
                    )
                    if high_reward_mask.any():
                        high_reward_count = high_reward_mask.float().sum().clamp_min(1.0)
                        high_reward_nonterminal = (
                            high_reward_mask & target_nonterminal_mask
                        )
                        self.track_data(
                            "TargetDiagnostics/high_reward_nonterminal_fraction",
                            (
                                high_reward_nonterminal.float().sum()
                                / high_reward_count
                            ).item(),
                        )
                        self.track_data(
                            "TargetDiagnostics/high_reward_bootstrap_max",
                            target_bootstrap_values[high_reward_mask].max().item(),
                        )
                        self.track_data(
                            "TargetDiagnostics/high_reward_reward_max",
                            sampled_rewards[high_reward_mask].max().item(),
                        )
                    else:
                        self.track_data(
                            "TargetDiagnostics/high_reward_nonterminal_fraction",
                            0.0,
                        )
                        self.track_data(
                            "TargetDiagnostics/high_reward_bootstrap_max",
                            0.0,
                        )
                        self.track_data(
                            "TargetDiagnostics/high_reward_reward_max",
                            0.0,
                        )

                if current_q_taken_values is not None:
                    self.track_data(
                        "CriticDiagnostics/current_q_taken_max",
                        current_q_taken_values.max().item(),
                    )
                    self.track_data(
                        "CriticDiagnostics/current_q_taken_min",
                        current_q_taken_values.min().item(),
                    )
                    self.track_data(
                        "CriticDiagnostics/current_q_taken_mean",
                        current_q_taken_values.mean().item(),
                    )

                for key, value in current_q_policy_diagnostics.items():
                    self.track_data(
                        f"CriticDiagnostics/current_{key}",
                        value.item(),
                    )

                for key, value in target_q_policy_diagnostics.items():
                    self.track_data(
                        f"CriticDiagnostics/target_{key}",
                        value.item(),
                    )

                for key, value in privileged_observation_stats.items():
                    self.track_data(
                        f"CriticPrivileged/{key}",
                        value.item(),
                    )

                self.track_data(
                    "Entropy/position_policy_entropy_mean",
                    position_policy_entropy.mean().item(),
                )
                self.track_data(
                    "Entropy/orientation_policy_entropy_mean",
                    orientation_policy_entropy.mean().item(),
                )
                self.track_data(
                    "Entropy/position_policy_entropy_fraction_of_max",
                    (position_policy_entropy.mean() / max(self._target_position_entropy, 1e-8)).item()
                    if torch.is_tensor(position_policy_entropy.mean() / max(self._target_position_entropy, 1e-8))
                    else float(position_policy_entropy.mean().item() / max(self._target_position_entropy, 1e-8)),
                )
                self.track_data(
                    "Entropy/orientation_policy_entropy_fraction_of_max",
                    (orientation_policy_entropy.mean() / max(self._target_orientation_entropy, 1e-8)).item()
                    if torch.is_tensor(orientation_policy_entropy.mean() / max(self._target_orientation_entropy, 1e-8))
                    else float(orientation_policy_entropy.mean().item() / max(self._target_orientation_entropy, 1e-8)),
                )
                self.track_data(
                    "Entropy/position_target",
                    float(self._target_position_entropy),
                )
                self.track_data(
                    "Entropy/orientation_target",
                    float(self._target_orientation_entropy),
                )

                if self.cfg.learn_entropy:
                    self.track_data(
                        "Entropy/position_alpha",
                        self._scalar_value(self._position_entropy_coefficient),
                    )
                    self.track_data(
                        "Entropy/orientation_alpha",
                        self._scalar_value(self._orientation_entropy_coefficient),
                    )
                    if position_entropy_loss is not None:
                        self.track_data(
                            "Loss/entropy_position",
                            position_entropy_loss.item(),
                        )
                    if orientation_entropy_loss is not None:
                        self.track_data(
                            "Loss/entropy_orientation",
                            orientation_entropy_loss.item(),
                        )
