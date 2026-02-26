from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .discrete_sac_cfg import DISCRETE_SAC_CFG


class DiscreteSAC(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: DISCRETE_SAC_CFG | None = None,
    ) -> None:
        """Discrete Soft Actor-Critic (SAC)."""
        self.cfg: DISCRETE_SAC_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=DISCRETE_SAC_CFG() if cfg is None else cfg,
        )

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
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        self._target_update_counter = 1

        self._entropy_coefficient = self.cfg.initial_entropy_value
        if self.cfg.learn_entropy:
            self._target_entropy = self.cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -np.log(self.action_space.n)
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self.cfg.learning_rate[2])
            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            self.critic1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.cfg.learning_rate[1])
            self.critic2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.cfg.learning_rate[1])

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic1_optimizer"] = self.critic1_optimizer
            self.checkpoint_modules["critic2_optimizer"] = self.critic2_optimizer

            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic1_scheduler = self.cfg.learning_rate_scheduler[1]
            self.critic2_scheduler = self.cfg.learning_rate_scheduler[1]
            self.entropy_scheduler = self.cfg.learning_rate_scheduler[2] if self.cfg.learn_entropy else None

            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            if self.critic1_scheduler is not None:
                self.critic1_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic1_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )
                self.critic2_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic2_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )
            if self.entropy_scheduler is not None:
                self.entropy_scheduler = self.cfg.learning_rate_scheduler[2](
                    self.entropy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[2]
                )

        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor

        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    @staticmethod
    def _action_distribution_from_outputs(outputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        net_output = outputs.get("net_output", None)
        if net_output is None:
            raise RuntimeError("Policy outputs must contain 'net_output' for DiscreteSAC")

        if torch.all(net_output >= 0):
            sums = net_output.sum(dim=-1, keepdim=True)
            if torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
                probs = torch.clamp(net_output, min=1e-8)
                log_probs = torch.log(probs)
                return probs, log_probs

        log_probs = F.log_softmax(net_output, dim=-1)
        return log_probs.exp(), log_probs

    def _get_states(self, observations: torch.Tensor, states: torch.Tensor | None) -> torch.Tensor:
        return observations if states is None else states

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self.memory is not None and hasattr(self.memory, "create_tensor"):
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)

            state_space = self.state_space if self.state_space is not None else self.observation_space
            self.memory.create_tensor(name="states", size=state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=state_space, dtype=torch.float32)

            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

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
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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

        if self.memory is not None:
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            states = self._get_states(observations, states)
            next_states = self._get_states(next_observations, next_states)

            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if timestep >= self.cfg.learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        for gradient_step in range(self.cfg.gradient_steps):
            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                    "states": self._state_preprocessor(sampled_next_states, train=True),
                }

                _, outputs = self.policy.act(inputs, role="policy")
                action_probs, action_log_probs = self._action_distribution_from_outputs(outputs)

                with torch.no_grad():
                    critic_1_values, _ = self.critic_1.act(inputs, role="critic_1")
                    critic_2_values, _ = self.critic_2.act(inputs, role="critic_2")
                    critic_values = torch.min(critic_1_values, critic_2_values)

                policy_loss = torch.sum(
                    action_probs * (self._entropy_coefficient * action_log_probs - critic_values.detach()), dim=1
                ).mean()

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self.cfg.policy_grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.policy_grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                with torch.no_grad():
                    _, next_outputs = self.policy.act(next_inputs, role="policy")
                    next_action_probs, next_action_log_probs = self._action_distribution_from_outputs(next_outputs)

                    target_q1_values, _ = self.target_critic_1.act(next_inputs, role="target_critic_1")
                    target_q2_values, _ = self.target_critic_2.act(next_inputs, role="target_critic_2")

                    target_q_values = torch.sum(
                        next_action_probs
                        * (
                            torch.min(target_q1_values, target_q2_values)
                            - self._entropy_coefficient * next_action_log_probs
                        ),
                        dim=1,
                    ).unsqueeze(1)

                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                critic_1_values, _ = self.critic_1.act(inputs, role="critic_1")
                critic_2_values, _ = self.critic_2.act(inputs, role="critic_2")

                gather_index = sampled_actions.long()
                if gather_index.dim() == 1:
                    gather_index = gather_index.unsqueeze(1)

                critic_1_values = torch.gather(critic_1_values, 1, gather_index)
                critic_2_values = torch.gather(critic_2_values, 1, gather_index)

                critic_1_loss = F.mse_loss(critic_1_values, target_values)
                critic_2_loss = F.mse_loss(critic_2_values, target_values)

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
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self.cfg.q_network_grad_norm_clip,
                )

            self.scaler.step(self.critic1_optimizer)
            self.scaler.step(self.critic2_optimizer)

            if self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    entropy = torch.sum(action_probs * action_log_probs, dim=-1, keepdim=True)
                    entropy_loss = -(
                        self.log_entropy_coefficient * (entropy.detach() + self._target_entropy)
                    ).mean()

                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.scaler.update()

            self._target_update_counter += 1
            if self._target_update_counter > self.cfg.steps_to_target_net_update:
                self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)
                self._target_update_counter = 1

            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic1_scheduler:
                self.critic1_scheduler.step()
            if self.critic2_scheduler:
                self.critic2_scheduler.step()
            if self.entropy_scheduler:
                self.entropy_scheduler.step()

            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic 1 loss", critic_1_loss.item())
                self.track_data("Loss / Critic 2 loss", critic_2_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                if self.cfg.learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self.policy_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                if self.critic1_scheduler:
                    self.track_data("Learning / Critic 1 learning rate", self.critic1_scheduler.get_last_lr()[0])
                if self.critic2_scheduler:
                    self.track_data("Learning / Critic 2 learning rate", self.critic2_scheduler.get_last_lr()[0])
