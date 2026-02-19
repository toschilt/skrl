from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger

from skrl.memories.torch import Memory
from skrl.models.torch import Model

from skrl.agents.agent import Agent

# fmt: off
# [start-config-dict-torch]
SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)
    "steps_to_target_net_update": 64,       # the number of steps to update the target q net

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "policy_grad_norm_clip": 0,         # clipping coefficient for the norm of the gradients
    "q_network_grad_norm_clip": 0,      # clipping coefficient for the norm of the gradients

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class DiscreteSAC(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]
        self._steps_to_target_net_update = self.cfg["steps_to_target_net_update"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._policy_grad_norm_clip = self.cfg["policy_grad_norm_clip"]
        self._q_network_grad_norm_clip = self.cfg["q_network_grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # target net update counter
        if self._steps_to_target_net_update > 0:
            self._target_update_counter = 1

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    # TODO: check if this is the standard for discrete actions
                    self._target_entropy = -np.log(self.action_space.n)
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic1_optimizer = torch.optim.Adam(
                self.critic_1.parameters(), lr=self._critic_learning_rate
            )
            self.critic2_optimizer = torch.optim.Adam(
                self.critic_2.parameters(), lr=self._critic_learning_rate
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic1_scheduler = self._learning_rate_scheduler(
                    self.critic1_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic2_scheduler = self._learning_rate_scheduler(
                    self.critic2_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic1_optimizer"] = self.critic1_optimizer
            self.checkpoint_modules["critic2_optimizer"] = self.critic2_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory (only for skrl Memory)
        if self.memory is not None and isinstance(self.memory, Memory):
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

        # always used (both memories)
        self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in tqdm.tqdm(
            range(self._gradient_steps),
            desc="Training..."
        ):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            # compute policy (actor) loss
            _, logp, _ = self.policy.act({"states": sampled_states}, role="policy")

            with torch.no_grad():
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states}, role="critic_2"
                )
                critic_value = torch.min(critic_1_values, critic_2_values)

            # sum over actions represents the exact expectation over actions
            # mean over states represents the Monte Carlo estimation of expectation over states
            policy_loss = torch.sum(
                logp.exp() * (self._entropy_coefficient * logp - critic_value.detach())
            , dim=1).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._policy_grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._policy_grad_norm_clip)

            self.policy_optimizer.step()

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            # compute target values
            with torch.no_grad():
                _, next_logp, _ = self.policy.act({"states": sampled_next_states}, role="policy")

                target_q1_values, _, _ = self.target_critic_1.act(
                    {"states": sampled_next_states}, role="target_critic_1"
                )
                target_q2_values, _, _ = self.target_critic_2.act(
                    {"states": sampled_next_states}, role="target_critic_2"
                )

                # Discrete SAC (marginalize over all possible actions)
                # Value: $$ V(s') = \sum_{a'} \pi(a' \mid s') \left[ Q(s', a') - \alpha \log \pi(a' \mid s') \right] $$
                target_q_values = torch.sum(
                    next_logp.exp() * (
                        torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * next_logp
                    )
                , dim=1).unsqueeze(1)
                target_values = (
                    sampled_rewards
                    + self._discount_factor
                    * (sampled_terminated | sampled_truncated).logical_not()
                    * target_q_values
                )

            # compute critic loss
            critic_1_values, _, _ = self.critic_1.act(
                {"states": sampled_states}, role="critic_1"
            )
            critic_2_values, _, _ = self.critic_2.act(
                {"states": sampled_states}, role="critic_2"
            )

            critic_1_values = torch.gather(
                critic_1_values, 1, sampled_actions.long().unsqueeze(1)
            )
            critic_2_values = torch.gather(
                critic_2_values, 1, sampled_actions.long().unsqueeze(1)
            )
            critic_1_loss = F.mse_loss(critic_1_values, target_values.detach()).mean()
            critic_2_loss = F.mse_loss(critic_2_values, target_values.detach()).mean()

            # optimization step (critic)
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_1_loss.backward()
            critic_2_loss.backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._q_network_grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._q_network_grad_norm_clip
                )

            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

            # entropy learning
            if self._learn_entropy:
                # In discrete settings, we can calculate the exact entropy over actions
                entropy = (logp * logp.exp()).sum(dim=-1)
                entropy_loss = -(
                    self.log_entropy_coefficient * (entropy.detach() + self._target_entropy)
                ).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_optimizer.step()

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self._target_update_counter += 1

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic1_scheduler.step()
                self.critic2_scheduler.step()

            # record data
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

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self._learning_rate_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic 1 learning rate", self.critic1_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic 2 learning rate", self.critic2_scheduler.get_last_lr()[0])

        if self._target_update_counter > self._steps_to_target_net_update:
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)
            self._target_update_counter = 1