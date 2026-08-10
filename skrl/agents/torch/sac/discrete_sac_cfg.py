from __future__ import annotations

import dataclasses

from .sac_cfg import SAC_CFG


@dataclasses.dataclass(kw_only=True)
class DISCRETE_SAC_CFG(SAC_CFG):
    """Configuration for the Discrete SAC agent."""

    steps_to_target_net_update: int = 64
    """Number of gradient steps between target critics soft updates."""

    policy_grad_norm_clip: float = 0
    """Policy gradient clipping coefficient by global norm."""

    q_network_grad_norm_clip: float = 0
    """Critic gradient clipping coefficient by global norm."""

    actor_learning_starts: int = 0
    """Timestep at which actor and entropy updates may begin."""

    actor_update_delay: int = 1
    """Number of critic gradient updates between actor updates."""

    skip_previous_done_transitions: bool = False
    """Do not store auto-reset bridge transitions following completed episodes."""

    sequential_critic_update: bool = False
    """Backpropagate the two critics one at a time to reduce peak memory."""

    def expand(self) -> None:
        """Expand the configuration."""
        super().expand()

        if self.policy_grad_norm_clip <= 0 and self.grad_norm_clip > 0:
            self.policy_grad_norm_clip = self.grad_norm_clip
        if self.q_network_grad_norm_clip <= 0 and self.grad_norm_clip > 0:
            self.q_network_grad_norm_clip = self.grad_norm_clip
