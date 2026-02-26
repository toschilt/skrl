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

    def expand(self) -> None:
        """Expand the configuration."""
        super().expand()

        if self.policy_grad_norm_clip <= 0 and self.grad_norm_clip > 0:
            self.policy_grad_norm_clip = self.grad_norm_clip
        if self.q_network_grad_norm_clip <= 0 and self.grad_norm_clip > 0:
            self.q_network_grad_norm_clip = self.grad_norm_clip
