"""Purpose: Provide shared, framework-neutral setup and safety primitives for Torch SAC agents.

Usage: Import these internal helpers from SAC implementations to preserve common
checkpoint registration, target setup, recursive replay transfer, and masked
discrete-action validation without introducing application dependencies.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from packaging import version

from skrl import config, logger

SAC_MODEL_NAMES = ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2")


def register_sac_models(agent: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Resolve standard SAC models and retain their established checkpoint keys."""
    models = tuple(agent.models.get(name, None) for name in SAC_MODEL_NAMES)
    for name, model in zip(SAC_MODEL_NAMES, models):
        agent.checkpoint_modules[name] = model
    return models


def broadcast_trainable_sac_models(policy: Any, critic_1: Any, critic_2: Any) -> None:
    """Synchronize the trainable SAC models when distributed Torch is enabled."""
    if not config.torch.is_distributed:
        return
    logger.info("Broadcasting models' parameters")
    for model in (policy, critic_1, critic_2):
        if model is not None:
            model.broadcast_parameters()


def create_grad_scaler(device: str | torch.device, mixed_precision: bool) -> torch.amp.GradScaler:
    """Create the version-compatible GradScaler used by the existing SAC agents."""
    device_type = torch.device(device).type
    if version.parse(torch.__version__) >= version.parse("2.4"):
        return torch.amp.GradScaler(device=device_type, enabled=mixed_precision)
    return torch.cuda.amp.GradScaler(enabled=mixed_precision)


def initialize_target_critics(target_critic_1: Any, target_critic_2: Any, critic_1: Any, critic_2: Any) -> None:
    """Freeze and copy paired target critics using SAC's initial Polyak value."""
    if target_critic_1 is not None and target_critic_2 is not None:
        target_critic_1.freeze_parameters(True)
        target_critic_2.freeze_parameters(True)
        target_critic_1.update_parameters(critic_1, polyak=1)
        target_critic_2.update_parameters(critic_2, polyak=1)


def configure_preprocessors(agent: Any) -> None:
    """Build optional preprocessors and retain their established checkpoint keys."""
    if agent.cfg.observation_preprocessor:
        agent._observation_preprocessor = agent.cfg.observation_preprocessor(
            **agent.cfg.observation_preprocessor_kwargs
        )
        agent.checkpoint_modules["observation_preprocessor"] = agent._observation_preprocessor
    else:
        agent._observation_preprocessor = agent._empty_preprocessor

    if agent.cfg.state_preprocessor:
        agent._state_preprocessor = agent.cfg.state_preprocessor(**agent.cfg.state_preprocessor_kwargs)
        agent.checkpoint_modules["state_preprocessor"] = agent._state_preprocessor
    else:
        agent._state_preprocessor = agent._empty_preprocessor


def get_states(observations: torch.Tensor, states: torch.Tensor | None) -> torch.Tensor:
    """Use observations as the state fallback retained by both SAC public APIs."""
    return observations if states is None else states


def move_to_device_recursive(value: Any, device: str | torch.device) -> Any:
    """Move tensor leaves of a nested replay value without changing container types."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [move_to_device_recursive(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device_recursive(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device_recursive(item, device) for key, item in value.items()}
    return value


def action_distribution_from_outputs(outputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize the established discrete policy output contracts."""
    probs = outputs.get("probs", None)
    log_probs = outputs.get("log_probs", None)
    if (probs is None) != (log_probs is None):
        raise RuntimeError("Policy outputs must provide both 'probs' and 'log_probs'")
    if probs is not None:
        return probs, log_probs

    logits = outputs.get("logits", outputs.get("net_output", None))
    if logits is None:
        raise RuntimeError(
            "Policy outputs must contain either ('probs', 'log_probs'), 'logits', or logits in 'net_output'"
        )
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.exp(), log_probs


def sanitize_discrete_action_indices(
    gather_index: torch.Tensor,
    invalid_mask: torch.Tensor | None,
    *,
    num_actions: int | None = None,
) -> torch.Tensor:
    """Replace invalid or out-of-range critic indices with the first valid action."""
    if num_actions is None:
        if invalid_mask is None:
            raise ValueError("num_actions is required when no invalid action mask is available")
        num_actions = invalid_mask.shape[1]
    if num_actions <= 0:
        raise ValueError("Discrete critics must expose at least one action")

    if invalid_mask is None:
        invalid_mask = torch.zeros((gather_index.shape[0], num_actions), dtype=torch.bool, device=gather_index.device)
    else:
        if invalid_mask.dim() != 2 or invalid_mask.shape != (gather_index.shape[0], num_actions):
            raise ValueError(
                f"Invalid action mask shape {tuple(invalid_mask.shape)}; expected {(gather_index.shape[0], num_actions)}"
            )
        invalid_mask = invalid_mask.to(device=gather_index.device, dtype=torch.bool).clone()

    all_invalid = invalid_mask.all(dim=1)
    if all_invalid.any():
        invalid_mask[all_invalid, 0] = False
    in_range = (gather_index >= 0) & (gather_index < num_actions)
    bounded_index = gather_index.clamp(min=0, max=num_actions - 1)
    chosen_invalid = ~in_range | torch.gather(invalid_mask, 1, bounded_index)
    first_valid = (~invalid_mask).to(dtype=torch.int64).argmax(dim=1, keepdim=True)
    return torch.where(chosen_invalid, first_valid, bounded_index)


def update_target_critics(agent: Any) -> None:
    """Apply the existing delayed paired-target update and counter reset."""
    agent._target_update_counter += 1
    if agent._target_update_counter > agent.cfg.steps_to_target_net_update:
        agent.target_critic_1.update_parameters(agent.critic_1, polyak=agent.cfg.polyak)
        agent.target_critic_2.update_parameters(agent.critic_2, polyak=agent.cfg.polyak)
        agent._target_update_counter = 1
