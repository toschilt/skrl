"""Purpose: Isolate factorized discrete SAC masking, entropy, and Q-value validation math.

Usage: Use these internal pure helpers from ``FactorizedDiscreteSACSimple``;
they accept Torch values only and intentionally contain no PathSim imports.
"""

from __future__ import annotations

from typing import Any

import torch


def extract_factorized_policy_outputs(outputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the policy's required position and orientation logits contract."""
    logits_pos = outputs.get("logits_pos", None)
    logits_rot = outputs.get("logits_rot", None)
    if logits_pos is None or logits_rot is None:
        raise RuntimeError("Policy outputs must contain 'logits_pos' [B, N_pos] and 'logits_rot' [B, N_pos, N_rot]")
    return logits_pos, logits_rot


def resolve_target_entropies(
    position_entropy: float | None, orientation_entropy: float | None, n_pos: int, n_rot: int
) -> tuple[float, float]:
    """Retain explicit entropy targets or resolve the established factor dimensions."""
    if position_entropy is None:
        position_entropy = float(torch.log(torch.tensor(float(n_pos))))
    if orientation_entropy is None:
        orientation_entropy = float(torch.log(torch.tensor(float(n_rot))))
    return position_entropy, orientation_entropy


def build_position_invalid_mask_from_states(states: Any) -> torch.Tensor:
    """Return the exact position-action mask stored with a replay state.

    The environment builds this padding mask together with the packed neighbor
    slots seen by the actor.  Do not infer an additional self-node mask from
    ``current_index``: local node IDs can be remapped by observation/graph
    variants, and the stay-and-rotate ablation deliberately leaves that slot
    valid.  The stored padding mask is therefore the single action-validity
    contract for both sampling and replay.
    """
    edge_padding_mask = states[7]
    if edge_padding_mask.dim() not in (2, 3):
        raise ValueError(
            "edge_padding_mask must have dim 2 or 3, "
            f"got {edge_padding_mask.dim()}"
        )
    return (
        edge_padding_mask.squeeze(1).bool()
        if edge_padding_mask.dim() == 3
        else edge_padding_mask.bool()
    )


def masked_position_distribution(
    logits_pos: torch.Tensor, invalid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce a normalized position distribution with a deterministic all-invalid fallback."""
    masked_logits = logits_pos.masked_fill(invalid_mask, -1e9)
    pos_probs = torch.softmax(masked_logits, dim=-1) * (~invalid_mask).float()
    normalized_probs = pos_probs / pos_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    fallback = torch.zeros_like(normalized_probs)
    fallback[:, 0] = 1.0
    normalized_probs = torch.where((~invalid_mask).any(dim=-1, keepdim=True), normalized_probs, fallback)
    return torch.log(normalized_probs.clamp_min(1e-8)), normalized_probs


def sanitize_position_actions(position_actions: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
    """Map invalid replay position actions to the first valid deterministic fallback."""
    position_actions = position_actions.view(-1).long().clamp(min=0, max=invalid_mask.shape[1] - 1)
    valid_mask = ~invalid_mask
    first_valid = valid_mask.float().argmax(dim=1).long()
    replacement = torch.where(valid_mask.any(dim=1), first_valid, torch.zeros_like(first_valid))
    batch_index = torch.arange(position_actions.shape[0], device=position_actions.device)
    return torch.where(invalid_mask[batch_index, position_actions], replacement, position_actions)


def conservative_q_penalty(
    q_all: torch.Tensor,
    q_taken: torch.Tensor,
    invalid_position_mask: torch.Tensor,
    temperature: float,
    invalid_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a CQL-style log-sum-exp penalty over valid factorized actions."""
    temperature = max(float(temperature), 1e-6)
    invalid_position_action_mask = invalid_position_mask.unsqueeze(-1).expand_as(q_all)
    invalid_action_mask = (
        invalid_position_action_mask
        if invalid_action_mask is None
        else invalid_action_mask.bool() | invalid_position_action_mask
    )
    action_logits = q_all.masked_fill(invalid_action_mask, -1e9).reshape(q_all.shape[0], -1) / temperature
    conservative_value = temperature * torch.logsumexp(action_logits, dim=1, keepdim=True)
    return (conservative_value - q_taken).mean()


def factorized_q_policy_diagnostics(
    q_all: torch.Tensor,
    position_probs: torch.Tensor,
    orientation_probs: torch.Tensor,
    invalid_position_mask: torch.Tensor,
    invalid_orientation_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize valid all-action Q values relative to the factorized policy."""
    q_all, position_probs, orientation_probs = q_all.detach(), position_probs.detach(), orientation_probs.detach()
    invalid_action_mask = invalid_position_mask.detach().bool().unsqueeze(-1).expand_as(q_all)
    if invalid_orientation_mask is not None:
        invalid_action_mask = invalid_action_mask | invalid_orientation_mask.detach().bool()
    masked_q_all = q_all.masked_fill(invalid_action_mask, -1e9)
    flat_q = masked_q_all.reshape(masked_q_all.shape[0], -1)
    q_max_per_sample, flat_argmax = torch.max(flat_q, dim=1)
    orientation_dim = int(q_all.shape[-1])
    action_probs = position_probs.unsqueeze(-1) * orientation_probs
    action_probs = action_probs.masked_fill(invalid_action_mask, 0.0)
    action_probs = action_probs / action_probs.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    q_policy_expectation = torch.sum(action_probs * q_all.masked_fill(invalid_action_mask, 0.0), dim=(1, 2))
    q_max_action_probability = torch.gather(
        action_probs.reshape(action_probs.shape[0], -1), 1, flat_argmax.view(-1, 1)
    ).squeeze(1)
    return {
        "q_all_max": q_max_per_sample.max(),
        "q_all_mean": masked_q_all[~invalid_action_mask].mean(),
        "q_policy_expectation_mean": q_policy_expectation.mean(),
        "q_policy_expectation_max": q_policy_expectation.max(),
        "q_max_minus_policy_expectation_mean": (q_max_per_sample - q_policy_expectation).mean(),
        "q_max_minus_policy_expectation_max": (q_max_per_sample - q_policy_expectation).max(),
        "q_max_action_position_mean": (flat_argmax // orientation_dim).float().mean(),
        "q_max_action_orientation_mean": (flat_argmax % orientation_dim).float().mean(),
        "q_max_action_probability_mean": q_max_action_probability.mean(),
        "q_max_action_probability_max": q_max_action_probability.max(),
    }
