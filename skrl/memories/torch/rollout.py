from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional, Union
import torch
import gymnasium
import numpy as np  # Added for safe type checking if needed

class RolloutBuffer:
    """
    Sequence-aware on-policy rollout memory compatible with PPO_RNN.
    
    Corrected for:
    1. Vectorized Environments (GAE shape mismatch fix).
    2. List/Numpy Observations (TypeError fix: explicitly converts data to Tensors).
    """

    def __init__(
        self,
        memory_size: int = None,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self._tensor_specs: Dict[str, Dict[str, Any]] = {}
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def __len__(self):
        return self.size

    def reset(self):
        self.buffer: List[List[Dict[str, torch.Tensor]]] = [
            [] for _ in range(self.num_envs)
        ]
        self._sequence_cache: Optional[List[List[Dict[str, torch.Tensor]]]] = None
        self.size = 0

    # ------------------------------------------------------------------
    # Tensor registration
    # ------------------------------------------------------------------

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gymnasium.Space],
        dtype: torch.dtype,
        keep_dimensions: bool = False,
    ):
        if isinstance(size, gymnasium.Space):
            size = size.shape if hasattr(size, "shape") and size.shape else (1,)

        if isinstance(size, int):
            size = (size,)
        elif size is None:
            size = (1,)

        self._tensor_specs[name] = {
            "size": tuple(size),
            "dtype": dtype,
            "keep_dimensions": keep_dimensions,
        }

    # ------------------------------------------------------------------
    # Storage (FIXED)
    # ------------------------------------------------------------------

    def add_samples(self, **tensors):
        """
        Store one transition per environment.

        Guarantees:
        - Multimodal data (list of tensors) keeps its structure
        - Per-env tensors are sliced correctly
        - Only leaf values are converted to tensors
        - Prevents silent shape corruption (critical for PPO_RNN)
        """

        # --------------------------------------------------
        # 1. Infer batch size
        # --------------------------------------------------
        batch_size = None

        for v in tensors.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break

        if batch_size is None:
            for v in tensors.values():
                if isinstance(v, (list, tuple)):
                    # multimodal case: list of tensors
                    if len(v) > 0 and isinstance(v[0], torch.Tensor):
                        batch_size = v[0].shape[0]
                        break
                    # per-env list
                    batch_size = len(v)
                    break

        if batch_size is None or batch_size == 0:
            return

        # --------------------------------------------------
        # 2. Store data per environment
        # --------------------------------------------------
        for env in range(batch_size):
            transition = {}

            for name, value in tensors.items():
                if value is None:
                    continue

                # ------------------------------------------
                # Case A: tensor batch [B, ...]
                # ------------------------------------------
                if isinstance(value, torch.Tensor):
                    val = value[env].detach().to(self.device)

                # ------------------------------------------
                # Case B: list / tuple
                # ------------------------------------------
                elif isinstance(value, (list, tuple)):

                    # 🔥 MULTIMODAL: list of tensors
                    if len(value) > 0 and isinstance(value[0], torch.Tensor):
                        val = [
                            v[env].detach().to(self.device)
                            for v in value
                        ]

                    # 🔹 Per-env list
                    elif len(value) == batch_size:
                        val = value[env]

                        # convert leaf if possible
                        if isinstance(val, torch.Tensor):
                            val = val.detach().to(self.device)
                        else:
                            try:
                                val = torch.as_tensor(val, device=self.device)
                            except Exception:
                                pass

                    # 🔹 Fallback (rare)
                    else:
                        val = value

                # ------------------------------------------
                # Case C: scalar / numpy / etc.
                # ------------------------------------------
                else:
                    try:
                        val = torch.as_tensor(value, device=self.device)
                    except Exception:
                        val = value

                transition[name] = val

            self.buffer[env].append(transition)
            self.size += 1

        self._sequence_cache = None

    # ------------------------------------------------------------------
    # Tensor access
    # ------------------------------------------------------------------

    def get_tensor_by_name(self, name: str) -> torch.Tensor:
        """
        Returns [Time, Num_Envs, *Dims] for GAE.
        """
        tensors = []
        lengths = [len(self.buffer[env]) for env in range(self.num_envs)]
        min_steps = min(lengths) if lengths else 0
        
        if min_steps == 0:
            return torch.empty(0, device=self.device)

        for env in range(self.num_envs):
            if not self.buffer[env]:
                continue
            env_tensor = torch.stack(
                [self.buffer[env][t][name] for t in range(min_steps)], 
                dim=0
            )
            tensors.append(env_tensor)

        return torch.stack(tensors, dim=1)

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor):
        # Structured input [Time, Num_Envs, ...]
        if tensor.dim() >= 2 and tensor.shape[1] == self.num_envs:
            steps = tensor.shape[0]
            for env in range(self.num_envs):
                for t in range(steps):
                    if t < len(self.buffer[env]):
                        self.buffer[env][t][name] = tensor[t, env].to(self.device)
        # Flattened fallback
        else:
            idx = 0
            for env in range(self.num_envs):
                for t in range(len(self.buffer[env])):
                    self.buffer[env][t][name] = tensor[idx].to(self.device)
                    idx += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_all(
        self,
        names: Tuple[str, ...],
        mini_batches: int = 1,
        sequence_length: int = 1,
    ):
        if self.size == 0:
            return []

        # Build flattened view once
        flat = self._build_flat_tensors(names)
        if flat is None:
            return []

        # Determine total size (using first tensor)
        shape_name = names[0] if names[0] != "observations" else names[1]
        total = flat[shape_name].shape[0] if isinstance(flat[shape_name], torch.Tensor) else flat[shape_name][0].shape[0]

        # Sequence mode
        if sequence_length > 1:
            valid_starts = total - sequence_length + 1
            if valid_starts <= 0:
                return []

            starts = torch.randperm(valid_starts, device=self.device)
            chunks = torch.chunk(starts, mini_batches) if mini_batches > 1 else [starts]

            return [
                self._sample_sequence_batch_flat(names, flat, c, sequence_length)
                for c in chunks
            ]

        # Flat mode
        else:
            indices = torch.randperm(total, device=self.device)
            chunks = torch.chunk(indices, mini_batches) if mini_batches > 1 else [indices]

            batches = []
            for c in chunks:
                batch = []
                for name in names:
                    data = flat[name]
                    if isinstance(data, torch.Tensor):
                        # single tensor, index directly
                        batch.append(data[c])
                    elif isinstance(data, list):
                        # multi-modal, index each modality tensor
                        batch.append([modality[c] for modality in data])
                    else:
                        raise TypeError(f"Unexpected type in flat tensors: {type(data)}")
                batches.append(batch)
            return batches

    def _build_flat_tensors(self, names):
        """
        Build [time * num_envs, ...] tensors once.
        This is the key to correct PPO behavior.
        """
        min_steps = min(len(self.buffer[env]) for env in range(self.num_envs))
        if min_steps == 0:
            return None

        flat = {}
        for name in names:
            per_env = []

            for env in range(self.num_envs):
                vals = []
                for t in range(min_steps):
                    v = self.buffer[env][t].get(name)

                    if v is None:
                        if name in self._tensor_specs:
                            spec = self._tensor_specs[name]
                            v = torch.zeros(
                                spec["size"],
                                dtype=spec["dtype"],
                                device=self.device,
                            )
                        else:
                            v = torch.zeros(1, device=self.device)
                    elif isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                    vals.append(v)

                if isinstance(vals[0], torch.Tensor):
                    per_env.append(torch.stack(vals, dim=0))  # [T, ...]
                elif isinstance(vals[0], list):
                    # vals: list of length T, each element is a list of n_modalities tensors
                    n_modalities = len(vals[0])

                    # Initialize a list of lists: one per modality
                    modality_lists = [[] for _ in range(n_modalities)]
                    
                    # Fill modality_lists with tensors from all timesteps
                    for timestep_vals in vals:  # each timestep_vals is a list of tensors
                        for i, tensor in enumerate(timestep_vals):
                            modality_lists[i].append(tensor.to(self.device))
                    
                    # Stack each modality over time dimension T
                    per_env.append([torch.stack(modality, dim=0) for modality in modality_lists])
                    
            # stack envs then flatten
            if isinstance(per_env[0], torch.Tensor):
                stacked = torch.stack(per_env, dim=1)  # [T, E, ...]
                flat[name] = stacked.reshape(-1, *stacked.shape[2:])  # [T*E, ...]
            else:  # multi-modal
                n_modalities = len(per_env[0])
                flat[name] = []
                for i in range(n_modalities):
                    # collect modality i from all envs
                    modality_envs = [per_env[env][i] for env in range(self.num_envs)]  # list of [T, *dims_i]
                    stacked = torch.stack(modality_envs, dim=1)  # [T, E, *dims_i]
                    flat[name].append(stacked.reshape(-1, *stacked.shape[2:]))  # [T*E, *dims_i] 

        return flat
    
    def _sample_sequence_batch_flat(
        self,
        names,
        flat,
        start_indices,
        sequence_length,
    ):
        """
        Returns tensors shaped:

            (batch, seq_len, ...)

        No env dimension leaks.
        """
        batch = []
        device = self.device

        # build sequence offsets once
        offsets = torch.arange(sequence_length, device=device)
        seq_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        # shape: [batch, seq_len]

        for name in names:
            tensor = flat[name]  # [N, ...]
            gathered = tensor[seq_indices]  # ✅ (batch, seq_len, ...)
            batch.append(gathered)

        return batch