from __future__ import annotations

from typing import Any

import csv
import datetime
import os

import gymnasium
import numpy as np
import torch

from skrl import config


class ReplayBuffer:
    """Tensor-backed replay buffer with ring semantics and skrl-compatible sampling API."""

    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        replacement: bool = True,
    ) -> None:
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = config.torch.parse_device(device)

        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory
        self.replacement = replacement

        if self.export_format not in ["pt", "npz", "csv"]:
            raise ValueError(f"Unsupported export format: '{self.export_format}'")

        self._tensor_specs: dict[str, dict[str, Any]] = {}
        self.graph_keys: list[str] = []

        self._storage: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        self._storage_kind: dict[str, str] = {}
        self._storage_meta: dict[str, Any] = {}

        self.pos = 0
        self.size = 0
        self.filled = False
        self.sampling_indexes: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.size

    def register_graph(self, graph_spec: dict[str, tuple[tuple[int], torch.dtype]]) -> None:
        self.graph_keys = list(graph_spec.keys())

    def create_tensor(
        self,
        name: str,
        *,
        size: int | tuple[int, ...] | list[int] | gymnasium.Space | None,
        dtype: torch.dtype | None = None,
        keep_dimensions: bool = False,
    ) -> bool:
        if name in self._tensor_specs:
            return False

        self._tensor_specs[name] = {
            "size": size,
            "dtype": dtype,
            "keep_dimensions": keep_dimensions,
        }
        return True

    def get_tensor_by_name(self, name: str) -> torch.Tensor | list[torch.Tensor]:
        if self.size == 0:
            return torch.empty(0, device=self.device)

        data = self._storage.get(name, None)
        if data is None:
            return torch.empty(0, device=self.device)

        indices = self._valid_indices()
        if isinstance(data, torch.Tensor):
            return data[indices]
        return [tensor[indices] for tensor in data]

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        data = self._storage.get(name, None)
        if data is None or not isinstance(data, torch.Tensor):
            return

        n = min(self.size, tensor.shape[0])
        if n <= 0:
            return

        indices = self._valid_indices()[:n]
        data[indices] = tensor[:n].to(self.device)

    def reset(self) -> None:
        self.pos = 0
        self.size = 0
        self.filled = False
        self.sampling_indexes = None

    def share_memory(self) -> None:
        for name, storage in self._storage.items():
            if isinstance(storage, torch.Tensor):
                if not storage.is_cuda:
                    storage.share_memory_()
            else:
                for tensor in storage:
                    if not tensor.is_cuda:
                        tensor.share_memory_()

    def get_sampling_indexes(self) -> torch.Tensor | None:
        return self.sampling_indexes

    def add_samples(self, **tensors: Any) -> None:
        if not tensors:
            return

        batch_size = self._infer_batch_size(tensors)
        if batch_size <= 0:
            return

        states = tensors.get("states", None)
        next_states = tensors.get("next_states", None)
        if isinstance(states, (list, tuple)) and self.graph_keys:
            tensors["states"] = dict(zip(self.graph_keys, states))
        if isinstance(next_states, (list, tuple)) and self.graph_keys:
            tensors["next_states"] = dict(zip(self.graph_keys, next_states))

        write_indices = (torch.arange(batch_size, device=self.device) + self.pos) % self.memory_size

        for name, value in tensors.items():
            if value is None:
                continue

            normalized, kind, meta = self._normalize_batch_value(value, batch_size)
            self._ensure_storage(name, normalized, kind, meta)
            self._write_batch(name, normalized, write_indices)

        self.pos = int((self.pos + batch_size) % self.memory_size)
        self.size = min(self.memory_size, self.size + batch_size)
        self.filled = self.size >= self.memory_size

        if self.export and self.filled:
            self.save(directory=self.export_directory, format=self.export_format)

    def sample(
        self,
        names: list[str],
        *,
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> list[list[torch.Tensor | list[torch.Tensor]]]:
        if sequence_length != 1:
            raise ValueError("Sequence sampling is not supported by ReplayBuffer")
        if self.size == 0:
            raise RuntimeError("Sampling from empty replay buffer")

        if self.replacement:
            indexes = torch.randint(0, self.size, (batch_size,), device=self.device)
        else:
            count = min(batch_size, self.size)
            indexes = torch.randperm(self.size, device=self.device)[:count]

        physical = self._logical_to_physical(indexes)
        self.sampling_indexes = physical

        if mini_batches > 1:
            chunks = torch.chunk(physical, mini_batches)
            return [self._sample_by_index(names, chunk) for chunk in chunks]
        return [self._sample_by_index(names, physical)]

    def sample_by_index(
        self,
        names: list[str],
        *,
        indexes: list | np.ndarray | torch.Tensor,
        mini_batches: int = 1,
    ) -> list[list[torch.Tensor | list[torch.Tensor]]]:
        if isinstance(indexes, np.ndarray):
            indexes = torch.from_numpy(indexes)
        elif isinstance(indexes, list):
            indexes = torch.tensor(indexes)

        indexes = indexes.to(self.device).long()
        self.sampling_indexes = indexes

        if mini_batches > 1:
            chunks = torch.chunk(indexes, mini_batches)
            return [self._sample_by_index(names, chunk) for chunk in chunks]
        return [self._sample_by_index(names, indexes)]

    def sample_all(
        self,
        names: list[str],
        *,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> list[list[torch.Tensor | list[torch.Tensor]]]:
        if sequence_length != 1:
            raise ValueError("Sequence sampling is not supported by ReplayBuffer")
        if self.size == 0:
            return []

        indexes = self._valid_indices()
        self.sampling_indexes = indexes

        if mini_batches > 1:
            chunks = torch.chunk(indexes, mini_batches)
            return [self._sample_by_index(names, chunk) for chunk in chunks]
        return [self._sample_by_index(names, indexes)]

    def _sample_by_index(self, names: list[str], indexes: torch.Tensor) -> list[torch.Tensor | list[torch.Tensor]]:
        return [self._stack_name(name, indexes) for name in names]

    def _stack_name(self, name: str, indexes: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        data = self._storage.get(name, None)
        if data is None:
            spec = self._tensor_specs.get(name, {})
            shape = spec.get("size", (1,))
            dtype = spec.get("dtype", torch.float32)
            if isinstance(shape, gymnasium.Space):
                shape = shape.shape if hasattr(shape, "shape") and shape.shape else (1,)
            elif isinstance(shape, int):
                shape = (shape,)
            elif shape is None:
                shape = (1,)
            elif isinstance(shape, list):
                shape = tuple(shape)
            elif not isinstance(shape, tuple):
                shape = (1,)
            return torch.zeros((indexes.shape[0], *shape), dtype=dtype, device=self.device)

        if isinstance(data, torch.Tensor):
            return data[indexes]

        return [tensor[indexes] for tensor in data]

    def _infer_batch_size(self, tensors: dict[str, Any]) -> int:
        for value in tensors.values():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                return value.shape[0] if value.ndim > 0 else 1
            if isinstance(value, np.ndarray):
                return value.shape[0] if value.ndim > 0 else 1
            if isinstance(value, dict) and value:
                first = next(iter(value.values()))
                if isinstance(first, torch.Tensor):
                    return first.shape[0] if first.ndim > 0 else 1
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                first = value[0]
                if isinstance(first, torch.Tensor):
                    if first.ndim > 0:
                        return first.shape[0]
                return len(value)
        return 0

    def _normalize_batch_value(self, value: Any, batch_size: int) -> tuple[torch.Tensor | list[torch.Tensor], str, Any]:
        if isinstance(value, torch.Tensor):
            return value.to(self.device), "tensor", None

        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, device=self.device), "tensor", None

        if isinstance(value, dict):
            keys = list(value.keys())
            tensors = []
            for key in keys:
                item = value[key]
                if isinstance(item, np.ndarray):
                    item = torch.as_tensor(item, device=self.device)
                elif not isinstance(item, torch.Tensor):
                    item = torch.as_tensor(item, device=self.device)
                tensors.append(item.to(self.device))
            return tensors, "dict", {"keys": keys}

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return torch.empty((batch_size, 0), device=self.device), "tensor", None

            if all(isinstance(item, torch.Tensor) for item in value):
                first = value[0]
                if first.ndim > 0 and first.shape[0] == batch_size:
                    return [item.to(self.device) for item in value], "list", None
                stacked = torch.stack([item.to(self.device) for item in value], dim=0)
                return stacked, "tensor", None

            if len(value) == batch_size:
                return torch.as_tensor(value, device=self.device), "tensor", None

            return torch.as_tensor(value, device=self.device), "tensor", None

        return torch.as_tensor(value, device=self.device), "tensor", None

    def _ensure_storage(self, name: str, value: torch.Tensor | list[torch.Tensor], kind: str, meta: Any) -> None:
        if name in self._storage:
            return

        self._storage_kind[name] = kind
        self._storage_meta[name] = meta

        if isinstance(value, torch.Tensor):
            shape = (self.memory_size, *value.shape[1:])
            self._storage[name] = torch.empty(shape, dtype=value.dtype, device=self.device)
            return

        self._storage[name] = [
            torch.empty((self.memory_size, *tensor.shape[1:]), dtype=tensor.dtype, device=self.device)
            for tensor in value
        ]

    def _write_batch(self, name: str, value: torch.Tensor | list[torch.Tensor], indexes: torch.Tensor) -> None:
        target = self._storage[name]
        if isinstance(target, torch.Tensor):
            target[indexes] = value
            return

        for tensor_target, tensor_value in zip(target, value):
            tensor_target[indexes] = tensor_value

    def _logical_to_physical(self, logical_indexes: torch.Tensor) -> torch.Tensor:
        if self.size < self.memory_size:
            return logical_indexes
        start = self.pos
        return (logical_indexes + start) % self.memory_size

    def _valid_indices(self) -> torch.Tensor:
        if self.size < self.memory_size:
            return torch.arange(self.size, device=self.device)
        start = self.pos
        return (torch.arange(self.size, device=self.device) + start) % self.memory_size

    def save(self, *, directory: str = "", format: str = "pt") -> None:
        if not directory:
            directory = self.export_directory

        os.makedirs(os.path.join(directory, "memories"), exist_ok=True)

        path = os.path.join(
            directory,
            "memories",
            "{}_memory_{}.{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
                hex(id(self)),
                format,
            ),
        )

        payload = {
            "memory_size": self.memory_size,
            "num_envs": self.num_envs,
            "pos": self.pos,
            "size": self.size,
            "filled": self.filled,
            "tensor_specs": self._tensor_specs,
            "storage_kind": self._storage_kind,
            "storage_meta": self._storage_meta,
            "storage": self._storage,
            "graph_keys": self.graph_keys,
        }

        if format == "pt":
            torch.save(payload, path)
        elif format == "npz":
            np.savez(path, payload=payload)
        elif format == "csv":
            if self.size == 0:
                return
            names = list(self._storage.keys())
            batches = self.sample_all(names, mini_batches=1)
            if not batches:
                return
            values = batches[0]

            with open(path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(names)
                for row_idx in range(self.size):
                    row = []
                    for item in values:
                        if isinstance(item, torch.Tensor):
                            row.append(item[row_idx].detach().cpu().flatten().tolist())
                        else:
                            row.append([tensor[row_idx].detach().cpu().flatten().tolist() for tensor in item])
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, path: str) -> None:
        if path.endswith(".pt"):
            payload = torch.load(path, map_location=self.device)
        elif path.endswith(".npz"):
            payload = np.load(path, allow_pickle=True)["payload"].item()
        else:
            raise ValueError(f"Unsupported format: {path}")

        # backward compatibility with old list-of-dicts format
        if isinstance(payload, list):
            self.reset()
            self._storage = {}
            self._storage_kind = {}
            self._storage_meta = {}
            for transition in payload:
                self.add_samples(**{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in transition.items()})
            return

        self.memory_size = payload["memory_size"]
        self.num_envs = payload["num_envs"]
        self.pos = payload["pos"]
        self.size = payload["size"]
        self.filled = payload["filled"]
        self._tensor_specs = payload.get("tensor_specs", {})
        self._storage_kind = payload.get("storage_kind", {})
        self._storage_meta = payload.get("storage_meta", {})
        self._storage = payload.get("storage", {})
        self.graph_keys = payload.get("graph_keys", [])
