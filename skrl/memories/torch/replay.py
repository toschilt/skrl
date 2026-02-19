from typing import Optional, Union, Tuple, List, Dict
import os
import csv
import datetime
import numpy as np
import torch

class ReplayBuffer:
    """
    Standalone, memory-efficient replay buffer for GPU graph environments.

    Features:
    - vectorized envs (num_envs > 1)
    - true ring buffer
    - no preallocation
    - GPU-resident tensors
    - no flattening
    - skrl-compatible sampling API
    """

    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        replacement: bool = True,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
    ):
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.replacement = replacement

        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory

        if export_format not in ["pt", "npz", "csv"]:
            raise ValueError(f"Unsupported export format: {export_format}")

        # ring buffer
        self.buffer: List[Dict[str, torch.Tensor]] = []
        self.pos = 0
        self.filled = False

        # graph structure
        self.graph_keys: List[str] = []

        # sampling
        self.sampling_indexes = None

    # ------------------------------------------------------------------
    # Graph registration
    # ------------------------------------------------------------------

    def register_graph(self, graph_spec: Dict[str, Tuple[Tuple[int], torch.dtype]]):
        """
        graph_spec:
            key -> (shape, dtype)
        """
        self.graph_keys = list(graph_spec.keys())

    # ------------------------------------------------------------------
    # Basic API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer.clear()
        self.pos = 0
        self.filled = False

    def share_memory(self):
        # nothing to do: tensors are already shared if on GPU
        pass

    def get_sampling_indexes(self):
        return self.sampling_indexes

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_samples(
        self,
        *,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        log_prob: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
    ):
        """
        Vectorized insert for SAC and PPO.

        Expected shapes:
            states[k]:      [num_envs, *graph_shape]
            next_states[k]: [num_envs, *graph_shape]

        Optional PPO fields:
            log_prob:       [num_envs, 1] log probabilities of actions
            values:         [num_envs, 1] state value estimates
            returns:        [num_envs, 1] computed returns
            advantages:     [num_envs, 1] computed advantages
        """

        # convert states/next_states from list/tuple to dict
        if isinstance(states, (list, tuple)):
            states = dict(zip(self.graph_keys, states))
        if isinstance(next_states, (list, tuple)):
            next_states = dict(zip(self.graph_keys, next_states))

        B = rewards.shape[0]  # number of environments

        for i in range(B):
            transition = {}

            # store graph nodes
            for k in self.graph_keys:
                transition[k] = states[k][i]
                transition[k + "_next"] = next_states[k][i]

            # store common fields
            transition["actions"] = actions[i]
            transition["rewards"] = rewards[i]
            transition["terminated"] = terminated[i]
            transition["truncated"] = truncated[i]

            # store PPO-specific fields if provided
            if log_prob is not None:
                transition["log_prob"] = log_prob[i]
            if values is not None:
                transition["values"] = values[i]
            if returns is not None:
                transition["returns"] = returns[i]
            if advantages is not None:
                transition["advantages"] = advantages[i]

            # add to buffer with ring behavior
            if len(self.buffer) < self.memory_size:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
                self.filled = True

            self.pos = (self.pos + 1) % self.memory_size

        # auto-save if enabled
        if self.export and self.filled:
            self.save(directory=self.export_directory, format=self.export_format)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        names: Tuple[str],
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> List[List[torch.Tensor]]:

        assert sequence_length == 1, "Sequence sampling not supported"

        size = len(self.buffer)
        if size == 0:
            raise RuntimeError("Sampling from empty replay buffer")

        if self.replacement:
            indexes = torch.randint(0, size, (batch_size,), device=self.device)
        else:
            perm = torch.randperm(size, device=self.device)
            indexes = perm[:batch_size]

        self.sampling_indexes = indexes

        if mini_batches > 1:
            chunks = torch.chunk(indexes, mini_batches)
            return [self._sample_by_index(names, c) for c in chunks]
        else:
            return [self._sample_by_index(names, indexes)]

    def sample_by_index(
        self,
        names: Tuple[str],
        indexes: Union[np.ndarray, torch.Tensor, list],
        mini_batches: int = 1,
    ):
        if isinstance(indexes, np.ndarray):
            indexes = torch.from_numpy(indexes).to(self.device)
        elif isinstance(indexes, list):
            indexes = torch.tensor(indexes, device=self.device)

        if mini_batches > 1:
            chunks = torch.chunk(indexes, mini_batches)
            return [self._sample_by_index(names, c) for c in chunks]
        else:
            return [self._sample_by_index(names, indexes)]

    def sample_all(
        self,
        names: Tuple[str],
        mini_batches: int = 1,
        sequence_length: int = 1,
    ):
        assert sequence_length == 1

        size = len(self.buffer)
        indexes = torch.arange(size, device=self.device)

        if mini_batches > 1:
            chunks = torch.chunk(indexes, mini_batches)
            return [self._sample_by_index(names, c) for c in chunks]
        else:
            return [self._sample_by_index(names, indexes)]

    # ------------------------------------------------------------------
    # Internal sampling helpers
    # ------------------------------------------------------------------

    def _sample_by_index(self, names, indexes):
        batch = []

        for name in names:
            if name == "states":
                batch.append(self._stack_graph(indexes, next_state=False))
            elif name == "next_states":
                batch.append(self._stack_graph(indexes, next_state=True))
            else:
                batch.append(
                    torch.stack([self.buffer[i.item()][name] for i in indexes], dim=0)
                )

        return batch

    def _stack_graph(self, indexes, next_state: bool):
        suffix = "_next" if next_state else ""
        out = []

        for k in self.graph_keys:
            out.append(
                torch.stack(
                    [self.buffer[i.item()][k + suffix] for i in indexes],
                    dim=0
                )
            )

        return out

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def save(self, directory: str = "", format: str = "pt"):
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

        if format == "pt":
            torch.save(self.buffer, path)

        elif format == "npz":
            np.savez(path, buffer=self.buffer)

        elif format == "csv":
            with open(path, "w") as f:
                writer = csv.writer(f)
                keys = self.buffer[0].keys()
                writer.writerow(keys)
                for item in self.buffer:
                    writer.writerow(
                        [item[k].detach().cpu().flatten().tolist() for k in keys]
                    )

        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, path: str):
        if path.endswith(".pt"):
            self.buffer = torch.load(path)
        elif path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            self.buffer = list(data["buffer"])
        else:
            raise ValueError(f"Unsupported format: {path}")