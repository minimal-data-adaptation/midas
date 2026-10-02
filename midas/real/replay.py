"""Thread-safe adapter around the shared MIDAS replay buffer."""

from __future__ import annotations

import threading

from midas.data.replay_buffer import ReplayBuffer


class ThreadSafeReplayBuffer:
    def __init__(self, observation_space, action_space, capacity: int) -> None:
        self.buffer = ReplayBuffer(observation_space, action_space, int(capacity))
        self.lock = threading.RLock()

    @property
    def size(self) -> int:
        return self.buffer.size

    @property
    def trajectory_count(self) -> int:
        return self.buffer._traj_counter

    def __len__(self) -> int:
        return len(self.buffer)

    def seed(self, seed: int) -> None:
        self.buffer.seed(seed)

    def get_rng_state(self):
        with self.lock:
            return self.buffer.get_rng_state()

    def set_rng_state(self, state) -> None:
        with self.lock:
            self.buffer.set_rng_state(state)

    def sample(self, batch_size: int):
        with self.lock:
            return self.buffer.sample(batch_size)

    def insert_trajectory(self, transitions: list[dict]) -> None:
        with self.lock:
            for transition in transitions:
                self.buffer.insert(transition)
            self.buffer.increment_traj_counter()

    def append_delta(self, path: str) -> None:
        with self.lock:
            self.buffer.append_delta(path)


__all__ = ["ThreadSafeReplayBuffer"]
