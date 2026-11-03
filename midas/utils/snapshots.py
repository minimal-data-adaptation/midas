"""Shared, pre-emption-safe replay snapshot primitives.

The simulation and real-world trainers use the same incremental HDF5 replay
format.  A JSON manifest is always written last and is therefore the durable
completion marker for a snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable


@dataclass
class BufferSnapshotState:
    """High-water mark and cumulative delta chain for one replay buffer."""

    prev_traj_count: int = 0
    prev_delta_files: list[str] = field(default_factory=list)


@dataclass
class SnapshotState:
    """Incremental snapshot state for online and success replay buffers."""

    online: BufferSnapshotState = field(default_factory=BufferSnapshotState)
    success: BufferSnapshotState = field(default_factory=BufferSnapshotState)


def seed_snapshot_state_from_resume(resume_info: dict[str, Any] | None) -> SnapshotState:
    """Recreate writer high-water marks from a v2-or-newer resume manifest."""

    state = SnapshotState()
    if resume_info is None or int(resume_info.get("format_version", 1)) < 2:
        return state
    train_state = resume_info["train_state"]
    for target, source in (
        (state.online, train_state.get("online", {})),
        (state.success, train_state.get("success", {})),
    ):
        target.prev_traj_count = int(source.get("traj_count", 0))
        target.prev_delta_files = list(source.get("delta_files", []) or [])
    return state


def atomic_save(write_fn: Callable[[str], None], final_path: str | os.PathLike[str]) -> None:
    """Write to ``<path>.tmp`` and atomically publish the finished file."""

    final = os.fspath(final_path)
    tmp = final + ".tmp"
    try:
        write_fn(tmp)
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def save_buffer_delta(
    buffer: Any,
    name: str,
    save_dir: str | os.PathLike[str],
    state: BufferSnapshotState,
) -> str | None:
    """Persist newly completed trajectories and advance ``state``.

    Returns the relative delta path, or ``None`` if the buffer did not advance.
    """

    if buffer is None:
        return None
    current = int(buffer._traj_counter)  # ReplayBuffer's persisted trajectory cursor.
    if current <= state.prev_traj_count:
        return None
    relative = Path("replay_buffer") / name / f"{state.prev_traj_count}_{current}.h5"
    absolute = Path(save_dir) / relative
    absolute.parent.mkdir(parents=True, exist_ok=True)
    since = state.prev_traj_count
    atomic_save(lambda path: buffer.save_delta(path, since), absolute)
    relative_str = str(relative)
    state.prev_delta_files.append(relative_str)
    state.prev_traj_count = current
    return relative_str


def write_json_manifest(
    manifest: dict[str, Any],
    path: str | os.PathLike[str],
) -> None:
    """Atomically write a JSON manifest with stable, human-readable output."""

    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)

    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    atomic_save(_write, final)


# Compatibility aliases retained for the existing simulation module and old
# tests. New code should use the public names above.
_BufferSnapshotState = BufferSnapshotState
_seed_snapshot_state_from_resume = seed_snapshot_state_from_resume
_atomic_save = atomic_save
_maybe_save_buffer_delta = save_buffer_delta


__all__ = [
    "BufferSnapshotState",
    "SnapshotState",
    "atomic_save",
    "save_buffer_delta",
    "seed_snapshot_state_from_resume",
    "write_json_manifest",
]
