"""Helpers for pre-emption-safe resume of `examples/train_sim_residual.py`.

Two on-disk formats are supported:

**v2 (incremental deltas, current).** A complete snapshot at step ``S`` under
run directory ``save_dir`` consists of:
    save_dir/checkpoint<S>/                                 (Orbax agent ckpt)
    save_dir/replay_buffer/online/<lo>_<hi>.pkl             (global delta pool;
    save_dir/replay_buffer/success/<lo>_<hi>.pkl              cumulative chain)
    save_dir/train_state/<S>.json                           (manifest, written last)

The JSON manifest carries ``format_version: 2`` and the cumulative
``online.delta_files`` / ``success.delta_files`` lists (relative paths),
plus ``traj_count`` / ``size`` mirrors. ``resolve_resume`` requires every
listed delta file to exist before declaring a snapshot complete.

**v1 (legacy full-buffer pickle).** Older snapshots have:
    save_dir/checkpoint<S>/
    save_dir/replay_buffer/<S>/online_replay_buffer.pkl
    save_dir/replay_buffer/<S>/success_replay_buffer.pkl    (optional)
    save_dir/train_state/<S>.json                           (no format_version)

These are still selectable on resume; the train-side path falls back to
``ReplayBuffer.restore`` for the full pickle.

In both formats the JSON is renamed atomically last and is the durable
"snapshot complete" marker.
"""

import json
import os
import pathlib
import shutil
from typing import List, Optional, TypedDict


class ResumeInfo(TypedDict):
    step: int
    agent_dir: str
    format_version: int
    online_buffer_path: Optional[str]      # legacy full pickle (v1)
    success_buffer_path: Optional[str]     # legacy full pickle (v1)
    online_delta_paths: List[str]          # v2 cumulative chain (absolute paths)
    success_delta_paths: List[str]         # v2 cumulative chain (absolute paths)
    online_traj_count: int
    success_traj_count: int
    train_state: dict


def _v1_paths(resume_dir_p: pathlib.Path, step: int):
    rb_dir = resume_dir_p / 'replay_buffer' / str(step)
    return {
        'agent_dir': resume_dir_p / f'checkpoint{step}',
        'online_buffer_path': rb_dir / 'online_replay_buffer.pkl',
        'success_buffer_path': rb_dir / 'success_replay_buffer.pkl',
        'train_state_path': resume_dir_p / 'train_state' / f'{step}.json',
    }


def _read_train_state(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _v1_missing(paths: dict) -> List[str]:
    missing = []
    if not paths['agent_dir'].exists():
        missing.append(f"agent_dir={paths['agent_dir']}")
    if not paths['online_buffer_path'].exists():
        missing.append(f"online_buffer={paths['online_buffer_path']}")
    return missing


def _v2_missing(resume_dir_p: pathlib.Path, step: int, train_state: dict) -> List[str]:
    missing = []
    agent_dir = resume_dir_p / f'checkpoint{step}'
    if not agent_dir.exists():
        missing.append(f"agent_dir={agent_dir}")

    online_files = train_state.get('online', {}).get('delta_files', []) or []
    for rel in online_files:
        if not (resume_dir_p / rel).exists():
            missing.append(f"online_delta={rel}")
    success_files = train_state.get('success', {}).get('delta_files', []) or []
    for rel in success_files:
        if not (resume_dir_p / rel).exists():
            missing.append(f"success_delta={rel}")
    return missing


def resolve_resume(resume_dir: str) -> ResumeInfo:
    resume_dir_p = pathlib.Path(resume_dir)
    if not resume_dir_p.exists():
        raise FileNotFoundError(f"resume_dir does not exist: {resume_dir_p}")

    ts_root = resume_dir_p / 'train_state'
    if not ts_root.is_dir():
        raise FileNotFoundError(
            f"resume_dir is missing train_state/ subdir: {ts_root}"
        )

    candidate_steps = sorted(
        (int(p.stem) for p in ts_root.glob('*.json') if p.stem.isdigit()),
        reverse=True,
    )
    if not candidate_steps:
        raise FileNotFoundError(
            f"No train_state/<step>.json files found under {ts_root}; "
            f"nothing to resume from."
        )

    skipped = []
    for step in candidate_steps:
        train_state_path = ts_root / f'{step}.json'
        try:
            train_state = _read_train_state(train_state_path)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((step, [f"train_state_unreadable: {e}"]))
            continue

        format_version = int(train_state.get('format_version', 1))
        if format_version >= 2:
            missing = _v2_missing(resume_dir_p, step, train_state)
            if missing:
                skipped.append((step, missing))
                continue
            agent_dir = resume_dir_p / f'checkpoint{step}'
            online_files = train_state.get('online', {}).get('delta_files', []) or []
            success_files = train_state.get('success', {}).get('delta_files', []) or []
            info: ResumeInfo = {
                'step': step,
                'agent_dir': str(agent_dir),
                'format_version': format_version,
                'online_buffer_path': None,
                'success_buffer_path': None,
                'online_delta_paths': [str(resume_dir_p / r) for r in online_files],
                'success_delta_paths': [str(resume_dir_p / r) for r in success_files],
                'online_traj_count': int(train_state.get('online', {}).get('traj_count', 0)),
                'success_traj_count': int(train_state.get('success', {}).get('traj_count', 0)),
                'train_state': train_state,
            }
        else:
            paths = _v1_paths(resume_dir_p, step)
            missing = _v1_missing(paths)
            if missing:
                skipped.append((step, missing))
                continue
            success_path = paths['success_buffer_path']
            info = {
                'step': step,
                'agent_dir': str(paths['agent_dir']),
                'format_version': 1,
                'online_buffer_path': str(paths['online_buffer_path']),
                'success_buffer_path': str(success_path) if success_path.exists() else None,
                'online_delta_paths': [],
                'success_delta_paths': [],
                'online_traj_count': 0,
                'success_traj_count': 0,
                'train_state': train_state,
            }

        if skipped:
            for partial_step, partial_missing in skipped:
                print(
                    f"resolve_resume: skipping partial snapshot at step="
                    f"{partial_step}; missing {partial_missing}"
                )
            print(f"resolve_resume: resuming from step={step} (format v{info['format_version']})")
        return info

    detail = ', '.join(f"step={s}: missing {m}" for s, m in skipped)
    raise FileNotFoundError(
        f"No complete snapshot found under {resume_dir_p}. Inspected: {detail}"
    )


def gc_old_snapshots(save_dir: str, current_step: int,
                     keep_every_n: Optional[int]) -> None:
    """Delete snapshots that are neither the current latest nor a milestone.

    With ``keep_every_n`` set, retain snapshots whose step is a multiple of
    ``keep_every_n`` plus the most recent (``current_step``); delete everything
    else.

    Selection is driven by ``train_state/*.json`` (the durable completion
    marker). Per-step ``replay_buffer/<S>/`` subdirs are deleted only for
    legacy v1 snapshots; v2 snapshots use a global delta pool under
    ``replay_buffer/online/`` and ``replay_buffer/success/`` which is
    cumulative — milestone JSONs continue to reference earlier deltas, so the
    pool is never touched here. Use ``sweep_orphan_deltas`` separately to
    clean files left by aborted writes.

    Within each candidate snapshot, deletion order is JSON first →
    ``replay_buffer/<S>/`` (v1 only) → ``checkpoint<S>/``, so a crash
    mid-delete leaves remnants invisible to ``resolve_resume`` rather than a
    half-deleted selectable snapshot.
    """
    if keep_every_n is None:
        return
    ts_root = os.path.join(save_dir, 'train_state')
    if not os.path.isdir(ts_root):
        return
    for entry in os.listdir(ts_root):
        stem, ext = os.path.splitext(entry)
        if ext != '.json' or not stem.isdigit():
            continue
        step = int(stem)
        if step == current_step or step % keep_every_n == 0:
            continue
        json_path = os.path.join(ts_root, entry)
        format_version = 1
        try:
            with open(json_path) as f:
                format_version = int(json.load(f).get('format_version', 1))
        except (OSError, json.JSONDecodeError):
            pass
        rb_dir = os.path.join(save_dir, 'replay_buffer', str(step))
        ckpt_dir = os.path.join(save_dir, f'checkpoint{step}')
        try:
            os.remove(json_path)
        except FileNotFoundError:
            pass
        if format_version < 2 and os.path.isdir(rb_dir):
            shutil.rmtree(rb_dir, ignore_errors=True)
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)


def sweep_orphan_deltas(save_dir: str) -> List[str]:
    """Unlink delta files under ``replay_buffer/{online,success}/`` that no
    surviving ``train_state/<S>.json`` references.

    Run once at training start. Cleans up files left by mid-save crashes
    where a delta was renamed into place but the JSON manifest never followed.
    Returns the list of files removed.
    """
    save_dir_p = pathlib.Path(save_dir)
    ts_root = save_dir_p / 'train_state'
    if not ts_root.is_dir():
        return []

    referenced: set = set()
    for json_path in ts_root.glob('*.json'):
        if not json_path.stem.isdigit():
            continue
        try:
            with open(json_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for section in ('online', 'success'):
            files = manifest.get(section, {}).get('delta_files', []) or []
            for rel in files:
                referenced.add((save_dir_p / rel).resolve())

    removed = []
    for sub in ('online', 'success'):
        pool = save_dir_p / 'replay_buffer' / sub
        if not pool.is_dir():
            continue
        for f in pool.iterdir():
            if not f.is_file() or f.suffix not in ('.h5', '.pkl'):
                continue
            if f.resolve() not in referenced:
                try:
                    f.unlink()
                    removed.append(str(f))
                except OSError:
                    pass
    return removed
