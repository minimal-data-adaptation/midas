"""LIBERO demo data loader for replay buffer pre-seeding.

Loads demonstration trajectories from LIBERO HDF5 files, runs Pi-0.5
inference to compute base actions for each observation, and populates
a ReplayBuffer with sliding-window transitions (stride=1) suitable
for residual RL training.

Two HDF5 layouts are auto-detected:

1. Robomimic layout (from LIBERO/scripts/create_dataset.py):
    data/demo_{i}/obs/agentview_rgb     (T, H, W, 3)  upside-down
    data/demo_{i}/obs/eye_in_hand_rgb   (T, H, W, 3)  upside-down
    data/demo_{i}/obs/ee_states         (T, 6)  eef_pos + axis_angle
    data/demo_{i}/obs/gripper_states    (T, gripper_dim)
    data/demo_{i}/actions               (T, 7)
    data/demo_{i}/rewards               (T,)
    data/demo_{i}/dones                 (T,)

2. Flat LeRobot-style layout (physical-intelligence/libero export):
    episode_{i:07d}/image               (T, H, W, 3)  already upright
    episode_{i:07d}/wrist_image         (T, H, W, 3)  already upright
    episode_{i:07d}/state               (T, 8)  ee_states(6) + gripper(2)
    episode_{i:07d}/actions             (T, 7)
    episode_{i:07d}/rewards             (T,)
    metadata attrs: task_description, action_dim, state_dim, ...

The robomimic frames are upside-down (live sim convention) and the
loader flips them to match the live pipeline. The flat-layout frames
are already upright and are passed through unflipped.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import PIL.Image
from openpi_client import image_tools

from midas.data import ReplayBuffer


InitPolicy = Literal["none", "provided", "builtin"]


@dataclass(frozen=True)
class TaskSpec:
    """Portable description of one externally generated LIBERO task."""

    name: str
    bddl_file: str
    language: str
    init_file: str | None = None


@dataclass(frozen=True)
class _ExternalSuiteSpec:
    name: str
    tasks: tuple[TaskSpec, ...]
    bddl_root: Path
    init_root: Path | None
    init_policy: Literal["none", "provided"]


class ExternalTaskSuite:
    """Minimal LIBERO benchmark-compatible view over generated BDDL files."""

    def __init__(self, spec: _ExternalSuiteSpec):
        self._spec = spec
        self.name = spec.name
        self.tasks = spec.tasks
        self.n_tasks = len(spec.tasks)

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task(self, task_id: int) -> TaskSpec:
        return self.tasks[task_id]

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task_bddl_file_path(self, task_id: int) -> str:
        return str(self._spec.bddl_root / self.tasks[task_id].bddl_file)

    def get_task_init_states(self, task_id: int):
        if self._spec.init_policy != "provided" or self._spec.init_root is None:
            raise RuntimeError(f"Suite {self.name!r} does not provide init states")
        init_file = self.tasks[task_id].init_file
        if init_file is None:
            raise RuntimeError(f"Task {task_id} in {self.name!r} has no init file")
        import torch

        # Generated suite manifests point to trusted local LIBERO init-state
        # files containing NumPy arrays, not neural-network weights. PyTorch
        # 2.6+ therefore needs the legacy object loader enabled explicitly.
        return torch.load(self._spec.init_root / init_file, weights_only=False)


_EXTERNAL_SUITES: dict[str, _ExternalSuiteSpec] = {}


def _safe_child(root: Path, relative: str, field: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be relative and cannot escape its root: {relative!r}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes its root: {relative!r}")
    return resolved


def register_bddl_suite(
    suite_name: str,
    tasks: list[TaskSpec],
    bddl_root: str | os.PathLike[str],
    init_root: str | os.PathLike[str] | None = None,
    init_policy: Literal["none", "provided"] = "none",
) -> None:
    """Validate and register an external BDDL suite in the current process."""
    if not suite_name or not tasks:
        raise ValueError("suite_name and at least one task are required")
    bddl_path = Path(bddl_root).expanduser().resolve()
    init_path = Path(init_root).expanduser().resolve() if init_root is not None else None
    if not bddl_path.is_dir():
        raise FileNotFoundError(f"BDDL root does not exist: {bddl_path}")
    if init_policy == "provided" and (init_path is None or not init_path.is_dir()):
        raise FileNotFoundError(f"Init-state root does not exist: {init_path}")

    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    for task in tasks:
        bddl_file = _safe_child(bddl_path, task.bddl_file, "bddl_file")
        if not bddl_file.is_file():
            raise FileNotFoundError(bddl_file)
        robosuite_parse_problem(str(bddl_file))
        if init_policy == "provided":
            if task.init_file is None:
                raise ValueError(f"Task {task.name!r} must name an init file")
            init_file = _safe_child(init_path, task.init_file, "init_file")
            if not init_file.is_file():
                raise FileNotFoundError(init_file)

    spec = _ExternalSuiteSpec(
        name=suite_name,
        tasks=tuple(tasks),
        bddl_root=bddl_path,
        init_root=init_path,
        init_policy=init_policy,
    )
    existing = _EXTERNAL_SUITES.get(suite_name.lower())
    if existing is not None and existing != spec:
        raise ValueError(f"Suite {suite_name!r} is already registered differently")
    _EXTERNAL_SUITES[suite_name.lower()] = spec


def get_task_suite(suite_name: str):
    """Resolve an external suite first, then an untouched LIBERO built-in suite."""
    external = _EXTERNAL_SUITES.get(suite_name.lower())
    if external is not None:
        return ExternalTaskSuite(external)
    from libero.libero import benchmark

    benchmark_cls = benchmark.get_benchmark_dict().get(suite_name.lower())
    if benchmark_cls is None:
        raise KeyError(f"Unknown LIBERO suite: {suite_name}")
    return benchmark_cls()


def get_suite_init_policy(suite_name: str) -> InitPolicy:
    external = _EXTERNAL_SUITES.get(suite_name.lower())
    return external.init_policy if external is not None else "builtin"


def load_suite_manifest(manifest_path: str | os.PathLike[str]) -> str:
    """Register a generated suite from ``suite_meta.json`` in a fresh process."""
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported suite manifest schema: {payload.get('schema_version')}")
    suite_root = path.parent
    tasks = [TaskSpec(**task) for task in payload["tasks"]]
    register_bddl_suite(
        payload["suite_name"],
        tasks,
        suite_root / payload.get("bddl_root", "bddl"),
        suite_root / payload["init_root"] if payload.get("init_root") else None,
        payload["init_policy"],
    )
    return payload["suite_name"]


def _hdf5_obs_to_pi_zero_input(
    agentview_rgb: np.ndarray,
    eye_in_hand_rgb: np.ndarray,
    ee_states: np.ndarray,
    gripper_states: np.ndarray,
    task_description: str,
    flip: bool = True,
) -> dict:
    """Convert HDF5 observation arrays to Pi-0.5 input format.

    Applies the same [::-1, ::-1] flip as obs_to_pi_zero_input in
    train_utils_sim_residual.py to match the live pipeline orientation.

    Args:
        agentview_rgb: (H, W, 3) uint8 image from HDF5.
        eye_in_hand_rgb: (H, W, 3) uint8 wrist image from HDF5.
        ee_states: (6,) = eef_pos(3) + axis_angle(3), already converted.
        gripper_states: (gripper_dim,) gripper joint positions.
        task_description: Language instruction for this task.

    Returns:
        Dict matching Pi-0.5 input schema.
    """
    if flip:
        img = np.ascontiguousarray(agentview_rgb[::-1, ::-1])
        wrist_img = np.ascontiguousarray(eye_in_hand_rgb[::-1, ::-1])
    else:
        img = np.ascontiguousarray(agentview_rgb)
        wrist_img = np.ascontiguousarray(eye_in_hand_rgb)

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 224, 224)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )

    state = np.concatenate([ee_states, gripper_states[:2]])

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _process_libero_image(
    agentview_rgb: np.ndarray,
    resize_image: int,
    flip: bool,
) -> np.ndarray:
    """Apply demo→replay-buffer pixel preprocessing (flip + resize).

    Factored out so the feature-precompute pass in
    ``load_libero_demos_to_buffer`` and the obs-builder
    ``_hdf5_obs_to_replay_obs`` both consume identical pixel arrays —
    otherwise cached features could disagree with what the encoder would
    have produced from the stored ``pixels`` slot.
    """
    if flip:
        img = np.ascontiguousarray(agentview_rgb[::-1, ::-1])
    else:
        img = np.ascontiguousarray(agentview_rgb)
    if resize_image > 0:
        img = np.array(
            PIL.Image.fromarray(img).resize(
                (resize_image, resize_image)
            )
        )
    return img


def _hdf5_obs_to_replay_obs(
    agentview_rgb: np.ndarray,
    ee_states: np.ndarray,
    gripper_states: np.ndarray,
    base_action_chunk: np.ndarray,
    resize_image: int,
    add_states: bool,
    use_vlm_embedding: bool = False,
    vlm_embedding: Optional[np.ndarray] = None,
    flip: bool = True,
    pixel_features: Optional[np.ndarray] = None,
) -> dict:
    """Convert HDF5 observation arrays to replay buffer observation format.

    Builds the same dict structure as collect_traj_residual in
    train_utils_sim_residual.py, but without the leading batch dimension
    (replay buffer stores per-transition).

    Args:
        agentview_rgb: (H, W, 3) uint8 image from HDF5.
        ee_states: (6,) eef_pos + axis_angle.
        gripper_states: (gripper_dim,) gripper joint positions.
        base_action_chunk: (chunk_len, action_dim) base actions from Pi-0.5.
        resize_image: Target image resolution (e.g. 128).
        add_states: Whether to include proprioceptive state in obs.
        use_vlm_embedding: Whether to include VLM embedding.
        vlm_embedding: (W,) mean-pooled VLM hidden state, required if
            use_vlm_embedding is True.

    Returns:
        Dict with 'pixels', optionally 'state', 'base_action', optionally
        'vlm_embedding', each with a trailing singleton dimension.
    """
    curr_image = _process_libero_image(agentview_rgb, resize_image, flip)

    obs_dict = {
        'pixels': curr_image[..., np.newaxis],
        'base_action': base_action_chunk[..., np.newaxis],
    }

    if add_states:
        state = np.concatenate([ee_states, gripper_states[:2]])
        obs_dict['state'] = state[..., np.newaxis]

    if use_vlm_embedding:
        assert vlm_embedding is not None, (
            "vlm_embedding required when use_vlm_embedding=True"
        )
        obs_dict['vlm_embedding'] = vlm_embedding[..., np.newaxis]

    if pixel_features is not None:
        # Cached frozen-encoder features (only set when
        # ``freeze_vision_encoder=True`` on the agent). Trailing singleton
        # dim mirrors the VLM-embedding storage convention so the same
        # cached-feature branch in PixelMultiplexer.encode handles both.
        obs_dict['pixel_features'] = pixel_features[..., np.newaxis]

    return obs_dict


def load_libero_demos_to_buffer(
    hdf5_path: str,
    replay_buffer: ReplayBuffer,
    agent_dp,
    variant,
    num_demos: int = -1,
    task_description: Optional[str] = None,
    agent_vlm=None,
    agent=None,
) -> ReplayBuffer:
    """Load LIBERO demo data into a replay buffer using sliding windows.

    For each demo of T timesteps with query_freq=Q, produces T-Q
    overlapping transitions (stride=1). Each transition stores:
      - obs at time t, next_obs at time t+Q
      - action chunk actions[t:t+Q]
      - base_action from Pi-0.5 inference at time t

    Args:
        hdf5_path: Path to LIBERO HDF5 demo file.
        replay_buffer: Target replay buffer to populate.
        agent_dp: Frozen Pi-0.5 policy for base action generation.
        variant: Training config (attrdict) with chunk_len, query_freq,
            residual_alpha, resize_image, add_states, etc.
        num_demos: Number of demos to load (-1 = all).
        task_description: Language instruction. If None, extracted from
            HDF5 problem_info attribute.

    Returns:
        The populated replay_buffer.
    """
    query_freq = variant.query_freq
    chunk_len = variant.chunk_len
    action_dim = variant.action_dim
    residual_alpha = variant.residual_alpha
    resize_image = variant.resize_image
    add_states = variant.add_states
    predict_a_exec = variant.get('predict_a_exec', False)
    reward_type = variant.get('reward_type', 'sparse')
    discount = variant.discount
    use_vlm_embedding = variant.get('use_vlm_embedding', False)
    # PixelMultiplexer.encode prioritizes vlm_embedding and ignores
    # pixel_features, so caching is only meaningful when VLM mode is off.
    cache_pixel_features = (
        variant.get('freeze_vision_encoder', False) and not use_vlm_embedding
    )
    actor_pop_base_actions = variant.get('actor_pop_base_actions', False)
    critic_pop_base_actions = variant.get('critic_pop_base_actions', True)
    skip_infer = actor_pop_base_actions and critic_pop_base_actions
    if cache_pixel_features:
        assert agent is not None and getattr(agent, 'pixel_features_dim', None), (
            "load_libero_demos_to_buffer requires agent= with a populated "
            "pixel_features_dim when freeze_vision_encoder=True so demo "
            "transitions can be cached the same way rollout transitions are."
        )
        pixel_features_dim = int(agent.pixel_features_dim)
    else:
        pixel_features_dim = None

    f = h5py.File(hdf5_path, 'r')

    # Auto-detect layout. Robomimic files have a top-level 'data' group;
    # the flat LeRobot export stores one 'episode_XXXXXXX' group per file
    # alongside a 'metadata' group.
    is_flat_layout = 'data' not in f and any(
        k.startswith('episode_') for k in f.keys()
    )

    if is_flat_layout:
        flip_images = False
        demo_keys = sorted(
            [k for k in f.keys() if k.startswith('episode_')],
            key=lambda k: int(k.split('_')[1]),
        )
        parent_grp = f

        if task_description is None:
            if 'metadata' in f and 'task_description' in f['metadata'].attrs:
                task_description = str(
                    f['metadata'].attrs['task_description']
                )
            else:
                raise ValueError(
                    f'Flat-layout HDF5 {hdf5_path} has no '
                    'metadata/task_description and no override was given.'
                )
            print(
                f'[Demo Loader] Task description from HDF5 metadata: '
                f'{task_description}'
            )
        if not task_description:
            raise ValueError(
                f'Empty task_description for {hdf5_path}; pi0.5 requires a '
                'non-empty language prompt.'
            )
    else:
        flip_images = True
        parent_grp = f['data']

        if task_description is None:
            problem_info = json.loads(parent_grp.attrs['problem_info'])
            task_description = problem_info['language_instruction']
            print(
                f'[Demo Loader] Task description from HDF5: {task_description}'
            )
        if not task_description:
            raise ValueError(
                f'Empty task_description for {hdf5_path}; pi0.5 requires a '
                'non-empty language prompt.'
            )

        demo_keys = sorted(
            [k for k in parent_grp.keys() if k.startswith('demo_')],
            key=lambda k: int(k.split('_')[1]),
        )

    total_demos = len(demo_keys)
    if num_demos > 0:
        n = min(num_demos, total_demos)
    else:
        n = total_demos
    print(
        f'[Demo Loader] Layout={"flat" if is_flat_layout else "robomimic"}, '
        f'loading {n}/{total_demos} demos from {hdf5_path}'
    )

    total_transitions = 0

    for demo_idx in range(n):
        demo_key = demo_keys[demo_idx]
        demo_grp = parent_grp[demo_key]

        if is_flat_layout:
            agentview_rgb = demo_grp['image'][()]
            eye_in_hand_rgb = demo_grp['wrist_image'][()]
            state = demo_grp['state'][()]
            if state.shape[-1] < 8:
                raise ValueError(
                    f'{demo_key}: expected state dim >=8 '
                    f'(ee6 + gripper2), got {state.shape}'
                )
            ee_states = state[:, :6]
            gripper_states = state[:, 6:]
            actions = demo_grp['actions'][()]
            rewards_raw = demo_grp['rewards'][()]
            T = actions.shape[0]
            dones = np.zeros(T, dtype=np.float32)
            dones[-1] = 1.0
        else:
            agentview_rgb = demo_grp['obs/agentview_rgb'][()]
            eye_in_hand_rgb = demo_grp['obs/eye_in_hand_rgb'][()]
            ee_states = demo_grp['obs/ee_states'][()]
            gripper_states = demo_grp['obs/gripper_states'][()]
            actions = demo_grp['actions'][()]
            dones = demo_grp['dones'][()]
            rewards_raw = demo_grp['rewards'][()]

        T = agentview_rgb.shape[0]
        assert T == actions.shape[0], (
            f"Mismatch: {T} obs vs {actions.shape[0]} actions in {demo_key}"
        )

        if T <= query_freq:
            print(f'[Demo Loader] Skipping {demo_key}: '
                  f'T={T} <= query_freq={query_freq}')
            continue

        # Run Pi-0.5 on each timestep to get base actions
        base_actions_all = np.zeros((T, chunk_len, action_dim),
                                    dtype=np.float32)
        vlm_embeddings_all = None
        if use_vlm_embedding:
            vlm_dim = variant.get('vlm_embedding_dim', 2048)
            vlm_embeddings_all = np.zeros((T, vlm_dim), dtype=np.float32)

        print(f'[Demo Loader] {demo_key}: running Pi-0.5 on {T} timesteps...')
        for t in range(T):
            if skip_infer:
                base_actions_all[t] = np.zeros(
                    (chunk_len, action_dim), dtype=np.float32
                )
                if use_vlm_embedding:
                    obs_pi = _hdf5_obs_to_pi_zero_input(
                        agentview_rgb[t], eye_in_hand_rgb[t],
                        ee_states[t], gripper_states[t],
                        task_description,
                        flip=flip_images,
                    )
                    vlm_source = agent_vlm or agent_dp
                    vlm_hs = vlm_source.get_prefix_rep(obs_pi)
                    vlm_hs = vlm_hs[0]
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]
                    vlm_embeddings_all[t] = np.mean(vlm_hs, axis=0)
            else:
                obs_pi = _hdf5_obs_to_pi_zero_input(
                    agentview_rgb[t], eye_in_hand_rgb[t],
                    ee_states[t], gripper_states[t],
                    task_description,
                    flip=flip_images,
                )
                vlm_source = agent_vlm or agent_dp
                return_vlm = use_vlm_embedding and agent_vlm is None
                result = agent_dp.infer(
                    obs_pi, return_vlm_embedding=return_vlm
                )
                base_actions_all[t] = result['actions'][:chunk_len]
                if use_vlm_embedding:
                    if agent_vlm is not None:
                        vlm_hs = vlm_source.get_prefix_rep(obs_pi)
                        vlm_hs = vlm_hs[0]
                    else:
                        vlm_hs = result['vlm_embedding'][0]
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]
                    vlm_embeddings_all[t] = np.mean(vlm_hs, axis=0)

        # Cached frozen-encoder features. Run the encoder on every processed
        # frame in the demo so the replay buffer stores the same
        # ``pixel_features`` slot it would have stored from a live rollout.
        # Without this the demo transitions would miss the key, the cached-
        # feature branch in PixelMultiplexer.encode would not fire, and the
        # encoder would be invoked on the train graph for demo batches —
        # defeating the speed-up.
        # Chunked rather than batched-over-T to avoid (a) DINO OOM spikes on
        # long episodes and (b) per-T-length JIT recompiles.
        pixel_features_all = None
        if cache_pixel_features:
            processed_pixels = np.stack(
                [_process_libero_image(agentview_rgb[t], resize_image, flip_images)
                 for t in range(T)],
                axis=0,
            )[..., np.newaxis]  # (T, H, W, 3, 1)
            feat_chunks = []
            chunk = 32
            for s in range(0, T, chunk):
                feat_chunks.append(np.asarray(
                    agent.compute_pixel_features(processed_pixels[s:s + chunk])
                ))
            pixel_features_all = np.concatenate(feat_chunks, axis=0)  # (T, D)

        # Sliding window transitions (stride=1)
        num_transitions = T - query_freq
        print(f'[Demo Loader] {demo_key}: creating {num_transitions} '
              f'transitions (T={T}, query_freq={query_freq})')

        # Pre-pass: query-level rewards then discounted return-to-go (mc_returns),
        # consumed by offline analysis. gamma is applied at the query-
        # transition level to match the stored per-transition discount.
        gamma_q = discount ** query_freq
        demo_rewards = np.zeros(num_transitions, dtype=np.float32)
        for t in range(num_transitions):
            is_terminal = (t + query_freq >= T - 1)
            if reward_type == 'sparse':
                demo_rewards[t] = 0.0 if is_terminal else -1.0
            else:
                demo_rewards[t] = float(np.sum(
                    rewards_raw[t:t + query_freq].astype(np.float32)))
        mc_returns_arr = np.zeros(num_transitions, dtype=np.float32)
        running = 0.0
        for t in range(num_transitions - 1, -1, -1):
            running = demo_rewards[t] + gamma_q * running
            mc_returns_arr[t] = running

        for t in range(num_transitions):
            t_next = t + query_freq

            obs = _hdf5_obs_to_replay_obs(
                agentview_rgb[t], ee_states[t], gripper_states[t],
                base_actions_all[t], resize_image, add_states,
                use_vlm_embedding,
                vlm_embeddings_all[t] if use_vlm_embedding else None,
                flip=flip_images,
                pixel_features=(
                    pixel_features_all[t] if pixel_features_all is not None
                    else None
                ),
            )
            next_obs = _hdf5_obs_to_replay_obs(
                agentview_rgb[t_next], ee_states[t_next],
                gripper_states[t_next],
                base_actions_all[t_next], resize_image, add_states,
                use_vlm_embedding,
                vlm_embeddings_all[t_next] if use_vlm_embedding else None,
                flip=flip_images,
                pixel_features=(
                    pixel_features_all[t_next] if pixel_features_all is not None
                    else None
                ),
            )

            # Action chunk for this window
            demo_actions_chunk = actions[t:t + query_freq]

            if predict_a_exec:
                stored_actions = demo_actions_chunk.astype(np.float32)
            else:
                # delta = (a_demo - base_slice) / alpha
                base_slice = base_actions_all[t, :query_freq]
                stored_actions = (
                    (demo_actions_chunk - base_slice) / residual_alpha
                ).astype(np.float32)

            # Next actions: aligned with next_obs at t_next = t + query_freq
            if t_next < num_transitions:
                next_demo_chunk = actions[t_next:t_next + query_freq]
                if predict_a_exec:
                    next_stored_actions = next_demo_chunk.astype(np.float32)
                else:
                    next_base_slice = base_actions_all[t_next, :query_freq]
                    next_stored_actions = (
                        (next_demo_chunk - next_base_slice) / residual_alpha
                    ).astype(np.float32)
            else:
                next_stored_actions = stored_actions

            # Terminal: last valid window in this demo
            is_terminal = (t_next >= T - 1)

            if reward_type == 'sparse':
                reward = 0.0 if is_terminal else -1.0
                mask = 0.0 if is_terminal else 1.0
            else:
                reward = float(np.sum(
                    rewards_raw[t:t + query_freq].astype(np.float32)
                ))
                mask = 0.0 if is_terminal else 1.0

            insert_dict = dict(
                observations=obs,
                next_observations=next_obs,
                actions=stored_actions,
                next_actions=next_stored_actions,
                rewards=reward,
                masks=mask,
                discount=discount ** query_freq,
                success_flag=1.0,
                old_log_probs=0.0,
                mc_returns=mc_returns_arr[t],
            )
            replay_buffer.insert(insert_dict)

        replay_buffer.increment_traj_counter()
        total_transitions += num_transitions

        # Free large per-demo arrays to avoid OOM
        del agentview_rgb, eye_in_hand_rgb, ee_states, gripper_states
        del actions, dones, base_actions_all
        if vlm_embeddings_all is not None:
            del vlm_embeddings_all
        if pixel_features_all is not None:
            del pixel_features_all

        print(f'[Demo Loader] {demo_key}: done. '
              f'Buffer size: {len(replay_buffer)}')

    f.close()
    print(f'[Demo Loader] Loaded {n} demos, '
          f'{total_transitions} total transitions, '
          f'buffer size: {len(replay_buffer)}')
    return replay_buffer


def build_and_save_demo_buffer(
    hdf5_path: str,
    save_path: str,
    agent_dp,
    variant,
    num_demos: int = -1,
    task_description: Optional[str] = None,
) -> ReplayBuffer:
    """Build a demo replay buffer from HDF5 and save to disk.

    Convenience wrapper that creates a ReplayBuffer, populates it via
    load_libero_demos_to_buffer, and saves the result for fast restoring
    on subsequent runs.

    Args:
        hdf5_path: Path to LIBERO HDF5 demo file.
        save_path: Path to save the replay buffer pickle.
        agent_dp: Frozen Pi-0.5 policy.
        variant: Training config.
        num_demos: Number of demos to load (-1 = all).
        task_description: Language instruction override.

    Returns:
        The populated and saved ReplayBuffer.
    """
    from training.train_sim import DummyEnvResidual

    dummy_env = DummyEnvResidual(variant)

    # Estimate capacity: ~300 timesteps per demo, stride=1
    estimated_demos = num_demos if num_demos > 0 else 50
    estimated_capacity = estimated_demos * 300
    replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(estimated_capacity),
    )
    replay_buffer.seed(variant.seed)

    load_libero_demos_to_buffer(
        hdf5_path=hdf5_path,
        replay_buffer=replay_buffer,
        agent_dp=agent_dp,
        variant=variant,
        num_demos=num_demos,
        task_description=task_description,
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    replay_buffer.save(save_path)
    print(f'[Demo Loader] Buffer saved to {save_path}')

    return replay_buffer
