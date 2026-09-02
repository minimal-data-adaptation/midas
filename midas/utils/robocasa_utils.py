"""RoboCasa demo data loader for replay buffer pre-seeding.

Loads demonstration trajectories from RoboCasa Groot/LeRobot datasets,
runs Pi-0.5 inference to compute base actions for each observation, and
populates a ReplayBuffer with sliding-window transitions (stride=1)
suitable for residual RL training.

Groot/LeRobot format:
    data/episode_chunk_{chunk}/episode_{id}.parquet   state/action per timestep
    videos/episode_chunk_{chunk}/episode_{id}_{cam}.mp4  video per camera
    meta/modality.json                                 column layout metadata
    meta/tasks.jsonl                                   task descriptions
    meta/info.json                                     dataset info (chunks_size)
    extras/episode_{id:06d}/ep_meta.json               scene metadata
"""

import json
import os
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import PIL.Image
from openpi_client import image_tools

from midas.data import ReplayBuffer


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_modality_metadata(dataset_path: pathlib.Path) -> dict:
    """Load and parse meta/modality.json for state/action column layout.

    Returns the raw parsed dict with keys: "state", "action", "video",
    "annotation".  Each sub-dict maps component name -> metadata including
    ``start`` and ``end`` indices into the parquet's concatenated vector.
    """
    modality_path = dataset_path / "meta" / "modality.json"
    with open(modality_path) as f:
        return json.load(f)


def _load_dataset_info(dataset_path: pathlib.Path) -> dict:
    """Load meta/info.json for dataset-level metadata (chunks_size, fps, etc.)."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def _load_tasks(dataset_path: pathlib.Path) -> pd.DataFrame:
    """Load meta/tasks.jsonl into a DataFrame indexed by task_index."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    with open(tasks_path) as f:
        tasks = [json.loads(line) for line in f]
    df = pd.DataFrame(tasks)
    return df.set_index("task_index")


def _build_reorder_indices(
    modality_meta: dict,
    modality: str,
    openpi_order: list[str],
) -> list[int]:
    """Build column reorder indices from Groot order to OpenPI order.

    Args:
        modality_meta: Parsed modality.json dict.
        modality: "state" or "action".
        openpi_order: Component names in OpenPI concatenation order.

    Returns:
        List of column indices to apply: ``reordered = raw[:, indices]``.
    """
    meta = modality_meta[modality]
    reorder = []
    for comp in openpi_order:
        entry = meta[comp]
        reorder.extend(range(entry["start"], entry["end"]))
    return reorder


# Component concatenation orders matching GrootOpenpiSingleDataset.__getitem__
# (groot_openpi_dataset.py lines 201-214).
OPENPI_STATE_ORDER = [
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "base_position",
    "base_rotation",
    "gripper_qpos",
]

OPENPI_ACTION_ORDER = [
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
    "base_motion",
    "control_mode",
]


# ---------------------------------------------------------------------------
# Episode data loading
# ---------------------------------------------------------------------------

def _load_video_frames(video_path: pathlib.Path) -> np.ndarray:
    """Decode every MP4 frame as RGB using OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    return np.stack(frames)

def _load_episode_data(
    dataset_path: pathlib.Path,
    episode_id: int,
    modality_meta: dict,
    dataset_info: dict,
    tasks_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load a single episode's images, states, actions, and prompt.

    Reads raw parquet + video files and reorders state/action from Groot
    order to OpenPI order.  Path patterns are read from info.json to
    handle varying dataset layouts (LeRobot v2 vs older formats).

    Args:
        dataset_path: Root path of the Groot dataset.
        episode_id: Integer episode index.
        modality_meta: Parsed modality.json dict.
        dataset_info: Parsed info.json dict (contains data_path,
            video_path, chunks_size).
        tasks_df: DataFrame from meta/tasks.jsonl (indexed by task_index).

    Returns:
        Tuple of (agentview_frames, wrist_frames, states, actions, prompt):
            agentview_frames: (T, H, W, 3) uint8 RGB
            wrist_frames: (T, H, W, 3) uint8 RGB
            states: (T, 16) float64 in OpenPI order
            actions: (T, 12) float64 in OpenPI order
            prompt: task description string
    """
    chunks_size = dataset_info.get("chunks_size", 1000)
    data_path_pattern = dataset_info["data_path"]
    video_path_pattern = dataset_info["video_path"]
    chunk_idx = episode_id // chunks_size

    # --- Parquet ---
    parquet_rel = data_path_pattern.format(
        episode_chunk=chunk_idx, episode_index=episode_id
    )
    parquet_path = dataset_path / parquet_rel
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet not found: {parquet_path}. "
            f"Check episode_id={episode_id}, chunks_size={chunks_size}, "
            f"data_path pattern='{data_path_pattern}'."
        )
    df = pd.read_parquet(parquet_path)
    T = len(df)

    # --- State: reorder Groot → OpenPI ---
    state_reorder = _build_reorder_indices(
        modality_meta, "state", OPENPI_STATE_ORDER
    )
    raw_states = np.stack(df["observation.state"].values)  # (T, 16) Groot order
    assert raw_states.shape == (T, len(state_reorder)), (
        f"State shape mismatch: expected (T, {len(state_reorder)}), "
        f"got {raw_states.shape}"
    )
    states = raw_states[:, state_reorder]  # (T, 16) OpenPI order

    # --- Action: reorder Groot → OpenPI ---
    action_reorder = _build_reorder_indices(
        modality_meta, "action", OPENPI_ACTION_ORDER
    )
    raw_actions = np.stack(df["action"].values)  # (T, 12) Groot order
    assert raw_actions.shape == (T, len(action_reorder)), (
        f"Action shape mismatch: expected (T, {len(action_reorder)}), "
        f"got {raw_actions.shape}"
    )
    actions = raw_actions[:, action_reorder]  # (T, 12) OpenPI order

    # --- Video frames (decord → RGB) ---
    # Resolve video keys from modality.json: short name → original_key
    video_meta = modality_meta.get("video", {})

    agentview_video_key = video_meta.get(
        "robot0_agentview_left", {}
    ).get("original_key", "observation.images.robot0_agentview_left")
    agentview_rel = video_path_pattern.format(
        episode_chunk=chunk_idx,
        episode_index=episode_id,
        video_key=agentview_video_key,
    )
    agentview_path = dataset_path / agentview_rel
    if not agentview_path.exists():
        raise FileNotFoundError(
            f"Agentview video not found: {agentview_path}"
        )
    agentview_frames = _load_video_frames(
        agentview_path
    )  # (T_vid, H, W, 3) RGB

    wrist_video_key = video_meta.get(
        "robot0_eye_in_hand", {}
    ).get("original_key", "observation.images.robot0_eye_in_hand")
    wrist_rel = video_path_pattern.format(
        episode_chunk=chunk_idx,
        episode_index=episode_id,
        video_key=wrist_video_key,
    )
    wrist_path = dataset_path / wrist_rel
    if not wrist_path.exists():
        raise FileNotFoundError(
            f"Wrist video not found: {wrist_path}"
        )
    wrist_frames = _load_video_frames(
        wrist_path
    )  # (T_vid, H, W, 3) RGB

    # Validate frame counts match parquet length
    assert agentview_frames.shape[0] == T, (
        f"Agentview frame count mismatch: video has {agentview_frames.shape[0]} "
        f"frames but parquet has {T} rows for episode {episode_id}"
    )
    assert wrist_frames.shape[0] == T, (
        f"Wrist frame count mismatch: video has {wrist_frames.shape[0]} "
        f"frames but parquet has {T} rows for episode {episode_id}"
    )

    # --- Prompt ---
    # Determine the annotation column name from modality metadata
    annotation_meta = modality_meta.get("annotation", {})
    annotation_key = "human.task_description"
    if annotation_key in annotation_meta:
        original_key = annotation_meta[annotation_key].get("original_key")
        if original_key is None:
            original_key = f"annotation.{annotation_key}"
    else:
        # Fallback: try the first annotation key
        if annotation_meta:
            first_key = next(iter(annotation_meta))
            original_key = annotation_meta[first_key].get(
                "original_key", f"annotation.{first_key}"
            )
        else:
            original_key = None

    if original_key is not None and original_key in df.columns:
        task_index = df[original_key].iloc[0]
        if hasattr(task_index, 'item'):
            task_index = task_index.item()
        prompt = tasks_df.loc[task_index]["task"]
    else:
        prompt = ""
        print(
            f"[RoboCasa Demo Loader] WARNING: Could not find annotation column "
            f"for episode {episode_id}, using empty prompt."
        )

    return agentview_frames, wrist_frames, states, actions, str(prompt)


# ---------------------------------------------------------------------------
# Episode resolution
# ---------------------------------------------------------------------------

def get_scene_filtered_demos(
    dataset_path: pathlib.Path,
    layout_and_style_ids,
    fixture_refs=None,
    object_categories=None,
    episode_ids=None,
) -> list[int]:
    """Resolve scene-matched episodes without depending on OpenPI internals."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            episode_ids_in_dataset = [
                json.loads(line)["episode_index"] for line in f
            ]
    else:
        manifest_path = dataset_path / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        episode_ids_in_dataset = [
            episode["episode_id"] for episode in manifest["episodes"]
        ]

    allowed_scenes = set(map(tuple, layout_and_style_ids))
    allowed_ids = set(episode_ids) if episode_ids is not None else None
    filtered = []

    for episode_id in episode_ids_in_dataset:
        if allowed_ids is not None and episode_id not in allowed_ids:
            continue

        meta_path = (
            dataset_path / "extras" / f"episode_{episode_id:06d}"
            / "ep_meta.json"
        )
        with open(meta_path) as f:
            ep_meta = json.load(f)

        if (ep_meta["layout_id"], ep_meta["style_id"]) not in allowed_scenes:
            continue
        if fixture_refs is not None and any(
            ep_meta.get("fixture_refs", {}).get(key) != value
            for key, value in fixture_refs.items()
        ):
            continue
        if object_categories is not None:
            object_cfgs = ep_meta.get("object_cfgs", [])
            main_category = (
                object_cfgs[0].get("info", {}).get("cat")
                if object_cfgs else None
            )
            if main_category not in object_categories:
                continue

        filtered.append(episode_id)

    filtered.sort()
    if not filtered:
        raise ValueError(
            f"No episodes match filters in dataset at {dataset_path}. "
            f"layout_and_style_ids={layout_and_style_ids}, "
            f"fixture_refs={fixture_refs}, "
            f"object_categories={object_categories}, "
            f"episode_ids={episode_ids}"
        )
    return filtered

def resolve_robocasa_demo_episode_ids(
    dataset_path: pathlib.Path,
    pi_data_config=None,
    eval_controller=None,
    episode_ids: Optional[list[int]] = None,
) -> list[int]:
    """Resolve which episodes to load for demo pre-seeding.

    Resolution priority:
    1. eval_controller.last_reset_info["episode_id"] (exact replay match)
    2. Explicit episode_ids argument
    3. pi_data_config eval pool filters via get_scene_filtered_demos()
    4. Raise ValueError

    Args:
        dataset_path: Root path of the Groot dataset.
        pi_data_config: LeRobotRobocasaDataConfig with eval pool filters.
        eval_controller: RoboCasaEvalResetController (optional).
        episode_ids: Explicit episode ID list (optional).

    Returns:
        Sorted list of episode IDs to load.
    """
    # Priority 1: exact episode from eval controller
    if eval_controller is not None:
        reset_info = eval_controller.last_reset_info
        if reset_info is not None and "episode_id" in reset_info:
            ep_id = reset_info["episode_id"]
            print(f"[RoboCasa Demo Loader] Using episode {ep_id} "
                  f"from eval controller last_reset_info")
            return [ep_id]

    # Priority 2: explicit episode IDs
    if episode_ids is not None:
        result = sorted(episode_ids)
        print(f"[RoboCasa Demo Loader] Using {len(result)} explicit episode IDs: "
              f"{result}")
        return result

    # Priority 3: eval pool filters from pi_data_config
    if pi_data_config is not None:
        layout_and_style_ids = getattr(
            pi_data_config, 'layout_and_style_ids', None
        )
        if layout_and_style_ids is None:
            raise ValueError(
                "pi_data_config.layout_and_style_ids is required for "
                "scene-filtered demo resolution."
            )

        eval_episode_ids = getattr(
            pi_data_config, 'eval_pool_episode_ids', None
        )
        eval_fixture_refs = getattr(
            pi_data_config, 'eval_pool_fixture_refs', None
        )
        eval_object_categories = getattr(
            pi_data_config, 'eval_pool_object_categories', None
        )

        result = get_scene_filtered_demos(
            dataset_path,
            layout_and_style_ids,
            fixture_refs=eval_fixture_refs,
            object_categories=eval_object_categories,
            episode_ids=eval_episode_ids,
        )
        print(f"[RoboCasa Demo Loader] Resolved {len(result)} episodes via "
              f"eval pool filters: {result}")
        return result

    raise ValueError(
        "No episode resolution method provided. Supply eval_controller, "
        "explicit episode_ids, or pi_data_config with eval pool filters."
    )


# ---------------------------------------------------------------------------
# Observation conversion helpers
# ---------------------------------------------------------------------------

def _groot_obs_to_pi_zero_input(
    agentview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    task_description: str,
) -> dict:
    """Convert dataset observation arrays to Pi-0.5 input format.

    Mirrors obs_to_pi_zero_input() in train_utils_sim_residual.py (lines
    92-112) for robocasa.  No spatial flip is applied (unlike LIBERO).

    Args:
        agentview_rgb: (H, W, 3) uint8 RGB image.
        wrist_rgb: (H, W, 3) uint8 RGB wrist image.
        state: (16,) float — already in OpenPI order.
        task_description: Language instruction.

    Returns:
        Dict matching Pi-0.5 input schema.
    """
    img = np.ascontiguousarray(agentview_rgb)
    wrist_img = np.ascontiguousarray(wrist_rgb)

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 224, 224)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _process_groot_image(
    agentview_rgb: np.ndarray,
    resize_image: int,
) -> np.ndarray:
    """Apply demo→replay-buffer pixel preprocessing (resize only — no flip).

    Factored out so the feature-precompute pass in
    ``load_robocasa_demos_to_buffer`` and the obs-builder
    ``_groot_obs_to_replay_obs`` consume identical pixel arrays — otherwise
    cached features could disagree with what the encoder would produce
    from the stored ``pixels`` slot.
    """
    curr_image = np.ascontiguousarray(agentview_rgb)
    if resize_image > 0:
        curr_image = np.array(
            PIL.Image.fromarray(curr_image).resize(
                (resize_image, resize_image)
            )
        )
    return curr_image


def _groot_obs_to_replay_obs(
    agentview_rgb: np.ndarray,
    state: np.ndarray,
    base_action_chunk: np.ndarray,
    resize_image: int,
    add_states: bool,
    use_vlm_embedding: bool = False,
    vlm_embedding: Optional[np.ndarray] = None,
    pixel_features: Optional[np.ndarray] = None,
) -> dict:
    """Convert dataset observation arrays to replay buffer format.

    Builds the same dict structure as add_online_data_to_buffer_residual
    in train_utils_sim_residual.py (lines 427-488).

    Args:
        agentview_rgb: (H, W, 3) uint8 RGB image.
        state: (16,) float — in OpenPI order.
        base_action_chunk: (chunk_len, action_dim) float32 — full cached
            chunk from Pi-0.5 (not just query_freq slice).
        resize_image: Target image resolution (e.g. 128).
        add_states: Whether to include proprioceptive state.
        use_vlm_embedding: Whether to include VLM embedding.
        vlm_embedding: (W,) mean-pooled VLM hidden state, required if
            use_vlm_embedding is True.

    Returns:
        Dict with 'pixels', 'base_action', optionally 'state' and
        'vlm_embedding', each with trailing singleton dimension.
    """
    curr_image = _process_groot_image(agentview_rgb, resize_image)

    obs_dict = {
        'pixels': curr_image[..., np.newaxis],
        'base_action': base_action_chunk[..., np.newaxis],
    }

    if add_states:
        obs_dict['state'] = state[..., np.newaxis]

    if use_vlm_embedding:
        assert vlm_embedding is not None, (
            "vlm_embedding required when use_vlm_embedding=True"
        )
        obs_dict['vlm_embedding'] = vlm_embedding[..., np.newaxis]

    if pixel_features is not None:
        obs_dict['pixel_features'] = pixel_features[..., np.newaxis]

    return obs_dict


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_robocasa_demos_to_buffer(
    dataset_path: pathlib.Path | str,
    replay_buffer: ReplayBuffer,
    agent_dp,
    variant,
    episode_ids: Optional[list[int]] = None,
    num_demos: int = -1,
    task_description: Optional[str] = None,
    pi_data_config=None,
    eval_controller=None,
    agent_vlm=None,
    agent=None,
) -> ReplayBuffer:
    """Load RoboCasa demo data into a replay buffer using sliding windows.

    For each episode of T timesteps with query_freq=Q, produces T-Q
    overlapping transitions (stride=1).  Each transition stores:
      - obs at time t, next_obs at time t+Q
      - action chunk actions[t:t+Q]
      - base_action from Pi-0.5 inference (full chunk_len chunk)

    Args:
        dataset_path: Root path of the Groot/LeRobot dataset.
        replay_buffer: Target replay buffer to populate.
        agent_dp: Frozen Pi-0.5 policy for base action generation.
        variant: Training config (attrdict).
        episode_ids: Explicit episode IDs to load (optional).
        num_demos: Number of episodes to load (-1 = all matching).
        task_description: Language instruction override.
        pi_data_config: LeRobotRobocasaDataConfig with eval pool filters.
        eval_controller: RoboCasaEvalResetController (optional).

    Returns:
        The populated replay_buffer.
    """
    dataset_path = pathlib.Path(dataset_path)

    # Load dataset metadata
    modality_meta = _load_modality_metadata(dataset_path)
    dataset_info = _load_dataset_info(dataset_path)
    tasks_df = _load_tasks(dataset_path)

    # Resolve episodes
    resolved_ids = resolve_robocasa_demo_episode_ids(
        dataset_path, pi_data_config, eval_controller, episode_ids
    )
    if num_demos > 0:
        resolved_ids = resolved_ids[:num_demos]

    total_episodes = len(resolved_ids)
    print(f"[RoboCasa Demo Loader] Loading {total_episodes} episodes: "
          f"{resolved_ids}")

    # Extract variant config
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
    cache_pixel_features = (
        variant.get('freeze_vision_encoder', False) and not use_vlm_embedding
    )
    actor_pop_base_actions = variant.get('actor_pop_base_actions', False)
    critic_pop_base_actions = variant.get('critic_pop_base_actions', True)
    skip_infer = actor_pop_base_actions and critic_pop_base_actions
    if cache_pixel_features:
        assert agent is not None and getattr(agent, 'pixel_features_dim', None), (
            "load_robocasa_demos_to_buffer requires agent= with a populated "
            "pixel_features_dim when freeze_vision_encoder=True so demo "
            "transitions match the cached-feature obs structure."
        )

    total_transitions = 0

    for ep_idx, episode_id in enumerate(resolved_ids):
        agentview_frames, wrist_frames, states, actions, prompt = (
            _load_episode_data(
                dataset_path, episode_id, modality_meta, dataset_info,
                tasks_df,
            )
        )

        # Use override task description if provided
        ep_prompt = task_description if task_description is not None else prompt

        T = agentview_frames.shape[0]
        assert T == actions.shape[0], (
            f"Mismatch: {T} obs vs {actions.shape[0]} actions in "
            f"episode {episode_id}"
        )

        if T <= query_freq:
            print(f"[RoboCasa Demo Loader] Skipping episode {episode_id}: "
                  f"T={T} <= query_freq={query_freq}")
            continue

        # Run Pi-0.5 on each timestep to get base actions
        base_actions_all = np.zeros(
            (T, chunk_len, action_dim), dtype=np.float32
        )
        vlm_embeddings_all = None
        if use_vlm_embedding:
            vlm_dim = variant.get('vlm_embedding_dim', 2048)
            vlm_embeddings_all = np.zeros((T, vlm_dim), dtype=np.float32)

        print(f"[RoboCasa Demo Loader] Episode {episode_id} "
              f"({ep_idx + 1}/{total_episodes}): "
              f"running Pi-0.5 on {T} timesteps...")

        for t in range(T):
            if skip_infer:
                base_actions_all[t] = np.zeros(
                    (chunk_len, action_dim), dtype=np.float32
                )
                if use_vlm_embedding:
                    obs_pi = _groot_obs_to_pi_zero_input(
                        agentview_frames[t], wrist_frames[t],
                        states[t], ep_prompt,
                    )
                    vlm_source = agent_vlm or agent_dp
                    vlm_hs = vlm_source.get_prefix_rep(obs_pi)
                    vlm_hs = vlm_hs[0]
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]
                    vlm_embeddings_all[t] = np.mean(vlm_hs, axis=0)
            else:
                obs_pi = _groot_obs_to_pi_zero_input(
                    agentview_frames[t], wrist_frames[t],
                    states[t], ep_prompt,
                )
                vlm_source = agent_vlm or agent_dp
                return_vlm = use_vlm_embedding and agent_vlm is None
                result = agent_dp.infer(
                    obs_pi, return_vlm_embedding=return_vlm
                )
                # The frozen Pi-0.5 policy emits the full RoboCasa action,
                # while MIDAS may intentionally train a leading subset via
                # --action_dim. Keep demo observations consistent with the
                # residual actor/replay-buffer shape used by online rollouts.
                base_actions_all[t] = result['actions'][
                    :chunk_len, :action_dim
                ]
                if use_vlm_embedding:
                    if agent_vlm is not None:
                        vlm_hs = vlm_source.get_prefix_rep(obs_pi)
                        vlm_hs = vlm_hs[0]
                    else:
                        vlm_hs = result['vlm_embedding'][0]
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]
                    vlm_embeddings_all[t] = np.mean(vlm_hs, axis=0)

        # See load_libero_demos_to_buffer for rationale: demos must store
        # cached frozen-encoder features under ``pixel_features`` so the
        # train graph's cached-feature branch fires for demo batches too.
        # Chunked to avoid DINO OOM and per-T JIT recompiles.
        pixel_features_all = None
        if cache_pixel_features:
            processed_pixels = np.stack(
                [_process_groot_image(agentview_frames[t], resize_image)
                 for t in range(T)],
                axis=0,
            )[..., np.newaxis]  # (T, H, W, 3, 1)
            feat_chunks = []
            chunk = 32
            for s in range(0, T, chunk):
                feat_chunks.append(np.asarray(
                    agent.compute_pixel_features(processed_pixels[s:s + chunk])
                ))
            pixel_features_all = np.concatenate(feat_chunks, axis=0)

        # RoboCasa demonstrations store the full 12-D policy action. When the
        # residual policy controls only a leading subset via --action_dim,
        # store the same reduced action shape used by online transitions.
        if actions.shape[-1] > action_dim:
            actions = actions[:, :action_dim]

        # Sliding window transitions (stride=1)
        num_transitions = T - query_freq
        print(f"[RoboCasa Demo Loader] Episode {episode_id}: "
              f"creating {num_transitions} transitions "
              f"(T={T}, query_freq={query_freq})")

        # Pre-pass: query-level rewards then discounted return-to-go (mc_returns)
        # for offline analysis; gamma applied at the query-transition level.
        gamma_q = discount ** query_freq
        demo_rewards = np.zeros(num_transitions, dtype=np.float32)
        for t in range(num_transitions):
            is_terminal = (t + query_freq >= T - 1)
            if reward_type == 'sparse':
                demo_rewards[t] = 0.0 if is_terminal else -1.0
            else:
                raise ValueError(f"Unsupported reward_type: {reward_type}")
        mc_returns_arr = np.zeros(num_transitions, dtype=np.float32)
        running = 0.0
        for t in range(num_transitions - 1, -1, -1):
            running = demo_rewards[t] + gamma_q * running
            mc_returns_arr[t] = running

        for t in range(num_transitions):
            t_next = t + query_freq

            obs = _groot_obs_to_replay_obs(
                agentview_frames[t], states[t],
                base_actions_all[t], resize_image, add_states,
                use_vlm_embedding,
                vlm_embeddings_all[t] if use_vlm_embedding else None,
                pixel_features=(
                    pixel_features_all[t] if pixel_features_all is not None
                    else None
                ),
            )
            next_obs = _groot_obs_to_replay_obs(
                agentview_frames[t_next], states[t_next],
                base_actions_all[t_next], resize_image, add_states,
                use_vlm_embedding,
                vlm_embeddings_all[t_next] if use_vlm_embedding else None,
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

            # Next actions (for t+1 window)
            if t + 1 < num_transitions:
                next_demo_chunk = actions[t + 1:t + 1 + query_freq]
                if predict_a_exec:
                    next_stored_actions = next_demo_chunk.astype(np.float32)
                else:
                    next_base_slice = base_actions_all[t + 1, :query_freq]
                    next_stored_actions = (
                        (next_demo_chunk - next_base_slice) / residual_alpha
                    ).astype(np.float32)
            else:
                next_stored_actions = stored_actions

            # Terminal: last valid window in this episode
            is_terminal = (t_next >= T - 1)

            if reward_type == 'sparse':
                reward = 0.0 if is_terminal else -1.0
                mask = 0.0 if is_terminal else 1.0
            else:
                raise ValueError(f"Unsupported reward_type: {reward_type}")

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

        # Free large per-episode arrays to avoid OOM
        del agentview_frames, wrist_frames, states, actions, base_actions_all
        if vlm_embeddings_all is not None:
            del vlm_embeddings_all
        if pixel_features_all is not None:
            del pixel_features_all

        print(f"[RoboCasa Demo Loader] Episode {episode_id}: done. "
              f"Buffer size: {len(replay_buffer)}")

    print(f"[RoboCasa Demo Loader] Loaded {total_episodes} episodes, "
          f"{total_transitions} total transitions, "
          f"buffer size: {len(replay_buffer)}")
    return replay_buffer


def load_robocasa_hdf5_demos_to_buffer(
    hdf5_path: str,
    replay_buffer: ReplayBuffer,
    agent_dp,
    variant,
    num_demos: int = -1,
    task_description: Optional[str] = None,
    agent_vlm=None,
) -> ReplayBuffer:
    """Load RoboCasa demo data from HDF5 into a replay buffer.

    Reads flat-layout HDF5 files (episode_NNNNNNN/{image, wrist_image, state,
    actions, rewards}) and populates a replay buffer with sliding-window
    transitions (stride=1).  Uses the robocasa-correct 16-dim state and
    reuses ``_groot_obs_to_pi_zero_input`` / ``_groot_obs_to_replay_obs``.

    Args:
        hdf5_path: Path to the HDF5 file.
        replay_buffer: Target replay buffer to populate.
        agent_dp: Frozen Pi-0.5 policy for base action generation.
        variant: Training config (attrdict).
        num_demos: Number of episodes to load (-1 = all).
        task_description: Language instruction override.
        agent_vlm: Optional separate VLM encoder.

    Returns:
        The populated replay_buffer.
    """
    import h5py

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
    actor_pop_base_actions = variant.get('actor_pop_base_actions', False)
    critic_pop_base_actions = variant.get('critic_pop_base_actions', True)
    skip_infer = actor_pop_base_actions and critic_pop_base_actions

    f = h5py.File(hdf5_path, 'r')

    # Discover episodes
    demo_keys = sorted(
        [k for k in f.keys() if k.startswith('episode_')],
        key=lambda k: int(k.split('_')[1]),
    )

    # Task description: from arg, then HDF5 metadata, then variant
    if task_description is None:
        if 'metadata' in f and 'task_description' in f['metadata'].attrs:
            task_description = str(f['metadata'].attrs['task_description'])
        else:
            task_description = str(variant.get('task_description', ''))
        print(f'[RoboCasa HDF5 Demo Loader] Task description: '
              f'{task_description}')
    if not task_description:
        raise ValueError(
            f'Empty task_description for {hdf5_path}; pi0.5 requires a '
            'non-empty language prompt.'
        )

    total_demos = len(demo_keys)
    n = min(num_demos, total_demos) if num_demos > 0 else total_demos
    print(f'[RoboCasa HDF5 Demo Loader] Loading {n}/{total_demos} demos '
          f'from {hdf5_path}')

    total_transitions = 0

    for demo_idx in range(n):
        demo_key = demo_keys[demo_idx]
        demo_grp = f[demo_key]

        # Load arrays
        agentview_frames = demo_grp['image'][()]        # (T, H, W, 3)
        wrist_frames = demo_grp['wrist_image'][()]      # (T, H, W, 3)
        states = demo_grp['state'][()]                  # (T, 16)
        actions = demo_grp['actions'][()]               # (T, action_dim)
        rewards_raw = demo_grp['rewards'][()]            # (T,)

        T = agentview_frames.shape[0]
        assert T == actions.shape[0], (
            f"Mismatch: {T} obs vs {actions.shape[0]} actions in {demo_key}"
        )

        if T <= query_freq:
            print(f'[RoboCasa HDF5 Demo Loader] Skipping {demo_key}: '
                  f'T={T} <= query_freq={query_freq}')
            continue

        # Run Pi-0.5 on each timestep to get base actions
        base_actions_all = np.zeros(
            (T, chunk_len, action_dim), dtype=np.float32
        )
        vlm_embeddings_all = None
        if use_vlm_embedding:
            vlm_dim = variant.get('vlm_embedding_dim', 2048)
            vlm_embeddings_all = np.zeros((T, vlm_dim), dtype=np.float32)

        print(f'[RoboCasa HDF5 Demo Loader] {demo_key} '
              f'({demo_idx + 1}/{n}): '
              f'running Pi-0.5 on {T} timesteps...')

        for t in range(T):
            if skip_infer:
                base_actions_all[t] = np.zeros(
                    (chunk_len, action_dim), dtype=np.float32
                )
                if use_vlm_embedding:
                    obs_pi = _groot_obs_to_pi_zero_input(
                        agentview_frames[t], wrist_frames[t],
                        states[t], task_description,
                    )
                    vlm_source = agent_vlm or agent_dp
                    vlm_hs = vlm_source.get_prefix_rep(obs_pi)
                    vlm_hs = vlm_hs[0]
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]
                    vlm_embeddings_all[t] = np.mean(vlm_hs, axis=0)
            else:
                obs_pi = _groot_obs_to_pi_zero_input(
                    agentview_frames[t], wrist_frames[t],
                    states[t], task_description,
                )
                vlm_source = agent_vlm or agent_dp
                return_vlm = use_vlm_embedding and agent_vlm is None
                result = agent_dp.infer(
                    obs_pi, return_vlm_embedding=return_vlm
                )
                base_actions_all[t] = result['actions'][:chunk_len, :action_dim]
                if use_vlm_embedding:
                    if agent_vlm is not None:
                        vlm_hs = vlm_source.get_prefix_rep(obs_pi)
                        vlm_hs = vlm_hs[0]
                    else:
                        vlm_hs = result['vlm_embedding'][0]
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]
                    vlm_embeddings_all[t] = np.mean(vlm_hs, axis=0)

        # Truncate demo actions to residual action_dim if needed
        if actions.shape[-1] > action_dim:
            actions = actions[:, :action_dim]

        # Sliding window transitions (stride=1)
        num_transitions = T - query_freq
        print(f'[RoboCasa HDF5 Demo Loader] {demo_key}: '
              f'creating {num_transitions} transitions '
              f'(T={T}, query_freq={query_freq})')

        for t in range(num_transitions):
            t_next = t + query_freq

            obs = _groot_obs_to_replay_obs(
                agentview_frames[t], states[t],
                base_actions_all[t], resize_image, add_states,
                use_vlm_embedding,
                vlm_embeddings_all[t] if use_vlm_embedding else None,
            )
            next_obs = _groot_obs_to_replay_obs(
                agentview_frames[t_next], states[t_next],
                base_actions_all[t_next], resize_image, add_states,
                use_vlm_embedding,
                vlm_embeddings_all[t_next] if use_vlm_embedding else None,
            )

            # Action chunk for this window
            demo_actions_chunk = actions[t:t + query_freq]

            if predict_a_exec:
                stored_actions = demo_actions_chunk.astype(np.float32)
            else:
                base_slice = base_actions_all[t, :query_freq]
                stored_actions = (
                    (demo_actions_chunk - base_slice) / residual_alpha
                ).astype(np.float32)

            # Next actions (for t+1 window)
            if t + 1 < num_transitions:
                next_demo_chunk = actions[t + 1:t + 1 + query_freq]
                if predict_a_exec:
                    next_stored_actions = next_demo_chunk.astype(np.float32)
                else:
                    next_base_slice = base_actions_all[t + 1, :query_freq]
                    next_stored_actions = (
                        (next_demo_chunk - next_base_slice) / residual_alpha
                    ).astype(np.float32)
            else:
                next_stored_actions = stored_actions

            # Terminal: last valid window in this episode
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
            )
            replay_buffer.insert(insert_dict)

        replay_buffer.increment_traj_counter()
        total_transitions += num_transitions

        # Free large per-episode arrays
        del agentview_frames, wrist_frames, states, actions, base_actions_all
        if vlm_embeddings_all is not None:
            del vlm_embeddings_all

        print(f'[RoboCasa HDF5 Demo Loader] {demo_key}: done. '
              f'Buffer size: {len(replay_buffer)}')

    f.close()
    print(f'[RoboCasa HDF5 Demo Loader] Loaded {n} demos, '
          f'{total_transitions} total transitions, '
          f'buffer size: {len(replay_buffer)}')
    return replay_buffer


def build_and_save_demo_buffer(
    dataset_path: pathlib.Path | str,
    save_path: str,
    agent_dp,
    variant,
    episode_ids: Optional[list[int]] = None,
    num_demos: int = -1,
    task_description: Optional[str] = None,
    pi_data_config=None,
    eval_controller=None,
) -> ReplayBuffer:
    """Build a demo replay buffer from Groot dataset and save to disk.

    Convenience wrapper that creates a ReplayBuffer, populates it via
    load_robocasa_demos_to_buffer, and saves the result for fast restoring
    on subsequent runs.

    Args:
        dataset_path: Root path of the Groot/LeRobot dataset.
        save_path: Path to save the replay buffer pickle.
        agent_dp: Frozen Pi-0.5 policy.
        variant: Training config.
        episode_ids: Explicit episode IDs to load.
        num_demos: Number of episodes to load (-1 = all matching).
        task_description: Language instruction override.
        pi_data_config: LeRobotRobocasaDataConfig with eval pool filters.
        eval_controller: RoboCasaEvalResetController (optional).

    Returns:
        The populated and saved ReplayBuffer.
    """
    from training.train_sim import DummyEnvResidual

    dummy_env = DummyEnvResidual(variant)

    # Estimate capacity: ~300 timesteps per episode, stride=1
    estimated_episodes = num_demos if num_demos > 0 else 50
    estimated_capacity = estimated_episodes * 300
    replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(estimated_capacity),
    )
    replay_buffer.seed(variant.seed)

    load_robocasa_demos_to_buffer(
        dataset_path=dataset_path,
        replay_buffer=replay_buffer,
        agent_dp=agent_dp,
        variant=variant,
        episode_ids=episode_ids,
        num_demos=num_demos,
        task_description=task_description,
        pi_data_config=pi_data_config,
        eval_controller=eval_controller,
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    replay_buffer.save(save_path)
    print(f"[RoboCasa Demo Loader] Buffer saved to {save_path}")

    return replay_buffer
