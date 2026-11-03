"""Eval reset controller for config-driven RoboCasa evaluation.

Manages a demo pool and produces per-reset metadata based on the configured
eval initialization mode. Supports these modes:

  - exact_state_replay: restore exact recorded MuJoCo state from a demo
  - fixture_pair_fresh_placement: use demo scene config, resample object placement
  - fixture_pair_object_pool: same fixture pair but fresh random objects

Each mode can be combined with:
  - keep_robot_pose: preserve or resample robot base pose
  - keep_object_cfg: preserve recorded object configs (same object, new placement)
                     vs let RoboCasa sample fresh objects
"""

import copy
import gzip
import json
import logging
import pathlib

import numpy as np

from midas.utils.reproducibility import capture_numpy_rng, restore_numpy_rng

logger = logging.getLogger(__name__)


class RoboCasaEvalResetController:
    """Loads a demo pool and produces reset metadata per eval mode."""

    VALID_MODES = (
        "exact_state_replay",
        "fixture_pair_fresh_placement",
        "fixture_pair_same_category",
        "fixture_pair_object_pool",
    )

    def __init__(
        self,
        dataset_path: pathlib.Path,
        eval_init_mode: str,
        layout_and_style_ids: list[tuple[int, int]] | None = None,
        eval_pool_episode_ids: list[int] | None = None,
        eval_pool_fixture_refs: dict[str, str] | None = None,
        eval_pool_object_categories: list[str] | None = None,
        keep_robot_pose: bool = False,
        robot_pose_noise: float = 0.0,
        object_pose_noise: float = 0.0,
        object_ori_noise: float = 0.0,
        rng_seed: int = 0,
    ):
        if eval_init_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown eval_init_mode={eval_init_mode!r}. "
                f"Must be one of {self.VALID_MODES}"
            )

        self._dataset_path = pathlib.Path(dataset_path)
        self._mode = eval_init_mode
        self._keep_robot_pose = keep_robot_pose
        self._robot_pose_noise = robot_pose_noise
        self._object_pose_noise = object_pose_noise
        self._object_ori_noise = object_ori_noise
        self._rng = np.random.default_rng(rng_seed)

        # Build eval pool
        if eval_pool_episode_ids is not None:
            self._pool_ids = sorted(eval_pool_episode_ids)
            if layout_and_style_ids is not None or eval_pool_fixture_refs is not None:
                logger.warning(
                    "eval_pool_episode_ids is set, so layout_and_style_ids and "
                    "eval_pool_fixture_refs are ignored for pool construction. "
                    "The explicit episode IDs are used as-is."
                )
        else:
            from midas.utils.robocasa_utils import get_scene_filtered_demos
            self._pool_ids = get_scene_filtered_demos(
                self._dataset_path,
                layout_and_style_ids=layout_and_style_ids or [],
                fixture_refs=eval_pool_fixture_refs,
                object_categories=eval_pool_object_categories,
            )

        if not self._pool_ids:
            raise ValueError(
                f"Eval pool is empty for dataset at {self._dataset_path}. "
                f"Check eval_pool filters."
            )

        # Load and cache ep_meta for each pool episode
        self._pool_metas = {}
        for ep_id in self._pool_ids:
            meta_path = self._dataset_path / "extras" / f"episode_{ep_id:06d}" / "ep_meta.json"
            with open(meta_path) as f:
                self._pool_metas[ep_id] = json.load(f)

        # For exact_state_replay, cache initial states, full trajectories, and model XMLs
        self._pool_initial_states = {}
        self._pool_all_states = {}
        self._pool_model_xmls = {}
        if self._mode == "exact_state_replay":
            for ep_id in self._pool_ids:
                states_path = self._dataset_path / "extras" / f"episode_{ep_id:06d}" / "states.npz"
                all_states = np.load(states_path)["states"]
                self._pool_initial_states[ep_id] = all_states[0]
                self._pool_all_states[ep_id] = all_states

                xml_path = self._dataset_path / "extras" / f"episode_{ep_id:06d}" / "model.xml.gz"
                with gzip.open(xml_path, "rt") as f:
                    self._pool_model_xmls[ep_id] = f.read()

        # Extract pool-wide object category set (for logging/diagnostics)
        self._pool_object_categories = set()
        for meta in self._pool_metas.values():
            obj_cfgs = meta.get("object_cfgs", [])
            if obj_cfgs:
                cat = obj_cfgs[0].get("info", {}).get("cat")
                if cat is not None:
                    self._pool_object_categories.add(cat)
        self._pool_object_categories = sorted(self._pool_object_categories)

        # Round-robin index for exact_state_replay
        self._replay_idx = 0
        self._last_reset_info = None

        print(
            f"RoboCasaEvalResetController: mode={self._mode}, "
            f"pool_size={len(self._pool_ids)}, "
            f"keep_robot_pose={self._keep_robot_pose}, "
            f"robot_pose_noise={self._robot_pose_noise}, "
            f"object_pose_noise={self._object_pose_noise}, "
            f"object_ori_noise={self._object_ori_noise}, "
            f"object_categories={self._pool_object_categories}"
        )

    @property
    def last_reset_info(self) -> dict | None:
        """The reset info dict from the most recent prepare_reset() call."""
        return self._last_reset_info

    def seed(self, seed: int) -> None:
        """Reset sampling and exact-replay ordering for an independent stream."""

        self._rng = np.random.default_rng(int(seed))
        self._replay_idx = 0
        self._last_reset_info = None

    def get_rng_state(self) -> dict:
        """Capture random state and the exact-replay cursor."""

        return {
            "rng": capture_numpy_rng(self._rng),
            "replay_idx": self._replay_idx,
        }

    def set_rng_state(self, state: dict) -> None:
        """Restore state returned by :meth:`get_rng_state`."""

        self._rng = restore_numpy_rng(self._rng, state["rng"])
        self._replay_idx = int(state["replay_idx"])
        self._last_reset_info = None

    def prepare_reset(self) -> dict:
        """Produce reset metadata for the next eval episode."""
        if self._mode == "exact_state_replay":
            info = self._prepare_exact_state_replay()
        elif self._mode == "fixture_pair_fresh_placement":
            info = self._prepare_fixture_pair_fresh_placement()
        elif self._mode == "fixture_pair_same_category":
            info = self._prepare_fixture_pair_same_category()
        elif self._mode == "fixture_pair_object_pool":
            info = self._prepare_fixture_pair_object_pool()
        self._last_reset_info = info
        return info

    def _prepare_exact_state_replay(self) -> dict:
        ep_id = self._pool_ids[self._replay_idx % len(self._pool_ids)]
        self._replay_idx += 1
        return {
            "mode": "exact_state_replay",
            "ep_meta": copy.deepcopy(self._pool_metas[ep_id]),
            "initial_state": self._pool_initial_states[ep_id],
            "all_states": self._pool_all_states[ep_id],
            "model_xml": self._pool_model_xmls[ep_id],
            "episode_id": ep_id,
        }

    def _apply_robot_pose_policy(self, ep_meta: dict) -> None:
        """Control robot pose variation in ep_meta.

        If keep_robot_pose=True and robot_pose_noise=0: keep recorded pose exactly.
        If keep_robot_pose=True and robot_pose_noise>0: add bounded noise to recorded pose.
        If keep_robot_pose=False: remove pose so Kitchen fully resamples it.
        """
        if not self._keep_robot_pose:
            ep_meta.pop("init_robot_base_pos", None)
            ep_meta.pop("init_robot_base_ori", None)
        elif self._robot_pose_noise > 0.0 and "init_robot_base_pos" in ep_meta:
            pos = np.array(ep_meta["init_robot_base_pos"], dtype=float)
            ori = np.array(ep_meta["init_robot_base_ori"], dtype=float)
            pos[:2] += self._rng.uniform(-self._robot_pose_noise, self._robot_pose_noise, size=2)
            ori[2] += self._rng.uniform(-self._robot_pose_noise, self._robot_pose_noise)
            ep_meta["init_robot_base_pos"] = pos.tolist()
            ep_meta["init_robot_base_ori"] = ori.tolist()

    def _prepare_fixture_pair_fresh_placement(self) -> dict:
        """Same scene, same fixture pair, same object category.

        Keeps object_cfgs (including info/mjcf_path) so the exact same object
        model is used, but RoboCasa resamples its position within the placement
        region. Robot pose is kept or resampled based on keep_robot_pose.

        Variation axes:
          - Object placement position within the counter region
          - Robot base pose (if keep_robot_pose=False)
        """
        ep_id = self._rng.choice(self._pool_ids)
        ep_meta = copy.deepcopy(self._pool_metas[ep_id])
        self._apply_robot_pose_policy(ep_meta)
        return {
            "mode": "fixture_pair_fresh_placement",
            "ep_meta": ep_meta,
            "episode_id": ep_id,
        }

    def _prepare_fixture_pair_same_category(self) -> dict:
        """Same scene, same fixture pair, different instance of the same object category.

        Removes mjcf_path from the main object's info but preserves the category
        in obj_groups, so create_obj samples a different physical instance of the
        same category (e.g. a different hot_dog mesh). Placement constraints are
        preserved from the original object_cfgs. Robot pose controlled by noise knobs.

        Variation axes:
          - Object instance (different mesh from same category)
          - Object placement position within the counter region
          - Robot base pose (controlled by keep_robot_pose + robot_pose_noise)
        """
        ep_id = self._rng.choice(self._pool_ids)
        ep_meta = copy.deepcopy(self._pool_metas[ep_id])
        self._apply_robot_pose_policy(ep_meta)

        obj_cfgs = ep_meta.get("object_cfgs", [])
        if obj_cfgs and "info" in obj_cfgs[0]:
            # Extract the category before removing info
            cat = obj_cfgs[0]["info"].get("cat", "all")
            obj_cfgs[0].pop("info")
            # Set obj_groups to the specific category so create_obj samples
            # a different instance of the same category
            obj_cfgs[0]["obj_groups"] = cat

        return {
            "mode": "fixture_pair_same_category",
            "ep_meta": ep_meta,
            "episode_id": ep_id,
        }

    def _prepare_fixture_pair_object_pool(self) -> dict:
        """Same scene, same fixture pair, but fresh random objects.

        Removes object_cfgs entirely so _create_objects falls through to
        _get_obj_cfgs(), which generates fresh, placement-compatible object
        configs from scratch. This varies the physical object while keeping
        the scene (layout, style, fixtures) anchored by the rest of ep_meta.

        Variation axes:
          - Object category and instance (fully random from obj_groups)
          - Object placement position
          - Robot base pose (if keep_robot_pose=False)
        """
        ep_id = self._rng.choice(self._pool_ids)
        ep_meta = copy.deepcopy(self._pool_metas[ep_id])
        self._apply_robot_pose_policy(ep_meta)
        ep_meta.pop("object_cfgs", None)
        return {
            "mode": "fixture_pair_object_pool",
            "ep_meta": ep_meta,
            "episode_id": ep_id,
        }
