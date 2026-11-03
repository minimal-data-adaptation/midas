"""Exercise simulator replay contracts without allocating a renderer."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def wrapper():
    module = pytest.importorskip("robocasa.wrappers.gym_wrapper")

    class Simulator:
        def reset(self):
            pass

        def set_state_from_flattened(self, state):
            self.state = np.array(state)

        def forward(self):
            pass

    class Environment:
        def __init__(self):
            self.sim = Simulator()
            self.language = "pick the ice cube"
            self.events = []
            self._cam_configs = {"wrong": {}}
            self.fixture_refs = {}
            self.fixtures = {}
            self.robots = [SimpleNamespace(
                robot_model=SimpleNamespace(naming_prefix="robot0_"),
                composite_controller=SimpleNamespace(
                    action_limits=(np.zeros(1), np.ones(1)), part_controllers={},
                    update_state=lambda: None,
                ),
            )]

        def set_ep_meta(self, meta):
            self.meta = meta

        def reset(self):
            self.events.append("rebuild")
            if self.meta:
                self.object_cfgs = deepcopy(self.meta["object_cfgs"])
                self.fixtures = {name: object() for name in self.meta["fixture_refs"].values()}
                self.cab = None
            return self._get_observations()

        def edit_model_xml(self, xml):
            self.events.append(("edit_xml", deepcopy(self._cam_configs)))
            return xml

        def reset_from_xml_string(self, xml):
            self.events.append("restore_xml")

        def unset_ep_meta(self):
            self.meta = {}

        def get_ep_meta(self):
            # Emulate stale runtime language after XML restoration.
            return {"lang": self.language}

        def _get_observations(self, force_update=False):
            return {
                **{f"{camera}_image": np.zeros((4, 4, 3), dtype=np.uint8)
                   for camera in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")},
                **{key: np.zeros(size) for key, size in (
                    ("robot0_gripper_qpos", 2), ("robot0_base_pos", 3),
                    ("robot0_base_quat", 4), ("robot0_base_to_eef_pos", 3),
                    ("robot0_base_to_eef_quat", 4),
                )},
            }

        def step(self, action):
            self.language = "different stale step instruction"
            return self._get_observations(), 0.0, False, {}

        def _check_success(self):
            return False

    result = module.RoboCasaGymEnv.__new__(module.RoboCasaGymEnv)
    result.env = Environment()
    result.env.meta = {}
    result._episode_language = None
    result._eval_reset_controller = None
    result.enable_render = True
    result.camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    result.render_obs_key = "robot0_agentview_left_image"
    result.key_converter = SimpleNamespace(
        map_obs=module.PandaOmronKeyConverter.map_obs,
        get_camera_config=module.PandaOmronKeyConverter.get_camera_config,
        unmap_action=lambda action: {},
    )
    return result


def replay_info(language="Pick the hot dog from the counter and place it in the cabinet."):
    return {
        "mode": "exact_state_replay", "episode_id": 32,
        "ep_meta": {
            "lang": language,
            "object_cfgs": [{"name": "obj", "info": {"cat": "hot_dog"}}],
            "fixture_refs": {"cab": "cab_3"},
            "cam_configs": {"recorded_camera": {"pos": [1, 2, 3]}},
        },
        "model_xml": "<mujoco/>", "initial_state": [1.0, 2.0],
    }


def test_exact_replay_rebuilds_scene_and_retains_recorded_language_on_steps(wrapper):
    info = replay_info()
    wrapper._eval_reset_controller = SimpleNamespace(prepare_reset=lambda: deepcopy(info))
    obs, metadata = wrapper.reset()
    assert wrapper.env.events == [
        "rebuild", ("edit_xml", info["ep_meta"]["cam_configs"]), "restore_xml",
    ]
    assert wrapper.env.object_cfgs[0]["info"]["cat"] == "hot_dog"
    assert wrapper.env.cab is wrapper.env.fixtures["cab_3"]
    np.testing.assert_array_equal(wrapper.env.sim.state, info["initial_state"])
    assert metadata["episode_id"] == 32
    assert metadata["reset_mode"] == "exact_state_replay"
    assert obs["annotation.human.task_description"] == info["ep_meta"]["lang"]
    for _ in range(2):
        obs, *_ = wrapper.step({})
        assert obs["annotation.human.task_description"] == info["ep_meta"]["lang"]


def test_fresh_object_reset_regenerates_language_and_caches_it_for_steps(wrapper):
    info = replay_info()
    info["mode"] = "fixture_pair_object_pool"
    wrapper._eval_reset_controller = SimpleNamespace(prepare_reset=lambda: deepcopy(info))
    for language in ("pick the banana", "pick the mug"):
        wrapper.env.language = language
        obs, _ = wrapper.reset()
        assert obs["annotation.human.task_description"] == language
        obs, *_ = wrapper.step({})
        assert obs["annotation.human.task_description"] == language


@pytest.mark.parametrize("language", [None, "", "  ", 12])
def test_invalid_replay_language_fails_before_scene_restoration(wrapper, language):
    wrapper._eval_reset_controller = SimpleNamespace(prepare_reset=lambda: replay_info(language))
    with pytest.raises(ValueError, match="replay episode 32"):
        wrapper.reset()
    assert wrapper.env.events == []


def test_random_reset_does_not_reuse_previous_episode_language(wrapper):
    wrapper._episode_language = "old episode prompt"
    obs, _ = wrapper.reset()
    assert obs["annotation.human.task_description"] == "pick the ice cube"


def test_replay_refreshes_stale_controller_origin_and_base_mode_goal(wrapper):
    from robosuite.controllers.parts.arm.osc import OperationalSpaceController

    class Arm(OperationalSpaceController):
        def __init__(self):
            self.input_ref_frame = "base"
            self.origin_pos = np.array([99.0, 99.0, 99.0])
            self.origin_ori = np.eye(3)
            self.ref_pos = np.array([14.0, 25.0, 1.0])
            self.ref_ori_mat = np.eye(3)
            self.goal_pos = np.ones(3) * 100
            self.goal_ori = np.eye(3)
            self._goal_update_mode = "desired"
            self.position_limits = None
            self.orientation_limits = None
            self.position_goals = []
            self.orientation_goals = []
            self.interpolator_pos = SimpleNamespace(
                set_goal=lambda goal: self.position_goals.append(np.array(goal))
            )
            self.interpolator_ori = SimpleNamespace(
                set_goal=lambda goal: self.orientation_goals.append(np.array(goal))
            )

        def update(self, force=False):
            assert force is True
            np.testing.assert_array_equal(self.origin_pos, [10.0, 20.0, 0.0])

    arm = Arm()
    composite = wrapper.env.robots[0].composite_controller
    composite.part_controllers = {"right": arm}
    composite.update_state = lambda: arm.update_origin(np.array([10.0, 20.0, 0.0]), np.eye(3))
    wrapper._eval_reset_controller = SimpleNamespace(prepare_reset=lambda: replay_info())
    wrapper.reset()
    np.testing.assert_allclose(arm.goal_pos, [4.0, 5.0, 1.0])
    np.testing.assert_allclose(arm.goal_ori, np.eye(3))
    np.testing.assert_allclose(arm.position_goals[-1], [4.0, 5.0, 1.0])
    np.testing.assert_allclose(arm.orientation_goals[-1], np.zeros(3))
    # A first base-mode action must accumulate from the restored pose.
    arm.set_goal_update_mode("desired")
    np.testing.assert_allclose(arm.compute_goal_pos(np.zeros(3)), [4.0, 5.0, 1.0])
