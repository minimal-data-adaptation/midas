from types import SimpleNamespace

import numpy as np
import pytest

from openpi.policies.yam_policy import YamInputs, YamOutputs


YAM_CONFIGS = {
    "pi05_yam_combined_lora": (None, None, None),
    "pi05_yam_pickplace_a_lora": (14512, None, None),
    "pi05_yam_pickplace_b_lora": (None, 14512, None),
    "pi05_yam_arrange_a_lora": (25632, None, None),
    "pi05_yam_arrange_all_lora": (None, None, None),
    "pi05_yam_wipe_a_lora": (43505, None, None),
    "pi05_yam_wipe_b_lora": (None, None, 43214),
    "pi05_yam_wipe_all_lora": (None, None, None),
}


def test_yam_policy_transform_shapes_and_missing_camera_error():
    data = {
        "images": {
            name: np.zeros((3, 4, 5), np.uint8) for name in ("top", "left_wrist", "right_wrist")
        },
        "state": np.zeros(14, np.float32),
        "actions": np.zeros((60, 32), np.float32),
        "prompt": "task",
    }
    transformed = YamInputs()(data)
    assert transformed["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert transformed["state"].shape == (14,)
    assert YamOutputs()(transformed)["actions"].shape == (60, 14)
    del data["images"]["top"]
    with pytest.raises(ValueError, match="top"):
        YamInputs()(data)


def test_yam_openpi_configs_are_registered_with_expected_partitions():
    from openpi.training import config

    for name, expected_filter in YAM_CONFIGS.items():
        train_config = config.get_config(name)
        data = train_config.data.create(train_config.assets_dirs, train_config.model)
        assert train_config.model.action_horizon == 60
        assert (
            data.filter_orig_traj_id_6_eq,
            data.filter_orig_traj_id_6_min,
            data.filter_orig_traj_id_6_max,
        ) == expected_filter


def test_feature_filter_maps_through_a_prompt_filter():
    pytest.importorskip("lerobot")
    from openpi.training.data_loader import FeatureRangeFilteredDataset, FilteredDataset

    rows = [
        {"task": "keep", "value": 0},
        {"task": "drop", "value": 1},
        {"task": "keep", "value": 2},
        {"task": "keep", "value": 3},
    ]

    class Columns:
        column_names = ["orig_traj_id_6"]

        def __getitem__(self, key):
            assert key == "orig_traj_id_6"
            return [10, 10, 20, 20]

    raw = SimpleNamespace(hf_dataset=Columns())
    prompt_filtered = FilteredDataset(rows, "keep")
    feature_filtered = FeatureRangeFilteredDataset(prompt_filtered, lerobot_dataset=raw, equal=20)
    assert [feature_filtered[i]["value"] for i in range(len(feature_filtered))] == [2, 3]
