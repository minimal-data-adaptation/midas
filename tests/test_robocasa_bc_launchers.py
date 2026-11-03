"""Exercise the BC launchers without starting training."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / "experiments/behavior_cloning/robocasa"
CASES = [
    (
        "run_bc_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32.sh",
        "pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32",
    ),
    (
        "run_bc_pick_place_fridge_drawer_to_shelf_l50_s37.sh",
        "pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37",
    ),
    ("run_bc_prepare_coffee_l25_s29.sh", "pi05_robocasa_bc_prepare_coffee_l25_s29"),
]

LIBERO_LAUNCHERS = ROOT / "experiments/behavior_cloning/libero"
LIBERO_TASKS = [
    "alphabet_soup_tomato_sauce_basket",
    "cream_cheese_butter_basket",
    "stove_mokapot",
    "black_bowl_bottom_drawer",
    "white_mug_plates",
    "book_caddy",
    "white_mug_chocolate_pudding",
    "alphabet_soup_cream_cheese_basket",
    "both_mokapots_stove",
    "yellow_white_mug_microwave",
]


@pytest.fixture
def launcher_env(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "record = {'args': args, 'cwd': os.getcwd(), "
        "'platform': os.environ.get('JAX_PLATFORMS'), "
        "'pythonpath': os.environ.get('PYTHONPATH'), "
        "'entity': os.environ.get('WANDB_ENTITY')}\n"
        "with open(os.environ['BC_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        "if os.environ.get('BC_TEST_FAIL_SCRIPT') == args[0]:\n"
        "    sys.exit(1)\n"
        "if args[0] == 'scripts/compute_norm_stats.py':\n"
        "    config = args[args.index('--config-name') + 1]\n"
        "    root = pathlib.Path(os.environ['OPENPI_ASSETS_ROOT'])\n"
        "    if config.startswith('pi05_libero10_'):\n"
        "        output = root / 'pi05_libero10_assets' / config / 'physical-intelligence/libero'\n"
        "    else:\n"
        "        output = root / config / config\n"
        "    output.mkdir(parents=True)\n"
        "    (output / 'norm_stats.json').write_text('{}')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MIDAS_", "OPENPI_", "WANDB_", "JAX_"))
    }
    environment.update(
        MIDAS_REPO_DIR=str(ROOT),
        OPENPI_DATASET_ROOT=str(tmp_path / "data"),
        OPENPI_ASSETS_ROOT=str(tmp_path / "assets"),
        OPENPI_CHECKPOINT_ROOT=str(tmp_path / "checkpoints"),
        OPENPI_PYTHON_BIN=str(fake_python),
        WANDB_ENTITY="bc-test-entity",
        MIDAS_RUN_ID="12345",
        JAX_PLATFORMS="cpu",
        BC_TEST_CALLS=str(tmp_path / "calls.jsonl"),
    )
    return environment


def run_launcher(script, environment, tmp_path):
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return [
        json.loads(line) for line in Path(environment["BC_TEST_CALLS"]).read_text().splitlines()
    ]


@pytest.mark.parametrize(("filename", "config"), CASES)
@pytest.mark.parametrize("resume", [False, True])
def test_task_launchers_preserve_configs_and_handle_resume(
    filename, config, resume, launcher_env, tmp_path
):
    exp_name = f"{config.removeprefix('pi05_')}_v1_s0_12345"
    if resume:
        run_dir = Path(launcher_env["OPENPI_CHECKPOINT_ROOT"]) / "robocasa_bc" / config / exp_name
        run_dir.mkdir(parents=True)
        stats = Path(launcher_env["OPENPI_ASSETS_ROOT"]) / config / config / "norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text("existing statistics")

    calls = run_launcher(LAUNCHERS / filename, launcher_env, tmp_path)
    if not resume:
        assert calls[0]["args"] == ["scripts/compute_norm_stats.py", "--config-name", config]
        assert calls[0]["platform"] == "cpu"
    else:
        assert stats.read_text() == "existing statistics"
    assert calls[-2]["args"] == ["scripts/trace_robocasa_bc.py", "--config-name", config]
    assert calls[-2]["platform"] == "cpu"
    expected = ["scripts/train.py", config, "--exp-name", exp_name, "--seed", "0"]
    if resume:
        expected.append("--resume")
    assert calls[-1]["args"] == expected
    assert calls[-1]["platform"] is None
    assert calls[-1]["cwd"] == str(ROOT / "openpi")
    assert calls[-1]["pythonpath"].startswith(str(ROOT / "openpi/src") + ":")
    assert calls[-1]["entity"] == "bc-test-entity"
    assert "--overwrite" not in calls[-1]["args"]


def test_overrides_and_storage_defaults(launcher_env, tmp_path):
    for key in (
        "OPENPI_DATASET_ROOT",
        "OPENPI_ASSETS_ROOT",
        "OPENPI_CHECKPOINT_ROOT",
        "WANDB_ENTITY",
    ):
        launcher_env.pop(key)
    launcher_env.update(
        MIDAS_DATA_ROOT=str(tmp_path / "storage"),
        OPENPI_BC_WANDB_ENABLED="0",
        OPENPI_BC_EXP_NAME="coffee_custom",
        OPENPI_BC_SEED="11",
        OPENPI_BC_BATCH_SIZE="2",
        OPENPI_BC_NUM_TRAIN_STEPS="10",
        OPENPI_BC_NUM_WORKERS="0",
        OPENPI_BC_SAVE_INTERVAL="5",
    )
    calls = run_launcher(LAUNCHERS / CASES[-1][0], launcher_env, tmp_path)
    assert calls[-1]["args"] == [
        "scripts/train.py",
        CASES[-1][1],
        "--exp-name",
        "coffee_custom",
        "--seed",
        "11",
        "--batch-size",
        "2",
        "--num-train-steps",
        "10",
        "--num-workers",
        "0",
        "--save-interval",
        "5",
        "--no-wandb-enabled",
    ]
    assert (
        tmp_path / "storage/openpi_bc_assets" / CASES[-1][1] / CASES[-1][1] / "norm_stats.json"
    ).is_file()


def test_repository_location_fallback(launcher_env, tmp_path):
    launcher_env.pop("MIDAS_REPO_DIR")
    calls = run_launcher(LAUNCHERS / CASES[0][0], launcher_env, tmp_path)
    assert calls[-1]["args"][1] == CASES[0][1]


@pytest.mark.parametrize("invalid", ["../old_run", ".", ".."])
def test_invalid_experiment_name_fails_before_commands(invalid, launcher_env, tmp_path):
    launcher_env["OPENPI_BC_EXP_NAME"] = invalid
    result = subprocess.run(
        ["bash", str(LAUNCHERS / CASES[0][0])],
        cwd=tmp_path,
        env=launcher_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "single directory name" in result.stderr
    assert not Path(launcher_env["BC_TEST_CALLS"]).exists()


@pytest.mark.parametrize("task", LIBERO_TASKS)
@pytest.mark.parametrize("resume", [False, True])
def test_libero_launchers_preserve_configs_and_handle_resume(
    task, resume, launcher_env, tmp_path
):
    # LIBERO uses the Hugging Face dataset, not a RoboCasa data directory.
    launcher_env.pop("OPENPI_DATASET_ROOT")
    config = f"pi05_libero10_{task}_ep1"
    exp_name = f"libero10_{task}_ep1_v1_s42_12345"
    if resume:
        run_dir = (
            Path(launcher_env["OPENPI_CHECKPOINT_ROOT"])
            / "pi05_checkpoints_libero10" / config / exp_name
        )
        run_dir.mkdir(parents=True)
        stats = (
            Path(launcher_env["OPENPI_ASSETS_ROOT"])
            / "pi05_libero10_assets" / config / "physical-intelligence/libero/norm_stats.json"
        )
        stats.parent.mkdir(parents=True)
        stats.write_text("existing statistics")

    calls = run_launcher(LIBERO_LAUNCHERS / f"run_bc_{task}_ep1.sh", launcher_env, tmp_path)
    expected = ["scripts/train.py", config, "--exp-name", exp_name, "--seed", "42"]
    if resume:
        expected.append("--resume")
        assert len(calls) == 1
        assert stats.read_text() == "existing statistics"
    else:
        assert len(calls) == 2
        assert calls[0]["args"] == ["scripts/compute_norm_stats.py", "--config-name", config]
        assert calls[0]["platform"] == "cpu"
    assert calls[-1]["args"] == expected
    assert calls[-1]["platform"] is None
    assert calls[-1]["cwd"] == str(ROOT / "openpi")
    assert calls[-1]["entity"] == "bc-test-entity"


def test_launchers_cover_all_current_bc_configs():
    source = (ROOT / "openpi/src/openpi/training/config.py").read_text()
    names = {
        kw.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"
        for kw in node.keywords
        if kw.arg == "name" and isinstance(kw.value, ast.Constant)
    }
    expected_libero = {f"pi05_libero10_{task}_ep1" for task in LIBERO_TASKS}
    expected_robocasa = {config for _, config in CASES}
    assert expected_libero | expected_robocasa <= names
    assert {name for name in names if name.startswith("pi05_libero10_")} == expected_libero
    assert {name for name in names if name.startswith("pi05_robocasa_bc_")} == expected_robocasa


def test_direct_libero_runner_and_fsdp_override(launcher_env, tmp_path):
    launcher_env.pop("MIDAS_REPO_DIR")
    launcher_env["OPENPI_BC_FSDP_DEVICES"] = "4"
    config = "pi05_libero10_both_mokapots_stove_ep1"
    result = subprocess.run(
        ["bash", str(LIBERO_LAUNCHERS / "run_bc.sh"), config],
        cwd=tmp_path, env=launcher_env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(launcher_env["BC_TEST_CALLS"]).read_text().splitlines()]
    assert calls[-1]["args"][-2:] == ["--fsdp-devices", "4"]


@pytest.mark.parametrize("domain", ["libero", "robocasa"])
@pytest.mark.parametrize("arguments,code", [([], 2), (["--help"], 0), (["bad-config"], 2)])
def test_runner_usage(domain, arguments, code, launcher_env, tmp_path):
    script = ROOT / "experiments/behavior_cloning" / domain / "run_bc.sh"
    result = subprocess.run(
        ["bash", str(script), *arguments],
        cwd=tmp_path, env=launcher_env, capture_output=True, text=True,
    )
    assert result.returncode == code
    assert not Path(launcher_env["BC_TEST_CALLS"]).exists()


@pytest.mark.parametrize("script", ["scripts/compute_norm_stats.py", "scripts/trace_robocasa_bc.py"])
def test_preflight_failure_prevents_training(script, launcher_env, tmp_path):
    launcher_env["BC_TEST_FAIL_SCRIPT"] = script
    result = subprocess.run(
        ["bash", str(LAUNCHERS / CASES[0][0])],
        cwd=tmp_path, env=launcher_env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    calls = [json.loads(line) for line in Path(launcher_env["BC_TEST_CALLS"]).read_text().splitlines()]
    assert not any(call["args"][0] == "scripts/train.py" for call in calls)


@pytest.mark.parametrize("variable", ["OPENPI_ASSETS_ROOT", "OPENPI_CHECKPOINT_ROOT", "OPENPI_DATASET_ROOT"])
def test_relative_paths_fail_before_commands(variable, launcher_env, tmp_path):
    launcher_env[variable] = "relative/path"
    result = subprocess.run(
        ["bash", str(LAUNCHERS / CASES[0][0])],
        cwd=tmp_path, env=launcher_env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "must be an absolute path" in result.stderr
    assert not Path(launcher_env["BC_TEST_CALLS"]).exists()
