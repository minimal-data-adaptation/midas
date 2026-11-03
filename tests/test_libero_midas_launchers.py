"""Validate all LIBERO MiDAS launch commands without GPUs or starting training."""

import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / "experiments/libero"
TASKS = [
    ("alphabet_soup_tomato_sauce_basket", "alphabet_soup_tomato_sauce_basket_ep1"),
    ("cream_cheese_butter_basket", "cream_cheese_butter_basket_ep1"),
    ("stove_mokapot", "stove_moka_pot_ep1"),
    ("black_bowl_bottom_drawer", "black_bowl_drawer_ep1"),
    ("white_mug_plates", "white_mug_left_yellow_mug_right_ep1"),
    ("book_caddy", "book_caddy_ep1"),
    ("white_mug_chocolate_pudding", "white_mug_plate_choc_pudding_ep1"),
    ("alphabet_soup_cream_cheese_basket", "alphabet_soup_cream_cheese_basket_ep1"),
    ("both_mokapots_stove", "put_both_moka_pots_on_the_stove_ep86"),
    ("yellow_white_mug_microwave", "yellow_white_mug_microwave_ep1"),
]


def options(arguments):
    assert len(arguments) % 2 == 0
    return dict(zip(arguments[::2], arguments[1::2], strict=True))


def reference_options():
    source = (LAUNCHERS / "legacy/run_midas_task8_both_mokapots_paper_v1.sh").read_text()
    command = source[source.index('"$PYTHON_BIN" -u -m training.launch_train_sim'):]
    tokens = shlex.split(command.replace("\\\n", ""))
    assert tokens[-1] == "${RESUME_ARGS[@]}"
    return options(tokens[4:-1])


@pytest.fixture
def launcher_env(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "record = {'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'gpu_mask': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'platform': os.environ.get('JAX_PLATFORMS'), "
        "'cache': os.environ.get('OPENPI_DATA_HOME'), "
        "'output_parent': os.environ.get('MIDAS_EXP_DIR')}\n"
        "pathlib.Path(os.environ['MIDAS_TEST_CALL']).write_text(json.dumps(record))\n"
    )
    fake_python.chmod(0o755)
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("MIDAS_", "OPENPI_", "WANDB_", "JAX_"))
        and key not in {"T_CKPT", "DEMO_HDF5", "LIBERO_CONFIG_PATH"}
    }
    checkpoint = tmp_path / "bc/16000"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    data = tmp_path / "data"
    demos = data / "expert_hdf5"
    demos.mkdir(parents=True)
    for task_id, (_, demo) in enumerate(TASKS):
        (demos / f"libero10_task{task_id}_{demo}.hdf5").touch()
    environment.update(
        MIDAS_REPO_DIR=str(ROOT),
        MIDAS_DATA_ROOT=str(data),
        MIDAS_PYTHON_BIN=str(fake_python),
        MIDAS_TEST_CALL=str(tmp_path / "call.json"),
        T_CKPT=str(checkpoint),
        WANDB_ENTITY="midas-test-entity",
        MIDAS_RUN_ID="12345",
        LIBERO_CONFIG_PATH=str(tmp_path / "libero-config"),
        CUDA_VISIBLE_DEVICES="3",
        JAX_PLATFORMS="cpu",
    )
    return environment


def launch(task_id, environment, tmp_path):
    slug, _ = TASKS[task_id]
    script = LAUNCHERS / f"run_midas_task{task_id}_{slug}_ep1.sh"
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=environment,
        capture_output=True, text=True,
    )
    return result


@pytest.mark.parametrize("task_id", range(10))
@pytest.mark.parametrize("resume", [False, True])
def test_all_task_launchers_keep_original_hyperparameters(
    task_id, resume, launcher_env, tmp_path
):
    slug, demo = TASKS[task_id]
    name = f"midas_pi05_base-ep1-{slug}-vlm-a-exec-paper_v1-s0-12345"
    output_parent = Path(launcher_env["MIDAS_DATA_ROOT"]) / "midas_exps/libero10/pi05_base_v1"
    run_dir = output_parent / name
    if resume:
        state = run_dir / "train_state/latest.json"
        state.parent.mkdir(parents=True)
        state.write_text("existing snapshot")

    result = launch(task_id, launcher_env, tmp_path)
    assert result.returncode == 0, result.stderr
    record = json.loads(Path(launcher_env["MIDAS_TEST_CALL"]).read_text())
    assert record["args"][:3] == ["-u", "-m", "training.launch_train_sim"]
    expected = reference_options()
    expected.update({
        "--exp_name": name,
        "--pi_05_config": f"pi05_libero10_{slug}_ep1",
        "--pi_05_ckpt_dir": launcher_env["T_CKPT"],
        "--task_id": str(task_id),
        "--demo_hdf5_path": str(
            Path(launcher_env["MIDAS_DATA_ROOT"]) / "expert_hdf5"
            / f"libero10_task{task_id}_{demo}.hdf5"
        ),
    })
    if resume:
        expected["--resume_dir"] = str(run_dir)
        assert state.read_text() == "existing snapshot"
    assert options(record["args"][3:]) == expected
    assert record["cwd"] == str(ROOT)
    assert record["output_parent"] == str(output_parent)
    assert record["gpu_mask"] == "3"  # Never replace the selected GPU mask with 0.
    assert record["platform"] is None


def test_overrides_and_repository_location_fallback(launcher_env, tmp_path):
    launcher_env.pop("MIDAS_REPO_DIR")
    launcher_env.update(
        MIDAS_SEED="11",
        MIDAS_EXP_NAME_OVERRIDE="task8_seed11_custom",
        MIDAS_EXP_DIR=str(tmp_path / "custom-runs"),
        OPENPI_DATA_HOME=str(tmp_path / "custom-cache"),
    )
    demo = tmp_path / "custom-demo.hdf5"
    demo.touch()
    launcher_env["DEMO_HDF5"] = str(demo)
    result = launch(8, launcher_env, tmp_path)
    assert result.returncode == 0, result.stderr
    record = json.loads(Path(launcher_env["MIDAS_TEST_CALL"]).read_text())
    args = options(record["args"][3:])
    assert args["--seed"] == "11"
    assert args["--exp_name"] == "task8_seed11_custom"
    assert args["--demo_hdf5_path"] == str(demo)
    assert record["output_parent"] == str(tmp_path / "custom-runs")
    assert record["cache"] == str(tmp_path / "custom-cache")


@pytest.mark.parametrize("variable,value", [
    ("T_CKPT", ""),
    ("T_CKPT", "/nonexistent/libero-test-checkpoint"),
    ("DEMO_HDF5", "/nonexistent/libero-test-demo.hdf5"),
    ("MIDAS_DATA_ROOT", "relative/data"),
    ("MIDAS_EXP_DIR", "relative/output"),
    ("MIDAS_EXP_NAME_OVERRIDE", "../old-run"),
    ("MIDAS_SEED", "-1"),
    ("WANDB_ENTITY", ""),
])
def test_bad_inputs_do_not_start_training(variable, value, launcher_env, tmp_path):
    launcher_env[variable] = value
    result = launch(8, launcher_env, tmp_path)
    assert result.returncode != 0
    assert not Path(launcher_env["MIDAS_TEST_CALL"]).exists()


def test_configs_match_the_benchmark_task_instructions():
    source = (ROOT / "openpi/src/openpi/training/config.py").read_text()
    configs = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "TrainConfig":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                name = kwargs.get("name")
                if isinstance(name, ast.Constant):
                    configs[name.value] = kwargs
    benchmark = ROOT / "LIBERO-PRO/libero/libero/benchmark/libero_suite_task_map.py"
    assignment = ast.parse(benchmark.read_text()).body[0]
    tasks = ast.literal_eval(assignment.value)["libero_10"]
    assert len(list(LAUNCHERS.glob("run_midas_task*.sh"))) == 10
    for task_id, (slug, _) in enumerate(TASKS):
        config = configs[f"pi05_libero10_{slug}_ep1"]
        data_kwargs = {kw.arg: kw.value for kw in config["data"].keywords}
        base_kwargs = {kw.arg: kw.value for kw in data_kwargs["base_config"].keywords}
        instruction = re.sub(r"^[A-Z_]+_SCENE\d+_", "", tasks[task_id]).replace("_", " ")
        assert ast.literal_eval(base_kwargs["filter_prompt"]) == instruction


def test_legacy_script_references_still_exist():
    scripts = list((LAUNCHERS / "legacy").glob("*"))
    assert len(scripts) == 9
    for script in scripts:
        source = script.read_text()
        assert "$SCRIPT_DIR/../../.." in source
        for reference in re.findall(r'\$REPO_DIR/(experiments/[^"\s]+)', source):
            assert (ROOT / reference).is_file(), (script, reference)
