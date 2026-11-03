"""Public launch scripts run directly and keep their experiment defaults."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = json.loads((ROOT / "tests/fixtures/experiment_cli_defaults.json").read_text())
ROBOCASA_TRAIN = [path for path in DEFAULTS if path.startswith("experiments/robocasa/")]
BC_CONFIGS = [
    "pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32",
    "pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37",
    "pi05_robocasa_bc_prepare_coffee_l25_s29",
]


@pytest.fixture
def portable_env(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "record = {'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'gpu_mask': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'entity': os.environ.get('WANDB_ENTITY')}\n"
        "with open(os.environ['PORTABLE_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
    )
    interpreter.chmod(0o755)
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("MIDAS_", "OPENPI_", "WANDB_", "BC_"))
        and key not in {
            "T_CKPT", "DEMO_HDF5", "CHECKPOINT_DIR", "BASE_POLICY_CHECKPOINT",
            "EVAL_ROOT", "OUTPUT_DIR", "PERTURBATION", "MODEL_SEED",
            "EVAL_NAME_OVERRIDE", "LIBERO_CONFIG_PATH", "ROBOCASA_EVAL_DATA_ROOT",
        }
    }
    checkpoint = tmp_path / "bc/12000"
    (checkpoint / "params").mkdir(parents=True)
    for config in BC_CONFIGS:
        stats = checkpoint / "assets" / config / "norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text("{}")
    midas_checkpoint = tmp_path / "midas/checkpoint200000"
    midas_checkpoint.mkdir(parents=True)
    data_root = tmp_path / "storage"
    data_root.mkdir()
    env.update(
        MIDAS_DATA_ROOT=str(data_root),
        MIDAS_PYTHON_BIN=str(interpreter),
        MIDAS_RUN_ID="portable-test",
        WANDB_ENTITY="public-test-entity",
        ROBOCASA_EVAL_DATA_ROOT=str(tmp_path / "replay-data"),
        LIBERO_CONFIG_PATH=str(tmp_path / "libero-config"),
        T_CKPT=str(checkpoint),
        BC_CHECKPOINT=str(checkpoint),
        BASE_POLICY_CHECKPOINT=str(checkpoint),
        CHECKPOINT_DIR=str(midas_checkpoint),
        CHECKPOINT_DIR_S11=str(midas_checkpoint),
        CHECKPOINT_DIR_S42=str(midas_checkpoint),
        MODEL_SEED="11",
        PERTURBATION="clean",
        PORTABLE_TEST_CALLS=str(tmp_path / "calls.jsonl"),
        CUDA_VISIBLE_DEVICES="2,3",
    )
    return env


def run_script(relative, environment, tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / relative)], env=environment, cwd=tmp_path,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return [
        json.loads(line)
        for line in Path(environment["PORTABLE_TEST_CALLS"]).read_text().splitlines()
    ]


def options(record):
    arguments = record["args"][3:]
    return {
        key: arguments[index + 1]
        for index, key in enumerate(arguments[:-1]) if key.startswith("--")
    }


@pytest.mark.parametrize("script", DEFAULTS)
def test_training_and_eval_defaults_are_preserved(script, portable_env, tmp_path):
    calls = run_script(script, portable_env, tmp_path)
    record = calls[-1]
    actual = options(record)
    for key, expected in DEFAULTS[script].items():
        assert actual[key] == expected, (script, key)
    assert record["cwd"] == str(ROOT)
    assert record["gpu_mask"] == "2,3"
    if script in ROBOCASA_TRAIN:
        assert record["entity"] == "public-test-entity"
        assert actual["--pi_05_ckpt_dir"] == portable_env["T_CKPT"]
    elif script.startswith("experiments/evals/"):
        assert actual["--pi_05_ckpt_dir"] == portable_env["BC_CHECKPOINT"]


@pytest.mark.parametrize("script", ROBOCASA_TRAIN)
def test_repeat_wrappers_keep_training_defaults(script, portable_env, tmp_path):
    repeat = script.removesuffix(".sh") + "_seed_repro_test_repeat.sh"
    calls = run_script(repeat, portable_env, tmp_path)
    actual = options(calls[-1])
    for key, expected in DEFAULTS[script].items():
        assert actual[key] == expected
    assert "seed_repro_test_repeat_portable-test" in actual["--exp_name"]


def test_evaluation_matrix_runs_sequentially(portable_env, tmp_path):
    matrix = "experiments/libero/legacy/run_midas_task8_checkpoint200000_eval_matrix.sh"
    calls = run_script(matrix, portable_env, tmp_path)
    assert len(calls) == 8
    names = [options(call)["--exp_name"] for call in calls]
    assert len(set(names)) == 8
    assert all(options(call)["--seed"] == "0" for call in calls)
    assert sum("s11" in name for name in names) == 4
    assert sum("s42" in name for name in names) == 4


@pytest.mark.parametrize("script", ROBOCASA_TRAIN)
def test_default_names_are_unique_and_include_process_id(script, portable_env, tmp_path):
    portable_env.pop("MIDAS_RUN_ID")
    first = options(run_script(script, portable_env, tmp_path)[-1])["--exp_name"]
    second = options(run_script(script, portable_env, tmp_path)[-1])["--exp_name"]
    assert first != second
    assert re.search(r"\d{8}T\d{6}_\d+$", first)
    assert re.search(r"\d{8}T\d{6}_\d+$", second)


@pytest.mark.parametrize("profile", ["libero", "robocasa"])
def test_active_environment_requires_no_conda(profile, portable_env, tmp_path):
    portable_env.pop("MIDAS_PYTHON_BIN")
    portable_env["PATH"] = str(tmp_path) + os.pathsep + portable_env["PATH"]
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; midas_select_python "$2"; printf "%s" "$PYTHON_BIN"',
         "environment-test", str(ROOT / "scripts/experiment_env.sh"), profile],
        env=portable_env, cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "python"


def test_public_scripts_have_no_scheduler_or_site_dependencies():
    experiment_root = ROOT / "experiments"
    assert not list(experiment_root.rglob("*.slurm"))
    forbidden = re.compile(
        r"SBATCH|SLURM_|\bsbatch\b|\bsqueue\b|\bsacct\b|\bscancel\b|"
        r"maxlab|carnegie-mellon|babel-|/data/hf_cache|/home/(?:skowshik|sreyas)"
    )
    paths = list(experiment_root.rglob("*.sh")) + list(experiment_root.rglob("*.md"))
    paths += [ROOT / "README.md", ROOT / "docs/EXPERIMENT_PATHS.md"]
    for path in paths:
        assert not forbidden.search(path.read_text()), path
    for path in experiment_root.rglob("*.sh"):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, (path, result.stderr)


def test_documented_bash_commands_are_valid():
    docs = [ROOT / "README.md", *list((ROOT / "experiments").rglob("*.md"))]
    for path in docs:
        for block in re.findall(r"```bash\n(.*?)```", path.read_text(), flags=re.S):
            result = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
            assert result.returncode == 0, (path, result.stderr)
            for token in shlex.split(block, comments=True):
                if token.startswith("experiments/") and token.endswith(".sh"):
                    assert (ROOT / token).is_file(), (path, token)
