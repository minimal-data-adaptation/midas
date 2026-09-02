import yaml

from training.evaluation import evaluate_sim
from training.evaluation.position_perturbation import (
    get_perturb_objects,
    sample_disk_perturbation,
)


def test_prepare_arguments_forces_eval_only_and_preserves_training_flags():
    arguments = evaluate_sim.prepare_arguments(
        ["--env", "cartpole", "--checkpoint_dir", "/tmp/checkpoint"]
    )
    assert arguments[:4] == [
        "--env",
        "cartpole",
        "--checkpoint_dir",
        "/tmp/checkpoint",
    ]
    assert arguments[-2:] == ["--eval_only", "1"]


def test_inline_perturbation_generation_forwards_manifest(tmp_path, monkeypatch):
    input_dir = tmp_path / "bddl" / "libero_10"
    input_dir.mkdir(parents=True)
    config = tmp_path / "language.yaml"
    config.write_text("libero_10: {}\n", encoding="utf-8")
    eval_config = tmp_path / "evaluation.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "bddl_files_path": str(tmp_path / "bddl"),
                "ood_task_configs": {"language": str(config)},
            }
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    calls = []

    def fake_generate_suite(**kwargs):
        calls.append(kwargs)
        return generated

    monkeypatch.setattr("training.perturbation.generate_suite", fake_generate_suite)
    arguments = evaluate_sim.prepare_arguments(
        [
            "--checkpoint_dir",
            "/tmp/checkpoint",
            "--eval_config_path",
            str(eval_config),
            "--use_language",
            "1",
            "--perturb_seed",
            "7",
        ]
    )

    assert calls[0]["input_dir"] == input_dir
    assert calls[0]["flags"].language is True
    assert calls[0]["seed"] == 7
    assert arguments[-4:] == [
        "--suite_manifest",
        str(generated / "suite_meta.json"),
        "--eval_only",
        "1",
    ]


def test_position_samples_stay_inside_requested_disk():
    import numpy as np

    rng = np.random.RandomState(3)
    samples = [sample_disk_perturbation(rng, 0.05) for _ in range(100)]
    assert all(sample[2] == 0 for sample in samples)
    assert all(np.linalg.norm(sample[:2]) <= 0.05 for sample in samples)


def test_known_moka_pot_task_has_auto_position_targets():
    objects = get_perturb_objects(
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    )
    assert objects == ["moka_pot_1", "moka_pot_2"]
