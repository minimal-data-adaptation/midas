import json
import random

import pytest

import training.perturbation as perturbation


BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language move the cup)
  (:fixtures main_table - table)
  (:objects cup_1 - cup bowl_1 - bowl)
  (:obj_of_interest cup_1 bowl_1)
  (:init (On cup_1 region_a) (On bowl_1 region_b))
  (:goal (And (On cup_1 bowl_1)))
)"""


def test_language_and_object_pipeline_is_deterministic():
    flags = perturbation.PerturbFlags(object=True, language=True)
    configs = {
        "object": {"suite": {"task": {"cup": ["mug"]}}},
        "language": {"suite": {"task": ["place the mug"]}},
    }
    first = perturbation.perturb_content(
        BDDL, suite_name="suite", task_name="task", flags=flags,
        configs=configs, rng=random.Random(3), validator=lambda _: None,
    )
    second = perturbation.perturb_content(
        BDDL, suite_name="suite", task_name="task", flags=flags,
        configs=configs, rng=random.Random(3), validator=lambda _: None,
    )
    assert first == second
    assert "mug_1" in first
    assert "(:language place the mug)" in first


def test_object_pipeline_replaces_table_qualified_region_references():
    source = BDDL.replace(
        "(On cup_1 region_a)",
        "(On cup_1 main_table_cup_init_region)",
    )
    transformed = perturbation.perturb_content(
        source,
        suite_name="suite",
        task_name="task",
        flags=perturbation.PerturbFlags(object=True),
        configs={"object": {"suite": {"task": {"cup": ["mug"]}}}},
        rng=random.Random(3),
        validator=lambda _: None,
    )

    assert "mug_1 - mug" in transformed
    assert "main_table_mug_init_region" in transformed
    assert "main_table_cup_init_region" not in transformed


def test_task_transform_rejects_combinations():
    with pytest.raises(ValueError, match="cannot be combined"):
        perturbation.perturb_content(
            BDDL, suite_name="suite", task_name="task",
            flags=perturbation.PerturbFlags(task=True, language=True),
            configs={"task": {}, "language": {}}, rng=random.Random(0),
        )


def test_atomic_suite_generation_and_cache(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    (inputs / "task.bddl").write_text(BDDL, encoding="utf-8")
    registered = []
    monkeypatch.setattr(perturbation, "register_bddl_suite", lambda *args, **kwargs: registered.append((args, kwargs)))
    flags = perturbation.PerturbFlags(language=True)
    configs = {"language": {"suite": {"task": ["new instruction"]}}}
    validator = lambda path: None
    first = perturbation.generate_suite(inputs, outputs, "suite", flags, configs, 7, validator=validator)
    second = perturbation.generate_suite(inputs, outputs, "suite", flags, configs, 7, validator=validator)
    assert first == second
    manifest = json.loads((first / "suite_meta.json").read_text(encoding="utf-8"))
    assert manifest["init_policy"] == "none"
    assert manifest["tasks"][0]["language"] == "new instruction"
    assert len(list(outputs.glob("suite_language_7_*"))) == 1
    assert registered
