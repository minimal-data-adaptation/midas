import ast
from pathlib import Path


def test_real_evaluator_has_no_training_or_update_path():
    source = Path("training/evaluation/evaluate_real.py").read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "training.train_real" not in imported | imported_from
    assert "midas.real.learner" not in imported | imported_from
    assert not any(
        isinstance(node, ast.Attribute) and node.attr.startswith("update_")
        for node in ast.walk(tree)
    )


def test_simulation_entry_points_do_not_import_real_stack():
    for path in (
        "training/launch_train_sim.py",
        "training/train_sim.py",
        "training/train_utils_sim.py",
        "training/evaluation/evaluate_sim.py",
    ):
        source = Path(path).read_text()
        assert "midas.real" not in source
        assert "yam_teleop" not in source
        assert "train_real" not in source
