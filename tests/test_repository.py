from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submodule_contract():
    text = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert text.count("branch = midas") == 3
    assert "https://github.com/shreyas-kowshik/openpi.git" in text
    assert "https://github.com/shreyas-kowshik/LIBERO-PRO.git" in text
    assert "https://github.com/shreyas-kowshik/robocasa.git" in text


def test_no_reconstructed_libero_bddl_path():
    retained = list((ROOT / "midas").rglob("*.py")) + list((ROOT / "training").rglob("*.py"))
    forbidden = "get_libero_path(" + repr("bddl_files")
    assert not [path for path in retained if forbidden in path.read_text(encoding="utf-8")]
