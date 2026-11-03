import re
import subprocess
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


def test_no_personalized_absolute_paths():
    absolute_path = re.compile(r"(?<![:/])/(?:[^\s\"'`]+)")
    private_names = ("skowshik", "sreyas", "maxlab")
    offenders = []

    repositories = [ROOT, *(ROOT / name for name in ("openpi", "LIBERO-PRO", "robocasa"))]
    for repository in repositories:
        if repository != ROOT and not (repository / ".git").exists():
            continue
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout.decode().split("\0")
        for relative in tracked:
            path = repository / relative
            if not relative or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for match in absolute_path.finditer(text):
                candidate = match.group(0).lower()
                if any(name in candidate for name in private_names):
                    label = path.relative_to(ROOT)
                    offenders.append(f"{label}: {match.group(0)}")

    assert not offenders, "Personalized absolute paths found:\n" + "\n".join(offenders)
