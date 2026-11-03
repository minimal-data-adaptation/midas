"""Download and integrity-check public MIDAS artifacts."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen


def load_artifacts_index() -> dict:
    index = files("midas").joinpath("artifacts_index.json")
    return json.loads(index.read_text(encoding="utf-8"))


def download_artifact(name: str, destination: str | Path) -> Path:
    record = load_artifacts_index()["artifacts"].get(name)
    if record is None:
        raise KeyError(f"Unknown MIDAS artifact: {name}")
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(record["url"], timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != record["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for artifact {name}")
    temporary.replace(destination)
    return destination
