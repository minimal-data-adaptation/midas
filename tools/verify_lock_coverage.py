#!/usr/bin/env python3
"""Verify that a profile lock is installed at the versions it records."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
from pathlib import Path

from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
LOCKS = {
    "libero": ROOT / "locks" / "lock-libero.txt",
    "robocasa": ROOT / "locks" / "lock-robocasa.txt",
    "ci-cpu": ROOT / "locks" / "lock-ci-cpu.txt",
}


def locked_versions(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if match:
            records[canonicalize_name(match.group(1))] = match.group(2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(LOCKS), required=True)
    args = parser.parse_args()
    expected = locked_versions(LOCKS[args.profile])
    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    missing = sorted(name for name in expected if name not in installed)
    mismatched = sorted(
        f"{name}: locked {version}, installed {installed[name]}"
        for name, version in expected.items()
        if name in installed and installed[name] != version
    )
    if missing or mismatched:
        details = [*(f"missing: {name}" for name in missing), *mismatched]
        raise SystemExit("Lock verification failed:\n  " + "\n  ".join(details))
    print(f"Verified {len(expected)} locked packages for {args.profile}.")


if __name__ == "__main__":
    main()
