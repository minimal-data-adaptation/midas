"""Deterministically generate validated LIBERO BDDL perturbation suites.

The transform configuration files use the same nested mapping as LIBERO-PRO:
``suite -> task -> transform-specific options``. Generated suites contain only
BDDL files, use ordinary ``env.reset()``, and can be loaded in another process
with ``midas.utils.libero_utils.load_suite_manifest``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml

from midas.utils.libero_utils import TaskSpec, register_bddl_suite


@dataclass(frozen=True)
class PerturbFlags:
    """Enabled BDDL transforms, applied in canonical field order."""

    swap: bool = False
    environment: bool = False
    object: bool = False
    language: bool = False
    task: bool = False

    def enabled(self) -> tuple[str, ...]:
        return tuple(name for name in self.__dataclass_fields__ if getattr(self, name))


def _outer_block_span(text: str, head: str) -> tuple[int, int] | None:
    start = text.find(head)
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError(f"Unbalanced BDDL block beginning with {head!r}")


def _replace_block(text: str, head: str, replacement: str) -> str:
    span = _outer_block_span(text, head)
    if span is None:
        raise ValueError(f"BDDL file has no {head} block")
    return text[: span[0]] + replacement + text[span[1] :]


def _task_config(config: Mapping, suite: str, task: str):
    return (config.get(suite, {}) or {}).get(task)


def _swap(text: str, config: Mapping, suite: str, task: str, rng: random.Random) -> str:
    task_config = _task_config(config, suite, task)
    if not task_config:
        raise ValueError(f"No swap configuration for {suite}/{task}")
    objects = re.findall(r"\b([A-Za-z][\w]*_\d+)\b", (_block(text, "(:obj_of_interest") or ""))
    locations = dict(re.findall(r"\(On\s+(\w+)\s+(\w+)\s*\)", text))
    rng.shuffle(objects)
    used: set[str] = set()
    changed = False
    for obj in objects:
        if obj in used:
            continue
        candidates = task_config.get(obj, task_config.get("__any__", [])) if isinstance(task_config, dict) else task_config
        candidates = [candidate for candidate in candidates or [] if candidate in locations and candidate not in used and candidate != obj]
        if not candidates or obj not in locations:
            continue
        other = rng.choice(candidates)
        first, second = locations[obj], locations[other]
        text, count_a = re.subn(rf"\(On\s+{re.escape(obj)}\s+{re.escape(first)}\s*\)", f"(On {obj} {second})", text, count=1)
        text, count_b = re.subn(rf"\(On\s+{re.escape(other)}\s+{re.escape(second)}\s*\)", f"(On {other} {first})", text, count=1)
        if count_a and count_b:
            used.update((obj, other))
            changed = True
    if not changed:
        raise ValueError(f"Swap configuration produced no change for {suite}/{task}")
    return text


def _block(text: str, head: str) -> str | None:
    span = _outer_block_span(text, head)
    return None if span is None else text[slice(*span)]


def _replace_objects(text: str, config: Mapping, suite: str, task: str, rng: random.Random) -> str:
    task_config = _task_config(config, suite, task)
    if not isinstance(task_config, dict) or not task_config:
        raise ValueError(f"No object configuration for {suite}/{task}")
    language = _block(text, "(:language")
    body = text.replace(language, "__MIDAS_LANGUAGE__", 1) if language else text
    changed = False
    for old, candidates in sorted(task_config.items()):
        if not old or not candidates:
            continue
        new = rng.choice(list(candidates))
        # Object categories also occur inside table-qualified region names,
        # e.g. ``kitchen_table_moka_pot_right_init_region``.  An underscore
        # before the category is therefore a valid boundary and must not
        # prevent replacement; otherwise declarations and region references
        # disagree in the generated BDDL.
        body, count = re.subn(
            rf"(?<![A-Za-z0-9]){re.escape(str(old))}(?=_|\b)", str(new), body
        )
        changed |= count > 0
    if not changed:
        raise ValueError(f"Object configuration produced no change for {suite}/{task}")
    return body.replace("__MIDAS_LANGUAGE__", language, 1) if language else body


def _replace_language(text: str, config: Mapping, suite: str, task: str, rng: random.Random) -> str:
    candidates = _task_config(config, suite, task)
    if not candidates:
        raise ValueError(f"No language configuration for {suite}/{task}")
    return _replace_block(text, "(:language", f"(:language {rng.choice(list(candidates))})")


def _replace_task(text: str, config: Mapping, suite: str, task: str, rng: random.Random) -> str:
    candidates = _task_config(config, suite, task)
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError(f"No task configuration for {suite}/{task}")
    language = rng.choice(sorted(candidates))
    selected = candidates[language]
    if not isinstance(selected, dict) or "goal" not in selected:
        raise ValueError(f"Malformed task configuration for {suite}/{task}: {language!r}")
    objects = selected.get("obj_of_interest", [])
    text = _replace_block(text, "(:language", f"(:language {language})")
    text = _replace_block(text, "(:goal", f"(:goal {selected['goal']})")
    return _replace_block(text, "(:obj_of_interest", "(:obj_of_interest\n    " + "\n    ".join(map(str, objects)) + "\n  )")


def _replace_environment(text: str, config: Mapping, suite: str, task: str, rng: random.Random) -> str:
    configured = _task_config(config, suite, task)
    if not configured:
        raise ValueError(f"No environment configuration for {suite}/{task}")
    current = str(configured[0] if isinstance(configured, list) else configured)
    environment_tokens = {
        "main_table": ("Tabletop", "table"),
        "kitchen_table": ("Kitchen_Tabletop", "kitchen_table"),
        "living_room_table": ("Living_Room_Tabletop", "living_room_table"),
        "study_table": ("Study_Tabletop", "study_table"),
        "floor": ("Floor", "floor"),
    }
    if current not in environment_tokens:
        raise ValueError(f"Unsupported current environment {current!r} for {suite}/{task}")
    replacement = rng.choice(sorted(set(environment_tokens) - {current}))
    updated = text.replace(current, replacement)
    token, fixture_type = environment_tokens[replacement]
    updated, count = re.subn(
        r"\(define\s*\(problem\s+LIBERO_[A-Za-z_]*\)",
        f"(define (problem LIBERO_{token}_Manipulation)",
        updated,
        count=1,
    )
    if not count:
        raise ValueError(f"Could not locate problem header in {suite}/{task}")
    fixtures = _block(updated, "(:fixtures")
    if fixtures is not None:
        rewritten = re.sub(
            rf"(^\s*{re.escape(replacement)}\s*-\s*)[A-Za-z_][A-Za-z0-9_]*",
            rf"\g<1>{fixture_type}",
            fixtures,
            count=1,
            flags=re.M,
        )
        updated = updated.replace(fixtures, rewritten, 1)
    return updated


_TRANSFORMS: dict[str, Callable[[str, Mapping, str, str, random.Random], str]] = {
    "swap": _swap,
    "environment": _replace_environment,
    "object": _replace_objects,
    "language": _replace_language,
    "task": _replace_task,
}

# Increment when transform semantics change so content-addressed suites are
# regenerated instead of reusing output produced by older transformation code.
_TRANSFORM_VERSION = 2


def perturb_content(
    content: str,
    *,
    suite_name: str,
    task_name: str,
    flags: PerturbFlags,
    configs: Mapping[str, Mapping],
    rng: random.Random,
    validator: Callable[[str], object] | None = None,
) -> str:
    """Apply enabled transforms and validate after every individual transform."""
    enabled = flags.enabled()
    if not enabled:
        raise ValueError("At least one perturbation flag is required")
    if flags.task and len(enabled) != 1:
        raise ValueError("The task transform cannot be combined with other transforms")
    for name in enabled:
        if name not in configs:
            raise ValueError(f"Missing --{name}-config for enabled transform")
        content = _TRANSFORMS[name](content, configs[name], suite_name, task_name, rng)
        if validator is not None:
            validator(content)
    return content


def _language(text: str) -> str:
    block = _block(text, "(:language")
    if block is None:
        return ""
    return re.sub(r"^\(:language\s*|\)$", "", block.strip(), flags=re.S).strip()


def _digest_inputs(input_files: list[Path], suite_name: str, flags: PerturbFlags, seed: int, configs: Mapping[str, Mapping]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "suite": suite_name,
                "flags": flags.enabled(),
                "seed": seed,
                "transform_version": _TRANSFORM_VERSION,
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(yaml.safe_dump(dict(configs), sort_keys=True).encode())
    for path in input_files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_suite(
    input_dir: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    suite_name: str,
    flags: PerturbFlags,
    configs: Mapping[str, Mapping],
    seed: int,
    *,
    validator: Callable[[str], object] | None = None,
) -> Path:
    """Generate or reuse an atomic content-addressed perturbation suite."""
    source = Path(input_dir).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    input_files = sorted(source.glob("*.bddl"))
    if not input_files:
        raise FileNotFoundError(f"No .bddl files found in {source}")
    if validator is None:
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        validator = robosuite_parse_problem
    digest = _digest_inputs(input_files, suite_name, flags, seed, configs)
    flag_label = "-".join(flags.enabled())
    cache_name = f"{suite_name}_{flag_label}_{seed}_{digest[:8]}"
    final_dir = destination_root / cache_name
    destination_root.mkdir(parents=True, exist_ok=True)
    lock_path = destination_root / f".{cache_name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = final_dir / "suite_meta.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("digest") == digest:
                cached_tasks = [TaskSpec(**task) for task in existing["tasks"]]
                register_bddl_suite(
                    existing["suite_name"], cached_tasks, final_dir / "bddl", init_policy="none"
                )
                return final_dir
            raise RuntimeError(f"Cache collision at {final_dir}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_name}.", dir=destination_root))
        try:
            bddl_root = temporary / "bddl"
            bddl_root.mkdir()
            rng = random.Random(seed)
            tasks: list[TaskSpec] = []
            output_hashes: dict[str, str] = {}
            for input_path in input_files:
                task_name = input_path.stem
                validation_path = temporary / f".validate_{input_path.name}"

                def validate_content(intermediate: str) -> None:
                    validation_path.write_text(intermediate, encoding="utf-8")
                    validator(str(validation_path))

                content = perturb_content(
                    input_path.read_text(encoding="utf-8"),
                    suite_name=suite_name,
                    task_name=task_name,
                    flags=flags,
                    configs=configs,
                    rng=rng,
                    validator=validate_content,
                )
                validation_path.unlink(missing_ok=True)
                output_path = bddl_root / input_path.name
                output_path.write_text(content, encoding="utf-8")
                validator(str(output_path))
                output_hashes[input_path.name] = hashlib.sha256(output_path.read_bytes()).hexdigest()
                tasks.append(TaskSpec(task_name, input_path.name, _language(content)))
            generated_name = cache_name.lower().replace("-", "_")
            manifest = {
                "schema_version": 1,
                "transform_version": _TRANSFORM_VERSION,
                "suite_name": generated_name,
                "source_suite": suite_name,
                "seed": seed,
                "flags": list(flags.enabled()),
                "digest": digest,
                "bddl_root": "bddl",
                "init_root": None,
                "init_policy": "none",
                "tasks": [asdict(task) for task in tasks],
                "output_sha256": output_hashes,
            }
            (temporary / "suite_meta.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.rename(temporary, final_dir)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    register_bddl_suite(generated_name, tasks, final_dir / "bddl", init_policy="none")
    return final_dir


def _load_yaml(path: str | None) -> Mapping:
    if path is None:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--seed", type=int, default=28)
    for name in _TRANSFORMS:
        parser.add_argument(f"--{name}", action="store_true")
        parser.add_argument(f"--{name}-config")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    flags = PerturbFlags(**{name: getattr(args, name) for name in _TRANSFORMS})
    configs = {name: _load_yaml(getattr(args, f"{name}_config")) for name in flags.enabled()}
    output = generate_suite(args.input_dir, args.output_root, args.suite_name, flags, configs, args.seed)
    print(output / "suite_meta.json")


if __name__ == "__main__":
    main()
