"""Evaluate a MIDAS checkpoint, with optional LIBERO perturbations.

Model, environment, and checkpoint construction is delegated to the training
launcher so evaluation cannot silently drift from the code that trained the
checkpoint. This wrapper additionally supports on-demand generation of
LIBERO BDDL perturbation suites.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

from training.launch_train_sim import main as train_main


_TRANSFORMS = ("swap", "environment", "object", "language", "task")


def _build_perturbation_parser() -> argparse.ArgumentParser:
    # Abbreviation must be disabled: otherwise argparse interprets the shared
    # ``--env`` flag as ``--environment_config`` and removes it from argv.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--eval_config_path",
        default="LIBERO-PRO/evaluation_config.yaml",
        help="LIBERO-PRO config used to locate default BDDL transform YAMLs.",
    )
    parser.add_argument(
        "--perturb_input_dir",
        default=None,
        help="Source BDDL directory; defaults to <bddl_files_path>/<suite>.",
    )
    parser.add_argument(
        "--perturb_output_root",
        default=os.environ.get("MIDAS_SUITE_CACHE", "~/.cache/midas/suites"),
        help="Cache root for generated, content-addressed suites.",
    )
    parser.add_argument("--perturb_seed", default=None, type=int)
    for name in _TRANSFORMS:
        parser.add_argument(f"--use_{name}", default=0, type=int)
        parser.add_argument(f"--{name}_config", default=None)
    return parser


def _option_value(arguments: list[str], option: str, default: str) -> str:
    """Read an option from an argv list without consuming it."""
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = option + "="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return default


def _resolve_config_path(value: str, eval_config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        eval_config_path.parent / path,
        eval_config_path.parent.parent / path,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _maybe_generate_suite(arguments: list[str]) -> list[str]:
    parser = _build_perturbation_parser()
    perturb_args, remaining = parser.parse_known_args(arguments)
    enabled = {
        name: bool(getattr(perturb_args, f"use_{name}")) for name in _TRANSFORMS
    }
    if not any(enabled.values()):
        return remaining
    if _option_value(remaining, "--env", "libero") != "libero":
        parser.error("BDDL perturbation flags are supported only with --env libero")
    if _option_value(remaining, "--suite_manifest", ""):
        parser.error("Pass either --suite_manifest or --use_<transform>, not both")

    eval_config_path = Path(perturb_args.eval_config_path).expanduser().resolve()
    payload = yaml.safe_load(eval_config_path.read_text(encoding="utf-8")) or {}
    suite_name = _option_value(remaining, "--task_suite_name", "libero_10")
    seed = (
        perturb_args.perturb_seed
        if perturb_args.perturb_seed is not None
        else int(_option_value(remaining, "--seed", "42"))
    )
    if perturb_args.perturb_input_dir:
        input_dir = Path(perturb_args.perturb_input_dir).expanduser().resolve()
    else:
        bddl_root = payload.get(
            "bddl_files_path", "LIBERO-PRO/libero/libero/bddl_files"
        )
        input_dir = _resolve_config_path(str(bddl_root), eval_config_path) / suite_name

    default_configs = payload.get("ood_task_configs", {}) or {}
    from training.perturbation import PerturbFlags, _load_yaml, generate_suite

    configs = {}
    for name, is_enabled in enabled.items():
        if not is_enabled:
            continue
        configured = (
            getattr(perturb_args, f"{name}_config") or default_configs.get(name)
        )
        if not configured:
            parser.error(
                f"--use_{name}=1 requires --{name}_config or an "
                f"ood_task_configs.{name} entry in --eval_config_path"
            )
        configs[name] = _load_yaml(
            str(_resolve_config_path(configured, eval_config_path))
        )

    generated = generate_suite(
        input_dir=input_dir,
        output_root=Path(perturb_args.perturb_output_root).expanduser(),
        suite_name=suite_name,
        flags=PerturbFlags(**enabled),
        configs=configs,
        seed=seed,
    )
    manifest = generated / "suite_meta.json"
    print(f"[Eval] generated perturbation suite: {manifest}")
    return [*remaining, "--suite_manifest", str(manifest)]


def prepare_arguments(argv: list[str]) -> list[str]:
    """Consume evaluator-only flags and force the shared eval-only path."""
    arguments = _maybe_generate_suite(list(argv))
    if "--eval_only" not in arguments:
        arguments.extend(["--eval_only", "1"])
    return arguments


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print("Evaluator-only LIBERO BDDL perturbation options:")
        _build_perturbation_parser().print_help()
        print("\nShared MIDAS training/evaluation options:")
        _, shared_arguments = _build_perturbation_parser().parse_known_args(arguments)
        train_main(shared_arguments)
        return
    train_main(prepare_arguments(arguments))


if __name__ == "__main__":
    main()
