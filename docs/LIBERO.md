# LIBERO

`LIBERO-PRO/` is the sole `libero` distribution in this repository. Set
`LIBERO_CONFIG_PATH` before importing it so its generated `config.yaml` is
outside the Git worktree. The default BDDL, init-state, and asset paths still
point into the relocatable submodule; edit only the dataset entry in the
generated config if demonstrations live elsewhere.

Training resolves each task through `get_task_bddl_file_path`; it never
reconstructs a BDDL path. Built-in suites use their packaged initial states.
Generated perturbation suites declare `init_policy=none` and use `env.reset()`.

Generate a deterministic language suite, for example:

```bash
python -m training.perturbation \
  --input-dir LIBERO-PRO/libero/libero/bddl_files/libero_10 \
  --output-root "$HOME/.cache/midas/suites" \
  --suite-name libero_10 --seed 28 --language \
  --language-config LIBERO-PRO/libero_ood/ood_language.yaml
```

The command prints a `suite_meta.json` path. Pass it to training/evaluation as
`--suite_manifest <path>`. Generation is validated, content-addressed,
lock-protected, and atomically published. The task transform is intentionally
mutually exclusive; other supported transforms execute in the fixed order
swap, environment, object, language.

## Checkpoint evaluation

Use `training.evaluation.evaluate_sim` and repeat every learner-shape and policy
flag from the training launch. In particular, values such as `--hidden_dims`,
`--resize_image`, `--predict_a_exec`, `--use_vlm_embedding`, actor/critic
architectures, query frequency, and chunk length must match the checkpoint.

Normal LIBERO evaluation:

```bash
python -m training.evaluation.evaluate_sim \
  --checkpoint_dir /path/to/run-or-checkpoint \
  --output_dir /path/to/eval/normal \
  --num_evals 50 \
  --task_suite_name libero_10 --task_id 8 \
  --pi_05_config <openpi-config> \
  --pi_05_ckpt_dir <base-policy-checkpoint> \
  <matching-MIDAS-training-flags>
```

The output contains `summary.csv`, `summary.json`, and one MP4 per rollout in
`videos/`. Use `--save_eval_videos 0` to omit MP4s. Built-in suites cycle over
fixed initialization states by default; `--round_robin_init_states 0` samples
them using `--seed`. Generated suites have no fixed init-state files and use
seeded stochastic BDDL placement.

An existing generated perturbation suite can be evaluated with:

```bash
python -m training.evaluation.evaluate_sim \
  <normal-evaluation-flags> \
  --suite_manifest /path/to/generated/suite_meta.json
```

Alternatively, the evaluator can generate and cache the suite itself. These
examples use transform paths from `LIBERO-PRO/evaluation_config.yaml`:

```bash
# Object replacement
python -m training.evaluation.evaluate_sim \
  <normal-evaluation-flags> --use_object 1 --perturb_seed 28

# Language paraphrase
python -m training.evaluation.evaluate_sim \
  <normal-evaluation-flags> --use_language 1 --perturb_seed 28

# Combined object + language perturbation
python -m training.evaluation.evaluate_sim \
  <normal-evaluation-flags> --use_object 1 --use_language 1 --perturb_seed 28
```

The analogous flags are `--use_swap`, `--use_environment`, and `--use_task`.
Override a transform config with `--object_config`, `--language_config`,
`--swap_config`, `--environment_config`, or `--task_config`. Override the source
BDDL directory with `--perturb_input_dir` and the cache root with
`--perturb_output_root` (default `~/.cache/midas/suites`). Task perturbation is
mutually exclusive with the other transforms.

Position plus yaw perturbations happen at reset time and therefore work with
both ordinary and generated suites:

```bash
# Known task: object names inferred from the task name
python -m training.evaluation.evaluate_sim \
  <normal-evaluation-flags> \
  --pos_perturb_radius 0.05 --pos_perturb_objects auto

# Explicit BDDL object names, useful for generated object suites
python -m training.evaluation.evaluate_sim \
  <normal-evaluation-flags> \
  --pos_perturb_radius 0.10 \
  --pos_perturb_objects moka_pot_1,moka_pot_2
```

The position helper perturbs XY within a disk and yaw within a radius-scaled
range, then rejection-samples for visibility, collision, height, drift, and
uprightness after physics settling. `--pos_perturb_settle_secs` controls the
pinned settle period and defaults to 5 seconds of simulation time.
