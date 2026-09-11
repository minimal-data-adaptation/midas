# Setup

Use the commands in the README for the normal setup. They are the tested source
of truth: Conda installs only Python 3.11.11 and pip, then the selected hashed
lock owns the complete dependency set.

## System prerequisites

On Ubuntu, install Git/Git LFS, build tools, FFmpeg, and common MuJoCo/OpenGL
libraries. Exact package names vary by distro; typical names include
`build-essential`, `cmake`, `ffmpeg`, `libegl1`, `libgl1`, `libglfw3`,
`libglew2.2`, and `patchelf`. GPU execution additionally needs an NVIDIA driver
that supports CUDA 12. JAX CUDA wheels contain CUDA user-space libraries; a full
CUDA toolkit is normally unnecessary.

## Profiles and locks

- `libero`: LIBERO-PRO plus `robosuite==1.4.1` and `mujoco==3.3.1`.
- `robocasa`: RoboCasa plus `robosuite==1.5.2` and `mujoco==3.3.1`.
- `ci-cpu`: the LIBERO profile plus test/build tooling.
- `real`: OpenPI plus LeRobot 0.3.3 and real server/evaluation dependencies;
  the externally supplied `yam_teleop` hardware package is intentionally not locked.

Rebuild locks only when dependency inputs intentionally change:

```bash
python -m pip install uv
uv pip compile requirements/libero.in -o locks/lock-libero.txt --generate-hashes --python-version 3.11
uv pip compile requirements/robocasa.in -o locks/lock-robocasa.txt --generate-hashes --python-version 3.11
uv pip compile requirements/ci-cpu.in -o locks/lock-ci-cpu.txt --generate-hashes --python-version 3.11
uv pip compile requirements/real.in -o locks/lock-real.txt --generate-hashes --python-version 3.11
```

`scripts/install_profile.sh` installs the chosen lock first and then installs
MIDAS, openpi, openpi-client, and the simulator fork editably with `--no-deps`.
`tools/verify_lock_coverage.py` verifies every locked package and version.
The real profile installs no simulator fork.

## Environment variables

- `LIBERO_CONFIG_PATH`: required before the first `libero` import; keeps the
  generated config out of the submodule.
- `OPENPI_DATA_HOME`: optional OpenPI download/cache root.
- `MIDAS_EXP_DIR`: experiment output root; defaults to `experiments`.
- `MUJOCO_GL=egl`: headless GPU rendering (`osmesa` is a CPU alternative).
- `MUJOCO_EGL_DEVICE_ID`: renderer GPU index on multi-GPU hosts.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: avoid reserving all GPU memory.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: optional JAX memory ceiling.

## `pip check` exceptions

The forks are deliberately consumed unchanged. The RoboCasa fork metadata pins
`numpy==2.2.5` and declares `lerobot==0.3.3`; MIDAS uses NumPy 1.26.4 to remain
compatible with OpenPI; the real profile installs LeRobot separately. OpenPI
metadata also declares `gym-aloha`, which is omitted because ALOHA is not used
by the simulation or YAM workflows. These metadata complaints are expected
after an otherwise successful profile setup. Runtime dependencies used by the
port are locked and covered by smoke tests.

If an import resolves to a different `libero`, uninstall that distribution and
rerun `python -m pip install --no-deps --no-build-isolation -e LIBERO-PRO`.
