# RoboCasa

Use the dedicated RoboCasa environment because its robosuite/MuJoCo versions
differ from LIBERO. After installing the profile, download assets once:

```bash
python robocasa/robocasa/scripts/download_kitchen_assets.py
```

`--robocasa_env_name` accepts a direct environment name or a registered
single-task set. If the selected OpenPI data config defines
`eval_init_mode="exact_state_replay"`, MIDAS builds and attaches the reset
controller from `training/robocasa_eval_reset.py`; evaluation then restores the
recorded simulator state and task metadata. Dataset and checkpoint paths come
from the OpenPI config/CLI and are never hard-coded.

Headless evaluation normally needs `MUJOCO_GL=egl`. Set
`MUJOCO_EGL_DEVICE_ID` when rendering and JAX should use different GPUs.
