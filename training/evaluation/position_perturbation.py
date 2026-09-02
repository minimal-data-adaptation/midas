"""Position + yaw perturbation utilities for LIBERO evaluation.

Perturbs target-object positions and yaw orientations in-place after
env.reset() / env.set_init_state(), without modifying any LIBERO-PRO code.

Perturbation levels (controlled by a single ``radius`` parameter):
    5 cm  →  ±5 cm XY  +  ±0.10 rad yaw (~6°)
    10 cm →  ±10 cm XY +  ±0.15 rad yaw (~9°)
    20 cm →  ±20 cm XY +  ±0.25 rad yaw (~14°)

Includes collision detection, z-position stability check, orientation
(upright) check, and rejection-resampling.
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_disk_perturbation(rng, radius):
    """Sample a 2D (XY) offset uniformly inside a disk of given radius.

    Z is left at zero so objects stay on the table surface.

    Args:
        rng: np.random.RandomState instance.
        radius: disk radius in meters.

    Returns:
        np.ndarray of shape (3,) with [dx, dy, 0].
    """
    angle = rng.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(rng.uniform())
    return np.array([r * np.cos(angle), r * np.sin(angle), 0.0])


def sample_yaw_perturbation(rng, max_yaw):
    """Sample a yaw angle uniformly in [-max_yaw, +max_yaw] radians."""
    return rng.uniform(-max_yaw, max_yaw)


def _quat_multiply(q1, q2):
    """Multiply two quaternions in MuJoCo [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _yaw_to_quat(yaw):
    """Convert a yaw angle (rotation about world z) to a [w,x,y,z] quaternion."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _get_object_geom_names(sim, object_name):
    """Return the set of MuJoCo geom names belonging to *object_name*.

    Robosuite prefixes all geom names with the object name, so we collect
    every geom whose name contains *object_name* as a substring.
    """
    geoms = set()
    for i in range(sim.model.ngeom):
        gname = sim.model.geom_id2name(i)
        if gname and object_name in gname:
            geoms.add(gname)
    return geoms


def _has_object_collision(env, object_names):
    """Check if any perturbed object collides with another scene object.

    Only flags contacts between a perturbed object and a *different* object
    (perturbed or non-perturbed) in the scene.  Robot and table contacts are
    ignored to avoid false positives.

    Must be called after ``sim.forward()`` so the contact array is populated.
    """
    sim = env.env.sim
    all_obj_names = list(env.env.objects_dict.keys())

    # Build geom-name sets for perturbed objects and all other scene objects.
    perturbed_geoms = {}  # obj_name -> set of geom names
    for name in object_names:
        perturbed_geoms[name] = _get_object_geom_names(sim, name)

    other_geoms = {}  # obj_name -> set of geom names
    for name in all_obj_names:
        if name not in perturbed_geoms:
            other_geoms[name] = _get_object_geom_names(sim, name)

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        g1 = sim.model.geom_id2name(contact.geom1)
        g2 = sim.model.geom_id2name(contact.geom2)

        for pname, pgeoms in perturbed_geoms.items():
            if g1 in pgeoms or g2 in pgeoms:
                other_g = g2 if g1 in pgeoms else g1
                # Check against other perturbed objects
                for oname, ogeoms in perturbed_geoms.items():
                    if oname != pname and other_g in ogeoms:
                        return True
                # Check against non-perturbed scene objects
                for oname, ogeoms in other_geoms.items():
                    if other_g in ogeoms:
                        return True
    return False


def _settle_with_pinned_robot(env, settle_steps):
    """Run *settle_steps* physics steps while holding the robot frozen.

    Without this the robot arm collapses under gravity during the settle
    period because no controller is active.
    """
    sim = env.env.sim
    robot = env.env.robots[0]
    robot_qpos_idx = robot._ref_joint_pos_indexes
    robot_qvel_idx = robot._ref_joint_vel_indexes
    saved_robot_qpos = np.array(sim.data.qpos[robot_qpos_idx])
    for _ in range(settle_steps):
        sim.step()
        # Pin robot joints back to their initial pose each step
        sim.data.qpos[robot_qpos_idx] = saved_robot_qpos
        sim.data.qvel[robot_qvel_idx] = 0.0
    sim.forward()


def _settle_with_pinned_objects(env, settle_steps, object_pin_info,
                               frozen_pin_info=None):
    """Run *settle_steps* physics steps while holding the robot frozen,
    pinning perturbed objects' XY/orientation, and freezing all other
    scene objects completely.

    Only allows perturbed objects' z-position to evolve under gravity,
    preventing bouncing-induced orientation changes from collisions with
    other objects (e.g. objects landing on drawers during 10cm/20cm
    perturbations) and preventing non-perturbed objects from shifting
    configuration during settling.

    Args:
        env: LIBERO OffScreenRenderEnv.
        settle_steps: number of low-level sim steps.
        object_pin_info: dict mapping obj_name -> dict with keys:
            'addr' (int): qpos start address for the 7-element free-joint state.
            'vel_addr' (int): qvel start address for the 6-element free-joint velocity.
            'target_xy' (np.ndarray, shape (2,)): desired XY position.
            'target_quat' (np.ndarray, shape (4,)): desired [w,x,y,z] quaternion.
        frozen_pin_info: optional dict mapping obj_name -> dict with keys:
            'addr' (int): qpos start address.
            'vel_addr' (int): qvel start address.
            'saved_qpos' (np.ndarray, shape (7,)): full qpos to restore each step.
            These objects are completely frozen (all DOF pinned).
    """
    sim = env.env.sim
    robot = env.env.robots[0]
    robot_qpos_idx = robot._ref_joint_pos_indexes
    robot_qvel_idx = robot._ref_joint_vel_indexes
    saved_robot_qpos = np.array(sim.data.qpos[robot_qpos_idx])
    for _ in range(settle_steps):
        sim.step()
        # Pin robot joints
        sim.data.qpos[robot_qpos_idx] = saved_robot_qpos
        sim.data.qvel[robot_qvel_idx] = 0.0
        # Pin each perturbed object's XY position and orientation;
        # only z-position and z-velocity evolve naturally under gravity.
        for info in object_pin_info.values():
            a = info['addr']
            va = info['vel_addr']
            sim.data.qpos[a:a + 2] = info['target_xy']        # pin x, y
            sim.data.qpos[a + 3:a + 7] = info['target_quat']  # pin quaternion
            sim.data.qvel[va] = 0.0                            # zero vx
            sim.data.qvel[va + 1] = 0.0                        # zero vy
            sim.data.qvel[va + 3:va + 6] = 0.0                 # zero wx, wy, wz
            sim.data.qvel[va + 2] *= 0.95                      # damp vz (reduce bouncing)
        # Freeze all non-perturbed objects completely (no movement at all).
        if frozen_pin_info:
            for info in frozen_pin_info.values():
                a = info['addr']
                va = info['vel_addr']
                sim.data.qpos[a:a + 7] = info['saved_qpos']
                sim.data.qvel[va:va + 6] = 0.0
    sim.forward()


def _free_settle_perturbed_objects(env, settle_steps, frozen_pin_info=None):
    """Brief settle with robot pinned, non-perturbed objects frozen,
    and perturbed objects completely free (all 6 DOF).

    This reveals physically unstable configurations that the pinned settle
    phase masks — objects resting on edges, interpenetrating other bodies,
    or in positions where they would immediately tip/slide once the pinning
    constraints are removed during an actual rollout.

    Args:
        env: LIBERO OffScreenRenderEnv.
        settle_steps: number of low-level sim steps.
        frozen_pin_info: optional dict mapping obj_name -> dict with keys
            'addr', 'vel_addr', 'saved_qpos' for objects to keep frozen.
    """
    sim = env.env.sim
    robot = env.env.robots[0]
    robot_qpos_idx = robot._ref_joint_pos_indexes
    robot_qvel_idx = robot._ref_joint_vel_indexes
    saved_robot_qpos = np.array(sim.data.qpos[robot_qpos_idx])
    for _ in range(settle_steps):
        sim.step()
        # Pin robot joints
        sim.data.qpos[robot_qpos_idx] = saved_robot_qpos
        sim.data.qvel[robot_qvel_idx] = 0.0
        # Freeze non-perturbed objects
        if frozen_pin_info:
            for info in frozen_pin_info.values():
                a = info['addr']
                va = info['vel_addr']
                sim.data.qpos[a:a + 7] = info['saved_qpos']
                sim.data.qvel[va:va + 6] = 0.0
    sim.forward()


def _object_is_upright(sim, obj_body_id, ref_up_axis=None, cos_threshold=0.95):
    """Check if an object has not tipped over relative to its original orientation.

    Some objects have body frames where the local z-axis does NOT point up
    (e.g. the alphabet_soup box).  To handle all objects correctly, this
    function accepts a *ref_up_axis* (int 0/1/2) indicating which column of
    the body rotation matrix pointed most upward in the **unperturbed** state.

    If *ref_up_axis* is ``None`` the function auto-detects by checking all
    three body axes and picking the one closest to world-up.  However it is
    more reliable to pass a pre-computed value.

    Returns True if the reference body axis still has a dot product with the
    world z-axis above *cos_threshold* (default 0.95 ≈ 18° tilt tolerance).
    """
    xmat = sim.data.body_xmat[obj_body_id].reshape(3, 3)
    if ref_up_axis is None:
        # Pick the body axis whose world-z component has the largest magnitude
        dots = np.abs(xmat[2, :])  # world-z component of each body axis
        ref_up_axis = int(np.argmax(dots))
    body_axis = xmat[:, ref_up_axis]
    return abs(body_axis[2]) >= cos_threshold


def perturb_object_positions(env, object_names, radius, rng, settle_secs=5.0):
    """Perturb named objects' XY positions and yaw, then regenerate observations.

    Perturbation magnitudes are tied to *radius*:
        radius (m)  |  XY offset      |  yaw offset
        0.05        |  ±5 cm disk      |  ±0.10 rad (~6°)
        0.10        |  ±10 cm disk     |  ±0.15 rad (~9°)
        0.20        |  ±20 cm disk     |  ±0.25 rad (~14°)

    After perturbation the physics are stepped for *settle_secs* seconds of
    sim-time so that objects come to rest on the table.  The robot arm is held
    frozen during settling to prevent it from collapsing under gravity.

    Rejection checks (up to 50 attempts):
      - Camera visibility: all perturbed objects must be in frame.
      - Collision: no perturbed object may overlap another scene object.
      - Z-position: objects must stay within 2 cm of their settled
        baseline z (computed once, without perturbation).  Catches objects
        that fell off surfaces or landed on other surfaces (e.g. drawers).
      - Upright: objects must remain upright (local z-axis within ~18° of
        world z-axis).

    Args:
        env: LIBERO OffScreenRenderEnv (env.env is the BDDLBaseDomain).
        object_names: list of BDDL object names to perturb (e.g. ["moka_pot_1"]).
        radius: perturbation disk radius in meters (e.g. 0.05 for 5 cm).
        rng: np.random.RandomState for reproducibility.
        settle_secs: seconds of sim-time to let physics settle after each
            perturbation attempt (default 5.0).

    Returns:
        obs: refreshed observation dict after perturbation.
    """
    MAX_RETRIES = 50
    Z_THRESHOLD = 0.02  # reject if object z deviates > 2 cm from settled baseline

    # Yaw perturbation lookup keyed by radius level
    _YAW_FOR_RADIUS = {0.05: 0.10, 0.10: 0.15, 0.20: 0.25}
    max_yaw = _YAW_FOR_RADIUS.get(round(radius, 2), radius * 2)
    sim = env.env.sim

    # Number of low-level sim steps for the requested settle duration.
    # MuJoCo default timestep = 0.002 s => 2500 steps for 5 s.
    settle_steps = int(settle_secs / sim.model.opt.timestep)

    # Save the *full* simulator state so we can restore between retries
    # (settling mutates qpos/qvel of every body, not just the perturbed ones).
    full_state = sim.get_state().flatten().copy()

    # Save original qpos slice and velocity address for each perturbed object
    original_qpos = {}
    for obj_name in object_names:
        obj = env.env.objects_dict[obj_name]
        joint_name = obj.joints[-1]
        addr = sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):
            addr = addr[0]
        vel_addr = sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(vel_addr, tuple):
            vel_addr = vel_addr[0]
        original_qpos[obj_name] = (joint_name, addr, vel_addr, sim.data.qpos[addr : addr + 7].copy())

    # Collect freeze info for ALL non-perturbed scene objects so they
    # cannot shift configuration during settling (e.g. bottle tipping).
    perturbed_set = set(object_names)
    frozen_obj_info = {}
    for obj_name, obj in env.env.objects_dict.items():
        if obj_name in perturbed_set:
            continue
        if not hasattr(obj, 'joints') or not obj.joints:
            continue
        joint_name = obj.joints[-1]
        addr = sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):
            addr = addr[0]
        vel_addr = sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(vel_addr, tuple):
            vel_addr = vel_addr[0]
        frozen_obj_info[obj_name] = {
            'addr': addr,
            'vel_addr': vel_addr,
            'saved_qpos': sim.data.qpos[addr:addr + 7].copy(),
        }

    # ------------------------------------------------------------------
    # Compute settled baseline z for each object (settle WITHOUT perturbation).
    # This lets us detect objects that fall during perturbed settling.
    # ------------------------------------------------------------------
    _settle_with_pinned_robot(env, settle_steps)
    baseline_z = {}
    ref_up_axis = {}
    for obj_name in object_names:
        body_id = env.env.obj_body_id[obj_name]
        baseline_z[obj_name] = float(sim.data.body_xpos[body_id][2])
        # Record which body axis points most upward in the unperturbed state.
        xmat = sim.data.body_xmat[body_id].reshape(3, 3)
        ref_up_axis[obj_name] = int(np.argmax(np.abs(xmat[2, :])))

    for attempt in range(MAX_RETRIES):
        # Restore full state (undoes any settling from prior attempt)
        sim.set_state_from_flattened(full_state)
        sim.forward()

        # Apply fresh random position + yaw perturbations and build pin info
        object_pin_info = {}
        for obj_name, (joint_name, addr, vel_addr, qpos_orig) in original_qpos.items():
            new_qpos = qpos_orig.copy()

            # XY perturbation (uniform disk)
            delta = sample_disk_perturbation(rng, radius)
            new_qpos[:3] += delta

            # Yaw perturbation (uniform ±max_yaw, composed with existing orientation)
            dyaw = sample_yaw_perturbation(rng, max_yaw)
            q_orig = new_qpos[3:7]  # [w, x, y, z]
            q_yaw = _yaw_to_quat(dyaw)
            new_qpos[3:7] = _quat_multiply(q_yaw, q_orig)

            sim.data.set_joint_qpos(joint_name, new_qpos)

            object_pin_info[obj_name] = {
                'addr': addr,
                'vel_addr': vel_addr,
                'target_xy': new_qpos[:2].copy(),
                'target_quat': new_qpos[3:7].copy(),
            }

        # Phase 1: Settle with pinned object XY/orientation (only z settles)
        # and all non-perturbed objects completely frozen.
        sim.forward()
        _settle_with_pinned_objects(env, settle_steps, object_pin_info,
                                   frozen_obj_info)

        # Phase 2: Free settle — release all DOF for perturbed objects so
        # physically unstable placements (resting on edges, on top of drawers,
        # interpenetrating other objects) are revealed before rollout begins.
        free_settle_steps = int(1.0 / sim.model.opt.timestep)  # 1 second
        _free_settle_perturbed_objects(env, free_settle_steps, frozen_obj_info)

        # Reject if any object is out of camera frame
        if not _all_objects_visible(env, object_names):
            continue

        # Reject if any perturbed object drifted in XY during free settle
        # (indicates the object was resting on an edge or unstable surface
        # and slid/fell when the pinning constraints were released).
        xy_ok = True
        for obj_name in object_names:
            body_id = env.env.obj_body_id[obj_name]
            cur_xy = sim.data.body_xpos[body_id][:2]
            target_xy = object_pin_info[obj_name]['target_xy']
            if np.linalg.norm(cur_xy - target_xy) > 0.02:  # 2 cm tolerance
                xy_ok = False
                break
        if not xy_ok:
            continue

        # Reject if any object's z deviates from settled baseline
        # (catches objects that fell off surfaces OR landed on other surfaces
        # like inside drawers or on top of cutting boards)
        z_ok = True
        for obj_name in object_names:
            body_id = env.env.obj_body_id[obj_name]
            cur_z = float(sim.data.body_xpos[body_id][2])
            if abs(cur_z - baseline_z[obj_name]) > Z_THRESHOLD:
                z_ok = False
                break
        if not z_ok:
            continue

        # Reject if any object has tipped over (checked AFTER free settle,
        # so objects that would tip when unpinned are caught)
        upright_ok = True
        for obj_name in object_names:
            body_id = env.env.obj_body_id[obj_name]
            if not _object_is_upright(sim, body_id,
                                      ref_up_axis=ref_up_axis[obj_name]):
                upright_ok = False
                break
        if not upright_ok:
            continue

        # Reject if colliding with other scene objects
        if not _has_object_collision(env, object_names):
            break
    else:
        warnings.warn(
            f"perturb_object_positions: could not find collision-free placement "
            f"after {MAX_RETRIES} attempts; using last sample."
        )

    # Regenerate observations
    env.env._post_process()
    env.env._update_observables(force=True)
    return env.env._get_observations()


_CAMERA_NAME = "agentview"
_VISIBILITY_MARGIN = 20  # pixels from image edge; object must be this far inside

def _is_visible_in_camera(sim, point_3d, img_w=224, img_h=224):
    """Check if a 3D world point projects inside the agentview camera frame.

    Uses the MuJoCo camera model to do an exact perspective projection.
    Returns False if the point falls outside the image (with margin).
    """
    cam_id = sim.model.camera_name2id(_CAMERA_NAME)

    # Camera pose in world frame
    cam_pos = sim.data.cam_xpos[cam_id]        # (3,)
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)  # world-to-cam columns

    # Transform point into camera frame
    #   cam_mat columns are the camera x/y/z axes in world frame
    #   MuJoCo camera looks along -z in its own frame
    p_cam = cam_mat.T @ (point_3d - cam_pos)

    # Behind camera
    if p_cam[2] >= 0:
        return False

    # Perspective projection
    fovy = sim.model.cam_fovy[cam_id]  # vertical FOV in degrees
    f = img_h / (2.0 * np.tan(np.radians(fovy) / 2.0))

    x_img = f * p_cam[0] / (-p_cam[2]) + img_w / 2.0
    y_img = f * p_cam[1] / (-p_cam[2]) + img_h / 2.0

    m = _VISIBILITY_MARGIN
    return (m <= x_img < img_w - m) and (m <= y_img < img_h - m)


def _all_objects_visible(env, object_names):
    """Return True if every named object projects inside the camera frame."""
    sim = env.env.sim
    for obj_name in object_names:
        body_id = env.env.obj_body_id[obj_name]
        pos = sim.data.body_xpos[body_id]
        if not _is_visible_in_camera(sim, pos):
            return False
    return True


# ---- Per-task convenience mapping ----------------------------------------

TASK_PERTURB_OBJECTS = {
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": [
        "moka_pot_1",
        "moka_pot_2",
    ],
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": [
        "alphabet_soup_1",
        "tomato_sauce_1",
    ],
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": [
        "akita_black_bowl_1",
    ],
}


def get_perturb_objects(task_name):
    """Return the list of objects to perturb for a known task, or raise."""
    if task_name not in TASK_PERTURB_OBJECTS:
        raise ValueError(
            f"No perturbation objects defined for task '{task_name}'. "
            f"Known tasks: {list(TASK_PERTURB_OBJECTS.keys())}"
        )
    return TASK_PERTURB_OBJECTS[task_name]
