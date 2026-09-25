#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import importlib
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from lerobot.lerobot_types import RobotObservation
from lerobot.utils.import_utils import _scipy_available

from .utils import _LazyAsyncVectorEnv

# scipy is only used for end-effector-pose composition (``--env.action_mode=ee``); guard it so this
# module (and its base-env unit tests, which mock the RoboTwin runtime) imports without scipy installed.
if _scipy_available:
    from scipy.spatial.transform import Rotation
else:
    Rotation = None

logger = logging.getLogger(__name__)

# Camera names as used by RoboTwin 2.0 inside get_obs()["observation"].
ROBOTWIN_CAMERA_NAMES: tuple[str, ...] = (
    "head_camera",
    "left_camera",
    "right_camera",
)
# The released RoboTwin LeRobot dataset uses ALOHA dataset names rather than
# simulator camera names. Keep this mapping explicit and fail closed so a
# policy can never silently receive black or mislabelled views.
ROBOTWIN_CAMERA_TO_DATASET: dict[str, str] = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
ROBOTWIN_TASK_CONFIGS: tuple[str, ...] = ("demo_clean", "demo_randomized")

ACTION_DIM = 14  # 7 DOF × 2 arms (joint-space control mode)
# End-effector-pose control mode: per arm [x, y, z, qx, qy, qz, qw, gripper] = 8, dual-arm = 16.
# Used by world-model policies (e.g. LingBot-VA) that predict eef-pose deltas executed via CuRobo IK.
EEF_ACTION_DIM = 16
# Absolute qpos contract for aloha-agilex at RoboTwin commit 0aeea2d. Each arm has six
# revolute joints with URDF limits [-10, 10], followed by one normalized gripper target [0, 1].
# The clean dataset stores this exact ordering. These are environment-space bounds, not the
# policy's normalized [-1, 1] representation; policy postprocessors must unnormalize first.
JOINT_ACTION_LOW = np.asarray(
    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0] * 2,
    dtype=np.float32,
)
JOINT_ACTION_HIGH = np.asarray(
    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0] * 2,
    dtype=np.float32,
)
# EEF mode remains a normalized delta-pose interface.
EEF_ACTION_LOW = -1.0
EEF_ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH: int | None = None
OFFICIAL_INSTRUCTION_ENV = "LEROBOT_ROBOTWIN_OFFICIAL_INSTRUCTION"
OFFICIAL_INSTRUCTION_TYPE_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_TYPE"
OFFICIAL_INSTRUCTION_MAX_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_MAX"


def _compose_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Compose a single-arm predicted delta pose onto the initial pose.

    ``new_pose`` / ``init_pose`` are 8-vectors ``[x, y, z, qx, qy, qz, qw, gripper]``. Translation
    is added, rotation is composed (``init_R * new_R``), and the gripper is taken from the
    prediction. Mirrors ``add_eef_pose`` in the upstream LingBot-VA RoboTwin client.
    """
    new_r = Rotation.from_quat(new_pose[3:7])
    init_r = Rotation.from_quat(init_pose[3:7])
    out_rot = (init_r * new_r).as_quat()
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def _add_init_eef_pose(delta_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Compose a dual-arm (16-d) predicted delta pose onto the initial eef pose, normalizing quats."""
    left = _compose_eef_pose(delta_pose[:8], init_pose[:8])
    right = _compose_eef_pose(delta_pose[8:], init_pose[8:])
    out = np.concatenate([left, right])
    # Normalize the two quaternions (indices 3:7 and 11:15) as the upstream client does.
    out[3:7] = out[3:7] / (np.linalg.norm(out[3:7]) + 1e-8)
    out[11:15] = out[11:15] / (np.linalg.norm(out[11:15]) + 1e-8)
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _validate_camera_mapping(
    camera_names: Sequence[str], mapping: Mapping[str, str] | None = None
) -> dict[str, str]:
    names = list(camera_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("camera_names must be non-empty and contain no duplicates")
    requested = dict(mapping or ROBOTWIN_CAMERA_TO_DATASET)
    unknown = sorted(set(names) - set(ROBOTWIN_CAMERA_TO_DATASET))
    if unknown:
        raise ValueError(f"Unknown RoboTwin simulator camera names: {unknown}")
    missing = sorted(set(names) - set(requested))
    if missing:
        raise ValueError(f"Explicit camera mapping is missing: {missing}")
    selected = {name: requested[name] for name in names}
    wrong = {
        name: value
        for name, value in selected.items()
        if ROBOTWIN_CAMERA_TO_DATASET[name] != value
    }
    if wrong:
        raise ValueError(
            "RoboTwin camera mapping must match the frozen dataset mapping; "
            f"observed {wrong}"
        )
    if len(set(selected.values())) != len(selected):
        raise ValueError("RoboTwin camera mapping destinations must be unique")
    return selected


def _arm_for_block(block: Any) -> str:
    return "left" if float(block.get_pose().p[0]) < 0 else "right"


def _robotwin_blocks_episode_info(task_name: str, env: Any) -> dict[str, str] | None:
    """Infer the episode-info dict used by RoboTwin's official instruction generator for block ranking."""
    if task_name == "blocks_ranking_rgb":
        return {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    if task_name == "blocks_ranking_size":
        return {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    return None


def _generate_robotwin_official_instruction(task_name: str, env: Any) -> str:
    """Generate language with RoboTwin's official task templates, matching its eval client."""
    fallback = task_name.replace("_", " ")
    episode_info = _robotwin_blocks_episode_info(task_name, env)
    if episode_info is None:
        logger.warning(
            "Official RoboTwin instruction is not implemented for task=%s; using %r.", task_name, fallback
        )
        return fallback

    try:
        # Part of the robotwin simulator repo, this is being pulled by the docker image running robotwin
        # see https://github.com/RoboTwin-Platform/RoboTwin/tree/main/description
        # Used to generate the official instructions
        from description.utils.generate_episode_instructions import generate_episode_descriptions
    except Exception:
        logger.warning(
            "Failed to import RoboTwin official instruction generator; using %r.", fallback, exc_info=True
        )
        return fallback

    instruction_type = os.environ.get(OFFICIAL_INSTRUCTION_TYPE_ENV, "seen")
    try:
        max_descriptions = int(os.environ.get(OFFICIAL_INSTRUCTION_MAX_ENV, "1000000"))
    except ValueError:
        max_descriptions = 1000000

    results = generate_episode_descriptions(task_name, [episode_info], max_descriptions=max_descriptions)
    if not results:
        logger.warning(
            "RoboTwin generated no official instructions for task=%s; using %r.", task_name, fallback
        )
        return fallback

    options = results[0].get(instruction_type) or results[0].get("seen") or results[0].get("unseen")
    if not options:
        logger.warning(
            "RoboTwin generated no %s official instructions for task=%s; using %r.",
            instruction_type,
            task_name,
            fallback,
        )
        return fallback

    return str(np.random.choice(options))


# D435 dims from task_config/_camera_config.yml (what demo_clean.yml selects).
DEFAULT_CAMERA_H = 240
DEFAULT_CAMERA_W = 320

# Task list from RoboTwin 2.0's `envs/` directory — mirrors upstream exactly
# (50 tasks as of main; earlier revisions had 60 with a different split).
# Keep this in sync with:
#   gh api /repos/RoboTwin-Platform/RoboTwin/contents/envs --paginate \
#     | jq -r '.[].name' | grep -E '\.py$' | grep -v '^_' | sed 's/\.py$//'
ROBOTWIN_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


_ROBOTWIN_SETUP_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_ROBOTWIN_HORIZON_CACHE: dict[str, int] = {}


def _load_robotwin_setup_kwargs(task_name: str, task_config: str) -> dict[str, Any]:
    """Build the kwargs dict RoboTwin's setup_demo expects.

    Mirrors the config loading done by RoboTwin's ``script/eval_policy.py``:
    reads the explicitly selected clean/randomized task config, resolves the embodiment file from
    ``_embodiment_config.yml``, loads the robot's own ``config.yml``, and
    reads camera dimensions from ``_camera_config.yml``.

    Uses ``aloha-agilex`` single-robot dual-arm by default (the only embodiment
    used by beat_block_hammer and most smoke-test tasks).
    """
    if task_config not in ROBOTWIN_TASK_CONFIGS:
        raise ValueError(
            f"task_config must be one of {ROBOTWIN_TASK_CONFIGS}; got {task_config!r}"
        )
    cache_key = (task_name, task_config)
    if cache_key in _ROBOTWIN_SETUP_CACHE:
        return dict(_ROBOTWIN_SETUP_CACHE[cache_key])

    import os

    import yaml  # type: ignore[import-untyped]
    from envs import CONFIGS_PATH  # type: ignore[import-not-found]

    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    if not isinstance(args, dict):
        raise ValueError(f"RoboTwin task config is not a mapping: {task_config}")

    # Resolve embodiment — demo_clean.yml uses [aloha-agilex] (dual-arm single robot)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) == 1:
        robot_file = embodiment_types[embodiment[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment must have 1 or 3 items, got {len(embodiment)}")

    with open(os.path.join(args["left_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["left_embodiment_config"] = yaml.safe_load(f)
    with open(os.path.join(args["right_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["right_embodiment_config"] = yaml.safe_load(f)

    # Camera dimensions
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)
    head_cam = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_cam]["h"]
    args["head_camera_w"] = camera_config[head_cam]["w"]

    # Headless overrides
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = task_config
    # Base_Task loads the pinned per-task horizon only in eval_mode. is_test is
    # consumed by each task's setup_demo signature and does not set eval_mode.
    args["eval_mode"] = True

    _ROBOTWIN_SETUP_CACHE[cache_key] = args
    return dict(args)


def _load_robotwin_episode_length(task_name: str) -> int:
    """Load the authoritative per-task evaluation horizon from pinned upstream."""
    if task_name in _ROBOTWIN_HORIZON_CACHE:
        return _ROBOTWIN_HORIZON_CACHE[task_name]
    import yaml  # type: ignore[import-untyped]
    from envs import CONFIGS_PATH  # type: ignore[import-not-found]

    path = os.path.join(CONFIGS_PATH, "_eval_step_limit.yml")
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict) or task_name not in values:
        raise ValueError(
            f"RoboTwin task {task_name!r} has no pinned horizon in {path}; "
            "refusing the upstream implicit 1000-step fallback"
        )
    value = values[task_name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid RoboTwin horizon for {task_name!r}: {value!r}")
    _ROBOTWIN_HORIZON_CACHE[task_name] = value
    return value


def _load_robotwin_task(task_name: str) -> type:
    """Dynamically import and return a RoboTwin 2.0 task class.

    RoboTwin tasks live in ``envs/<task_name>.py`` relative to the repository
    root and are expected to be on ``sys.path`` after installation.
    """
    try:
        module = importlib.import_module(f"envs.{task_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import RoboTwin task '{task_name}'. "
            "Ensure RoboTwin 2.0 is installed and its 'envs/' directory is on PYTHONPATH. "
            "See the RoboTwin installation guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html"
        ) from e
    task_cls = getattr(module, task_name, None)
    if task_cls is None:
        raise AttributeError(f"Task class '{task_name}' not found in envs/{task_name}.py")
    return task_cls


class RoboTwinEnv(gym.Env):
    """Gymnasium wrapper around a single RoboTwin 2.0 task.

    RoboTwin uses a custom SAPIEN-based API (``setup_demo`` / ``get_obs`` /
    ``take_action`` / ``check_success``) rather than the standard gym interface.
    This class bridges that API to Gymnasium so that ``lerobot-eval`` can drive
    RoboTwin exactly like LIBERO or Meta-World.

    The underlying SAPIEN environment is created lazily on the first ``reset()``
    call *inside the worker process*.  This is required for
    ``gym.vector.AsyncVectorEnv`` compatibility: SAPIEN allocates EGL/GPU
    contexts that must not be forked from the parent process.

    Observations
    ------------
    The ``pixels`` dict uses the raw RoboTwin camera names as keys (e.g.
    ``"head_camera"``, ``"left_camera"``). ``preprocess_observation`` in
    ``envs/utils.py`` then converts these to ``observation.images.<cam>``.

    Actions
    -------
    Joint mode consumes a 14-dim absolute target: six arm qpos values followed by a
    normalized gripper value for each arm. Arm bounds come from the pinned aloha-agilex
    URDF (``[-10, 10]``); grippers use ``[0, 1]``.

    Autograd
    --------
    ``setup_demo`` and ``take_action`` drive CuRobo's Newton trajectory
    optimizer, which calls ``cost.backward()`` internally. lerobot_eval wraps
    the rollout in ``torch.no_grad()``, so both call sites re-enable grad.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
        camera_name_mapping: Mapping[str, str] | None = None,
        observation_height: int | None = None,
        observation_width: int | None = None,
        episode_length: int | None = DEFAULT_EPISODE_LENGTH,
        task_config: str = "demo_clean",
        render_mode: str = "rgb_array",
        action_mode: str = "joint",
    ):
        super().__init__()
        self.task_name = task_name
        self.task = task_name  # used by add_envs_task() in utils.py
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self._reset_stride = n_envs
        self._next_seed = int(episode_index)
        if task_config not in ROBOTWIN_TASK_CONFIGS:
            raise ValueError(
                f"task_config must be one of {ROBOTWIN_TASK_CONFIGS}; got {task_config!r}"
            )
        self.task_config = task_config
        self.condition = "clean" if task_config == "demo_clean" else "randomized"
        # "joint": 14-d joint-space actions via take_action(action). "ee": 16-d end-effector-pose
        # deltas (added onto the episode's initial eef pose) executed via take_action(.., "ee") + IK.
        if action_mode not in ("joint", "ee"):
            raise ValueError(f"action_mode must be 'joint' or 'ee'; got {action_mode!r}")
        self.action_mode = action_mode
        self._action_dim = EEF_ACTION_DIM if action_mode == "ee" else ACTION_DIM
        self._init_eef_pose: np.ndarray | None = None
        self.camera_names = list(camera_names)
        self.camera_name_mapping = _validate_camera_mapping(
            self.camera_names, camera_name_mapping
        )
        # Default to D435 dims (the camera type baked into task_config/demo_clean.yml).
        # The YAML-driven lookup is deferred to reset() so construction doesn't
        # import RoboTwin's `envs` module — fast-tests run without RoboTwin installed.
        self.observation_height = observation_height or DEFAULT_CAMERA_H
        self.observation_width = observation_width or DEFAULT_CAMERA_W
        if episode_length is not None and (
            isinstance(episode_length, bool) or not isinstance(episode_length, int) or episode_length <= 0
        ):
            raise ValueError("episode_length must be a positive integer or None")
        self.episode_length = episode_length
        self._registered_episode_length: int | None = None
        # Filled from pinned upstream on reset before lerobot_eval reads it.
        self._max_episode_steps = episode_length
        self.horizon_source = "explicit_override" if episode_length is not None else "pinned_upstream"
        self.render_mode = render_mode

        self._env: Any | None = None  # deferred — created on first reset() inside worker
        self._step_count: int = 0

        image_spaces = {
            self.camera_name_mapping[cam]: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(image_spaces),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32),
            }
        )
        if self.action_mode == "joint":
            action_low: float | np.ndarray = JOINT_ACTION_LOW
            action_high: float | np.ndarray = JOINT_ACTION_HIGH
        else:
            action_low = EEF_ACTION_LOW
            action_high = EEF_ACTION_HIGH
        self.action_space = spaces.Box(
            low=action_low, high=action_high, shape=(self._action_dim,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """Create the SAPIEN environment on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own EGL/GPU context rather than inheriting a stale one from the
        parent process (which causes crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return
        task_cls = _load_robotwin_task(self.task_name)
        self._env = task_cls()

    def _get_obs(self) -> RobotObservation:
        assert self._env is not None, "_get_obs called before _ensure_env()"
        raw = self._env.get_obs()
        cameras_raw = raw.get("observation", {})

        images: dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            cam_data = cameras_raw.get(cam)
            img = cam_data.get("rgb") if cam_data else None
            if img is None:
                raise KeyError(
                    f"RoboTwin observation is missing required simulator camera {cam!r}; "
                    f"available={sorted(cameras_raw)}"
                )
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] != 3:
                img = img[..., :3]
            expected_shape = (self.observation_height, self.observation_width, 3)
            if img.shape != expected_shape:
                raise ValueError(
                    f"RoboTwin camera {cam!r} has shape {img.shape}, expected {expected_shape}"
                )
            images[self.camera_name_mapping[cam]] = img

        ja = raw.get("joint_action") or {}
        vec = ja.get("vector")
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).ravel()
            joint_state = (
                arr[:ACTION_DIM] if arr.size >= ACTION_DIM else np.zeros(ACTION_DIM, dtype=np.float32)
            )
        else:
            joint_state = np.zeros(ACTION_DIM, dtype=np.float32)

        return {"pixels": images, "agent_pos": joint_state}

    def _read_eef_pose(self) -> np.ndarray:
        """Read the current 16-d dual-arm eef pose [left(xyz+quat)+grip, right(xyz+quat)+grip]."""
        assert self._env is not None, "_read_eef_pose called before _ensure_env()"
        ep = self._env.get_obs()["endpose"]
        pose = (
            list(ep["left_endpose"])
            + [ep["left_gripper"]]
            + list(ep["right_endpose"])
            + [ep["right_gripper"]]
        )
        return np.asarray(pose, dtype=np.float64)

    def reset(self, seed: int | None = None, **kwargs) -> tuple[RobotObservation, dict]:
        self._ensure_env()
        super().reset(seed=seed)
        assert self._env is not None  # set by _ensure_env() above

        actual_seed = self._next_seed if seed is None else int(seed)
        registered_horizon = _load_robotwin_episode_length(self.task_name)
        if self.episode_length is not None and self.episode_length > registered_horizon:
            raise ValueError(
                f"Explicit horizon {self.episode_length} exceeds pinned upstream horizon "
                f"{registered_horizon} for {self.task_name}; the simulator would stop first"
            )
        self._registered_episode_length = registered_horizon
        self._max_episode_steps = self.episode_length or registered_horizon
        setup_kwargs = _load_robotwin_setup_kwargs(self.task_name, self.task_config)
        setup_kwargs.update(seed=actual_seed, is_test=True)
        with torch.enable_grad():
            self._env.setup_demo(**setup_kwargs)
        upstream_horizon = getattr(self._env, "step_lim", None)
        if upstream_horizon != registered_horizon:
            raise RuntimeError(
                f"RoboTwin runtime horizon drift for {self.task_name}: "
                f"runtime={upstream_horizon!r}, pinned={registered_horizon}"
            )
        # Advance only after a successful setup. A failed reset remains retryable
        # with the same seed; one terminal transition therefore consumes no seed.
        self._next_seed = actual_seed + self._reset_stride
        self._step_count = 0

        use_official_instruction = self.task_name in {"blocks_ranking_rgb", "blocks_ranking_size"}
        if _env_flag(OFFICIAL_INSTRUCTION_ENV, default=use_official_instruction):
            self.task_description = _generate_robotwin_official_instruction(self.task_name, self._env)
            if hasattr(self._env, "set_instruction"):
                self._env.set_instruction(instruction=self.task_description)
            logger.info("RoboTwin official instruction | task=%s | %s", self.task_name, self.task_description)
        else:
            self.task_description = self.task_name.replace("_", " ")

        # In eef mode the policy predicts pose deltas relative to the initial eef pose.
        if self.action_mode == "ee":
            self._init_eef_pose = self._read_eef_pose()

        obs = self._get_obs()
        return obs, {
            "is_success": False,
            "task": self.task_name,
            "task_config": self.task_config,
            "condition": self.condition,
            "episode_seed": actual_seed,
            "episode_horizon": self._max_episode_steps,
            "registered_episode_horizon": registered_horizon,
            "horizon_source": self.horizon_source,
            "camera_mapping": dict(self.camera_name_mapping),
        }

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        assert self._env is not None, "step() called before reset()"
        if action.ndim != 1 or action.shape[0] != self._action_dim:
            raise ValueError(f"Expected 1-D action of shape ({self._action_dim},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("RoboTwin action must contain only finite values")
        executed_action = np.array(action, copy=True)
        action_saturation: dict[str, Any] = {
            "indices": [],
            "original": [],
            "executed": [],
            "max_delta": 0.0,
        }
        if self.action_mode == "joint":
            arm_indices = np.asarray([*range(6), *range(7, 13)], dtype=np.int64)
            below = arm_indices[action[arm_indices] < JOINT_ACTION_LOW[arm_indices]]
            above = arm_indices[action[arm_indices] > JOINT_ACTION_HIGH[arm_indices]]
            if below.size or above.size:
                violations = sorted(set(below.tolist() + above.tolist()))
                raise ValueError(
                    "RoboTwin arm action is outside the pinned absolute-qpos bounds "
                    f"at indices {violations}; arm clipping is forbidden"
                )
            gripper_indices = np.asarray([6, 13], dtype=np.int64)
            executed_action[gripper_indices] = np.clip(action[gripper_indices], 0.0, 1.0)
            saturated = gripper_indices[executed_action[gripper_indices] != action[gripper_indices]]
            if saturated.size:
                original_values = action[saturated].astype(float).tolist()
                executed_values = executed_action[saturated].astype(float).tolist()
                action_saturation = {
                    "indices": saturated.tolist(),
                    "original": original_values,
                    "executed": executed_values,
                    "max_delta": max(
                        abs(original - executed)
                        for original, executed in zip(original_values, executed_values, strict=True)
                    ),
                }

        with torch.enable_grad():
            if self.action_mode == "ee":
                ee_action = _add_init_eef_pose(
                    np.asarray(executed_action, dtype=np.float64), self._init_eef_pose
                )
                self._env.take_action(ee_action, action_type="ee")
            elif hasattr(self._env, "take_action"):
                self._env.take_action(executed_action)
            else:
                self._env.step(executed_action)

        self._step_count += 1

        is_success = bool(getattr(self._env, "eval_success", False))
        if not is_success and hasattr(self._env, "check_success"):
            is_success = bool(self._env.check_success())

        obs = self._get_obs()
        reward = float(is_success)
        terminated = is_success
        assert self._max_episode_steps is not None
        truncated = not terminated and self._step_count >= self._max_episode_steps

        info: dict[str, Any] = {
            "task": self.task_name,
            "is_success": is_success,
            "step": self._step_count,
            "task_config": self.task_config,
            "condition": self.condition,
            "episode_horizon": self._max_episode_steps,
            "horizon_source": self.horizon_source,
            "camera_mapping": dict(self.camera_name_mapping),
            "action_saturation": action_saturation,
        }
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task_name,
                "is_success": is_success,
                "action_saturation": dict(action_saturation),
            }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_env()
        obs = self._get_obs()
        # Prefer head camera for rendering; fall back to first available.
        if "cam_high" in obs["pixels"]:
            return obs["pixels"]["cam_high"]
        return next(iter(obs["pixels"].values()))

    def close(self) -> None:
        if self._env is not None:
            if hasattr(self._env, "close_env"):
                import contextlib

                with contextlib.suppress(TypeError):
                    self._env.close_env()
            self._env = None


# ---- Multi-task factory --------------------------------------------------------


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    camera_name_mapping: Mapping[str, str],
    observation_height: int,
    observation_width: int,
    episode_length: int | None,
    task_config: str,
    action_mode: str = "joint",
) -> list[Callable[[], RoboTwinEnv]]:
    """Return n_envs factory callables for a single task."""

    def _make_one(episode_index: int) -> RoboTwinEnv:
        return RoboTwinEnv(
            task_name=task_name,
            episode_index=episode_index,
            n_envs=n_envs,
            camera_names=camera_names,
            camera_name_mapping=camera_name_mapping,
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
            task_config=task_config,
            action_mode=action_mode,
        )

    return [partial(_make_one, i) for i in range(n_envs)]


def create_robotwin_envs(
    task: str,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
    camera_name_mapping: Mapping[str, str] | None = None,
    observation_height: int = DEFAULT_CAMERA_H,
    observation_width: int = DEFAULT_CAMERA_W,
    episode_length: int | None = DEFAULT_EPISODE_LENGTH,
    task_config: str = "demo_clean",
    action_mode: str = "joint",
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboTwin 2.0 environments.

    Returns:
        ``dict[task_name][0] -> VectorEnv`` — one entry per task, each wrapping
        ``n_envs`` parallel rollouts.

    Args:
        task: Comma-separated list of task names (e.g. ``"beat_block_hammer"``
            or ``"beat_block_hammer,click_bell"``).
        n_envs: Number of parallel rollouts per task.
        env_cls: Vector env constructor (e.g. ``gym.vector.AsyncVectorEnv``).
        camera_names: Simulator cameras to include in observations.
        camera_name_mapping: Explicit simulator-to-dataset mapping. Values must
            match `head/left/right -> cam_high/cam_left_wrist/cam_right_wrist`.
        observation_height: Pixel height for all cameras.
        observation_width: Pixel width for all cameras.
        episode_length: Optional shorter smoke horizon. None uses pinned upstream
            `_eval_step_limit.yml`; values above the registered limit are rejected.
        task_config: Explicit benchmark condition: `demo_clean` or `demo_randomized`.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be callable (e.g. gym.vector.AsyncVectorEnv).")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    task_names = [t.strip() for t in str(task).split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one RoboTwin task name.")

    unknown = [t for t in task_names if t not in ROBOTWIN_TASKS]
    if unknown:
        raise ValueError(f"Unknown RoboTwin tasks: {unknown}. Available tasks: {sorted(ROBOTWIN_TASKS)}")
    mapping = _validate_camera_mapping(camera_names, camera_name_mapping)
    if task_config not in ROBOTWIN_TASK_CONFIGS:
        raise ValueError(
            f"task_config must be one of {ROBOTWIN_TASK_CONFIGS}; got {task_config!r}"
        )

    logger.info(
        "Creating RoboTwin envs | tasks=%s | condition=%s | n_envs(per task)=%d",
        task_names,
        task_config,
        n_envs,
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv
    is_sync = env_cls is gym.vector.SyncVectorEnv
    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for task_name in task_names:
        fns = _make_env_fns(
            task_name=task_name,
            n_envs=n_envs,
            camera_names=list(camera_names),
            camera_name_mapping=mapping,
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
            task_config=task_config,
            action_mode=action_mode,
        )
        if is_async:
            lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
            if cached_obs_space is None:
                cached_obs_space = lazy.observation_space
                cached_act_space = lazy.action_space
                cached_metadata = lazy.metadata
            out[task_name][0] = lazy
        elif is_sync:
            out[task_name][0] = gym.vector.SyncVectorEnv(
                fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
            )
        else:
            out[task_name][0] = env_cls(fns)
        logger.info("Built vec env | task=%s | n_envs=%d", task_name, n_envs)

    return {k: dict(v) for k, v in out.items()}
