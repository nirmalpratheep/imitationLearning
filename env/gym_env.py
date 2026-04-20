"""
env/gym_env.py — Gymnasium wrapper for RaceEnvironment, compatible with SB3.

Observation space: Dict
    image   : Box(0.0, 1.0, (3, 64, 64), float32)  — normalised CHW image
    scalars : Box(-inf, inf, (9,), float32)          — speed, ang_vel, 5×rays, wp_sin, wp_cos

Action space: Box(-1.0, 1.0, (2,), float32) — [accel, steer]

Supports two parallelism modes:

  DummyVecEnv  (default, --num-envs N)
    All envs live in the main process and share one CurriculumSampler.
    Curriculum advancement is reflected immediately at every reset().

  SubprocVecEnv  (--subproc)
    Each env runs in its own subprocess (parallel env stepping).
    Sampler cannot be shared across processes, so each env tracks a
    `frontier_level` int that the main-process callback syncs via
    vec_env.set_attr('frontier_level', new_level) after each advance.
"""

from __future__ import annotations

import random
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.models import DriveAction


class RaceGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around RaceEnvironment.

    Parameters
    ----------
    sampler        : shared CurriculumSampler (DummyVecEnv mode).
                     Pass None for SubprocVecEnv mode.
    frontier_level : current curriculum frontier index (SubprocVecEnv mode).
                     Updated externally via vec_env.set_attr().
    replay_frac    : fraction of episodes replayed from already-mastered tracks.
    max_steps      : episode step limit
    laps_target    : episode ends after this many laps
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sampler=None,
        frontier_level: int = 0,
        replay_frac: float = 0.3,
        max_steps: int = 3000,
        laps_target: int = 1,
        shared_level=None,
    ):
        super().__init__()
        self._sampler       = sampler
        self.frontier_level = frontier_level   # writable via set_attr in subprocess mode
        self._shared_level  = shared_level     # multiprocessing.Value for ParallelEnv mode
        self._replay_frac   = replay_frac
        self._max_steps     = max_steps
        self._laps_target   = laps_target
        self._race_env: Optional[Any] = None
        self._current_track: Optional[Any] = None

        # Episode accumulators
        self._ep_reward: float = 0.0
        self._ep_length: int   = 0
        self._ep_crashes: int  = 0
        self._ep_laps: int     = 0
        self._ep_on_track: int = 0

        self.observation_space = spaces.Dict({
            "image":   spaces.Box(0.0, 1.0, shape=(3, 64, 64), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(9,),   dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)

        # Lazy import keeps subprocess workers headless
        from env.environment import RaceEnvironment  # noqa: PLC0415

        track = self._sample_track()
        track.build()
        self._race_env = RaceEnvironment(
            track, self._max_steps, self._laps_target, use_image=True
        )
        self._current_track = track

        self._ep_reward   = 0.0
        self._ep_length   = 0
        self._ep_crashes  = 0
        self._ep_laps     = 0
        self._ep_on_track = 0

        raw = self._race_env.reset()
        return self._to_obs(raw), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        accel = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        raw = self._race_env.step(DriveAction(accel=accel, steer=steer))

        reward = float(raw.reward)
        self._ep_reward += reward
        self._ep_length += 1

        if raw.metadata:
            self._ep_laps    = raw.metadata.get("lap",     self._ep_laps)
            self._ep_crashes = raw.metadata.get("crashes", self._ep_crashes)
            if raw.metadata.get("on_track", True):
                self._ep_on_track += 1

        terminated = bool(raw.done)
        truncated  = False

        info: Dict[str, Any] = {
            "track_level": self._current_track.level,
            "track_name":  self._current_track.name,
        }

        if terminated:
            on_track_pct = 100.0 * self._ep_on_track / max(self._ep_length, 1)
            info["episode"] = {
                "r": self._ep_reward,
                "l": self._ep_length,
                "t": 0.0,
            }
            info["episode_reward"]  = self._ep_reward
            info["episode_crashes"] = self._ep_crashes
            info["episode_laps"]    = self._ep_laps
            info["on_track_pct"]    = on_track_pct

        return self._to_obs(raw), reward, terminated, truncated, info

    # ── Track sampling ────────────────────────────────────────────────────────

    def _sample_track(self):
        """
        DummyVecEnv  → delegate to the shared CurriculumSampler.
        SubprocVecEnv → use frontier_level + replay_frac locally.
        """
        if self._sampler is not None:
            return self._sampler.sample()

        # Subprocess mode: replicate the sampler's replay logic locally.
        from game.rl_splits import TRAIN  # noqa: PLC0415
        fl = max(0, min(
            self._shared_level.value if self._shared_level is not None else self.frontier_level,
            len(TRAIN) - 1,
        ))
        if fl > 0 and random.random() < self._replay_frac:
            idx = random.randint(0, fl - 1)   # replay a mastered track
        else:
            idx = fl                           # train on the frontier
        return TRAIN[idx]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_obs(self, raw) -> Dict[str, np.ndarray]:
        img     = raw.image.transpose(2, 0, 1).astype(np.float32) / 255.0
        scalars = np.array(raw.scalars, dtype=np.float32)
        return {"image": img, "scalars": scalars}
