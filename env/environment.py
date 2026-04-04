"""
OpenEnv server-side environment wrapping the car racing game.

Observation: 64×64 egocentric headlight image + [speed, on_track, sin, cos].
The image is rendered offscreen (no display required).
"""

import math
from typing import Any, Optional

import numpy as np
import pygame

from openenv.core.env_server import Environment

from game.rl_splits import CarEnv
from game.tracks import TrackDef, SCREEN_W, SCREEN_H
from game.oval_racer import draw_headlights
from .models import DriveAction, RaceObservation

# Egocentric view parameters
_VIEW_PX = 120   # world-pixel square captured around the car before scaling
_OUT_PX  = 64    # output image size (64×64)
_GRASS   = (45, 110, 45)


class RaceEnvironment(Environment[DriveAction, RaceObservation, dict]):
    """
    Wraps game.rl_splits.CarEnv as an OpenEnv Environment.

    Parameters
    ----------
    track       : TrackDef (must already be built, or call track.build() first)
    max_steps   : episode step limit
    laps_target : episode ends after this many laps
    use_image   : if False, image field of RaceObservation will be None
                  (useful for fast debugging / unit tests)
    """

    def __init__(
        self,
        track: TrackDef,
        max_steps: int = 3000,
        laps_target: int = 3,
        use_image: bool = True,
    ):
        self._env = CarEnv(track, max_steps=max_steps, laps_target=laps_target)
        self._use_image = use_image
        self._episode_id: Optional[str] = None

        # Offscreen surface reused every step — allocated once
        if use_image:
            self._surf = pygame.Surface((SCREEN_W, SCREEN_H))

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RaceObservation:
        self._episode_id = episode_id
        obs = self._env.reset()
        return self._to_obs(obs, done=False, reward=0.0)

    def step(
        self,
        action: DriveAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RaceObservation:
        obs, reward, done, info = self._env.step([action.accel, action.steer])
        return self._to_obs(obs, done=done, reward=reward, metadata=info)

    @property
    def state(self) -> dict:
        t = self._env.track
        return {
            "episode_id": self._episode_id,
            "track_level": t.level,
            "track_name": t.name,
            "laps": self._env.laps,
            "step": self._env._step,
        }

    # ── Image rendering ──────────────────────────────────────────────────────

    def _render_headlight_image(self) -> np.ndarray:
        """
        Render a 64×64 RGB egocentric view centred on the car.

        Pipeline:
          1. Blit track surface onto offscreen canvas
          2. Draw headlight cone
          3. Crop _VIEW_PX × _VIEW_PX square around the car (grass-padded at borders)
          4. Rotate so the car always faces UP (forward = top of image)
          5. Re-crop centre after rotation padding
          6. Scale to _OUT_PX × _OUT_PX
          7. Return as (H, W, 3) uint8 numpy array
        """
        x = self._env._x
        y = self._env._y
        angle = self._env._angle   # degrees; 0=right, 90=down (pygame y-down)

        # 1 & 2 — render to offscreen surface
        self._surf.blit(self._env.track.surface, (0, 0))
        draw_headlights(self._surf, x, y, angle)

        # 3 — crop around car, padding with grass if near screen edge
        half = _VIEW_PX // 2
        canvas = pygame.Surface((_VIEW_PX, _VIEW_PX))
        canvas.fill(_GRASS)
        src = pygame.Rect(int(x) - half, int(y) - half, _VIEW_PX, _VIEW_PX)
        clipped = src.clip(pygame.Rect(0, 0, SCREEN_W, SCREEN_H))
        if clipped.width > 0 and clipped.height > 0:
            canvas.blit(self._surf, (clipped.x - src.x, clipped.y - src.y), clipped)

        # 4 — rotate so forward (angle) maps to UP (270° in pygame convention)
        rotated = pygame.transform.rotate(canvas, -(angle - 270))

        # 5 — re-crop centre (rotation adds padding)
        rw, rh = rotated.get_size()
        cx2, cy2 = rw // 2, rh // 2
        inner = pygame.Rect(cx2 - half, cy2 - half, _VIEW_PX, _VIEW_PX)
        inner = inner.clip(rotated.get_rect())
        cropped = pygame.Surface((_VIEW_PX, _VIEW_PX))
        cropped.fill(_GRASS)
        cropped.blit(rotated, (inner.x - (cx2 - half), inner.y - (cy2 - half)), inner)

        # 6 — scale to output size
        scaled = pygame.transform.scale(cropped, (_OUT_PX, _OUT_PX))

        # 7 — pygame surfarray is (W, H, 3); transpose to (H, W, 3)
        return pygame.surfarray.array3d(scaled).transpose(1, 0, 2)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _to_obs(
        self,
        obs: list,
        done: bool,
        reward: float,
        metadata: dict = None,
    ) -> RaceObservation:
        image = self._render_headlight_image() if self._use_image else None
        return RaceObservation(
            image=image,
            speed=obs[4],
            on_track=obs[5],
            sin_angle=obs[2],
            cos_angle=obs[3],
            done=done,
            reward=reward,
            metadata=metadata or {},
        )
