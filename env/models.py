"""
Action and Observation types for the car racing environment.

Observation design notes
------------------------
Dropped from original 7-float vector:
    x, y        — absolute screen coords, track-specific, hurt generalisation
    gate_side   — distance to start/finish gate, meaningless on unseen tracks

Kept:
    speed       — controls braking/throttle decisions
    on_track    — reactive penalty signal
    sin/cos     — heading orientation (redundant with image but cheap to include)

Added:
    image       — 64×64 RGB egocentric headlight view (car always faces up).
                  CNN reads track shape ahead; generalises to any unseen layout
                  because it only depends on local visual context, not absolute position.
"""

from typing import Any, Dict, Optional

import numpy as np
from pydantic import ConfigDict
from openenv.core.env_server.types import Action, Observation


class DriveAction(Action):
    """Continuous driving action."""
    accel: float             # -1 (brake) .. +1 (throttle)
    steer: float             # -1 (left)  .. +1 (right)
    metadata: Dict[str, Any] = {}


class RaceObservation(Observation):
    """
    Combined image + scalar observation.

    image       : (64, 64, 3) uint8 numpy array — egocentric headlight view,
                  car faces up, forward is towards the top of the image.
                  None when use_image=False.
    speed       : speed / max_speed  (≈ 0..1)
    on_track    : 1.0 on track, 0.0 off track
    sin_angle   : sin of absolute heading (orientation context)
    cos_angle   : cos of absolute heading
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Optional[Any] = None   # np.ndarray (64, 64, 3) uint8
    speed: float = 0.0
    on_track: float = 1.0
    sin_angle: float = 0.0
    cos_angle: float = 1.0
