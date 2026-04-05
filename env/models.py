"""
Action and Observation types for the car racing environment.

Observation design notes
------------------------
Dropped from original 7-float vector:
    x, y        — absolute screen coords, track-specific, hurt generalisation
    gate_side   — distance to start/finish gate, meaningless on unseen tracks
    on_track    — binary (0/1), tells agent it crashed AFTER the fact; no lookahead
    sin/cos     — absolute global heading; encodes track layout, hurts generalisation

Kept:
    speed           — controls braking/throttle decisions
    angular_velocity — egocentric turn rate; no global orientation leak

Replaced on_track with 5 raycasts:
    ray_left        — clearance 90° left  of heading  (lateral, right now)
    ray_front_left  — clearance 45° left  of heading  (diagonal lookahead)
    ray_front       — clearance straight  ahead        (forward lookahead)
    ray_front_right — clearance 45° right of heading  (diagonal lookahead)
    ray_right       — clearance 90° right of heading  (lateral, right now)

    All rays: 1.0 = boundary MAX px away (clear), 0.0 = boundary at car (edge/off).
    Binary on_track told the agent it crashed AFTER crossing.
    Raycasts tell the agent HOW FAR it is from each boundary BEFORE crossing.

Added:
    image       — 64×64 RGB egocentric headlight view (car always faces up).
                  CNN reads track shape ahead; generalises to any unseen layout.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict
from env._openenv_compat import Action, Observation


class DriveAction(Action):
    """Continuous driving action."""
    accel: float             # -1 (brake) .. +1 (throttle)
    steer: float             # -1 (left)  .. +1 (right)
    metadata: Dict[str, Any] = {}


class RaceObservation(Observation):
    """
    Combined image + scalar observation.

    image            : (64, 64, 3) uint8 numpy array — egocentric headlight view.
                       None when use_image=False.
    speed            : speed / max_speed  (≈ 0..1)
    angular_velocity : degrees turned last step / STEER_DEG  (≈ -1..1, egocentric)
    ray_left         : clearance to left boundary   (0=at edge, 1=MAX away)
    ray_front_left   : clearance front-left diagonal
    ray_front        : clearance straight ahead
    ray_front_right  : clearance front-right diagonal
    ray_right        : clearance to right boundary
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Optional[Any] = None   # np.ndarray (64, 64, 3) uint8

    # scalar branch — 7 values total
    speed: float = 0.0
    angular_velocity: float = 0.0
    ray_left: float = 1.0
    ray_front_left: float = 1.0
    ray_front: float = 1.0
    ray_front_right: float = 1.0
    ray_right: float = 1.0

    @property
    def scalars(self) -> List[float]:
        """Convenience: flat list for feeding directly into the MLP encoder."""
        return [
            self.speed,
            self.angular_velocity,
            self.ray_left,
            self.ray_front_left,
            self.ray_front,
            self.ray_front_right,
            self.ray_right,
        ]
