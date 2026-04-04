"""
Headless automated test for all 16 tracks.
Exit 0 if all pass, 1 if any fail.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys
import math
import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from game.tracks import TRACKS, SCREEN_W, SCREEN_H

ACCEL     = 0.13
STEER_DEG = 2.7

all_pass = True

for track in TRACKS:
    name = f"Lv{track.level}: {track.name}"
    try:
        # 1. Build must not raise
        track.build()

        # 2. surface not None, correct size
        assert track.surface is not None, "surface is None"
        assert track.surface.get_size() == (SCREEN_W, SCREEN_H), \
            f"surface size {track.surface.get_size()} != ({SCREEN_W},{SCREEN_H})"

        # 3. mask not None
        assert track.mask is not None, "mask is None"

        # 4. start_pos is on track
        sx, sy = track.start_pos
        assert track.on_track(sx, sy), \
            f"start_pos {track.start_pos} not on track"

        # 5. gate_side at start_pos ≈ 0
        gs = track.gate_side(sx, sy)
        assert abs(gs) < 2.0, \
            f"gate_side at start_pos = {gs:.4f}, expected < 2.0"

        # 6. Simulate 150 steps
        x     = float(sx)
        y     = float(sy)
        angle = float(track.start_angle)
        speed = 0.0
        max_speed = track.max_speed

        for step in range(150):
            accel = 1   # constant throttle
            steer = math.sin(step * 0.15) * 0.5  # gentle sinusoidal steer

            speed_ratio = min(abs(speed) / max_speed, 1.0) if max_speed > 0 else 0
            angle += steer * STEER_DEG * max(0.3, speed_ratio)
            speed  = min(speed + ACCEL, max_speed)
            speed  = max(0.0, speed - 0.038)  # friction

            rad = math.radians(angle)
            x  += speed * math.cos(rad)
            y  += speed * math.sin(rad)

            # on_track check (no crash required)
            _ = track.on_track(x, y)

        print(f"PASS  {name}")

    except Exception as e:
        print(f"FAIL  {name}: {e}")
        all_pass = False

print()
if all_pass:
    print("All 16 tracks PASSED.")
    sys.exit(0)
else:
    print("Some tracks FAILED.")
    sys.exit(1)
