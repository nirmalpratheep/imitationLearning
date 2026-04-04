# Game

Pygame-based curriculum car racer used as the simulation backend for RL training.

## Running

```bash
# From project root
uv run python main.py          # start at track 1
uv run python main.py 5        # start at track 5

# As a module
uv run python -m game.curriculum_game 5
```

## Controls

| Key | Action |
|-----|--------|
| Arrow keys | Drive (up=throttle, down=brake, left/right=steer) |
| N / P | Next / previous track |
| 1 – 9 | Jump to track number |
| R | Restart (counts as an attempt) |
| ESC | Quit |

## Tracks

16 tracks across 4 difficulty tiers. Each level narrows the road, tightens turns, or increases speed cap.

| Tier | Levels | Shape | Description |
|------|--------|-------|-------------|
| A — Easy | 1–4 | Full ellipses | Wide to narrow ovals, superspeedway |
| B — Medium-Easy | 5–8 | Rounded rectangles | Stadium oval, tight rectangle |
| C — Medium-Hard | 9–12 | Two-arc layouts | Hairpin, chicane, double-hairpin, asymmetric |
| D — Hard | 13–16 | Polygon circuits | L-shape, T-notch, complex circuit, master challenge |

## Game Rules

- Complete **one full lap** without touching the white fence border.
- Touching the fence **or** pressing R = restart from start, attempt counter +1.
- Cross the start/finish line cleanly to finish. A summary screen shows stats.

## HUD

A single top bar shows: track name · speed · attempt count · lap time · total time · distance · max speed. Timer starts on first key press, not on load.

## File Structure

```
game/
  oval_racer.py        Original single-oval game. Exports draw_headlights, draw_car,
                       SCREEN_W, SCREEN_H used by curriculum_game and env/.
  tracks.py            16 TrackDef objects. Each knows its waypoints, road width,
                       start position/angle, speed cap, and on-track mask.
  curriculum_game.py   Main playable game. RaceState drives the lap logic,
                       reset-on-crash, finish detection, and HUD rendering.
  rl_splits.py         CarEnv (gym-style wrapper), CurriculumSampler, Evaluator,
                       and TRAIN / VAL / TEST splits for RL training.
  test_tracks.py       Headless test: builds all 16 tracks and simulates 150 steps
                       each. Run with: uv run python -m game.test_tracks
```

## Physics Constants

| Constant | Value | Effect |
|----------|-------|--------|
| `ACCEL` | 0.13 | Throttle acceleration per frame |
| `BRAKE_DECEL` | 0.22 | Braking deceleration per frame |
| `FRICTION` | 0.038 | Passive speed decay per frame |
| `STEER_DEG` | 2.7 | Degrees rotated per steer step |
| `max_speed` | 3.0–4.5 | Per-track speed cap (px/frame) |

Speed is in px/frame. Multiply by FPS (60) to get px/s shown in HUD.

## Finish Line Detection

Two-phase gate crossing to handle fast cars reliably:

1. **Arm** — wait until `gate_side > 50 px` ahead (car is clearly past the gate going forward).
2. **Trigger** — detect `prev_side < 0` and `curr_side >= 0` with `speed > 0.3`.

This prevents the car from triggering on spawn or when reversing back over the line.

## Track Metadata (used by reward)

Each `TrackDef` computes three values at construction time:

| Field | Formula | Purpose |
|-------|---------|---------|
| `optimal_dist` | Waypoint polygon perimeter (px) | Theoretical shortest lap path |
| `par_time_steps` | `optimal_dist / (max_speed × 0.7)` | Expected lap frames at 70% speed |
| `complexity` | `(115 / width) × (max_speed / 3.0)` | Difficulty multiplier (1.0 → 3.45) |

## RL Interface (`rl_splits.py`)

`CarEnv` exposes a gym-style API:

```python
from game.rl_splits import make_env, TRAIN

env = make_env(TRAIN[0])
obs = env.reset()        # [x/W, y/H, sin, cos, speed/max, on_track, gate_side]
obs, reward, done, info = env.step([accel, steer])
# info keys: lap, on_track, step, crashes, lap_dist, out_of_bounds
```

### Reward Function

All terms scale with `C = track.complexity` so the curriculum `threshold` stays
meaningful across all 16 tracks without manual tuning.

| Term | Trigger | Value |
|------|---------|-------|
| Forward pulse | Every step | `+speed/max_speed × 0.01` |
| Off-track | Every step off road | `−0.5 × C` |
| Crash event | on→off boundary crossing | `−5.0 × C` |
| Lap completion | Gate crossed cleanly | `+50 × time_ratio × dist_ratio × C` |
| Out of bounds | Terminal | `−100 × C` |

**Lap completion breakdown:**

```
time_ratio = clamp(par_time_steps / actual_lap_steps,  0.5, 2.0)
dist_ratio = clamp(optimal_dist   / actual_lap_dist,   0.5, 1.5)
```

- Faster than par → `time_ratio > 1` → reward multiplied up (max 2×)
- Shorter path → `dist_ratio > 1` → reward multiplied up (max 1.5×)
- Best possible lap: `50 × 2.0 × 1.5 × C = 150 × C`
- Worst completed lap: `50 × 0.5 × 0.5 × C = 12.5 × C`

**Complexity examples:**

| Track | Width | Max speed | C |
|-------|-------|-----------|---|
| 1 — Wide Oval | 115 | 3.0 | 1.00 |
| 8 — Small Oval | 60 | 3.2 | 2.03 |
| 14 — T-Notch | 58 | 4.0 | 2.66 |
| 16 — Master Challenge | 50 | 4.5 | 3.45 |
