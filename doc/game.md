# Game Simulation

The `game/` package is a pure Pygame racing simulation with no ML dependencies.
It provides procedural tracks, car physics, raycasting sensors, and lap detection.

## Files

| File | Description |
|------|-------------|
| `tracks.py` | 10 procedural `TrackDef` objects — waypoints, surfaces, masks |
| `rl_splits.py` | `CarEnv` (physics + raycasting + reward), `CurriculumSampler`, TRAIN split |
| `oval_racer.py` | Car rendering, headlight cone drawing |
| `curriculum_game.py` | Interactive human-playable game (arrow keys) |

---

## Tracks

10 tracks across 3 difficulty groups:

```
Group A — Easy ovals           tracks  1, 2, 3, 4
Group B — Rectangular shapes   tracks  5, 6, 7, 8
Group C — Hairpins & chicanes  tracks  9, 10
```

```
Track  Name                Width  MaxSpd  Key challenge
─────────────────────────────────────────────────────────────
 1     Wide Oval           115    3.0     Nothing — just drive
 2     Standard Oval        85    3.5     Slightly narrower
 3     Narrow Oval          58    3.5     Precision needed
 4     Superspeedway        85    4.5     High speed, elliptical
 5     Rounded Rectangle    90    3.5     First corners
 6     Stadium Oval         80    4.0     Tight end-caps
 7     Tight Rectangle      65    3.5     Sharp 90° corners
 8     Small Oval           60    3.2     Small radius
 9     Hairpin Track        75    3.5     First hairpin
10     Chicane Track        70    3.5     Chicane section
```

### TrackDef

Each track is a `TrackDef` object defined in `tracks.py`:

- **`waypoints`**: List of `(x, y)` centreline points forming a closed polygon
- **`width`**: Road width in pixels (or `segment_widths` for variable-width tracks)
- **`start_pos`** / **`start_angle`**: Spawn location and heading
- **`max_speed`**: Speed cap for this track
- **`complexity`**: Difficulty multiplier = `(base_width / effective_width) * (max_speed / base_speed)`

`build()` renders the track onto a pygame Surface (dark grey tarmac, white boundary lines, checkered start/finish) and creates a collision mask.

---

## Car Physics

Located in `CarEnv` inside `rl_splits.py`. Constant parameters:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `ACCEL` | 0.13 | Forward acceleration per step |
| `BRAKE_DECEL` | 0.22 | Braking deceleration per step |
| `FRICTION` | 0.038 | Passive speed loss per step |
| `STEER_DEG` | 2.7° | Degrees turned per unit steer input |

Steering rate scales with speed — at low speed (< 30% max), steering is reduced to `0.3 × STEER_DEG`. Off-track, speed is multiplied by 0.80 each step (mud drag).

---

## Raycasting

CarEnv casts 5 rays from the car at fixed angles relative to heading:

```
Ray angles: -90°, -45°, 0°, +45°, +90°
Max length: 120 px
Step size:  2 px
```

Each ray returns a normalised distance in `[0, 1]`:
- `1.0` = boundary is 120 px away (clear road)
- `0.0` = boundary is at the car (on the edge)

The left/right rays provide lateral clearance; diagonal/front rays provide lookahead.

---

## Waypoint GPS

The track centreline waypoints are used internally for reward and for the `wp_sin` / `wp_cos` observation:

```python
next_idx = (current_wp + 10) % n_waypoints   # look ahead ~10 waypoints
relative_angle = atan2(dy, dx) - car_heading
wp_sin = sin(relative_angle)   # < 0 → waypoint is left, > 0 → right
wp_cos = cos(relative_angle)   # ≈ 1 → waypoint is straight ahead
```

This acts as a relative GPS compass toward the track ahead, invariant to absolute position.

---

## Playing Interactively

```bash
uv run python main.py        # Track 1
uv run python main.py 5      # Track 5
```

Arrow keys for steering and acceleration. The HUD shows speed, lap count, and track name.
