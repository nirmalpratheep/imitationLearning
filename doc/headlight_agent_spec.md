# Headlight-Based Rule Agent — Specification

## Overview

A rule-based control agent for the oval racer that uses the car's headlight cone
as its sole sensor. When a cone ray intersects a white lane boundary, the agent
applies a corrective steering input. No learning, no reward shaping — pure
geometry-driven control.

---

## 1. Sensor: Headlight Cone

| Parameter | Value |
|-----------|-------|
| Origin | Car centre (x, y) |
| Direction | Car heading angle |
| Spread | 60° total (±30° from heading) |
| Length | 60 px |
| Rays sampled | Left edge, Right edge, Centre |

The cone is defined by three rays cast from the car's nose:

```
left_ray  = heading − 30°
centre_ray = heading
right_ray = heading + 30°
```

Each ray is sampled at **N = 10 evenly spaced points** along its length
(6 px apart). A "hit" is recorded at the first sample point that lies on or
outside a white boundary ellipse.

---

## 2. White Line Detection (Geometry)

The track has two ellipses: **outer** and **inner**. A ray point `(px, py)` is
considered a hit when:

```
outer hit:  ellipse_value(px, py, OUTER_RX, OUTER_RY) >= 0.95   # near/past outer edge
inner hit:  ellipse_value(px, py, INNER_RX, INNER_RY) <= 1.05   # near/past inner edge
```

where `ellipse_value = ((px − CX) / RX)² + ((py − CY) / RY)²`

Values are thresholded slightly inside the boundary so the agent reacts before
crossing, giving it time to correct.

---

## 3. Rules

### 3.1 Single-ray hits

| Condition | Action |
|-----------|--------|
| Left ray hits outer wall | Steer right |
| Right ray hits outer wall | Steer left |
| Left ray hits inner wall | Steer left |
| Right ray hits inner wall | Steer right |

### 3.2 Both-edge hits (tight section)

| Condition | Action |
|-----------|--------|
| Both rays hit outer wall | Hard steer toward centre + reduce throttle |
| Both rays hit inner wall | Hard steer away from centre + reduce throttle |
| Left hits outer AND right hits inner | Reduce speed, no steer change |

### 3.3 Centre ray hit (imminent collision)

| Condition | Action |
|-----------|--------|
| Centre ray hits any wall | Emergency: full brake + steer away from wall |

### 3.4 No hit (clear road)

| Condition | Action |
|-----------|--------|
| No rays hit anything | Full throttle, no steer change |

---

## 4. Steering Magnitude

Correction strength scales with **how close** the hit is (hit distance `d`):

```
proximity = 1 − (d / CONE_LEN)        # 0 = far, 1 = at car nose
steer_input = BASE_STEER × (1 + proximity × URGENCY_SCALE)
```

Suggested defaults:

| Parameter | Value |
|-----------|-------|
| `BASE_STEER` | 1.0 (one steer step) |
| `URGENCY_SCALE` | 2.0 |
| Max steer clamp | 3.0 |

---

## 5. Throttle Policy

```
if centre ray hit:
    throttle = -1   (brake)
elif both edges hit:
    throttle = 0    (coast)
else:
    throttle = +1   (accelerate)
```

Speed is capped by the existing `MAX_SPEED` in the physics engine.

---

## 6. Agent Step Loop

```
every frame:
  1. cast 3 rays from car position along heading ± 30°
  2. sample each ray at 10 points
  3. for each point, evaluate outer/inner ellipse values
  4. record first hit (type: outer/inner, side: left/centre/right, distance: d)
  5. look up rule table → get (steer, throttle)
  6. scale steer by proximity
  7. pass (accel, steer) to car.update()
```

---

## 7. Implementation Notes

- Ray casting reuses `_in_ellipse()` already defined in `game.py`.
- No state memory needed — each frame is fully determined by current ray hits.
- The agent can be toggled with a key (e.g. **A**) so human and agent modes
  can be compared on the same lap.
- Log `(hit_side, hit_type, distance, steer_applied)` each frame for later
  analysis or imitation learning dataset collection.

---

## 8. Limitations & Next Steps

| Limitation | Possible fix |
|------------|--------------|
| Reacts only to nearest wall, not lane centre | Add centre-of-track error term |
| No speed-dependent lookahead | Scale `CONE_LEN` with current speed |
| Oscillates in straight sections | Add a deadband: only steer if proximity > 0.3 |
| Cannot plan for tight curves | Extend to 5 rays spread across full cone |
