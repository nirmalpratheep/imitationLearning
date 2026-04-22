# Environment & Reward Design

The `env/` package wraps the Pygame simulation as both an **OpenEnv** server-side
environment and a **Gymnasium** `Dict`-observation environment for PPO training.

## Files

| File | Description |
|------|-------------|
| `environment.py` | `RaceEnvironment` — OpenEnv `Environment` subclass, renders headlight image |
| `gym_env.py` | `RaceGymEnv` — Gymnasium wrapper with `Dict` obs (image + scalars), supports `ParallelEnv` |
| `encoder.py` | `ImpalaCNN` + `RaceEncoder` — image (256-d) + scalars (32-d) → 288-d features |
| `curriculum.py` | `CurriculumBuilder` — bridges `CurriculumSampler` with `RaceEnvironment` |
| `models.py` | `DriveAction`, `RaceObservation`, `RaceState` (Pydantic / OpenEnv types) |
| `server/app.py` | FastAPI entry point for OpenEnv HTTP/WS serving |

---

## Observation Space

### Image — Egocentric Headlight View

A 64×64 RGB crop centred on the car, rotated so the car always faces up.
This makes the visual input invariant to absolute position and heading —
the CNN learns "a wall is approaching from the right" rather than
"at pixel (450, 515) the track curves left."

**Rendering pipeline** (per step in `RaceEnvironment`):
```
game state (x, y, angle)
  → blit track surface to offscreen 900×600 Surface
  → draw headlight cone (60° wide, 60 px ahead)
  → crop 120×120 px around car (grass-padded at edges)
  → rotate so heading = UP
  → re-crop centre 120×120 after rotation padding
  → scale to 64×64
  → surfarray → (64, 64, 3) uint8
```

In `RaceGymEnv`, the image is transposed to CHW and normalised:
`(3, 64, 64) float32` in `[0, 1]`.

### Scalars — 9-dimensional vector

```
obs.angular_velocity : float  rotational speed (gyroscope)
obs.speed            : float  forward speed / max_speed (speedometer)
obs.ray_left         : float  boundary distance at -90° (lateral left)
obs.ray_front_left   : float  boundary distance at -45° (diagonal)
obs.ray_front        : float  boundary distance at  0° (straight ahead)
obs.ray_front_right  : float  boundary distance at +45° (diagonal)
obs.ray_right        : float  boundary distance at +90° (lateral right)
obs.wp_sin           : float  sin of relative bearing to next waypoint
obs.wp_cos           : float  cos of relative bearing to next waypoint
```

**Design rationale**: The 5 raycasts provide explicit geometric distance to
boundaries _before_ a crash (replacing the old binary `on_track` flag).
The waypoint bearings give a relative GPS compass toward the track ahead.
`angular_velocity` and `speed` are proprioceptive sensors that any real car has.

### Gymnasium Space

```python
observation_space = Dict({
    "image":   Box(0.0, 1.0, shape=(3, 64, 64), dtype=float32),
    "scalars": Box(-inf, inf, shape=(9,),         dtype=float32),
})
action_space = Box(-1.0, 1.0, shape=(2,), dtype=float32)  # [accel, steer]
```

---

## Reward Shaping

All rewards are kept small to prevent value-function explosion and keep
log-std (policy exploration) receiving meaningful gradients.

| Event | Reward | Purpose |
|-------|--------|---------|
| Every step | −0.005 | Efficiency pressure — don't dawdle |
| Forward speed (on-track) | `speed_norm * 0.10` | Must drive forward to earn reward |
| Reverse speed | `speed_norm * 0.10` (negative) | Penalise going backwards |
| Waypoint advance (forward) | +0.25 per waypoint | Dense directional signal toward lap completion |
| Waypoint regress (backward) | −0.25 per waypoint | Penalise wrong-way driving |
| Lap completed | +10.0 | Major bonus for completing a lap |
| Off-track / crash | −15.0, **episode ends** | Strong deterrent against leaving the road |
| Out of bounds | −15.0, **episode ends** | Stay on screen |

### Waypoint Progress

The track centreline waypoints form a closed loop. Each step, the nearest
waypoint is found. Advancing forward earns `+0.25` per waypoint crossed;
going backward costs `−0.25`. This provides dense gradient toward lap
completion — without it, the only lap signal is the `+10` bonus at the finish.

### Lap Detection

Two-phase arm/trigger to reliably detect gate crossings:
1. **Arm**: Car travels 50 px past the start/finish gate
2. **Trigger**: Car crosses back through the gate (`prev_side < 0 → curr_side ≥ 0`)
   with `speed > 0.3` and `lap_dist ≥ 80% of optimal_dist` (anti-shortcut gate)

### Episode Termination

- Car goes off-track (crash)
- Car leaves screen bounds
- `max_steps` exceeded (default: 3000)
- `laps_target` laps completed (default: 1)

---

## Model Architecture

```
ImpalaCNN(image 3×64×64)
    → 3 blocks × (Conv → MaxPool → ResBlock × 2)
    → channels: 3 → 16 → 32 → 32
    → flatten 2048 → FC → 256-d

MLP(9 scalars)
    → Linear(9→32) → ReLU → Linear(32→32) → ReLU → 32-d

Concat → 288-d feature vector
    |→ actor_mean   Linear(288, 2)   → [accel, steer] mean
    |   actor_log_std  Parameter(2)  → shared log std
    |   → IndependentNormal → sample action
    |→ critic        Linear(288, 1)  → scalar value estimate
```

- **Parameter sharing**: The `RaceEncoder` (ImpalaCNN + MLP) is shared between actor and critic heads
- **Init**: Orthogonal weights (gain √2 for encoder, 0.01 for actor, 1.0 for critic)
- **Actor bias**: `mean.bias[0] = 0.3` — gentle forward accel so the car stays moving during exploration
- **Log-std**: Starts at −1.0 (std ≈ 0.37), unbounded

### ImpalaCNN vs Nature CNN

ImpalaCNN uses residual skip connections (`x + f(x)`) in each block, giving
gradients a direct path to early conv layers. This results in ~3–5× better
sample efficiency compared to Nature CNN at the same inference cost.

---

## OpenEnv Integration

```python
from openenv import connect

env = connect("http://localhost:8000")
obs = env.reset()
while not obs.done:
    action = my_agent(obs)
    obs = env.step(action)
```

Or use the Gymnasium wrapper directly for local training:

```python
from env.gym_env import RaceGymEnv
env = RaceGymEnv()
obs, info = env.reset()
```

For containerised serving:
```bash
docker build -t curriculum-car-racer .
docker run -p 8000:8000 curriculum-car-racer
```
