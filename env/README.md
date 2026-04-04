# Environment

OpenEnv-compatible RL environment wrapping the car racing game. Provides a typed
observation (egocentric headlight image + scalar features), typed action, curriculum
builder, and an Impala CNN encoder ready to plug into a PPO actor-critic.

## Quick Start

```python
from env import CurriculumBuilder, DriveAction

builder = CurriculumBuilder()

# Training loop
env = builder.next_env()
obs = env.reset()

total_reward = 0.0
while not obs.done:
    # obs.image   : (64, 64, 3) uint8 numpy array
    # obs.speed   : float  0..1
    # obs.on_track: float  1.0 / 0.0
    action = DriveAction(accel=1.0, steer=0.0)
    obs = env.step(action)
    total_reward += obs.reward

advanced = builder.record(total_reward)   # auto-advances curriculum when ready
print(builder.status)
```

## File Structure

```
env/
  models.py       DriveAction and RaceObservation (Pydantic, OpenEnv-compatible)
  environment.py  RaceEnvironment — server-side wrapper around game.rl_splits.CarEnv
  client.py       RaceEnvClient  — OpenEnv WebSocket client
  encoder.py      ImpalaCNN + RaceEncoder (PyTorch) for PPO actor-critic
  curriculum.py   CurriculumBuilder — wraps rl_splits TRAIN/VAL/TEST splits
```

## Observation Space

`RaceObservation` has two parts that feed different network branches:

### Image — `obs.image` → CNN encoder
- Shape: `(64, 64, 3)` uint8
- **Egocentric**: car always faces up, track geometry is heading-invariant
- Rendering pipeline per step:
  1. Blit track surface to offscreen canvas
  2. Draw headlight cone (60° spread, 60 px ahead)
  3. Crop 120×120 px square centred on car (grass-padded at borders)
  4. Rotate so car heading maps to UP
  5. Re-crop centre after rotation padding
  6. Scale to 64×64

### Scalars — `obs.speed / on_track / sin_angle / cos_angle` → MLP encoder

| Field | Range | Purpose |
|-------|-------|---------|
| `speed` | 0..1 | Speed / max_speed. Controls braking decisions. |
| `on_track` | 0 or 1 | Reactive penalty signal. |
| `sin_angle` | −1..1 | Absolute heading orientation. |
| `cos_angle` | −1..1 | Absolute heading orientation. |

**Dropped from original CarEnv obs** (would hurt generalisation to unseen tracks):

| Dropped field | Why |
|--------------|-----|
| `x`, `y` | Absolute screen position — track-specific, causes overfitting |
| `gate_side` | Distance to start/finish gate — meaningless on unseen layouts |

## Action Space

`DriveAction(accel, steer)` — continuous, both clamped to `[−1, 1]` inside `CarEnv.step`.

| Field | Range | Effect |
|-------|-------|--------|
| `accel` | +1 | Full throttle |
| `accel` | −1 | Brake |
| `steer` | +1 | Steer right |
| `steer` | −1 | Steer left |

## Reward Function

Defined in `game/rl_splits.py:CarEnv.step`. All terms scale with
`C = track.complexity` so the curriculum `threshold` stays meaningful across
all 16 tracks without manual tuning (Track 1 → C=1.0, Track 16 → C=3.45).

| Term | Trigger | Value | Goal |
|------|---------|-------|------|
| Forward pulse | Every step | `+speed/max_speed × 0.01` | Prevent stalling |
| Off-track | Every step off road | `−0.5 × C` | Stay on road |
| Crash event | on→off transition | `−5.0 × C` | Fewer boundary hits |
| Lap completion | Gate crossed cleanly | `+50 × time_ratio × dist_ratio × C` | Fast + short path |
| Out of bounds | Terminal | `−100 × C` | Don't leave screen |

**Lap completion bonus** rewards efficiency on two axes simultaneously:

```
time_ratio = clamp(par_time_steps / actual_lap_steps,  0.5, 2.0)
dist_ratio = clamp(optimal_dist   / actual_lap_dist,   0.5, 1.5)
```

| Performance | time_ratio | dist_ratio | Lap bonus (C=1) |
|------------|-----------|-----------|-----------------|
| Faster than par, tight racing line | 2.0 | 1.5 | +150 |
| On-par time, near-optimal path | 1.0 | 1.0 | +50 |
| Slow, meandering path | 0.5 | 0.5 | +12.5 |

**Curriculum window note:** with complexity scaling, the same `threshold=30.0`
works across all tracks — a competent lap on Track 16 naturally scores ~3.45×
higher than the same quality lap on Track 1.

## Encoder

`RaceEncoder` fuses both observation branches into a single feature vector for PPO:

```
image (64×64×3)
  └─► ImpalaCNN  →  256-d
                         ├─► cat  →  288-d  →  Actor / Critic heads
scalars (4,)              │
  └─► MLP 4→32→32  →  32-d
```

```python
import torch
from env import RaceEncoder

encoder = RaceEncoder()           # out_features = 288
img     = torch.zeros(4, 3, 64, 64)   # batch of 4, normalised 0..1
scalars = torch.zeros(4, 4)
features = encoder(img, scalars)  # (4, 288)
```

### ImpalaCNN vs Nature CNN

| | Nature CNN (DQN) | ImpalaCNN (IMPALA) |
|---|---|---|
| Architecture | 3 plain conv layers | 3 blocks × (Conv + MaxPool + 2 ResBlocks) |
| Skip connections | None | Yes — `x = x + residual(x)` in each block |
| Gradient flow | Vanishes in early layers | Direct path back through shortcuts |
| Sample efficiency | Baseline | ~3–5× better on visual RL tasks |
| Inference cost | Fast | Same (equivalent depth) |

## Curriculum Builder

Based on the 16-track split in `game/rl_splits.py`:

| Split | Tracks | Purpose |
|-------|--------|---------|
| TRAIN | 1,2, 5,6, 9,10, 13,14 | 2 per tier, curriculum ordered easy→hard |
| VAL | 3, 7, 11, 15 | 1 per tier — performance gating, never trained on |
| TEST | 4, 8, 12, 16 | 1 per tier — held-out, final evaluation only |

```python
from env import CurriculumBuilder

builder = CurriculumBuilder(
    threshold=30.0,  # mean reward needed to advance (same value works all tracks due to complexity scaling)
    window=50,       # rolling window size — advance only after 50 consecutive episodes exceed threshold
                     # too small (e.g. 5)  → advances on lucky streaks, policy not stable yet
                     # too large (e.g. 500) → stays on mastered track too long, slows curriculum
    replay_frac=0.3, # 30% of episodes replay mastered tracks (prevents forgetting)
    use_image=True,  # set False to skip image rendering (fast unit tests / ablations)
)

env = builder.next_env()          # samples frontier (or replay) track
builder.record(episode_reward)    # auto-advances when threshold met

for env in builder.val_envs():    # evaluate on held-out VAL tracks
    ...

print(builder.status)             # "Frontier: track 2 'Standard Oval' [2/8] ..."
print(builder.is_complete)        # True when all TRAIN tracks mastered
```

## OpenEnv Client (Remote Server)

To run the environment as a server and connect from a remote training process:

```python
# server — start with: openenv serve env.environment:RaceEnvironment
# client
from env import RaceEnvClient, DriveAction

async with RaceEnvClient(base_url="http://localhost:8000") as client:
    result = await client.reset()
    result = await client.step(DriveAction(accel=1.0, steer=0.0))

# or synchronously
with RaceEnvClient(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    result = client.step(DriveAction(accel=1.0, steer=0.0))
```

## Headless Mode (parallel training)

Set these env vars before importing pygame to run without a display:

```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
```

`RaceEnvironment` renders entirely to offscreen `pygame.Surface` objects, so no
display is needed at any point.
