# Curriculum Car Racer — OpenEnv Environment

A Pygame car racing environment with **curriculum learning**, egocentric headlight vision,
and 20 procedural tracks of increasing difficulty. Built for the
[OpenEnv Challenge](https://github.com/meta-pytorch/OpenEnv).

```
game/        Pygame simulation — 20 tracks, car physics, curriculum splits
env/         OpenEnv + Gymnasium wrappers, ImpalaCNN encoder, curriculum builder
training/    PPO training scripts (TorchRL recommended), monitor, test utilities
```

---

## Quick Start

```bash
# Requires Python 3.12+ and uv
git clone <repo-url>
cd curriculum-car-racer
uv sync

# Play interactively (arrow keys)
uv run python main.py
uv run python main.py 5      # start on track 5

# Train with PPO (GPU recommended)
bash training/cmd

# Serve as OpenEnv API
uvicorn env.server.app:app --host 0.0.0.0 --port 8000

# Or via Docker
docker build -t curriculum-car-racer .
docker run -p 8000:8000 curriculum-car-racer
```

---

## Folder Structure

### `game/` — Simulation

Pure Pygame racing simulation. No ML dependencies.

| File | Description |
|------|-------------|
| `oval_racer.py` | Car physics, rendering primitives |
| `tracks.py` | 20 procedural `TrackDef` objects |
| `rl_splits.py` | `CarEnv`, `CurriculumSampler`, train/val/test splits |
| `curriculum_game.py` | Interactive human-playable game |

### `env/` — OpenEnv / Gymnasium

Wraps `game/` for RL training and OpenEnv API serving.

| File | Description |
|------|-------------|
| `environment.py` | `RaceEnvironment` — OpenEnv server-side interface |
| `gym_env.py` | `RaceGymEnv` — Gymnasium-compatible wrapper |
| `encoder.py` | `ImpalaCNN` + `RaceEncoder` (image + scalars → 288-d) |
| `curriculum.py` | `CurriculumBuilder` — track progression manager |
| `models.py` | `DriveAction`, `RaceObservation`, `RaceState` (Pydantic) |
| `server/app.py` | FastAPI entry point for OpenEnv HTTP/WS serving |

### `training/` — Scripts

See [training/README.md](training/README.md) for full documentation.

| Script | Description |
|--------|-------------|
| `train_torchrl.py` | **TorchRL PPO** — recommended training script |
| `train_cleanrl.py` | CleanRL custom PPO — legacy reference |
| `monitor.py` | W&B monitor with auto-restart on training stalls |
| `test_video.py` | Render a single inference episode to MP4 |
| `push_to_hub.py` | Upload environment to HuggingFace Hub |

---

## Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `image` | (3, 64, 64) float32 | Egocentric headlight view — car always faces up |
| `scalars` | (9,) float32 | speed, angular_velocity, 5 raycasts, wp_sin, wp_cos |

## Action Space

`Box(-1.0, 1.0, (2,))` — `[accel, steer]`

## Reward Structure

| Event | Reward |
|-------|--------|
| Every step | −0.1 (efficiency) |
| Heading toward waypoint | +(1 + wp_cos) / 2 × 20 (0–20) |
| Moving backward past waypoint | −10 |
| Lap completed | +100 |
| Off-track / crash | −100, episode ends |

---

## Curriculum

10 training tracks across 5 difficulty tiers:

| Tier | Tracks | Description |
|------|--------|-------------|
| Easy | 1, 2 | Wide ovals |
| Medium-Easy | 5, 6 | Rectangular shapes |
| Medium-Hard | 9, 10 | Hairpins & chicanes |
| Hard | 13, 14 | Complex polygons |
| Single-lane | 17, 18 | One-car-wide lanes |

Agent advances when rolling mean reward > `threshold × track.complexity`.
5 validation tracks and 5 held-out test tracks are never trained on.

---

## Architecture

```
image (3,64,64) ──► ImpalaCNN ──────────► 256-d
                     3 residual blocks
scalars (9,) ──────► MLP (9→32→32) ──────► 32-d
                                            │
                                     concat 288-d
                                            │
                              ┌─────────────┴─────────────┐
                           actor head                critic head
                         (288→2 Gaussian)           (288→1 scalar)
```

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

Or use the Gymnasium wrapper directly:

```python
from env.gym_env import RaceGymEnv
env = RaceGymEnv()
obs, info = env.reset()
```

---

## W&B Metrics

| Key | Description |
|-----|-------------|
| `episode/reward` | Total reward per episode |
| `episode/laps` | Laps completed |
| `episode/crashes` | Off-track exits |
| `curriculum/level` | Current frontier track |
| `curriculum/rolling_mean` | Rolling mean vs threshold |
| `ppo/approx_kl` | KL divergence (healthy < 0.05) |
| `ppo/explained_variance` | Value quality (1.0 = perfect) |
| `inference/track_XX_*` | Per-track video at each video interval |

---

## Documentation

| File | Contents |
|------|----------|
| [training/README.md](training/README.md) | Training flags, resume, video rendering |
| [game/README.md](game/README.md) | Game physics, controls, track system |
| [env/README.md](env/README.md) | Observation space, encoder, curriculum API |
| [doc/](doc/) | Architecture, curriculum, encoder deep-dives |
