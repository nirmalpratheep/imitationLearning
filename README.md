# Curriculum Car Racer — OpenEnv Student Challenge Submission

> **OpenEnv Student Challenge** · Reinforcement Learning · Curriculum Learning · PyTorch · TorchRL

A Pygame car racing environment where a PPO agent learns to drive entirely from scratch — no human demonstrations, no pre-training, no privileged map information. Starting from random noise on a wide oval, the agent progressively masters 10 tracks of increasing difficulty through automated curriculum gating, ending with tight hairpins and a chicane layout.

![Agent racing all 10 curriculum tracks](inference_videos/collage.gif)

*Agent driving all 10 curriculum tracks simultaneously — trained with TorchRL PPO from pure reward.*

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Tracks mastered | **10 / 10** (100%) |
| Total training steps | ~1.3M environment steps |
| Final policy | Zero crashes, ≥1 lap on every track, greedy (deterministic) |
| Observation | 64×64 egocentric image + 9 scalars |
| Action | Continuous `[accel, steer]` in `[−1, 1]²` |
| Architecture | ImpalaCNN + MLP → shared encoder → Gaussian actor + critic |

---

## Quick Start

```bash
# Requires Python 3.12+ and uv
git clone https://github.com/NirmalPratheep/curriculum-car-racer
cd curriculum-car-racer
uv sync

# Play interactively (arrow keys)
uv run python main.py            # Track 1 (Wide Oval)
uv run python main.py 9          # Track 9 (Hairpin)

# Run inference on all tracks from a trained checkpoint
uv run python inference/inference.py \
  --checkpoint checkpoints/ppo_torchrl_final.pt

# Train from scratch (GPU recommended, ~1-3 hours)
bash training/cmd

# Serve as an OpenEnv HTTP API
uvicorn env.server.app:app --host 0.0.0.0 --port 8000

# Or via Docker
docker build -t curriculum-car-racer .
docker run -p 8000:8000 curriculum-car-racer
```

---

## Repository Layout

```
curriculum-car-racer/
├── game/                    # Pure Pygame simulation — no ML dependencies
│   ├── tracks.py            # 10 procedural TrackDef objects
│   ├── rl_splits.py         # CarEnv (physics + reward + raycasting), CurriculumSampler
│   ├── oval_racer.py        # Car rendering, headlight cone
│   └── curriculum_game.py   # Human-playable interactive mode
├── env/                     # OpenEnv + Gymnasium wrappers
│   ├── environment.py       # RaceEnvironment — OpenEnv Environment subclass
│   ├── gym_env.py           # RaceGymEnv — Gymnasium Dict wrapper
│   ├── encoder.py           # ImpalaCNN + RaceEncoder (image+scalars → 288-d)
│   ├── curriculum.py        # CurriculumBuilder — track progression manager
│   ├── models.py            # DriveAction, RaceObservation, RaceState (Pydantic)
│   └── server/app.py        # FastAPI entry point for OpenEnv HTTP/WS serving
├── training/                # Training scripts
│   ├── train_torchrl.py     # Main PPO training with curriculum + priority replay
│   ├── monitor.py           # W&B live monitor
│   ├── test_video.py        # Render single inference episode to MP4
│   └── push_to_hub.py       # Upload to HuggingFace Hub
├── inference/
│   └── inference.py         # Greedy eval — runs all tracks, saves MP4s
├── doc/                     # Architecture & design documentation
│   ├── game.md              # Simulation, tracks, physics, raycasting
│   ├── environment.md       # Observation, reward, model architecture
│   └── training.md          # Training process, hyperparameters, W&B metrics
├── openenv.yaml             # OpenEnv environment manifest
└── Dockerfile               # Container for API serving
```

---

## The 10-Track Curriculum

Tracks are ordered by difficulty. The agent trains on the **frontier track** while replaying mastered tracks to prevent forgetting.

| # | Name | Width | Max Speed | Challenge |
|---|------|-------|-----------|-----------|
| 1 | Wide Oval | 115 px | 3.0 | Baseline — just drive |
| 2 | Standard Oval | 85 px | 3.5 | Narrower road |
| 3 | Narrow Oval | 58 px | 3.5 | Precision required |
| 4 | Superspeedway | 85 px | 4.5 | High speed, elongated |
| 5 | Rounded Rectangle | 90 px | 3.5 | First real corners |
| 6 | Stadium Oval | 80 px | 4.0 | Tight end-caps |
| 7 | Tight Rectangle | 65 px | 3.5 | Sharp 90° corners |
| 8 | Small Oval | 60 px | 3.2 | Small radius turns |
| 9 | Hairpin Track | 75 px | 3.5 | Hairpin turn |
| 10 | Chicane Track | 70 px | 3.5 | S-bend chicane section |

```
Group A — Easy ovals         (Tracks 1–4)   Width from 115 → 58 px
Group B — Rectangular shapes (Tracks 5–8)   First 90° corners, stadium curves
Group C — Hairpins & chicanes(Tracks 9–10)  Hairpin reversals, S-bends
```

---

## Observation Space

The agent observes **two modalities**, fused by the encoder before the actor/critic heads:

### Egocentric Headlight Image — `(3, 64, 64) float32`

A 64×64 RGB crop centred on the car, always rotated so the car faces upward. This makes the visual input invariant to absolute position and heading — the network learns "wall approaching from the right" rather than "at pixel (450, 515)."

```
game state (x, y, angle)
  → blit track surface to offscreen 900×600 canvas
  → draw 60°-wide headlight cone, 60 px ahead
  → crop 120×120 px around car (grass-padded at edges)
  → rotate so heading = UP
  → re-crop centre 120×120 after rotation padding
  → scale to 64×64
  → surfarray → (64, 64, 3) uint8 → CHW float32 / 255
```

### Scalar Sensors — `(9,) float32`

```
angular_velocity   gyroscope (rotational speed)
speed              forward speed / max_speed
ray_left           boundary distance at −90° relative to heading
ray_front_left     boundary distance at −45°
ray_front          boundary distance at   0° (straight ahead)
ray_front_right    boundary distance at  +45°
ray_right          boundary distance at  +90°
wp_sin             sin(bearing to next waypoint − car heading)
wp_cos             cos(bearing to next waypoint − car heading)
```

The 5 raycasts replace the old binary `on_track` flag with explicit, continuous boundary lookahead — the agent can "see" a wall before hitting it. `wp_sin / wp_cos` encode a relative GPS compass, invariant to absolute position.

### Gymnasium Space

```python
observation_space = Dict({
    "image":   Box(0.0, 1.0, shape=(3, 64, 64), dtype=float32),
    "scalars": Box(-inf, inf, shape=(9,),         dtype=float32),
})
action_space = Box(-1.0, 1.0, shape=(2,), dtype=float32)  # [accel, steer]
```

---

## Reward Design

All rewards are kept small to prevent value-function explosion:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Every step | −0.005 | Efficiency pressure |
| Forward speed (on-track) | `speed_norm × 0.10` | Must drive forward to earn reward |
| Reverse speed | `speed_norm × 0.10` (negative) | Penalise going backwards |
| Waypoint advance (forward) | +0.25 per waypoint crossed | Dense directional signal |
| Waypoint regress (backward) | −0.25 per waypoint lost | Penalise wrong-way driving |
| Lap completed | **+10.0** | Major completion bonus |
| Off-track / crash | **−15.0**, episode ends | Strong deterrent |
| Out of bounds | **−15.0**, episode ends | Stay on screen |

**Lap detection** uses a two-phase arm/trigger gate: the car must first travel 50 px *past* the start line (arming phase), then cross back through it having covered ≥80% of the optimal lap distance (anti-shortcut).

---

## Model Architecture

```
image (3, 64, 64)
    └─► ImpalaCNN
        ├─ Block 1: Conv(3→16)   → MaxPool → ResBlock × 2
        ├─ Block 2: Conv(16→32)  → MaxPool → ResBlock × 2
        └─ Block 3: Conv(32→32)  → MaxPool → ResBlock × 2
        → Flatten(2048) → Linear → ReLU → 256-d

scalars (9,)
    └─► MLP: Linear(9→32) → ReLU → Linear(32→32) → ReLU → 32-d

                        Concat → 288-d (RaceEncoder)
                                    │
                    ┌───────────────┴───────────────┐
                 Actor head                     Critic head
          Linear(288→2) + log_std           Linear(288→1)
          IndependentNormal                 scalar value V(s)
          → [accel, steer] ∈ [−1,1]²
```

**Design choices:**
- **ImpalaCNN** (3 residual blocks) over Nature CNN: skip connections give gradients a direct path to early conv layers — ~3–5× better sample efficiency
- **Shared encoder** between actor and critic: the bottleneck 288-d vector is jointly optimised
- **Orthogonal init**: gain √2 for encoder, 0.01 for actor head (small initial actions), 1.0 for critic
- **Actor bias init**: `mean.bias[0] = 0.3` — gentle initial forward acceleration, avoids the agent spinning in place during early exploration
- **Log-std**: initialised at −1.0 (std ≈ 0.37, moderate initial exploration), unbounded during training

---

## Training — TorchRL PPO with Curriculum Gating

```bash
uv run python -u training/train_torchrl.py \
  --num-envs 16 \
  --rollout-steps 8192 \
  --batch-size 1024 \
  --ppo-epochs 10 \
  --compile \
  --window 20 \
  --video-interval 100000 \
  --total-steps 300_000_000
```

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Parallel environments | 16 | TorchRL `ParallelEnv` |
| Rollout steps | 8192 | Frames before each PPO update |
| Minibatch size | 1024 | PPO minibatches per rollout |
| PPO epochs | 10 | Passes per rollout |
| Learning rate | 3e-4 | Adam (eps=1e-5) |
| Discount γ | 0.99 | |
| GAE λ | 0.95 | Generalised Advantage Estimation |
| Clip ε | 0.2 | PPO clipping range |
| Value loss coef | 0.5 | |
| Entropy coef | 0.01 | Exploration bonus |
| Max grad norm | 0.5 | Gradient clipping |
| Target KL | 0.1 | Early-stop per PPO epoch |

### Curriculum Advancement

Every `--eval-interval-steps` (default 25k), the training loop runs **greedy (deterministic) episodes** on every track:

1. A track **passes** if the agent completes ≥1 lap with 0 crashes
2. The frontier **advances** when all tracks up to and including the current one pass
3. If **all 10 tracks pass simultaneously**, training is complete

### Anti-Forgetting Replay

```
70% of episodes → current frontier track
30% of episodes → round-robin through all mastered tracks
```

Round-robin ensures every mastered track gets equal coverage, preventing catastrophic forgetting as the curriculum grows.

### Priority Replay

When greedy eval finds a regression (a previously mastered track now failing), that track is added to a **priority replay** list. Workers dedicate an additional 30% of episodes to these failing tracks until the next eval clears them. This recovers from regressions within one eval interval.

---

## OpenEnv Integration

This environment is fully compliant with the [OpenEnv standard](https://openenv.dev).

### As an HTTP API

```python
from openenv import connect

env = connect("http://localhost:8000")
obs = env.reset()
while not obs.done:
    action = my_agent(obs)   # DriveAction(accel=..., steer=...)
    obs = env.step(action)
```

### Locally (Gymnasium)

```python
from env.gym_env import RaceGymEnv

env = RaceGymEnv(track_level=5)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Manifest (`openenv.yaml`)

```yaml
name: curriculum-car-racer
version: 0.1.0
entry_point: env.server.app:app
action_type: env.models:DriveAction
observation_type: env.models:RaceObservation
state_type: env.models:RaceState
```

---

## W&B Training Curves

![Episode metrics — reward, laps, crashes across curriculum](doc/Training-Episode-wandb.png)

![PPO internals — value loss, policy loss, entropy, explained variance](doc/Training-PPO-wandb.png)

Key metrics logged during training:

| Metric | Description |
|--------|-------------|
| `episode/reward` | Total reward per episode |
| `episode/laps` | Laps completed per episode |
| `episode/crashes` | Off-track exits per episode |
| `curriculum/level` | Current frontier track index (0-based) |
| `curriculum/greedy_pass` | 1 when all tracks pass greedy eval |
| `curriculum/priority_n_tracks` | Tracks currently in priority replay |
| `ppo/explained_variance` | Value quality (1.0 = perfect predictor) |
| `ppo/entropy` | Policy entropy (exploration health) |
| `system/steps_per_sec` | Training throughput |

---

## Documentation

| Document | Contents |
|----------|----------|
| [doc/game.md](doc/game.md) | Simulation details — 10 tracks, car physics, raycasting, waypoint GPS |
| [doc/environment.md](doc/environment.md) | OpenEnv/Gymnasium wrappers, observation design, reward shaping, model architecture |
| [doc/training.md](doc/training.md) | Training process, curriculum progression, priority replay, all hyperparameters, W&B reference |

---

## HuggingFace

- **Blog post**: [Teaching an RL Agent to Race from Scratch with Curriculum Learning and OpenEnv](https://huggingface.co/blog/NirmalPratheep/curriculum-car-racer)
- **Model / Environment Hub**: `NirmalPratheep/curriculum-car-racer`

---

## Citation

```bibtex
@misc{pratheep2026curriculumcarracer,
  title   = {Curriculum Car Racer: Automated Curriculum RL with OpenEnv},
  author  = {Nirmal Pratheep},
  year    = {2026},
  url     = {https://github.com/NirmalPratheep/curriculum-car-racer}
}
```
