# Imitation Learning — Curriculum Car Racer

Pygame car racing game + Gymnasium/SB3 PPO curriculum training pipeline.

## Structure

```
game/          Pygame simulation, 20 tracks, CarEnv, train/val/test splits
env/           Gymnasium wrapper, headlight image obs, ImpalaCNN encoder, curriculum builder
train_sb3.py   SB3 PPO training (main training script)
run_training.py Automated pipeline: monitors W&B, restarts on stalls/failures
train.py       Legacy custom CleanRL PPO (kept for reference)
main.py        Interactive game entry point
```

---

## Quick Start

### Install

```bash
# Requires Python 3.12+ and uv
uv sync
```

### Play the game interactively

```bash
uv run python main.py       # track 1 (default)
uv run python main.py 5     # track 5
```

---

## Training

### Single run — SB3 PPO

```bash
# Default: 4 envs, 5M steps, W&B online
uv run python train_sb3.py

# More environments (faster collection), longer run
uv run python train_sb3.py --num-envs 8 --total-steps 10_000_000

# Offline (no internet required for W&B)
uv run python train_sb3.py --wandb-offline

# Resume from a checkpoint
uv run python train_sb3.py --resume checkpoints/ppo_sb3_step00250000_lvl02.zip
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num-envs` | 4 | Parallel environments (all in-process via DummyVecEnv) |
| `--total-steps` | 5 000 000 | Total env steps |
| `--rollout-steps` | 2048 | Steps collected per env per PPO update |
| `--lr` | 3e-4 | Initial learning rate (linearly decayed to `--lr-min`) |
| `--gamma` | 0.99 | Discount factor |
| `--threshold` | 30.0 | Mean reward needed to advance to the next curriculum track |
| `--video-interval` | 25 000 | Log inference videos to W&B every N steps (0 = off) |
| `--resume` | — | Path to a `.zip` checkpoint to continue from |

### Automated pipeline (monitors + auto-restarts)

```bash
uv run python run_training.py                     # 10M steps
uv run python run_training.py --total-steps 5_000_000
uv run python run_training.py --fresh             # ignore existing checkpoints
uv run python run_training.py --dry-run           # print commands, don't run
```

`run_training.py` monitors W&B metrics every 60 s and:
- Restarts with softer hyperparameters when the curriculum stalls
- Hard-restarts on NaN / frozen policy
- Stops when all 10 curriculum tracks are mastered

---

## Multi-environment support

`train_sb3.py` uses SB3's `DummyVecEnv` (all envs in the same process, stepped sequentially). This lets all workers **share the curriculum sampler** — curriculum advancement is reflected immediately at every env's next `reset()`.

```
Main process
  ├─ DummyVecEnv
  │    ├─ RaceGymEnv[0]  ──┐
  │    ├─ RaceGymEnv[1]  ──┤── shared CurriculumSampler
  │    ├─ RaceGymEnv[2]  ──┤
  │    └─ RaceGymEnv[3]  ──┘
  └─ CurriculumWandbCallback  (advances sampler, logs W&B, saves checkpoints)
```

Use `--num-envs N` to set N. Recommended values:

| Hardware | `--num-envs` |
|----------|-------------|
| CPU only  | 4–8 |
| GPU (CUDA) | 8–16 |

> `--subproc` flag enables SB3's `SubprocVecEnv` (each env in its own subprocess).
> Curriculum state is managed in the main process but **not synced to subprocesses**
> between steps — use `DummyVecEnv` (default) for correct curriculum behaviour.

---

## Reward structure

Based on [RL-CarNavigationAgent/citymap_assignment.py](https://github.com/nirmalpratheep/RL-CarNavigationAgent/blob/main/citymap_assignment.py):

| Event | Reward |
|-------|--------|
| Every step | **−0.1** (efficiency pressure) |
| Advancing toward next waypoint | **+(1 + wp_cos) / 2 × 20** (0–20, max when aimed straight at waypoint) |
| Moving backward through waypoints | **−10** (distance penalty) |
| Lap completed | **+100** (target-reached bonus) |
| Off-track (crash) | **−100**, episode ends |
| Out of screen bounds | **−100**, episode ends |

`wp_cos` = cosine of the angle between the car's heading and the direction to the next waypoint.  
`wp_cos = 1` → car is aimed straight at the waypoint → maximum heading reward.  
`wp_cos = −1` → car is heading away → heading reward is 0 (but no extra penalty unless waypoint index also decreases).

---

## Curriculum

10 training tracks across 5 difficulty tiers:

| Tier | Tracks | Description |
|------|--------|-------------|
| A-easy | 1, 2 | Wide ovals |
| B-medium-easy | 5, 6 | Rectangular shapes |
| C-medium-hard | 9, 10 | Hairpins & chicanes |
| D-hard | 13, 14 | Complex polygons |
| E-single-lane | 17, 18 | One-car-wide lanes |

The agent starts on track 1. It advances to the next track once its rolling mean episode reward (last `--window` episodes) exceeds `--threshold × track.complexity`.

5 validation tracks (3, 7, 11, 15, 19) and 5 held-out test tracks (4, 8, 12, 16, 20) are never trained on.

---

## Observation space

Each step returns a `Dict` with two entries:

| Key | Shape | Description |
|-----|-------|-------------|
| `image` | (3, 64, 64) float32 | Egocentric headlight view; car always faces up |
| `scalars` | (9,) float32 | speed, angular_velocity, 5 raycasts, wp_sin, wp_cos |

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
                           (288→2 Gaussian)          (288→1 scalar)
```

---

## W&B metrics

| Key | Description |
|-----|-------------|
| `episode/reward` | Total reward per episode |
| `episode/laps` | Laps completed |
| `episode/crashes` | Off-track exits |
| `episode/on_track_pct` | % steps on track |
| `ppo/policy_loss` | Clipped surrogate loss |
| `ppo/explained_variance` | Value function quality (1.0 = perfect) |
| `ppo/approx_kl` | KL divergence (target < 0.02) |
| `curriculum/level` | Current frontier track index |
| `curriculum/rolling_mean` | Rolling mean reward vs threshold |
| `inference/track_XX_*` | Per-track video every `--video-interval` steps |

---

## Documentation

| File | Contents |
|------|----------|
| [game/README.md](game/README.md) | Game controls, tracks, physics, RL interface |
| [env/README.md](env/README.md) | Observation space, action space, encoder, curriculum API |
