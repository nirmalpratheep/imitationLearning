# PPO Training Guide

## Reward Function

All reward terms are multiplied by `complexity = (115/track.width) × (max_speed/3.0)`.
This keeps the curriculum `threshold` meaningful across all 16 tracks without manual tuning.
Track 1 → C=1.0, Track 16 → C=3.45.

| Term | When | Value | Purpose |
|------|------|-------|---------|
| Forward pulse | Every step | `+speed/max_speed × 0.01` | Breaks ties, prevents stalling |
| Off-track | Every step off road | `−0.5 × C` | Continuous pressure to stay on road |
| Crash event | On→off boundary crossing | `−5.0 × C` | Penalises each individual crash |
| Lap completion | Gate crossed cleanly | `+50 × time_ratio × dist_ratio × C` | Main learning signal |
| Out of bounds | Terminal | `−100 × C` | Prevents driving off screen |

### Lap completion components

```
time_ratio = clamp(par_time_steps / actual_lap_steps,  0.5, 2.0)
dist_ratio = clamp(optimal_dist   / actual_lap_dist,   0.5, 1.5)
```

- **`par_time_steps`**: expected lap time in frames at 70% of `max_speed` (accounts for corners)
- **`optimal_dist`**: track centreline perimeter from waypoints (the theoretical shortest path)
- Faster than par → `time_ratio > 1` → reward scales up (max 2×)
- Shorter path → `dist_ratio > 1` → reward scales up (max 1.5×)
- Both bad → still gets at least `50 × 0.5 × 0.5 × C = 12.5 × C` for completing the lap

### Why this design

| Goal | How it's achieved |
|------|------------------|
| Fewer crashes | `−5×C` crash event penalty dominates accumulation |
| Shorter distance | `dist_ratio = optimal/actual` — tighter racing line scores higher |
| Faster lap time | `time_ratio = par/actual` — beating par is rewarded up to 2× |
| Complexity scaling | All terms × C — harder tracks give proportionally larger signals, `threshold=30` stays valid everywhere |

---

## Algorithm Choice

**PPO (Proximal Policy Optimisation)** is recommended for this environment.

| Algorithm | Suitable? | Reason |
|-----------|-----------|--------|
| PPO | ✓ Best fit | Stable with visual inputs, continuous actions, curriculum |
| DQN | Partial | Requires discretising actions; worse on continuous control |
| SAC | Possible | Off-policy — harder to combine with curriculum sampling |
| A3C | Avoid | Unstable with image obs; PPO supersedes it |

---

## Network Architecture

```python
import torch.nn as nn
from env.encoder import RaceEncoder

class PPOActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RaceEncoder()          # out: 288-d
        D = self.encoder.out_features         # 288

        # Actor: outputs mean and log_std for accel and steer
        self.actor_mean    = nn.Linear(D, 2)
        self.actor_log_std = nn.Parameter(torch.zeros(2))

        # Critic: single scalar value estimate
        self.critic = nn.Linear(D, 1)

    def forward(self, img, scalars):
        feat = self.encoder(img, scalars)
        return self.actor_mean(feat), self.actor_log_std, self.critic(feat)
```

Actions are sampled from `Normal(mean, exp(log_std))` and clamped to `[-1, 1]`.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | Adam optimiser |
| LR schedule | Linear decay to 1e-5 | Over full training horizon |
| Discount γ | 0.99 | |
| GAE λ | 0.95 | Advantage estimation |
| Clip ε | 0.2 | PPO clipping |
| Value loss coeff | 0.5 | |
| Entropy coeff | 0.01 → 0.001 | Anneal from phase 2 onward |
| Rollout steps | 2048 | Steps collected before each update |
| Mini-batch size | 256 | |
| PPO epochs | 4 | Updates per rollout |
| Max grad norm | 0.5 | Gradient clipping |

---

## Observation Preprocessing

```python
# Normalise image to float32 [0, 1]
img = obs.image.astype(np.float32) / 255.0       # (64, 64, 3)
img = torch.from_numpy(img).permute(2, 0, 1)     # (3, 64, 64)  CHW

# Scalars already in [-1, 1]
scalars = torch.tensor([
    obs.speed,
    obs.on_track,
    obs.sin_angle,
    obs.cos_angle,
], dtype=torch.float32)
```

---

## Headless Training (Parallel Envs)

Set env vars **before** importing pygame:

```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
```

`RaceEnvironment` renders entirely to offscreen `pygame.Surface` objects — no
display is touched. Run N environments in parallel by instantiating N
`RaceEnvironment` objects. For true multiprocessing, wrap each in a subprocess
and communicate via queue or `multiprocessing.Pipe`.

Expected throughput (image obs, CPU only):

| Parallel envs | Steps/sec | Time to 3 M steps |
|---------------|-----------|-------------------|
| 1 | ~60 | ~14 h |
| 4 | ~240 | ~3.5 h |
| 8 (GPU render) | ~480 | ~1.7 h |
| 16 (offscreen) | ~900 | ~55 min |

---

## Training Loop Sketch

```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import torch
from env import CurriculumBuilder, DriveAction
from env.encoder import RaceEncoder

builder = CurriculumBuilder(threshold=30.0, window=50)
model   = PPOActorCritic()
optim   = torch.optim.Adam(model.parameters(), lr=3e-4)

for episode in range(MAX_EPISODES):
    env      = builder.next_env()
    obs      = env.reset()
    rollout  = []
    total_r  = 0.0

    while not obs.done:
        img     = preprocess_image(obs.image)
        scalars = preprocess_scalars(obs)

        with torch.no_grad():
            mean, log_std, value = model(img.unsqueeze(0), scalars.unsqueeze(0))
            dist   = torch.distributions.Normal(mean, log_std.exp())
            action = dist.sample().clamp(-1, 1)
            logp   = dist.log_prob(action).sum(-1)

        obs = env.step(DriveAction(accel=action[0, 0].item(),
                                   steer=action[0, 1].item()))
        rollout.append((img, scalars, action, logp, value, obs.reward, obs.done))
        total_r += obs.reward

    # PPO update on rollout...
    ppo_update(model, optim, rollout)
    advanced = builder.record(total_r)

    if advanced:
        print(f"Advanced! {builder.status}")
        val_scores = evaluate(model, builder.val_envs())
        print(f"Val mean reward: {val_scores:.1f}")
```

---

## Logging Checklist

Track these metrics in TensorBoard / W&B:

```
Training
  episode_reward         total reward per episode
  episode_length         steps per episode
  laps_completed         laps per episode
  on_track_fraction      fraction of steps on track
  frontier_level         current curriculum level (0–7)

PPO internals
  policy_loss
  value_loss
  entropy
  approx_kl             should stay < 0.02; if > 0.05, reduce lr
  clip_fraction         fraction of clipped updates

Evaluation (on VAL tracks after each advance)
  val/mean_reward
  val/completion_rate    fraction of episodes with ≥ 1 lap
  val/mean_laps
```

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Agent drives in circles | Reward too sparse, no lap bonus | Add `+50` lap bonus |
| KL divergence explodes | LR too high | Reduce to 1e-4 |
| Agent hugs inner wall | `on_track` reward not penalising enough | Increase off-track penalty |
| Policy collapses after curriculum advance | Forgetting previous tracks | Increase `replay_frac` to 0.4 |
| Image obs not helping vs flat obs | Image normalisation wrong | Confirm `/ 255.0` and CHW order |
| Training very slow | Display running | Confirm `SDL_VIDEODRIVER=dummy` |
