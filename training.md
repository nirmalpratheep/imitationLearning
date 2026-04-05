# Training Reference — PPO Car Racer

Source: `train.py`  
Algorithm: Proximal Policy Optimisation (PPO) with GAE, linear annealing, curriculum, and W&B logging.

---

## 1. Model Architecture

```
ImpalaCNN(image 3×64×64)
    └─► feature map
MLP(7 scalars)
    └─► scalar embedding
Concat → 288-d feature vector
    ├─► actor_mean   Linear(288, 2)   → [accel, steer] mean
    │   actor_log_std  Parameter(2)   → shared log std
    │   → Normal distribution → sample action
    └─► critic        Linear(288, 1)  → scalar value estimate
```

- **Actions**: `[accel, steer]` continuous, clamped to `[-1, 1]`
- **Distribution**: Gaussian with learned mean, shared log-std parameter
- **Init**: orthogonal weights — `gain=0.01` for actor (small initial actions), `gain=1.0` for critic

---

## 2. Observation

| Component | Shape | Description |
|-----------|-------|-------------|
| `image` | `(3, 64, 64)` | Headlight cone crop, RGB, normalised to `[0, 1]` |
| `scalars` | `(7,)` | Speed, heading, wall distances, lap progress, etc. |

Images are converted `HWC → CHW` and divided by `255.0` before feeding to the network.

---

## 3. PPO Parameters

### 3.1 Data Collection

#### `--total-steps` (default: `5_000_000`)
Total environment steps across all parallel envs. Main loop runs `while global_step < total_steps`. With `N=4` envs, `global_step += N` each inner step, so `5M / 4 = 1.25M` actual rollout iterations.

#### `--rollout-steps` (default: `2048`)
Steps collected **per env** before each gradient update. With `N=4`, each update consumes `2048 × 4 = 8192` transitions. Larger = more stable gradient estimates but slower updates. Smaller = more frequent but noisier updates.

#### `--num-envs` (default: `4`)
Parallel environments stepped sequentially in the collection loop. Increases effective batch size (`T × N`) and throughput. Recommended: `os.cpu_count() // 2` on CPU-only machines.

---

### 3.2 Optimisation

#### `--ppo-epochs` (default: `4`)
Full passes over the rollout buffer per update. More epochs = more gradient steps from the same data = better sample efficiency. Too many = policy drifts from the data it was collected under. `target-kl` early-stop guards against this.

#### `--minibatch-size` (default: `256`)
Rollout buffer (`T×N = 8192`) is shuffled and sliced into minibatches of this size per epoch. `8192 / 256 = 32` minibatch updates per epoch. Smaller = noisier gradients, more updates; larger = more stable, fewer updates.

#### `--lr` (default: `3e-4`)
Initial Adam learning rate. Linearly annealed during training:
```python
lr_now = lr + (lr_min - lr) * (global_step / total_steps)
```
Starts at `3e-4`, decays to `lr_min` by the end of training.

#### `--lr-min` (default: `1e-5`)
Final learning rate at `progress = 1.0`. At 50% of training, LR ≈ `1.55e-4`. The decay is purely linear — no warmup, no cosine schedule.

---

### 3.3 Advantage Estimation

#### `--gamma` (default: `0.99`)
Discount factor. Controls how much future rewards are worth relative to immediate ones. `γ=0.99` means a reward 100 steps away is worth `0.99^100 ≈ 0.37` of its face value. Used in the TD delta:
```python
delta = reward + gamma * next_value * (1 - done) - current_value
```

#### `--gae-lambda` (default: `0.95`)
GAE smoothing factor. Trades off bias vs. variance in the advantage estimate:
- `λ = 0` → pure TD(0): low variance, high bias
- `λ = 1` → Monte Carlo returns: high variance, low bias
- `λ = 0.95` → standard sweet spot

Used in the backwards advantage accumulation:
```python
advantages[t] = delta + gamma * gae_lambda * (1 - done) * lastgae
returns = advantages + values
```

---

### 3.4 Policy Update

#### `--clip-eps` (default: `0.2`)
PPO clipping range. The probability ratio `r = new_prob / old_prob` is clipped to `[0.8, 1.2]`. Prevents the policy from updating too aggressively in a single step:
```python
pg_loss = max(-advantage * ratio,
              -advantage * clamp(ratio, 1-clip_eps, 1+clip_eps)).mean()
```

#### `--vf-coef` (default: `0.5`)
Weight of value function loss in the combined loss:
```python
loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
```
Value loss (MSE) naturally has larger magnitude than policy loss, so `0.5` balances their gradient contributions.

#### `--ent-coef-start` (default: `0.01`)
Initial entropy bonus coefficient. Entropy is subtracted from the loss (negative sign), rewarding the policy for staying exploratory early in training.

#### `--ent-coef-end` (default: `0.001`)
Final entropy coefficient at end of training. Annealed linearly alongside LR:
```python
ent_now = ent_coef_start + (ent_coef_end - ent_coef_start) * progress
```
Reduces from `0.01 → 0.001` so the policy commits to good actions rather than continuing to explore.

#### `--max-grad-norm` (default: `0.5`)
Gradient clipping threshold. After `loss.backward()`, gradients are globally rescaled if their norm exceeds `0.5`:
```python
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```
Prevents exploding gradients. Logged as `ppo/grad_norm` — if consistently near `0.5`, the model is being clipped frequently and LR may be too high.

#### `--target-kl` (default: `0.02`)
Early-stop threshold per epoch. If the Schulman KL approximation exceeds this value, the epoch loop breaks immediately:
```python
approx_kl = ((ratio - 1) - logratio).mean()
if approx_kl > target_kl:
    early_stopped = True; break
```
Guards against catastrophic policy updates mid-epoch. If `ppo/early_stopped = 1` appears frequently in W&B, reduce `clip-eps` or `lr`.

---

### 3.5 Parameter Summary

| Parameter | Default | Guards against |
|-----------|---------|---------------|
| `total-steps` | 5,000,000 | — |
| `rollout-steps` | 2,048 | Noisy gradient estimates |
| `num-envs` | 4 | Low throughput |
| `ppo-epochs` | 4 | Stale data reuse |
| `minibatch-size` | 256 | Memory / gradient noise |
| `lr` | 3e-4 | — |
| `lr-min` | 1e-5 | Overshooting late training |
| `gamma` | 0.99 | Short-sighted policy |
| `gae-lambda` | 0.95 | Bias/variance tradeoff |
| `clip-eps` | 0.2 | Policy update too large |
| `vf-coef` | 0.5 | Value loss dominating |
| `ent-coef-start` | 0.01 | Premature convergence |
| `ent-coef-end` | 0.001 | Excess exploration late |
| `max-grad-norm` | 0.5 | Exploding gradients |
| `target-kl` | 0.02 | Catastrophic mid-epoch update |

---

## 4. Curriculum Parameters

#### `--threshold` (default: `30.0`)
Rolling mean reward required to advance to the next track. Scaled by each track's `complexity` factor:
```python
effective_threshold = threshold * frontier.complexity
```

#### `--window` (default: `50`)
Number of recent episodes used to compute the rolling mean for curriculum advancement. `deque(maxlen=50)` — older episodes automatically drop off.

#### `--replay-frac` (default: `0.3`)
Fraction of episodes replayed from already-mastered tracks (anti-forgetting). `0.3` means 30% of env resets draw from previous tracks, 70% from the current frontier.

#### `--val-episodes` (default: `10`)
Episodes run per validation track after each curriculum advance. Validation is greedy (no exploration noise). Results logged as `val/track_N_reward`, `val/track_N_completion`.

---

## 5. W&B Metrics Explained

| Metric | Healthy range | What it tells you |
|--------|--------------|-------------------|
| `ppo/policy_loss` | Small, near zero | Clipped surrogate loss — large negative = big policy shift |
| `ppo/value_loss` | Decreasing over time | MSE between predicted and actual returns |
| `ppo/entropy` | Slowly decreasing | Higher = more exploration; too low = policy collapsed |
| `ppo/approx_kl` | < 0.02 | Policy divergence from collected data; spikes = unstable update |
| `ppo/clip_fraction` | 0.05 – 0.2 | Fraction of ratios that hit the clip bound; too high = LR too big |
| `ppo/explained_variance` | Rising toward 1.0 | How well value function predicts returns; < 0 = useless critic |
| `ppo/grad_norm` | < 0.5 | Gradient magnitude before clipping; consistently at 0.5 = clipping |
| `ppo/early_stopped` | Rare (0) | 1 = KL exceeded target mid-epoch; frequent = reduce LR or clip-eps |

---

## 6. Checkpointing

Saves every `--checkpoint-interval` steps (default `500,000`) to:
```
checkpoints/ppo_step<NNNNNNNN>_lvl<LL>.pt
```

Checkpoint contains: model weights, optimizer state, global step, curriculum level, reward window, episode count, W&B run ID.

Resume with:
```bash
python train.py --resume checkpoints/ppo_step00500000_lvl02.pt
```
W&B run is automatically continued using the stored `wandb_run_id`.

---

## 7. Training Phases (from spec)

| Phase | Steps | Goal | Reward active |
|-------|-------|------|--------------|
| Survival | 0 – 500k | Stay on track | `on_track`, `off_track` only |
| Speed | 500k – 1.5M | Complete laps faster | + speed bonus + lap bonus |
| Fine-tune | 1.5M – 3M | Consistent lap times | All, entropy coef → 0.001 |

---

## 8. Usage

```bash
# Basic
python train.py

# More envs for faster data collection
python train.py --num-envs 8

# Longer run with custom W&B project
python train.py --total-steps 5_000_000 --wandb-project my-racer

# Resume from checkpoint
python train.py --resume checkpoints/ppo_step00500000_lvl02.pt

# No internet (offline W&B)
python train.py --wandb-offline

# Faster CPU inference (PyTorch >= 2.0)
python train.py --compile
```
