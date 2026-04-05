# Training the Curriculum Car Racer

> The agent starts on a wide oval, barely able to keep from flying off the road.
> By the end it threads two-car-width choke points at speed, brakes for hairpins,
> and finds the shortest racing line through complex polygon circuits — all from a
> single 64×64 image and seven sensor readings, learned entirely from reward.

There are three files that matter for training:

```
train.py       ← the only file you need to run and tune
doc/train.md   ← this file (human-maintained notes and guidance)
checkpoints/   ← saved model snapshots (auto-created)
```

`train.py` is a single self-contained script. It owns the PPO loop, the
curriculum progression, validation gating, W&B logging, and checkpointing.
You should not need to touch any other file to train.

---

## Quick start

```bash
# install deps (first time only)
pip install wandb torch

# default run — 5 M steps, 4 envs, logs to W&B
python train.py

# recommended CPU run — use all cores
python train.py --num-envs 8 --compile

# target run — 3 M steps, named project
python train.py --total-steps 3_000_000 --wandb-project my-racer --num-envs 8

# resume a previous run
python train.py --resume checkpoints/ppo_step00500000_lvl02.pt

# no internet / offline machine
python train.py --wandb-offline
```

After training starts you will see a W&B URL in the terminal. Open it in a
browser to watch every metric in real time. If you ran with `--wandb-offline`,
sync later with `wandb sync checkpoints/wandb/`.

---

## What the agent is optimising

The reward function has four components that directly map to the four objectives:

| Objective | Reward signal | What to watch in W&B |
|-----------|--------------|----------------------|
| **Fewer attempts** (fewer crashes) | `−5 × complexity` per on→off track transition | `episode/crashes` ↓ |
| **Faster lap time** | `+50 × (par_time / actual_time)` clamped [0.5, 2.0] | `episode/length` ↓ |
| **Higher speed** | `+speed/max_speed × 0.01` every step | `episode/on_track_pct` ↑ |
| **Shorter distance** | `+50 × (optimal_dist / actual_dist)` clamped [0.5, 1.0] | `episode/laps` ↑ |

All four are combined into a single lap-completion bonus:

```
lap_bonus = 50 × time_ratio × dist_ratio × complexity
```

`complexity` scales with track difficulty so the threshold stays meaningful
on both easy ovals and tight choke circuits. You do **not** need to change
the reward function — the objectives emerge from this single formula.

---

## Curriculum

Training progresses through 20 tracks in 5 difficulty tiers:

```
Tier A — Easy ovals           tracks  1– 4    Wide Oval → Superspeedway
Tier B — Rectangular shapes   tracks  5– 8    Rounded Rect → Small Oval
Tier C — Hairpins & chicanes  tracks  9–12    Hairpin → Asymmetric
Tier D — Complex polygons     tracks 13–16    L-Shape → Master Challenge
Tier E — Choke+catch-up       tracks 17–20    Two-Strait → Canyon Pass
```

**TRAIN** (10 tracks): 1,2, 5,6, 9,10, 13,14, 17,18 — the agent trains here.  
**VAL** (5 tracks): 3, 7, 11, 15, 19 — used only for gating, never trained on.  
**TEST** (5 tracks): 4, 8, 12, 16, 20 — held out entirely, run once at the end.

### How advancement works

The agent spends 70% of episodes on the current *frontier* track and 30% on
randomly sampled *mastered* tracks (anti-forgetting replay). It advances when
its rolling mean reward over the last `--window` episodes exceeds the
effective threshold:

```
effective_threshold = --threshold × track.complexity
```

Each advancement triggers a validation run on all 5 VAL tracks. The result
appears in W&B as `val/mean_reward` and a per-track table. If val performance
is weak, the agent has not generalised — consider increasing `--window` or
`--replay-frac` before continuing.

```
curriculum/level        current frontier position (0 = Track 1, 9 = Track 18)
curriculum/rolling_mean rolling mean over last window episodes
curriculum/threshold    effective threshold at this point (scales with difficulty)
curriculum/is_replay    1 = anti-forgetting episode, 0 = frontier episode
```

### Watching progression in W&B

Plot `curriculum/track_level` on the x-axis against `episode/reward` to see
the reward profile per track. Each step-change in `curriculum/track_level`
is a curriculum advancement event. If the line stalls for many thousands of
steps without advancing, the agent is stuck — see the failure modes section.

---

## Multi-environment training

`--num-envs N` runs N environments in the same process, all sharing one model.
Each rollout step does a single batched forward pass across all N envs, then
steps each env sequentially. The PPO update trains on `rollout-steps × N`
samples per update.

```
total samples per update = --rollout-steps × --num-envs
```

All N environments share the same curriculum — episode rewards from any env
count toward curriculum advancement.

**Choosing N for CPU:**
- Start with `--num-envs 4` (the default).
- Increase toward `os.cpu_count() // 2` for diminishing returns at the env-stepping
  bottleneck (pure Python game physics, no GIL release).
- `--num-envs 8` is a reliable all-core setting on a typical 8-core laptop.
- Use `--compile` alongside multi-env for an additional ~20–40% speedup from
  `torch.compile` (requires PyTorch ≥ 2.0).

**CPU thread allocation** is set automatically:
- `torch.set_num_threads(os.cpu_count())` — all cores for tensor operations
- `torch.set_num_interop_threads(cpu_count // 2)` — inter-op parallelism

---

## Checkpoints

Checkpoints are saved automatically every `--checkpoint-interval` steps
(default: every 500 k steps) and as a final `ppo_final.pt` at the end.

```
checkpoints/
  ppo_step00500000_lvl02.pt
  ppo_step01000000_lvl04.pt
  ...
  ppo_final.pt
```

Each `.pt` file stores the complete training state needed for exact resumption:

```python
{
  "step":             int,        # global step at save time
  "curriculum_level": int,        # 0-based frontier index
  "model":            state_dict,
  "optimizer":        state_dict,
  "args":             dict,       # all CLI args used for this run
  # resume state
  "reward_window":    list,       # rolling reward deque (for curriculum)
  "episode_num":      int,        # total episodes completed
  "sampler_idx":      int,        # curriculum sampler frontier index
  "sampler_rewards":  list,       # sampler's internal reward window
  "wandb_run_id":     str,        # W&B run ID for chart continuity
}
```

---

## Resuming a run

Pass `--resume` with the path to any checkpoint file:

```bash
python train.py --resume checkpoints/ppo_step01000000_lvl04.pt
```

Everything is restored automatically:
- Model weights and optimizer state
- `global_step` counter (training continues from that step)
- Curriculum frontier position and reward window
- Episode counter
- W&B run (charts continue on the same run, not a new one)

The `--total-steps` budget is respected relative to the original count, so
a run targeting 5 M steps resumed at step 1 M will train for another 4 M steps.

You can change `--num-envs` or `--compile` on resume without affecting the model.
Changing PPO hyperparameters (lr, clip-eps, etc.) takes effect immediately.

---

## Loading a checkpoint for inference

```python
import torch
from train import PPOActorCritic

ckpt  = torch.load("checkpoints/ppo_final.pt", map_location="cpu")
model = PPOActorCritic()
model.load_state_dict(ckpt["model"])
model.eval()
```

---

## All training arguments

```
python train.py --help
```

| Argument | Default | What it controls |
|----------|---------|-----------------|
| `--total-steps` | `5_000_000` | Total environment steps |
| `--rollout-steps` | `2048` | Steps collected per env per PPO update |
| `--num-envs` | `4` | Number of parallel environments |
| `--ppo-epochs` | `4` | Update passes per rollout |
| `--minibatch-size` | `256` | Minibatch size per gradient step |
| `--lr` | `3e-4` | Initial learning rate (Adam) |
| `--lr-min` | `1e-5` | Final LR after linear decay |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE smoothing parameter |
| `--clip-eps` | `0.2` | PPO clipping range |
| `--vf-coef` | `0.5` | Value loss weight |
| `--ent-coef-start` | `0.01` | Entropy coefficient at step 0 |
| `--ent-coef-end` | `0.001` | Entropy coefficient at final step |
| `--max-grad-norm` | `0.5` | Gradient clipping norm |
| `--target-kl` | `0.02` | KL early-stop threshold per update |
| `--threshold` | `30.0` | Base curriculum advancement threshold |
| `--window` | `50` | Rolling reward window for advancement |
| `--replay-frac` | `0.3` | Fraction of episodes from mastered tracks |
| `--val-episodes` | `10` | Episodes per VAL track after each advance |
| `--checkpoint-interval` | `500_000` | Steps between checkpoint saves |
| `--checkpoint-dir` | `checkpoints` | Directory for saved models |
| `--resume` | `None` | Path to a `.pt` checkpoint to resume from |
| `--compile` | `False` | Enable `torch.compile` (PyTorch ≥ 2.0) |
| `--seed` | `42` | RNG seed |
| `--device` | `cuda/cpu` | Training device (auto-detected) |
| `--wandb-project` | `curriculum-car-racer` | W&B project name |
| `--wandb-run-name` | `None` | W&B run name (auto-generated if None) |
| `--wandb-offline` | `False` | Disable W&B network calls |

---

## 50k-step monitoring checklist — MANDATORY

**This check is not optional.** At every 50k step boundary a health report
must be produced. The automated monitor (`monitor.py`) does this continuously,
but you should also read and act on its output at every 50k mark.

### Run the monitor alongside every training run

Always start `monitor.py` together with `train.py`:

```bash
# Terminal 1 — training
python train.py --num-envs 8 --compile --lr 3e-4 --max-grad-norm 1.0 \
                --target-kl 0.05 --checkpoint-interval 250000

# Terminal 2 — monitor (auto-detects latest wandb run)
python monitor.py --interval 60
```

The monitor polls every 60 seconds, prints a live metrics row, and fires a
PASS/FAIL report at every 50k boundary automatically. Check `monitor.log`
(or the terminal) at each 50k mark.

### What a healthy 50k report looks like

```
======================================================================
  PASS  250k check at step 252,416
======================================================================

  step= 252,416  reward=  -125.4  on_track= 92.0%  ev= 0.690  kl= 0.0032
                 grad=   1.5  stopped=0  lvl=0  sps=72
```

The full health table at each check should look like this (200k example):

```
Metric                   Value        Target at 200k   Status
---------------------------------------------------------------------------
episode/reward           -125         > -200           PASS - ahead of target
episode/on_track_pct     93%          > 85%            PASS
ppo/approx_kl            0.0032       < 0.02           PASS
ppo/early_stopped        0            not always 1     PASS
ppo/grad_norm            1.46         < 12             PASS - excellent, dropped from 19
ppo/explained_variance   0.44         > 0.85           WARN - critic catching up (not fail)
curriculum/rolling_mean  -246 vs 30   closing          OK - still progressing
```

All hard metrics within range, EV warn on first occurrence → no action, continue training.

### What a failing 50k report looks like

```
======================================================================
  FAIL  200k check at step 204,312
  FAIL  episode/reward -975.4 < -200
  WARN  grad_norm 91.7 > 12

  Fix — kill train.py then run:
    python train.py --num-envs 8 --compile --checkpoint-interval 250000 \
                    --ent-coef-start 0.02 --max-grad-norm 2.0 --lr 1e-4
======================================================================
```

Kill train.py, run the fix command exactly as printed, restart monitor.

### EV (explained variance) — warn vs fail rule

A single EV dip below threshold is a **WARN**, not a **FAIL**. During rapid
improvement phases the policy improves faster than the critic can track,
causing a temporary EV dip. This is normal.

- **WARN** (first low check) → watch next check, no action
- **FAIL** (second consecutive low check) → apply `--vf-coef 1.0`

### Pass/fail thresholds

| Step | `episode/reward` | `on_track_pct` | `explained_var` | `approx_kl` | `grad_norm` | `early_stopped` |
|------|-----------------|----------------|----------------|-------------|-------------|-----------------|
| 50k  | > −800          | > 60%          | > 0.5          | 0.001–0.02  | < 30        | not always 1    |
| 100k | > −500          | > 70%          | > 0.7          | 0.001–0.02  | < 20        | not always 1    |
| 150k | > −300          | > 80%          | > 0.8          | 0.001–0.02  | < 15        | not always 1    |
| 200k | > −200          | > 85%          | > 0.85         | 0.001–0.02  | < 12        | not always 1    |
| 250k | > −150          | > 88%          | > 0.85         | 0.001–0.02  | < 10        | not always 1    |
| 300k | > −100          | > 90%          | > 0.88         | 0.001–0.02  | < 10        | not always 1    |
| 400k | > 0             | > 92%          | > 0.90         | 0.001–0.02  | < 8         | not always 1    |
| 500k | > 20            | > 93%          | > 0.90         | 0.001–0.015 | < 8         | not always 1    |
| 750k | > 50            | > 94%          | > 0.92         | 0.001–0.015 | < 6         | not always 1    |
| 1M   | > 80            | > 95%          | > 0.93         | 0.001–0.015 | < 6         | not always 1    |
| 1.5M | > 120           | > 95%          | > 0.94         | 0.001–0.015 | < 5         | not always 1    |
| 2M   | > 150           | > 96%          | > 0.95         | 0.001–0.012 | < 5         | not always 1    |
| 3M   | > 180           | > 96%          | > 0.95         | 0.001–0.012 | < 4         | not always 1    |
| 4M   | > 200           | > 97%          | > 0.96         | 0.001–0.01  | < 4         | not always 1    |
| 5M   | > 200           | > 97%          | > 0.96         | 0.001–0.01  | < 4         | Final — run TEST eval |

### Decision tree — what to do when a check fails

**Any metric is NaN → kill immediately, fix the bug, restart from scratch.**
NaN metrics mean no learning is happening. Do not continue.

**`early_stopped` = 1 on every update AND `approx_kl` = 0 or NaN:**
Early stop fires before any gradient step. Policy is frozen.
→ Kill, resume last checkpoint with `--lr 1e-4 --max-grad-norm 2.0`
→ If no checkpoint yet, restart from scratch with those flags.

**`grad_norm` above threshold for 2 consecutive checks:**
Gradients are too large; clipping is truncating every update.
→ At next checkpoint, resume with `--max-grad-norm 2.0 --lr 1e-4`

**`episode/reward` flat or regressing for 2 consecutive checks:**
Agent is stuck. Try in order:
1. `--ent-coef-start 0.02` (more exploration)
2. `--threshold` reduced by 20% (easier curriculum gate)
3. `--replay-frac 0.4` (more anti-forgetting)
→ Apply via `--resume` from the latest checkpoint. No need to restart scratch.

**`on_track_pct` < threshold:**
Agent is crashing too often. Reward signal may be too sparse.
→ Check `episode/crashes`; if > 5, raise `--ent-coef-start 0.03`
→ If `on_track_pct` < 50% past 200k, consider reward shaping (see below)

**`explained_variance` below threshold past 200k steps:**
Value function not learning. Advantage estimates are noise.
→ Resume with `--vf-coef 1.0`

**`curriculum/level` not advancing past expected:**
Agent stuck on a track tier.
→ See failure modes section. Primary fix: `--threshold` × 0.8

### Also check at every milestone
- `system/steps_per_sec` not dropping — sustained drop means memory pressure
- `val/completion_rate` increases after every curriculum advance
- No `ppo/policy_loss` NaN — if NaN appears, kill immediately

---

## Sanity checks — run these in the first 5 minutes

**Do this before walking away from a new run.** Open W&B and verify these
three signals within the first 1–2 PPO updates (~2–3 min with `--num-envs 8`):

| Metric | Expected at step ~16 k | Red flag |
|--------|----------------------|----------|
| `ppo/policy_loss` | finite number (positive or negative) | **NaN** |
| `ppo/approx_kl` | 0.001 – 0.02 | NaN or always exactly 0.02 |
| `ppo/entropy` | ~2.5 – 2.8 (Gaussian, 2-D action) | NaN or 0 |
| `ppo/early_stopped` | 0 most of the time | **always 1** |
| `ppo/explained_variance` | near 0 is fine early | below −1 after 50 k steps |
| `system/steps_per_sec` | > 50 | < 10 (something is blocking) |

**If `ppo/policy_loss` is NaN and `ppo/early_stopped` is always 1:**  
The KL check fires before any gradient step is taken — no learning is
happening at all. Most common cause: a log-probability mismatch between the
rollout collection and the PPO update (e.g. actions stored in the buffer do
not match the actions log-probs were computed on). Kill the run, fix the bug,
restart from scratch.

**If `ppo/entropy` is NaN:**  
Same root cause as above — the distribution computation has a numerical
problem. Check for action clamping that creates a mismatch between stored
and recomputed log-probs.

---

## What healthy training looks like over time

These four metrics are the primary indicators. Check them at 0, 100k, 500k,
and 1M steps to confirm the run is on track.

### Episode reward (`episode/reward`) — the most important signal

| Steps | Expected range | Action if outside range |
|-------|---------------|------------------------|
| 0–50k | −2000 to −200 | Too early to act |
| 50k–200k | trending upward | If flat, raise `--ent-coef-start` to 0.02 |
| 200k–600k | approaching 0 | If still −1000, check crashes and on_track_pct |
| 600k+ | positive, climbing | If stuck below 0, see failure modes |

`episode/reward` does **not** decrease monotonically — expect noisy upward trend
over thousands of episodes, not each episode.

### PPO explained variance (`ppo/explained_variance`) — value function health

- **Target:** ≥ 0.7 after the first few thousand steps, ≥ 0.9 by 500k steps
- **Healthy pattern:** jumps quickly to 0.7+ in the first 1–2 updates, then
  slowly climbs toward 0.95
- **Red flag:** stays below 0.1 after 500k steps → value function not learning,
  raise `--vf-coef` to 1.0
- **Red flag:** drops from 0.9 back to 0.1 → catastrophic forgetting after a
  curriculum advance, raise `--replay-frac` to 0.4

### PPO approx KL (`ppo/approx_kl`) — update step size

- **Target:** stays in the range 0.001–0.02 throughout training
- **Healthy pattern:** small oscillations, never sustained above 0.02
- `ppo/early_stopped` flipping to 1 occasionally is normal (KL clipped one
  minibatch); **always 1** is a bug
- **Red flag:** regularly > 0.05 → reduce `--lr` to 1e-4

### PPO policy loss (`ppo/policy_loss`) — note: not a simple ↓ metric

Policy loss in PPO is **not** like supervised loss. It does not decrease
monotonically and should not be used alone to judge training health.
Use it only to detect catastrophic events:

- Sudden spike to large positive value → policy collapsed, reduce `--lr`
- Stuck at exactly 0.0 → early stopping every update, policy not updating

### Expected trajectory summary

```
Metric                 Early (0–50k)   Mid (200k–1M)   Late (2M–5M)
─────────────────────────────────────────────────────────────────────
episode/reward         −2000 to −200   −500 to +50     +50 to +300
ppo/explained_var      0.5 to 0.8      0.8 to 0.95     0.9 to 0.98
ppo/approx_kl          0.001–0.02      0.001–0.02      0.001–0.015
ppo/entropy            ~2.8            2.5 to 2.8      2.0 to 2.5
curriculum/level       0               2 to 5          6 to 9
```

---

## W&B dashboard setup

The script configures a custom x-axis (`global_step`) for every metric group.
Recommended panels to create:

**Training health**
- `ppo/approx_kl` — should stay below 0.02; if it spikes, reduce `--lr`
- `ppo/explained_variance` — should trend toward 0.9+; below 0 means the
  value function is useless
- `ppo/entropy` — should decrease slowly from ~2.8 toward ~2.0; a sudden
  collapse means the policy has converged prematurely

**Objectives (the four metrics you care about)**
- `episode/reward` ↑ — primary signal; noisy upward trend over thousands of episodes
- `episode/crashes` ↓ — fewer boundary crossings per episode
- `episode/length` ↓ — shorter episodes = faster laps
- `episode/on_track_pct` ↑ — proxy for speed (only gets reward while on track)
- `episode/laps` ↑ — direct lap completion rate

**Curriculum**
- `curriculum/track_level` — staircase shape as the agent advances
- `curriculum/rolling_mean` vs `curriculum/threshold` on the same chart —
  you can visually see the agent approaching the threshold before each advance

**Throughput**
- `system/steps_per_sec` — watch this after changing `--num-envs` to confirm
  the speedup; if it stops scaling, you have hit the env-stepping bottleneck

**Validation (logged at every curriculum advance)**
- `val/mean_reward` — should increase with each advance
- `val/completion_rate` — target > 0.8 before considering the run successful

---

## Reward shaping (worst-case tuning)

The reward function is the first lever to pull when the agent completely fails
to make progress — before touching PPO hyperparameters.

The four components in `env/environment.py` and their worst-case adjustments:

| Component | Default weight | If agent ignores it |
|-----------|---------------|---------------------|
| Crash penalty | `−5 × complexity` | Raise to `−10 × complexity` |
| Lap time bonus | `+50 × time_ratio` (clamped 0.5–2.0) | Lower clamp floor to 0.25 so partial-lap progress still pays |
| Speed reward | `+speed/max_speed × 0.01` per step | Raise multiplier to `0.05` for early-tier tracks |
| Distance bonus | `+50 × dist_ratio` (clamped 0.5–1.0) | Raise weight to `+75` if agent takes wildly suboptimal paths |

**Rule of thumb:** reward shaping is a last resort. Try curriculum threshold
reduction (`--threshold`) and exploration increase (`--ent-coef-start`) first,
since those require no code changes. Only edit the reward function when the
agent's behaviour is qualitatively wrong (e.g., deliberately crashing to end
episodes early, or spinning in place indefinitely).

---

## Hyperparameter tuning

Only change one thing at a time. The default values are well-calibrated for
this environment — most runs do not need tuning.

**If `ppo/approx_kl` regularly exceeds 0.05**
→ Reduce `--lr` to `1e-4`. If it still spikes, reduce `--ppo-epochs` to 2.

**If `curriculum/rolling_mean` plateaus well below threshold for > 200 k steps**
→ The agent is stuck. Options in order of preference:
  1. Increase `--replay-frac` to `0.4` (more anti-forgetting)
  2. Reduce `--threshold` by 20% for the current tier
  3. Increase `--ent-coef-start` to `0.02` (more exploration)

**If `ppo/explained_variance` stays below 0.1 after 500 k steps**
→ Value function is not learning. Increase `--vf-coef` to `1.0`.

**If `episode/crashes` is not decreasing on Tier E (choke tracks)**
→ The choke is too hard to learn by reward alone. Increase `--rollout-steps`
  to `4096` so the agent sees more complete choke-passage sequences per update.

**For faster iteration (debugging, not full training)**

```bash
python train.py --total-steps 200_000 --rollout-steps 512 --window 20 \
                --val-episodes 3 --checkpoint-interval 0 --wandb-offline \
                --num-envs 4
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Agent drives in circles, never completes a lap | Reward too sparse early on | Lower `--threshold` for Track 1 to 10 |
| `ppo/approx_kl` explodes after curriculum advance | New track is much harder, policy overcorrects | Reduce `--lr` to `1e-4` |
| `episode/crashes` stuck at > 5 | `--ent-coef` too low, agent not exploring | Raise `--ent-coef-start` to `0.03` |
| Performance on early tracks collapses after Track 10 | Catastrophic forgetting | Raise `--replay-frac` to `0.4` |
| `val/completion_rate` never rises above 0.3 | Agent memorising, not generalising | Training is working as expected — wait for more steps on Tier B/C |
| `ppo/explained_variance` negative | Value function diverged | Reduce `--lr` by 10× and restart from last checkpoint with `--resume` |
| `system/steps_per_sec` does not increase with more envs | Env-stepping bottleneck (pure Python) | This is expected; `--num-envs 8` is the practical ceiling |
| Training very slow (< 30 steps/sec on CPU) | Display driver active | Confirm `SDL_VIDEODRIVER=dummy` is set (train.py sets it automatically) |
| Resume fails with pickle error | Checkpoint saved with old train.py | Re-run from scratch; old checkpoints lack `reward_window` and `sampler_idx` |

---

## Expected training timeline

At 100 steps/sec (single env, CPU):

| Milestone | ~Steps | ~Wall time |
|-----------|--------|-----------|
| Stays on Track 1 consistently | 100 k | 17 min |
| Completes Track 1 lap | 200 k | 33 min |
| Advances through Tier A (tracks 1–2) | 600 k | 1.7 h |
| Advances through Tier B (tracks 5–6) | 1.2 M | 3.3 h |
| Advances through Tier C (tracks 9–10) | 2.0 M | 5.6 h |
| Advances through Tier D (tracks 13–14) | 3.0 M | 8.3 h |
| Advances through Tier E (tracks 17–18) | 4.5 M | 12.5 h |
| Full 5 M step run complete | 5.0 M | 13.9 h |

With `--num-envs 8 --compile` on an 8-core CPU expect roughly 3–4× throughput
(~300–400 steps/sec), cutting the timeline to 3–5 hours for a full run.

---

## What `train.py` does not do

These are intentional omissions — add them when needed, not before:

- **No subprocess-parallel envs** — envs are stepped sequentially in one process;
  true subprocess workers (à la `SubprocVecEnv`) would require the game objects
  to be picklable and add significant complexity for marginal CPU gain
- **No expert data recording** — run `train.py` until curriculum level ≥ 6,
  then add `demos.append((obs.image, action))` inside the rollout loop to
  collect demonstrations for the BC / IQL / Diffusion Policy pipeline
- **No TEST evaluation** — run separately with `builder.test_envs()` after
  training is complete; the test split is held out and should not be touched
  until final evaluation
