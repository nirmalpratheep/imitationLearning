---
title: "Teaching an RL Agent to Race from Scratch: Curriculum Learning with OpenEnv"
thumbnail: /blog/assets/curriculum-car-racer/collage.gif
# To publish on HuggingFace: upload inference_videos/collage.gif to
# https://huggingface.co/blog/assets/curriculum-car-racer/collage.gif
authors:
  - user: NirmalPratheep
---

# Teaching an RL Agent to Race from Scratch: Curriculum Learning with OpenEnv

*An OpenEnv Student Challenge submission — building a production-ready RL environment from the ground up, then solving it with TorchRL PPO.*

---

What does it take to teach an AI agent to drive? Not with a map, not with pre-recorded human demonstrations, and not with privileged information about the track — just a small image, nine sensor readings, and a reward signal that says *"you crashed, try again."*

That was the challenge I set for myself: build a complete RL environment using OpenEnv, design a curriculum that lets an agent grow from helpless wandering to confident racing, and train a PPO policy that masters all 10 tracks without forgetting the earlier ones. This post walks through every decision — the environment design, the curriculum mechanics, the architecture, and the training innovations that made it work.

![Agent driving all 10 curriculum tracks simultaneously](../inference_videos/collage.gif)

*The final trained policy — zero crashes, at least one completed lap on every track, evaluated deterministically.*

---

## The Problem: Why Curriculum Learning?

Reinforcement learning from scratch is hard. A car racing agent faces a compounding challenge: it must simultaneously learn steering, speed control, crash avoidance, *and* track navigation — all from sparse reward. If you drop an untrained agent onto a tight hairpin track, it crashes immediately and learns almost nothing. The reward signal is too sparse for meaningful gradient flow.

The solution is **curriculum learning**: start with the easiest possible version of the task and only advance when the agent has genuinely mastered it. The agent builds up generalizable driving skills on wide, forgiving ovals — then transfers those skills to progressively tighter corners, higher speeds, and more complex track geometries.

This isn't a new idea in RL, but implementing it *robustly* — with automatic advancement, anti-forgetting replay, and regression recovery — is where the interesting engineering lives.

---

## The Environment: 10 Tracks, 3 Difficulty Groups

The simulation is built in **Pygame** with no ML dependencies. Each of the 10 tracks is a `TrackDef` — a procedurally generated centreline of waypoints that gets rendered into a dark-grey tarmac surface with white boundary lines and a checkered start/finish.

```
Group A — Easy ovals         (Tracks 1–4)   Width 115 → 58 px, speeds 3.0–4.5
Group B — Rectangular shapes (Tracks 5–8)   First 90° corners, stadium curves
Group C — Hairpins & chicanes(Tracks 9–10)  Hairpin reversals, S-bend chicane
```

| Track | Name | Width | Max Speed | Key Challenge |
|-------|------|-------|-----------|---------------|
| 1 | Wide Oval | 115 px | 3.0 | Nothing — just drive |
| 2 | Standard Oval | 85 px | 3.5 | Slightly narrower |
| 3 | Narrow Oval | 58 px | 3.5 | Precision required |
| 4 | Superspeedway | 85 px | 4.5 | High speed, elongated |
| 5 | Rounded Rectangle | 90 px | 3.5 | First real corners |
| 6 | Stadium Oval | 80 px | 4.0 | Tight end-caps |
| 7 | Tight Rectangle | 65 px | 3.5 | Sharp 90° corners |
| 8 | Small Oval | 60 px | 3.2 | Small radius |
| 9 | Hairpin Track | 75 px | 3.5 | Hairpin reversal |
| 10 | Chicane Track | 70 px | 3.5 | S-bend chicane |

Track complexity scales with road width and speed: `complexity = (base_width / road_width) × (speed / base_speed)`. This multiplier gates curriculum advancement — the agent needs a higher greedy-eval score on harder tracks before it's allowed to move on.

### Car Physics

The physics are intentionally simple and fast to simulate at scale:

```
ACCEL       = 0.13   px/step² forward acceleration
BRAKE_DECEL = 0.22   px/step² braking
FRICTION    = 0.038  px/step  passive drag
STEER_DEG   = 2.7°   per unit steer input
```

Steering rate scales with speed (reduced to 30% at very low speed) — so the agent can't spin in place at full steer rate, which helps prevent a degenerate spinning-in-circles local optimum.

### OpenEnv Compliance

The environment is fully compliant with the **OpenEnv** standard. Serving it as an HTTP API takes one command:

```bash
uvicorn env.server.app:app --host 0.0.0.0 --port 8000
```

Any OpenEnv client can then interact with it:

```python
from openenv import connect

env = connect("http://localhost:8000")
obs = env.reset()
while not obs.done:
    action = my_agent(obs)    # DriveAction(accel=..., steer=...)
    obs = env.step(action)
```

The `openenv.yaml` manifest declares the types so clients can auto-discover the environment's interface without reading the source code.

---

## Observation Design: What Does the Agent See?

The agent receives two inputs every step — fused by the encoder before reaching the actor and critic heads.

### 1. Egocentric Headlight Image (64×64 RGB)

The key design decision: the image is **always rotated so the car faces upward**. This makes the visual input invariant to absolute position and heading.

```
Raw game state (x, y, angle)
  → blit track surface to 900×600 offscreen canvas
  → draw 60°-wide headlight cone, 60 px ahead
  → crop 120×120 px centred on car (grass-padded at screen edges)
  → rotate so car heading = UP
  → re-crop 120×120 after rotation padding
  → scale to 64×64
  → array3d → (H, W, 3) uint8
```

The network never has to learn "this pixel coordinate means I'm near the left wall." It only needs to learn "a bright boundary in the right half of my view means steer left." This is the same principle as egocentric observation in robotics — it dramatically improves sample efficiency and generalisation across tracks.

### 2. Scalar Sensors (9 floats)

```python
[
  angular_velocity,   # gyroscope — rotational speed
  speed,              # speedometer — speed / max_speed
  ray_left,           # raycast at −90° (boundary distance, normalised 0→1)
  ray_front_left,     # raycast at −45°
  ray_front,          # raycast at   0° (straight ahead)
  ray_front_right,    # raycast at  +45°
  ray_right,          # raycast at  +90°
  wp_sin,             # sin(bearing to next waypoint − car heading)
  wp_cos,             # cos(bearing to next waypoint − car heading)
]
```

The **5 raycasts** replaced an earlier binary `on_track` flag. With binary feedback, the agent only learns it's off-track *after* crashing. With raycasts, it can see a wall approaching at 100 px and start steering away — the difference between an anticipatory driver and a reactive one.

The **waypoint bearing** (`wp_sin`, `wp_cos`) acts as a relative GPS compass. It points toward the centreline 10 waypoints ahead, in coordinates relative to the car's current heading. `wp_cos ≈ 1` means the waypoint is straight ahead; `wp_sin < 0` means it's to the left. This gives the agent intentional directional signal even in the middle of a straight — without exposing raw map coordinates.

---

## Reward Design: Keeping Gradients Meaningful

Early experiments used large reward magnitudes (±300 for crashes, +200 for lap completion). Value function targets in that range cause value loss to dominate the gradient, which crowds out the policy loss signal and causes entropy collapse. The final reward table uses magnitudes 1/20 of those initial values:

| Event | Reward | Purpose |
|-------|--------|---------|
| Every step | −0.005 | Efficiency pressure |
| Forward speed | `speed_norm × 0.10` | Must move to earn reward |
| Reverse speed | `speed_norm × 0.10` (negative) | Penalise going backwards |
| Waypoint advance | +0.25 per wp | Dense signal toward lap completion |
| Waypoint regress | −0.25 per wp | Wrong-way penalty |
| Lap completed | **+10.0** | Major bonus |
| Crash / off-track | **−15.0**, done | Hard deterrent |
| Out of bounds | **−15.0**, done | Stay on screen |

The waypoint progress reward was the single most impactful addition. Without it, the only lap signal is the +10 bonus at the finish line — extremely sparse for a 3000-step episode. With +0.25 per waypoint crossed, the agent gets a gradient *in the direction of the finish line* on every step of every episode, even episodes it never completes.

### Lap Detection: No Shortcuts

A naive "did the car cross the start line" check is easy to exploit. The final implementation uses a two-phase arm/trigger gate:

1. **Arm**: Car must travel >50 px *past* the start line going forward
2. **Trigger**: Car must cross back through the start line with `speed > 0.3` AND having covered ≥80% of the track's optimal lap distance

The 80% distance gate makes shortcutting across the infield impossible — the car genuinely has to drive the track.

---

## Model Architecture: ImpalaCNN + Shared Encoder

```
image (3, 64, 64) ──► ImpalaCNN
                       Block 1: Conv(3→16)  → MaxPool → ResBlock × 2
                       Block 2: Conv(16→32) → MaxPool → ResBlock × 2
                       Block 3: Conv(32→32) → MaxPool → ResBlock × 2
                       Flatten(2048) → Linear → ReLU ──────────► 256-d
                                                                     │
scalars (9,) ──────► MLP: Linear(9→32) → ReLU → Linear(32→32) ─► 32-d
                                                                     │
                                                              Concat 288-d
                                                              (RaceEncoder)
                                                                     │
                                              ┌──────────────────────┤
                                         Actor head            Critic head
                                      Linear(288→2)           Linear(288→1)
                                      + log_std (param)        scalar V(s)
                                      IndependentNormal
                                      → [accel, steer]
```

### Why ImpalaCNN over Nature CNN?

Nature CNN (3 conv layers, no skip connections) is the standard for Atari-scale inputs. ImpalaCNN adds **residual skip connections** in each block: `output = x + conv_block(x)`. This gives gradients a direct path from the actor/critic heads back to the early convolution layers — with Nature CNN on long networks, gradients vanish before reaching the early layers.

In practice: ImpalaCNN converges in ~3–5× fewer environment steps on this task. At the same inference cost per step, the wall-clock training time dropped from several days to a few hours.

### Initialisation Details

Small but impactful:
- **Orthogonal init** for all linear layers (gain √2 for encoder layers)
- **Actor head** gain 0.01 — very small initial action means (near-zero mean accel/steer), preventing the agent from immediately driving full-speed into a wall
- **Actor bias[0] = 0.3** — gentle initial forward acceleration, so the car is moving from step 1 rather than sitting still
- **Log-std = −1.0** initially (std ≈ 0.37) — moderate initial exploration, not too random

---

## Training: TorchRL PPO with Curriculum Gating

Training uses [TorchRL](https://github.com/pytorch/rl) PPO with 16 parallel worker processes:

```bash
uv run python -u training/train_torchrl.py \
  --num-envs 16       \   # 16 parallel workers
  --rollout-steps 8192 \  # frames before each PPO update
  --batch-size 1024    \  # PPO minibatch size
  --ppo-epochs 10      \  # passes per rollout
  --compile               # torch.compile for ~30% throughput boost
```

### The Three-Layer Curriculum

The curriculum has three interlocking mechanisms:

**1. Greedy Evaluation Gating**

Every 25k steps, all 10 tracks are evaluated with a deterministic (greedy) policy — no exploration noise. A track *passes* if the agent completes ≥1 lap with 0 crashes. The frontier advances only when every track up to the current one passes. If all 10 pass simultaneously, training terminates.

This is stricter than using the rolling training reward (which mixes exploration noise). Greedy eval measures what the policy *actually* knows, not how lucky it got.

**2. Anti-Forgetting Replay**

```
70% of episodes → current frontier track
30% of episodes → round-robin through all mastered tracks
```

Round-robin means each mastered track gets equal coverage — early tracks don't get starved as the curriculum grows. Without this, the agent reliably regresses on Track 1 by the time it reaches Track 7.

**3. Priority Replay**

When greedy eval finds a regression (a previously mastered track now failing), that track is written to a shared memory array. All worker processes check this array and dedicate an additional 30% of their episodes to priority-replaying the failing track. Regressions are typically recovered within one 25k-step eval interval.

### W&B Training Curves

![Episode metrics — reward, laps, crashes across curriculum](Training-Episode-wandb.png)

![PPO internals — value loss, policy loss, entropy, explained variance](Training-PPO-wandb.png)

The curriculum level stepping is clearly visible in the episode reward chart — each plateau is the agent consolidating skills on a new frontier track, and each jump is when it advances. The PPO entropy chart shows healthy exploration throughout (no entropy collapse), and explained variance rises toward 0.9+ as the value function accurately models the curriculum.

---

## Key Innovations

### 1. Egocentric Rotation as Data Augmentation

Rotating the observation so the car always faces up is equivalent to a learned data augmentation that's built directly into the environment rendering. It means the CNN never needs to learn "I'm in the top-left corner of the screen and about to hit the wall" — it only needs to learn "there's a wall in the upper-left of my view." The effective training distribution is ~4× richer for free.

### 2. Waypoint GPS as Relative Compass

Early experiments included raw waypoint coordinates in the scalar vector. These don't transfer across tracks — a policy trained on Track 1 learns "the waypoint at (700, 300) means go right" which is meaningless on Track 9.

The `(wp_sin, wp_cos)` encoding is track-agnostic: it only encodes *where the track goes relative to where I'm pointing*, not where I am in absolute space. This is the key to cross-track generalisation.

### 3. Soft Curriculum with Anti-Forgetting

The 70/30 split between frontier and replay was tuned empirically. Lower replay fractions (e.g., 90/10) caused clear catastrophic forgetting — the agent would master Track 7 but crash on Track 1. Higher replay fractions (e.g., 50/50) slowed frontier progress significantly. 70/30 with round-robin distribution hit the sweet spot.

### 4. Priority Replay for Regression Recovery

Without priority replay, a regression on an early track would typically persist for 5–10 eval intervals before being recovered. With priority replay, the agent re-encounters the failing track immediately and recovers within 1–2 intervals. This makes the curriculum monotonically advancing rather than oscillating.

### 5. Scaled Reward Magnitudes

Keeping all rewards in [−15, +10] was non-obvious but critical. Value function targets in this range let the critic converge quickly (explained variance > 0.8 within 200k steps). Larger magnitudes caused value loss to overwhelm the policy gradient — the policy would barely update because the gradient was dominated by trying to fit the value function.

---

## Results

The final policy, evaluated deterministically on all 10 tracks:

| Track | Laps | Crashes | Status |
|-------|------|---------|--------|
| 1 — Wide Oval | 1 | 0 | ✅ PASS |
| 2 — Standard Oval | 1 | 0 | ✅ PASS |
| 3 — Narrow Oval | 1 | 0 | ✅ PASS |
| 4 — Superspeedway | 1 | 0 | ✅ PASS |
| 5 — Rounded Rectangle | 1 | 0 | ✅ PASS |
| 6 — Stadium Oval | 1 | 0 | ✅ PASS |
| 7 — Tight Rectangle | 1 | 0 | ✅ PASS |
| 8 — Small Oval | 1 | 0 | ✅ PASS |
| 9 — Hairpin Track | 1 | 0 | ✅ PASS |
| 10 — Chicane Track | 1 | 0 | ✅ PASS |

**10/10 tracks — zero crashes, minimum one lap — in ~1.3M environment steps.**

Training on 16 parallel workers with `torch.compile` takes approximately 1–3 hours on a single GPU.

---

## Try It Yourself

Everything is open source. You can run the trained agent in 3 commands:

```bash
git clone https://github.com/NirmalPratheep/curriculum-car-racer
cd curriculum-car-racer
uv sync

# Inference on all 10 tracks
uv run python inference/inference.py \
  --checkpoint checkpoints/ppo_torchrl_final.pt
```

Or play interactively yourself:

```bash
uv run python main.py    # Track 1 — Wide Oval
uv run python main.py 9  # Track 9 — Hairpin
```

Or train from scratch:

```bash
bash training/cmd
```

The environment itself is also usable independently via OpenEnv:

```bash
# Start the API server
uvicorn env.server.app:app --host 0.0.0.0 --port 8000

# Connect from any OpenEnv client
python -c "
from openenv import connect
env = connect('http://localhost:8000')
obs = env.reset()
print(obs.image.shape, obs.wp_cos)
"
```

---

## What I Would Do Next

**Richer observations**: The current 5-ray sensor gives good boundary detection but misses detail around tight corners. A 12-ray or 180° sweep would help with the chicane.

**Multi-agent**: The OpenEnv architecture supports multiple simultaneous clients. Adding a second car as an obstacle (or a racing opponent) would make the task significantly harder and more interesting.

**Continuous curriculum**: Instead of discrete frontier advancement, sample tracks according to a probability distribution that concentrates on the current difficulty boundary — similar to automatic curriculum generation (ACG) or teacher-student setups.

**Sim-to-real transfer**: The physics model is simple enough that the observation design (egocentric image + relative GPS) should transfer to a real remote-controlled car with a camera and proximity sensors.

---

## Links

- **GitHub**: [NirmalPratheep/curriculum-car-racer](https://github.com/NirmalPratheep/curriculum-car-racer)
- **HuggingFace Hub**: [NirmalPratheep/curriculum-car-racer](https://huggingface.co/NirmalPratheep/curriculum-car-racer)
- **OpenEnv**: [openenv.dev](https://openenv.dev)
- **TorchRL**: [pytorch/rl](https://github.com/pytorch/rl)

---

*Submitted to the OpenEnv Student Challenge, April 2026.*
*All code, weights, and training logs are open source.*
