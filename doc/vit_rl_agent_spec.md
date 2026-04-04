# ViT + RL Agent from Headlight Cone Images — Specification

## Overview

Instead of hand-crafted ray-geometry rules, capture the headlight cone region
as a pixel crop each frame, encode it with a Vision Transformer (ViT), and train
a Reinforcement Learning policy on top. The agent learns what to do purely from
visual lane information — the same signal a human driver uses.

---

## 1. Observation Pipeline

```
game frame
  └─► crop headlight cone region  (image patch)
        └─► resize to 64×64 or 128×128
              └─► ViT encoder  → embedding vector (D=256 or 512)
                    └─► RL policy head  → (steer, throttle)
```

### 1.1 Cone Crop

- Take a rectangular bounding box around the headlight cone in screen space.
- Cone is 60° wide, 60 px long → bounding box ≈ 70×70 px at car nose.
- **Rotate the crop** to align with the car heading before feeding to ViT —
  this removes heading from the image and forces the model to learn lane geometry,
  not absolute orientation.

| Crop parameter | Value |
|----------------|-------|
| Size (raw) | ~70×70 px |
| Resized to | 64×64 px |
| Channels | RGB or Grayscale |
| Rotation aligned | Yes (heading = up) |

### 1.2 ViT Encoder

Patch the 64×64 image into tokens and encode with a small ViT:

| ViT parameter | Small config | Medium config |
|---------------|-------------|---------------|
| Image size | 64×64 | 128×128 |
| Patch size | 8×8 | 16×16 |
| Num patches | 64 | 64 |
| Embedding dim | 128 | 256 |
| Transformer depth | 4 layers | 6 layers |
| Heads | 4 | 8 |
| Output | CLS token → 128-d | CLS token → 256-d |

For this game a **pretrained ViT is not needed** — the visual domain is simple
(dark tarmac, white lines, yellow cone). Train from scratch.

---

## 2. Action Space

| Output | Type | Range |
|--------|------|-------|
| Steer | Discrete: {left, none, right} | 3 classes |
| Throttle | Discrete: {brake, coast, full} | 3 classes |

Combined: **9 discrete actions** — clean for PPO/DQN, avoids continuous
action instability early in training.

---

## 3. Reward Function

```
r = +0.5   if on_track and speed > 1.0        (moving forward on road)
r = +1.0   if lap completed
r = −1.0   if off_track
r = −0.1   per frame while stationary          (penalise stopping)
r = +0.05  × normalised_speed                 (encourage speed)
```

Keep reward sparse-ish — avoid over-shaping that teaches a local optimum
(e.g. always hugging the inner wall).

---

## 4. Training Setup

### 4.1 Algorithm

**PPO (Proximal Policy Optimisation)** recommended:
- Stable with visual inputs
- Works well with discrete actions
- Less sensitive to hyperparameters than DQN

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | 3e-4 |
| Clip epsilon | 0.2 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Rollout steps | 2048 |
| Mini-batch size | 256 |
| Epochs per update | 4 |
| Entropy coefficient | 0.01 |

### 4.2 Training Phases

**Phase 1 — Survival (0–500k steps)**
Goal: stay on track.
Reward: only `on_track` and `off_track` signals.
Expected behaviour: learns to steer away from white lines.

**Phase 2 — Speed (500k–1.5M steps)**
Goal: complete laps faster.
Reward: add speed bonus + lap completion bonus.
Expected behaviour: learns to take racing line, not just survive.

**Phase 3 — Fine-tune (1.5M–3M steps)**
Goal: consistent lap times.
Reduce entropy coefficient to 0.001 to exploit learned policy.

### 4.3 Data Collection Rate

At 60 FPS with `pygame`, one A100/3090 can run 8–16 parallel envs headlessly:

| Setup | Steps/sec | Time to 3M steps |
|-------|-----------|-----------------|
| 1 env, CPU | ~60 | ~14 hours |
| 8 envs, GPU render | ~480 | ~1.7 hours |
| 16 envs, offscreen | ~900 | ~55 minutes |

Run `pygame` with `os.environ["SDL_VIDEODRIVER"] = "offscreen"` for headless
parallel training.

---

## 5. Pros

| Pro | Detail |
|-----|--------|
| **No manual feature engineering** | ViT learns what features matter from pixels — white line thickness, curvature, distance — without being told |
| **Generalises to track changes** | If track colour, width, or shape changes, the policy can adapt with minimal retraining |
| **Attention is interpretable** | ViT attention maps show which pixels drove the decision — useful for debugging |
| **Scales to richer observations** | Easy to widen the crop to include more track context without redesigning the agent |
| **Learns non-obvious strategies** | May discover racing lines or anticipatory steering that rule-based agent cannot express |
| **Imitation learning ready** | The same image+action pairs from the rule agent can be used to pre-train (behavioural cloning) before RL fine-tuning |

---

## 6. Cons

| Con | Detail |
|-----|--------|
| **Sample inefficient** | ViT+RL needs millions of frames; rule agent is instant |
| **Training instability** | Visual RL can collapse or plateau — requires careful reward tuning |
| **Compute cost** | GPU required for reasonable training time; CPU-only training is very slow |
| **Overfit to track** | Agent trained on one oval may fail on a new layout without retraining |
| **Credit assignment is hard** | A crash 10 seconds after a bad steer is hard for the agent to attribute correctly |
| **Crop alignment is critical** | If rotation normalisation is wrong, the ViT encodes absolute heading instead of lane geometry — policy breaks |
| **Patch size sensitivity** | 64×64 with 8×8 patches gives only 64 tokens; thin white lines may be missed if patch straddles the line |

---

## 7. Recommended Baseline Comparison

Before full ViT+RL training, run these baselines in order:

```
1. Rule agent (headlight rays)          ← already specced
2. Behavioural cloning on rule agent    ← supervised, fast, good initialisation
3. PPO from scratch (no ViT, flat obs)  ← validates reward function
4. PPO with ViT encoder                 ← full system
```

Each baseline isolates a different failure mode.

---

## 8. Implementation Checklist

- [ ] Add headless render mode to `game.py` (`SDL_VIDEODRIVER=offscreen`)
- [ ] Add `get_observation()` method: returns rotated 64×64 cone crop as numpy array
- [ ] Wrap game as a Gymnasium `Env` (`reset`, `step`, `render`)
- [ ] Implement small ViT encoder in PyTorch (or use `timm.create_model("vit_tiny_patch8_64")`)
- [ ] Wire ViT output into PPO actor-critic heads
- [ ] Log attention maps every 50k steps (Weights & Biases / TensorBoard)
- [ ] Save checkpoint at each phase boundary

---

## 9. Generalisation to New Tracks

### 9.1 What Transfers vs. What Doesn't

The model has two parts with different transfer behaviour:

| Component | Transfers? | Reason |
|-----------|-----------|--------|
| ViT encoder | Yes | Learns visual patterns (white line on dark tarmac) — same on any track |
| Policy head | No | Learns timing tied to this track's curvature |

**Practical rule:** freeze the encoder, re-initialise and retrain only the policy
head on a new track. This is 5–10× faster than training from scratch.

| Scenario | Encoder | Policy | Works? |
|----------|---------|--------|--------|
| Same oval, different car speed | ✓ | ~✓ | Yes |
| Wider/narrower oval | ✓ | Partially | Fine-tune needed |
| New shape (chicane, L-shape) | ✓ | ✗ | Retrain policy |
| Different lane colour | ✗ | ✗ | Retrain both |

---

### 9.2 How to Make It More Generalisable

#### 1. Rotation-Normalised Crop (non-negotiable)

The crop **must** be rotated so the car heading always points up before feeding
to the ViT. Without this, the encoder learns "I am turning left at the bottom of
this oval" instead of "a white line is approaching from the right." The first
never transfers; the second always does.

#### 2. Domain Randomisation (biggest gain)

Train on many track variations simultaneously so the policy never overfits to
one layout. Randomise per episode:

```
- Track width       (±20%)
- Oval size         (scale 0.7× – 1.4×)
- Aspect ratio      (rounder vs. more elongated)
- Lane colour       (white → yellow → light grey)
- Grass colour      (green → brown → grey)
- Car speed cap     (vary MAX_SPEED)
```

If the agent survives all combinations, a real new track is just another sample
from the distribution.

#### 3. Add Speed as Input

The policy currently infers timing purely from vision. Speed-dependent timing
is impossible to learn from a fixed crop alone — the same white-line-position
means "turn now" at high speed but "turn later" at low speed. Fix: concatenate
normalised speed scalar to the CLS token before the policy head.

#### 4. Auxiliary Losses on the ViT Encoder

Force the encoder to learn geometry, not track identity, by adding supervised
auxiliary tasks during PPO training:

| Auxiliary task | What it forces the ViT to learn |
|---------------|--------------------------------|
| Predict distance to nearest white line | Metric lane geometry |
| Predict heading error to lane centre | Corrective direction |
| Predict on-track / off-track | Safety boundary awareness |

Gradients from both auxiliary and PPO losses shape the same ViT weights.

#### 5. Self-Supervised Pretraining

Before any RL, pretrain the ViT with self-supervised learning on random track
crops (no labels needed):

- Use **SimCLR or BYOL**: two augmented views of the same crop should have
  similar embeddings.
- Augmentations: colour jitter, blur, small rotations, brightness shift.

This gives the encoder a track-agnostic head start. Freeze it and only train
the policy head per new track.

#### 6. Wider Multi-Ray Observation

Extend the cone from 3 rays to 7–9, covering a wider forward arc. On a tight
curve the agent then sees the wall curving away, not just the immediate line —
giving earlier signal for sharper turns.

#### 7. Curriculum Training

```
Stage 1: Wide track, gentle oval          ← easy, builds basic lane following
Stage 2: Narrower track, faster speed     ← forces precision
Stage 3: Randomised shapes and colours    ← forces generalisation
Stage 4: Held-out test track (unseen)     ← zero-shot evaluation
```

Never evaluate on a track seen during training.

#### 8. What Not To Do

| Temptation | Problem |
|------------|---------|
| Train longer on one track | Deepens overfitting, does not help transfer |
| Increase ViT depth | More capacity to memorise, not generalise |
| Dense reward shaping | Policy exploits reward on this track specifically |
| Fine-tune encoder on each new track | Destroys transferable visual features |

#### 9. Priority Order

```
1. Rotation-normalise the crop          ← non-negotiable
2. Domain randomise track geometry      ← biggest generalisation gain
3. Add speed as input                   ← fixes timing dependence
4. Auxiliary loss (distance to wall)    ← structures the embedding
5. Curriculum training                  ← prevents early collapse
6. Self-supervised pretraining          ← adds compute, high payoff
```

Steps 1–3 alone give significant zero-shot transfer to new oval shapes.
Steps 4–6 push toward full track-type generalisation.

---

## 10. Key Risk

> **The cone crop is tiny.** At 60 px length and 60° spread, the agent sees
> only 1–2 track widths ahead. On a fast oval, this may not give enough reaction
> time. Consider increasing `CONE_LEN` to 120 px for RL training even if the
> visual headlight stays at 60 px for the human-facing game.
