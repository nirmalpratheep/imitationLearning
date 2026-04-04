# Observation Design

## Why not the raw 7-float vector?

The original `CarEnv` observation from `rl_splits.py`:

```
[x/W,  y/H,  sin(angle),  cos(angle),  speed/max,  on_track,  gate_side/500]
```

| Feature | Problem for generalisation |
|---------|--------------------------|
| `x`, `y` | Absolute screen position. Agent learns "turn here on this oval" not "a wall is approaching". Completely fails on unseen tracks. |
| `gate_side` | Distance to the start/finish gate of the current track. Meaningless on a different layout. |
| `on_track` | Binary reactive signal — agent only learns "I crashed" not "I'm about to crash". No lookahead. |
| `sin/cos angle` | Orientation only. Tells the agent which way it faces, not where the track goes. |
| `speed` | Fine. Speed-dependent timing is always useful. |

**Core problem**: no forward perception of track shape. Without knowing what's
ahead the agent can only react after hitting a wall.

---

## Egocentric Headlight Image

The headlight image gives the agent local visual context of the road ahead,
invariant to absolute position and track layout.

### Why egocentric (car faces up)?

If we feed raw screen pixels the agent learns:
> "At pixel (450, 515) the track curves left"

If we rotate the crop so the car always faces up, the agent learns:
> "When the left track edge enters from the top-right, steer left"

The second representation is the same on every track and every position. The
CNN will generalise; the first will not.

### Rendering pipeline (per step)

```
game state (x, y, angle)
  │
  ▼
blit track.surface to offscreen pygame.Surface(900×600)
  │
  ▼
draw_headlights(surf, x, y, angle)        ← 60° cone, 60 px ahead, yellow fill
  │
  ▼
crop 120×120 px centred on car            ← world context around the car
  │
  ▼
pygame.transform.rotate(crop, -(angle-270))  ← heading → UP
  │
  ▼
re-crop centre 120×120 after rotation padding
  │
  ▼
pygame.transform.scale → 64×64            ← fixed output size
  │
  ▼
surfarray.array3d → (64, 64, 3) uint8     ← numpy array
```

### What the agent sees

```
Forward direction (up)
         ▲
         │
  ┌──────┴──────┐
  │  ██ track ██│  ← dark grey tarmac
  │ ░░░░░░░░░░░ │  ← yellow headlight cone
  │░ white line ░│  ← track boundary ahead
  │  ████████   │
  │    [car]    │  ← car position (centre-bottom)
  └─────────────┘
```

Straight road: track fills the centre, white lines near the top-left and top-right.
Left turn ahead: right track edge creeps inward from the top.
Sharp corner: one white line dominates one side.

---

## Final Observation

### Image branch → ImpalaCNN
```
obs.image   : (64, 64, 3)  uint8   normalise to float32 / 255 before network
```

### Scalar branch → MLP
```
obs.speed      : float  speed / max_speed   (≈ 0..1)
obs.on_track   : float  1.0 on road, 0.0 off road
obs.sin_angle  : float  sin of absolute heading
obs.cos_angle  : float  cos of absolute heading
```

`sin/cos angle` are kept despite being somewhat heading-specific because they
help the agent understand orientation context (e.g. am I going backwards?) at
essentially zero extra cost.

---

## Comparison with Vision Transformer

Using a ViT on this 64×64 image is possible but not recommended for training from scratch:

| Aspect | CNN (Impala) | ViT |
|--------|-------------|-----|
| Sample efficiency | High — inductive locality bias matches the task | Low — needs millions of samples without pretraining |
| 64×64 input | Ideal | Weak — patch_size=8 gives 64 tokens, limited context |
| Task requirement | Local edge detection (CNN strength) | Global attention (not needed here) |
| Inference speed | Fast | Slower |

**Recommendation**: ImpalaCNN for training from scratch. If a pretrained ViT
backbone is available and fine-tuned, the hybrid (CNN stem + 1-layer transformer)
becomes competitive. See `doc/encoder.md`.
