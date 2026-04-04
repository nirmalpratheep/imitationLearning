# Encoder

## RaceEncoder

Fuses the two observation branches into a single feature vector for PPO.

```
image (B, 3, 64, 64)
  └─► ImpalaCNN                →  (B, 256)
                                          cat  →  (B, 288)  →  actor / critic
scalars (B, 4)                            │
  └─► Linear(4→32) → ReLU → Linear(32→32) →  (B,  32)
```

```python
from env.encoder import RaceEncoder
import torch

encoder  = RaceEncoder()                   # out_features = 288
img      = obs_image.float() / 255.0      # (B, 3, 64, 64)
scalars  = torch.stack([obs.speed, obs.on_track, obs.sin_angle, obs.cos_angle], dim=-1)
features = encoder(img, scalars)           # (B, 288)
```

---

## Nature CNN vs ImpalaCNN

### Nature CNN (Mnih et al., DQN 2015)

Plain feedforward stack of three convolution layers followed by a fully-connected layer.

```
Input (3, 64, 64)
  Conv2d(3→32,  8×8, stride=4)  →  (32, 15, 15)
  ReLU
  Conv2d(32→64, 4×4, stride=2)  →  (64,  6,  6)
  ReLU
  Conv2d(64→64, 3×3, stride=1)  →  (64,  4,  4)
  ReLU
  Flatten  →  1024
  Linear(1024→512)
  ReLU
  Output: (512,)
```

No skip connections. Gradients must traverse every layer on the way back.
In deeper networks, or with a complex visual task, the first conv layer receives
vanishingly small gradients — it stops learning useful features early in training.

---

### ImpalaCNN (Espeholt et al., IMPALA 2018)

Three blocks, each containing a convolution, max pool, and **two residual sub-blocks**.

```
Input (3, 64, 64)

Block 1: channels 3→16
  Conv2d(3→16,  3×3, pad=1)   →  (16, 64, 64)
  MaxPool2d(3×3, stride=2)     →  (16, 32, 32)
  ResBlock(16)
    x = x + [ReLU→Conv(3×3)→ReLU→Conv(3×3)](x)
  ResBlock(16)

Block 2: channels 16→32
  Conv2d(16→32, 3×3, pad=1)   →  (32, 32, 32)
  MaxPool2d(3×3, stride=2)     →  (32, 16, 16)
  ResBlock(32)
  ResBlock(32)

Block 3: channels 32→32
  Conv2d(32→32, 3×3, pad=1)   →  (32, 16, 16)
  MaxPool2d(3×3, stride=2)     →  (32,  8,  8)
  ResBlock(32)
  ResBlock(32)

ReLU
Flatten  →  2048
Linear(2048→256)
ReLU
Output: (256,)
```

### The skip connection

```python
# ResBlock forward
def forward(self, x):
    return x + self.net(x)    # skip: gradient has a direct path back
```

The shortcut `x + f(x)` means that `∂L/∂x = ∂L/∂(output) · (1 + ∂f/∂x)`.
The `1` term is a direct gradient highway — the first conv block always receives
a healthy gradient signal no matter how deep the network is. Early filters keep
learning throughout training instead of stagnating.

---

### Side-by-side comparison

| Property | Nature CNN | ImpalaCNN |
|----------|-----------|-----------|
| Skip connections | None | 2 per block (6 total) |
| Gradient flow to early layers | Degrades with depth | Preserved via shortcuts |
| Parameters (for this config) | ~800 k | ~200 k (deeper but narrower) |
| Feature map at output | 4×4 | 8×8 (more spatial detail) |
| Sample efficiency | Baseline | ~3–5× better (benchmark: ProcGen, DM-Control) |
| Inference latency | Slightly faster | Negligible difference |
| Implementation complexity | Simple | Moderate (extra residual class) |

### When to switch to ViT

| Condition | Recommendation |
|-----------|---------------|
| Training from scratch, limited GPU budget | ImpalaCNN |
| Pretrained ViT checkpoint available | Hybrid (CNN stem + 1-layer transformer) |
| Need attention maps for interpretability | Hybrid or full ViT |
| Track variety > 20 distinct layouts | Hybrid — global context helps on complex shapes |

---

## Hybrid (CNN stem + Transformer)

For future use if the agent struggles with complex polygon tracks (levels 13–16)
where seeing both sides of a narrow passage simultaneously matters.

```
Input (3, 64, 64)
  Conv(3→16, 4×4 stride=2)   → (16, 32, 32)   ReLU
  Conv(16→32, 4×4 stride=2)  → (32, 16, 16)   ReLU
  Conv(32→32, 4×4 stride=2)  → (32,  8,  8)   ReLU
                                 ↓
         reshape: (B, 64 tokens, 32 dims)
                                 ↓
         TransformerEncoderLayer(d_model=32, nhead=4, 1 layer)
                                 ↓
         flatten (B, 64×32=2048) → Linear(2048→256)
```

Gives global spatial attention over the 8×8 feature grid (64 tokens) at
modest compute cost. Expect ~3× more samples vs ImpalaCNN to converge.
