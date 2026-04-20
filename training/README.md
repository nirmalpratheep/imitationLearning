# Training Scripts

All training scripts live here. Run them from the **project root** with `uv run`.

## Scripts

| Script | Framework | Status |
|--------|-----------|--------|
| `train_torchrl.py` | TorchRL PPO | **Recommended** |
| `train_cleanrl.py` | CleanRL custom PPO | Legacy / reference |
| `monitor.py` | W&B auto-monitor | Run alongside training |
| `test_video.py` | Inference renderer | Evaluate a checkpoint |

---

## Quick Start

```bash
# Clone and install
git clone <repo>
cd curriculum-car-racer
uv sync

# Train (GPU recommended)
bash training/cmd

# Or manually
uv run python -u training/train_torchrl.py \
  --num-envs 16 \
  --rollout-steps 4096 \
  --batch-size 512 \
  --total-steps 300_000_000
```

## Key Flags (`train_torchrl.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--num-envs` | 4 | Parallel worker processes |
| `--total-steps` | 5 000 000 | Total environment steps |
| `--rollout-steps` | 2048 | Frames per PPO update (across all envs) |
| `--batch-size` | 64 | PPO minibatch size |
| `--ppo-epochs` | 10 | Update passes per rollout |
| `--lr` | 3e-4 | Adam learning rate |
| `--threshold` | 30.0 | Rolling reward to advance curriculum |
| `--video-interval` | 25 000 | Steps between W&B video logs |
| `--resume` | — | Path to `.pt` checkpoint |
| `--compile` | off | Enable `torch.compile` |

## Resume from Checkpoint

```bash
uv run python -u training/train_torchrl.py \
  --resume checkpoints/ppo_torchrl_step00500000_lvl02.pt \
  --total-steps 300_000_000
```

## Render Inference Video

```bash
uv run python training/test_video.py --track 1
uv run python training/test_video.py --checkpoint checkpoints/ppo_torchrl_final.pt --track 5
```

## Run W&B Monitor (auto-restart on stall)

```bash
# In a separate terminal alongside training
uv run python training/monitor.py
```
