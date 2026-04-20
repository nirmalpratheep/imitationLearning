"""
train.py — CleanRL-style PPO training for the curriculum car racer.

W&B metrics logged
──────────────────
  episode/reward            total reward per episode
  episode/length            steps in the episode
  episode/laps              laps completed
  episode/crashes           on→off track transitions
  episode/on_track_pct      % of steps spent on track

  ppo/policy_loss           clipped surrogate loss
  ppo/value_loss            MSE value loss
  ppo/entropy               policy entropy (higher = more exploration)
  ppo/approx_kl             Schulman KL approximation (target < 0.02)
  ppo/clip_fraction         fraction of ratios that were clipped
  ppo/explained_variance    how well value fn predicts returns (1.0 = perfect)
  ppo/learning_rate         current LR (linearly decayed)
  ppo/entropy_coef          current entropy coefficient (linearly annealed)
  ppo/grad_norm             gradient norm before clipping

  curriculum/level          0-based frontier index within TRAIN tracks
  curriculum/track_level    TrackDef.level (1-20)
  curriculum/track_name     TrackDef.name
  curriculum/tier           difficulty tier (A-easy … E-single-lane)
  curriculum/rolling_mean   mean reward over last `window` episodes
  curriculum/threshold      effective threshold for the current track
  curriculum/is_replay      1 if this episode is an anti-forgetting replay

  val/mean_reward           aggregate mean reward across all VAL tracks
  val/mean_laps             aggregate mean laps
  val/completion_rate       fraction of VAL episodes with ≥ 1 lap
  val/track_N_reward        per-track breakdowns (N = track level)
  val/track_N_completion

  system/steps_per_sec      throughput
  system/elapsed_hours      wall-clock time

Usage
─────
  python train.py
  python train.py --num-envs 8
  python train.py --total-steps 5_000_000 --wandb-project my-racer
  python train.py --resume checkpoints/ppo_step00500000_lvl02.pt
  python train.py --wandb-offline          # no internet needed
"""

import argparse
import math
import os
import random
import statistics
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.amp import autocast, GradScaler
import wandb

# ── Headless pygame — must come before any game/env import ───────────────────
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from env import CurriculumBuilder, DriveAction          # noqa: E402
from env.encoder import RaceEncoder                     # noqa: E402
from env.subproc_vec_env import SubprocVecEnv           # noqa: E402
from game.rl_splits import difficulty_of                # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # W&B
    g = p.add_argument_group("W&B")
    g.add_argument("--wandb-project",  default="curriculum-car-racer")
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-offline",  action="store_true",
                   help="Log offline (no internet required)")

    # Training budget
    g = p.add_argument_group("Training budget")
    g.add_argument("--total-steps",    type=int,   default=5_000_000)
    g.add_argument("--rollout-steps",  type=int,   default=2048,
                   help="Steps collected per env per policy update")
    g.add_argument("--num-envs",       type=int,   default=4,
                   help="Number of parallel environments (CPU: use os.cpu_count()//2)")

    # PPO
    g = p.add_argument_group("PPO")
    g.add_argument("--ppo-epochs",     type=int,   default=4)
    g.add_argument("--minibatch-size", type=int,   default=1024)
    g.add_argument("--lr",             type=float, default=3e-4)
    g.add_argument("--lr-min",         type=float, default=1e-5,
                   help="Final LR after linear decay")
    g.add_argument("--gamma",          type=float, default=0.99)
    g.add_argument("--gae-lambda",     type=float, default=0.95)
    g.add_argument("--clip-eps",       type=float, default=0.2)
    g.add_argument("--vf-coef",        type=float, default=0.5)
    g.add_argument("--ent-coef-start", type=float, default=0.01)
    g.add_argument("--ent-coef-end",   type=float, default=0.001,
                   help="Entropy coef is linearly annealed to this value")
    g.add_argument("--max-grad-norm",  type=float, default=0.5)
    g.add_argument("--target-kl",      type=float, default=0.02,
                   help="Early-stop PPO epochs if approx KL exceeds this")

    # Curriculum
    g = p.add_argument_group("Curriculum")
    g.add_argument("--threshold",      type=float, default=30.0)
    g.add_argument("--window",         type=int,   default=50)
    g.add_argument("--replay-frac",    type=float, default=0.3)
    g.add_argument("--val-episodes",   type=int,   default=10,
                   help="Episodes per VAL track after each curriculum advance")

    # Checkpointing & resume
    g = p.add_argument_group("Checkpointing")
    g.add_argument("--checkpoint-interval", type=int, default=500_000,
                   help="Save a .pt every N global steps (0 = disabled)")
    g.add_argument("--video-interval",    type=int, default=25_000,
                   help="Log inference videos every N global steps (0 = only at checkpoints)")
    g.add_argument("--checkpoint-dir",      default="checkpoints")
    g.add_argument("--keep-checkpoints",  type=int, default=5,
                   help="Keep only the last N checkpoints on disk (0 = keep all)")
    g.add_argument("--resume",              default=None,
                   help="Path to a checkpoint .pt file to resume from")

    # Misc
    g = p.add_argument_group("Misc")
    g.add_argument("--seed",    type=int, default=42)
    g.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile the model for faster CPU inference (requires PyTorch ≥ 2.0)")
    g.add_argument("--subproc", action="store_true",
                   help="Run each environment in its own subprocess (parallel env stepping). "
                        "Recommended for GPU runs to improve utilisation.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PPOActorCritic(nn.Module):
    """
    ImpalaCNN(image) + MLP(7 scalars) → 288-d features → actor + critic heads.

    Actor  : Gaussian with learned mean, shared log-std parameter.
    Critic : single scalar value estimate.
    Actions: [accel, steer], each clamped to [-1, 1].
    """

    def __init__(self):
        super().__init__()
        self.encoder       = RaceEncoder()              # → (B, 288)
        D                  = self.encoder.out_features  # 288

        self.actor_mean    = nn.Linear(D, 2)
        # Start with smaller std (exp(-1) ≈ 0.37) so the initial policy
        # doesn't steer wildly.  With std=1.0 the car veers off-track before
        # it can learn anything.
        # Clamped in forward pass to [-3.0, -0.3] (std ∈ [0.05, 0.74]) to
        # prevent the entropy bonus from pushing std back toward 1.0.
        self.actor_log_std = nn.Parameter(torch.full((2,), -1.0))
        self.critic        = nn.Linear(D, 1)

        # Orthogonal init keeps early training stable
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        # Bias the initial policy toward gentle forward acceleration.
        # With brake_force(0.22) > accel_force(0.13), a zero-mean Gaussian
        # causes net deceleration → car stops → zero reward → no learning.
        # Bias of 0.3 keeps the car moving at moderate speed so it stays
        # on-track long enough to receive on-track and progress rewards.
        with torch.no_grad():
            self.actor_mean.bias[0] = 0.3   # accel dimension
        nn.init.orthogonal_(self.critic.weight,     gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def get_value(self, img: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(img, scalars))

    def get_action_and_value(
        self,
        img:     torch.Tensor,
        scalars: torch.Tensor,
        action:  torch.Tensor = None,
    ):
        feat    = self.encoder(img, scalars)
        mean    = self.actor_mean(feat)
        log_std = self.actor_log_std.clamp(-3.0, -0.3)
        std     = log_std.exp().expand_as(mean)
        dist    = Normal(mean, std)

        if action is None:
            action = dist.sample()

        logp    = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value   = self.critic(feat)
        return action, logp, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_tensors(obs, device: torch.device):
    """RaceObservation → (img (3,64,64), scalars (7,)) float32 tensors."""
    img     = (torch.from_numpy(obs.image.copy())
               .float().div(255.0)
               .permute(2, 0, 1)          # HWC → CHW
               .to(device))
    scalars = torch.tensor(obs.scalars, dtype=torch.float32, device=device)
    return img, scalars


def obs_list_to_arrays(obs_list):
    """Batch-convert obs_list → numpy float32 arrays (CPU only, no GPU allocation).
    Returns imgs (N, 3, H, W) float32 and scalars (N, 7) float32.
    Accepts both RaceObservation objects and _StepResult namedtuples."""
    imgs = np.stack([obs.image for obs in obs_list], axis=0)          # (N, H, W, 3) uint8
    imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0     # (N, 3, H, W) float32
    # _StepResult.scalars is already a (7,) float32 array; RaceObservation.scalars is a list
    scas_raw = [obs.scalars for obs in obs_list]
    scas = np.array(scas_raw, dtype=np.float32)                       # (N, 7)
    return imgs, scas


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """EV = 1 - Var(returns - values) / Var(returns).  1.0 = perfect, ≤ 0 = useless."""
    var_y = torch.var(returns)
    return float("nan") if var_y == 0 else (1.0 - torch.var(returns - values) / var_y).item()


@torch.no_grad()
def run_validation(
    model:      PPOActorCritic,
    builder:    CurriculumBuilder,
    n_episodes: int,
    device:     torch.device,
) -> dict:
    """
    Greedy rollout on every VAL track.
    Returns aggregate + per-track metrics suitable for wandb.log().
    """
    model.eval()
    all_rewards, all_laps, all_done = [], [], []
    per_track = []

    for env in builder.val_envs():
        track      = env._env.track
        ep_rewards = []
        ep_laps    = []

        for _ in range(n_episodes):
            obs     = env.reset()
            total_r = 0.0
            while not obs.done:
                img, sca = obs_to_tensors(obs, device)
                act, _, _, _ = model.get_action_and_value(img.unsqueeze(0), sca.unsqueeze(0))
                act_clamped = act.clamp(-1.0, 1.0)
                obs = env.step(DriveAction(accel=act_clamped[0, 0].item(), steer=act_clamped[0, 1].item()))
                total_r += obs.reward
            ep_rewards.append(total_r)
            ep_laps.append(env._env.laps)

        comp = sum(1 for l in ep_laps if l >= 1) / n_episodes
        per_track.append({
            "level":           track.level,
            "name":            track.name,
            "tier":            difficulty_of(track),
            "mean_reward":     float(np.mean(ep_rewards)),
            "mean_laps":       float(np.mean(ep_laps)),
            "completion_rate": comp,
        })
        all_rewards.extend(ep_rewards)
        all_laps.extend(ep_laps)
        all_done.extend([l >= 1 for l in ep_laps])

    model.train()
    return {
        "per_track":       per_track,
        "mean_reward":     float(np.mean(all_rewards)),
        "mean_laps":       float(np.mean(all_laps)),
        "completion_rate": sum(all_done) / len(all_done),
    }


def _topdown_frame(race_env) -> np.ndarray:
    """
    Render a top-down (bird's-eye) view of the track with the car overlaid.
    Returns a (300, 450, 3) uint8 numpy array (half of 900×600 screen).
    """
    import pygame
    surf = race_env._env.track.surface.copy()
    x    = int(race_env._env._x)
    y    = int(race_env._env._y)
    ang  = race_env._env._angle

    # Car body — red circle
    pygame.draw.circle(surf, (220, 50, 50), (x, y), 8)
    # Heading arrow — yellow line
    rad = math.radians(ang)
    tip = (int(x + 16 * math.cos(rad)), int(y + 16 * math.sin(rad)))
    pygame.draw.line(surf, (255, 220, 0), (x, y), tip, 3)

    # Scale 900×600 → 450×300
    small = pygame.transform.scale(surf, (450, 300))
    # surfarray gives (W, H, C); transpose to (H, W, C)
    return pygame.surfarray.array3d(small).transpose(1, 0, 2).copy()


@torch.no_grad()
def log_inference_videos(
    model:       PPOActorCritic,
    builder:     CurriculumBuilder,
    device:      torch.device,
    global_step: int,
    video_dir:   str = "inference_videos",
    max_steps:   int = 2000,
    frame_skip:  int = 4,
) -> None:
    """
    Run one greedy episode on every TRAIN track up to the current frontier,
    render top-down frames, save locally as MP4, and log to W&B.

    Called at every --video-interval so you can watch the agent improve over
    time on all tracks it has been trained on.

    Parameters
    ----------
    video_dir  : directory for local MP4 files (auto-created)
    max_steps  : cap the episode at this many steps (default 2000 ≈ 2 laps)
    frame_skip : only capture every Nth frame to keep video size manageable
    """
    import pygame
    import imageio.v3 as iio
    from game.rl_splits import _ensure_pygame, TRAIN
    from env.environment import RaceEnvironment

    _ensure_pygame()
    model.eval()
    os.makedirs(video_dir, exist_ok=True)

    # Render frontier track + all mastered tracks (index 0 … current_level)
    n_tracks = builder.current_level + 1
    video_logs = {}

    for track in TRAIN[:n_tracks]:
        track.build()
        env = RaceEnvironment(track, max_steps=max_steps, laps_target=2, use_image=True)
        obs = env.reset()

        frames = [_topdown_frame(env)]   # capture first frame at reset
        step   = 0

        while not obs.done:
            img, sca = obs_to_tensors(obs, device)
            act, _, _, _ = model.get_action_and_value(img.unsqueeze(0), sca.unsqueeze(0))
            obs  = env.step(DriveAction(
                accel=act.clamp(-1, 1)[0, 0].item(),
                steer=act.clamp(-1, 1)[0, 1].item(),
            ))
            step += 1
            if step % frame_skip == 0:
                frames.append(_topdown_frame(env))

        # (T, H, W, C) uint8
        video = np.stack(frames, axis=0)

        # Save locally as MP4
        track_slug = track.name.replace(' ', '_')
        filename = f"step{global_step:08d}_track{track.level:02d}_{track_slug}.mp4"
        local_path = os.path.join(video_dir, filename)
        iio.imwrite(local_path, video, fps=20, codec="libx264", plugin="pyav")

        key = f"inference/track_{track.level:02d}_{track_slug}"
        video_logs[key] = wandb.Video(video, fps=20, format="mp4")

    video_logs["global_step"] = global_step
    wandb.log(video_logs, step=global_step)
    model.train()
    print(f"  Saved {n_tracks} video(s) to {video_dir}/")


def save_checkpoint(path: str, model, optimizer, global_step: int,
                    curriculum_level: int, args, reward_window, episode_num,
                    sampler_idx, sampler_rewards, sampler_crashes=None,
                    sampler_laps=None, wandb_run_id=None):
    torch.save({
        "step":             global_step,
        "curriculum_level": curriculum_level,
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "args":             vars(args),
        # resume state
        "reward_window":    list(reward_window),
        "episode_num":      episode_num,
        "sampler_idx":      sampler_idx,
        "sampler_rewards":  list(sampler_rewards),
        "sampler_crashes":  list(sampler_crashes) if sampler_crashes is not None else [],
        "sampler_laps":     list(sampler_laps)    if sampler_laps    is not None else [],
        "wandb_run_id":     wandb_run_id,
    }, path)


def load_checkpoint(path: str):
    print(f"\n  [RESUME] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  [RESUME] Resuming from step {ckpt['step']:,}  "
          f"curriculum level {ckpt['curriculum_level']}\n")
    return ckpt


def prune_checkpoints(checkpoint_dir: str, keep: int):
    """Delete oldest checkpoints, keeping only the most recent `keep` files."""
    if keep <= 0:
        return
    import glob as _glob
    pts = sorted(_glob.glob(os.path.join(checkpoint_dir, "ppo_step*.pt")))
    for old in pts[:-keep]:
        os.remove(old)
        print(f"  [PRUNE] Removed old checkpoint: {os.path.basename(old)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── CPU resource maximisation ─────────────────────────────────────────────
    if args.device == "cpu":
        n_cpu = os.cpu_count() or 1
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(max(1, n_cpu // 2))
        print(f"CPU mode: {n_cpu} intra-op threads, {max(1, n_cpu//2)} inter-op threads")

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    N = args.num_envs   # number of parallel environments

    # ── Load checkpoint (if resuming) ─────────────────────────────────────────
    ckpt = load_checkpoint(args.resume) if args.resume else None

    # ── W&B setup ─────────────────────────────────────────────────────────────
    wandb_kwargs = dict(
        project = args.wandb_project,
        name    = args.wandb_run_name,
        config  = vars(args),
        mode    = "offline" if args.wandb_offline else "online",
    )
    if ckpt and ckpt.get("wandb_run_id"):
        wandb_kwargs["id"]     = ckpt["wandb_run_id"]
        wandb_kwargs["resume"] = "allow"

    run = wandb.init(**wandb_kwargs)

    # Custom step axis so all charts share the same x-axis
    wandb.define_metric("global_step")
    for prefix in ("episode", "ppo", "curriculum", "val", "system"):
        wandb.define_metric(f"{prefix}/*", step_metric="global_step")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    if args.checkpoint_interval > 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model     = PPOActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if ckpt:
        # Load on plain (uncompiled) model so key names always match
        state = ckpt["model"]
        # Strip _orig_mod. prefix if checkpoint was saved from a compiled model
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])

    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    wandb.watch(model, log="gradients", log_freq=200)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} parameters  |  Device: {device}  |  Envs: {N}")

    # ── Curriculum ────────────────────────────────────────────────────────────
    builder = CurriculumBuilder(
        threshold   = args.threshold,
        window      = args.window,
        replay_frac = args.replay_frac,
        use_image   = True,
    )

    # Restore curriculum state from checkpoint
    if ckpt:
        builder._sampler._idx = ckpt["sampler_idx"]
        builder._sampler._rewards = deque(ckpt["sampler_rewards"],
                                          maxlen=args.window)
        builder._sampler._crashes = deque(ckpt.get("sampler_crashes", []),
                                          maxlen=args.window)
        builder._sampler._laps    = deque(ckpt.get("sampler_laps", []),
                                          maxlen=args.window)

    # ── Rollout storage — shape (T, N, …) pre-allocated on CPU ───────────────
    T    = args.rollout_steps
    _pin = device.type == "cuda"

    def _buf(*shape):
        t = torch.zeros(*shape)
        return t.pin_memory() if _pin else t

    # Image/scalar stored as numpy — avoids D2H round-trip during collection
    buf_imgs_np    = np.zeros((T, N, 3, 64, 64), dtype=np.float32)
    buf_scalars_np = np.zeros((T, N, 9),         dtype=np.float32)
    buf_actions   = _buf(T, N, 2)
    buf_logps     = _buf(T, N)
    buf_rewards   = _buf(T, N)
    buf_dones     = _buf(T, N)
    buf_values    = _buf(T, N)

    # ── Per-env episode state ─────────────────────────────────────────────────
    if args.subproc:
        vec_env = SubprocVecEnv(N, max_steps=500_000, laps_target=1)
        init_tracks = [builder._sampler.sample() for _ in range(N)]
        init_results = vec_env.reset([t.level for t in init_tracks])
        # Convert _StepResult to a lightweight namespace compatible with obs_list API
        obs_list      = list(init_results)
        current_track = list(init_tracks)
        envs          = None   # not used in subproc mode
    else:
        vec_env       = None
        envs          = [builder.next_env() for _ in range(N)]
        obs_list      = [env.reset()        for env in envs]
        current_track = [env._env.track     for env in envs]

    is_replay = [False] * N

    ep_reward   = [0.0] * N
    ep_length   = [0]   * N
    ep_laps     = [0]   * N
    ep_crashes  = [0]   * N
    ep_on_track = [0]   * N

    # ── Global counters (restored from checkpoint if resuming) ────────────────
    global_step  = ckpt["step"]         if ckpt else 0
    episode_num  = ckpt["episode_num"]  if ckpt else 0
    update_num   = 0
    start_time   = time.time()

    reward_window = deque(
        ckpt["reward_window"] if ckpt else [],
        maxlen=args.window,
    )

    # Next checkpoint threshold (skip past already-saved checkpoints on resume)
    if args.checkpoint_interval > 0:
        next_ckpt = args.checkpoint_interval
        while next_ckpt <= global_step:
            next_ckpt += args.checkpoint_interval
    else:
        next_ckpt = float("inf")

    # Next video threshold (independent of checkpoints)
    if args.video_interval > 0:
        next_video = args.video_interval
        while next_video <= global_step:
            next_video += args.video_interval
    else:
        next_video = float("inf")

    frontier = builder._sampler.frontier_track
    env_mode = f"subproc ×{N}" if args.subproc else f"sequential ×{N}"
    print(f"Total steps : {args.total_steps:,}  (resuming from {global_step:,})" if ckpt
          else f"Total steps : {args.total_steps:,}")
    print(f"Envs        : {env_mode}")
    print(f"Rollout     : {T} steps x {N} envs = {T*N} per update  |  "
          f"Minibatch: {args.minibatch_size}  |  Epochs: {args.ppo_epochs}")
    print(f"Curriculum  : threshold={args.threshold}  window={args.window}  replay={args.replay_frac}")
    print(f"Frontier    : track {frontier.level} '{frontier.name}'")
    print(f"W&B         : {run.url}\n")
    print(f"{'Step':>10}  {'lvl':>5}  {'Name':<20}  {'SPS':>9}  "
          f"{'KL':>10}  {'Entropy':>10}  {'PG loss':>10}  {'VF loss':>8}  GPU")
    print("-" * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    while global_step < args.total_steps:

        # ── Anneal LR and entropy coef linearly ──────────────────────────────
        progress = global_step / args.total_steps          # 0.0 → 1.0
        lr_now   = args.lr + (args.lr_min - args.lr) * progress
        ent_now  = args.ent_coef_start + (args.ent_coef_end - args.ent_coef_start) * progress
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # ── Collect rollout (T steps × N envs) ───────────────────────────────
        model.eval()

        for step in range(T):
            # Batch obs → numpy (CPU only), then single H2D upload for inference
            imgs_np, scas_np = obs_list_to_arrays(obs_list)
            imgs = torch.from_numpy(imgs_np).to(device, non_blocking=True)
            scas = torch.from_numpy(scas_np).to(device, non_blocking=True)

            with torch.no_grad():
                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    actions, logps, _, values = model.get_action_and_value(imgs, scas)

            # Store numpy directly — no D2H round-trip
            buf_imgs_np[step]    = imgs_np
            buf_scalars_np[step] = scas_np
            buf_actions[step] = actions.cpu()
            buf_logps[step]   = logps.cpu()
            buf_values[step]  = values.squeeze(-1).cpu()

            # ── Step envs (parallel subprocess or sequential in-process) ─────
            # Accel is fixed at 1.0: the car always moves forward at full throttle.
            # The policy only controls steering; the accel output dimension is ignored.
            if vec_env is not None:
                steer = actions.clamp(-1.0, 1.0).cpu()
                vec_env.step_async(
                    [(1.0, steer[n, 1].item()) for n in range(N)]
                )
                step_results = vec_env.step_wait()
            else:
                step_results = []
                for n in range(N):
                    steer_n = actions[n, 1].clamp(-1.0, 1.0).item()
                    step_results.append(
                        envs[n].step(
                            DriveAction(accel=1.0, steer=steer_n)
                        )
                    )

            for n, obs in enumerate(step_results):
                obs_list[n]          = obs
                buf_rewards[step, n] = obs.reward
                buf_dones[step, n]   = float(obs.done)

                ep_reward[n]  += obs.reward
                ep_length[n]  += 1
                if obs.metadata:
                    ep_laps[n]    = obs.metadata.get("lap",     ep_laps[n])
                    ep_crashes[n] = obs.metadata.get("crashes", ep_crashes[n])
                    if obs.metadata.get("on_track", True):
                        ep_on_track[n] += 1

                # ── Episode end ──────────────────────────────────────────────
                if obs.done:
                    episode_num     += 1
                    on_track_pct     = ep_on_track[n] / max(ep_length[n], 1) * 100.0
                    reward_window.append(ep_reward[n])
                    rolling_mean     = statistics.mean(reward_window)

                    sampler   = builder._sampler
                    frontier  = sampler.frontier_track
                    threshold = args.threshold * frontier.complexity

                    wandb.log({
                        "global_step":              global_step,
                        "episode/reward":           ep_reward[n],
                        "episode/length":           ep_length[n],
                        "episode/laps":             ep_laps[n],
                        "episode/crashes":          ep_crashes[n],
                        "episode/on_track_pct":     on_track_pct,
                        "episode/number":           episode_num,
                        "curriculum/level":         builder.current_level,
                        "curriculum/track_level":   current_track[n].level,
                        "curriculum/track_name":    current_track[n].name,
                        "curriculum/tier":          difficulty_of(current_track[n]),
                        "curriculum/rolling_mean":   rolling_mean,
                        "curriculum/threshold":      threshold,
                        "curriculum/is_replay":      int(is_replay[n]),
                        "curriculum/crashes_per_ep": builder._sampler.rolling_crashes,
                        "curriculum/laps_per_ep":    builder._sampler.rolling_laps,
                        "curriculum/crash_free":     int(all(c == 0 for c in builder._sampler._crashes)),
                        "curriculum/all_lapped":     int(all(l >= 1 for l in builder._sampler._laps)),
                    }, step=global_step)

                    # ── Curriculum advancement check ─────────────────────────
                    advanced = builder.record(ep_reward[n], ep_crashes[n], ep_laps[n])

                    if advanced:
                        new_frontier = builder._sampler.frontier_track
                        print(f"\n\n  > CURRICULUM ADVANCE ->  "
                              f"Track {new_frontier.level} '{new_frontier.name}'  "
                              f"[level {builder.current_level}/{len(builder._sampler.tracks)-1}]")
                        print(f"    rolling_mean={rolling_mean:.2f}  "
                              f"threshold={threshold:.2f}\n")

                        wandb.log({
                            "global_step":                  global_step,
                            "curriculum/level":             builder.current_level,
                            "curriculum/advanced_to_level": new_frontier.level,
                            "curriculum/advanced_to_name":  new_frontier.name,
                            "curriculum/advanced_to_tier":  difficulty_of(new_frontier),
                        }, step=global_step)

                        # ── Validation after each advance ────────────────────
                        print(f"  Running validation ({args.val_episodes} eps/track)…")
                        val = run_validation(model, builder, args.val_episodes, device)

                        val_log = {
                            "global_step":         global_step,
                            "val/mean_reward":     val["mean_reward"],
                            "val/mean_laps":       val["mean_laps"],
                            "val/completion_rate": val["completion_rate"],
                        }
                        for pt in val["per_track"]:
                            lvl = pt["level"]
                            val_log[f"val/track_{lvl}_reward"]     = pt["mean_reward"]
                            val_log[f"val/track_{lvl}_laps"]       = pt["mean_laps"]
                            val_log[f"val/track_{lvl}_completion"] = pt["completion_rate"]
                        wandb.log(val_log, step=global_step)

                        table = wandb.Table(
                            columns=["track_level", "name", "tier",
                                     "mean_reward", "mean_laps", "completion_%"],
                            data=[
                                [pt["level"], pt["name"], pt["tier"],
                                 round(pt["mean_reward"], 2),
                                 round(pt["mean_laps"],   2),
                                 round(pt["completion_rate"] * 100, 1)]
                                for pt in val["per_track"]
                            ],
                        )
                        wandb.log({"val/per_track_table": table}, step=global_step)

                        print(f"  Val: mean_reward={val['mean_reward']:.1f}  "
                              f"completion={val['completion_rate']*100:.0f}%\n")

                    # ── Reset env for next episode ────────────────────────────
                    ep_reward[n] = ep_length[n] = ep_laps[n] = 0
                    ep_crashes[n] = ep_on_track[n] = 0

                    if vec_env is not None:
                        next_track       = builder._sampler.sample()
                        obs_list[n]      = vec_env.reset_one(n, next_track.level)
                        current_track[n] = next_track
                    else:
                        envs[n]          = builder.next_env()
                        current_track[n] = envs[n]._env.track
                        obs_list[n]      = envs[n].reset()

                    is_replay[n] = (
                        current_track[n].level != builder._sampler.frontier_track.level
                    )

            global_step += N

        # ── Compute advantages with GAE (vectorised over N envs) ─────────────
        with torch.no_grad():
            imgs_next_np, scas_next_np = obs_list_to_arrays(obs_list)
            imgs_next = torch.from_numpy(imgs_next_np).to(device, non_blocking=True)
            scas_next = torch.from_numpy(scas_next_np).to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=(scaler is not None)):
                next_values = model.get_value(imgs_next, scas_next).squeeze(-1).cpu()  # (N,)
            next_dones  = torch.tensor([float(obs_list[n].done) for n in range(N)])

        advantages = torch.zeros(T, N)
        lastgae    = torch.zeros(N)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - next_dones          # (N,)
                nextval         = next_values               # (N,)
            else:
                nextnonterminal = 1.0 - buf_dones[t]        # (N,)
                nextval         = buf_values[t + 1]         # (N,)

            delta       = (buf_rewards[t]
                           + args.gamma * nextval * nextnonterminal
                           - buf_values[t])
            advantages[t] = lastgae = (delta
                                       + args.gamma * args.gae_lambda
                                       * nextnonterminal * lastgae)
        returns = advantages + buf_values

        # ── PPO update — flatten (T, N, …) → (T*N, …) ───────────────────────
        model.train()

        TN     = T * N
        # Single pinned H2D transfer for image/scalar buffers
        b_img  = (torch.from_numpy(buf_imgs_np.reshape(TN, 3, 64, 64))
                  .pin_memory().to(device, non_blocking=True))
        b_sca  = (torch.from_numpy(buf_scalars_np.reshape(TN, 9))
                  .pin_memory().to(device, non_blocking=True))
        b_act  = buf_actions.view(TN, 2).to(device, non_blocking=True)
        b_logp = buf_logps.view(TN).to(device, non_blocking=True)
        b_adv  = advantages.view(TN).to(device, non_blocking=True)
        b_ret  = returns.view(TN).to(device, non_blocking=True)
        b_val  = buf_values.view(TN).to(device, non_blocking=True)

        # Normalise advantages for stable updates
        b_adv  = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        pg_losses, v_losses, entropies   = [], [], []
        approx_kls, clip_fracs           = [], []
        grad_norms                       = []
        early_stopped                    = False

        indices = np.arange(TN)
        for epoch in range(args.ppo_epochs):
            if early_stopped:
                break
            np.random.shuffle(indices)

            for start in range(0, TN, args.minibatch_size):
                mb = indices[start : start + args.minibatch_size]

                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    _, new_logp, ent, new_val = model.get_action_and_value(
                        b_img[mb], b_sca[mb], b_act[mb]
                    )
                    new_val = new_val.squeeze()

                    logratio  = new_logp - b_logp[mb]
                    ratio     = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()

                    if approx_kl > args.target_kl:
                        early_stopped = True
                        break

                    mb_adv   = b_adv[mb]
                    pg_loss  = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps),
                    ).mean()
                    v_loss   = 0.5 * (new_val - b_ret[mb]).pow(2).mean()
                    ent_loss = ent.mean()
                    loss     = pg_loss + args.vf_coef * v_loss - ent_now * ent_loss

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent_loss.item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                grad_norms.append(grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm))

        update_num += 1

        # ── Log PPO + system metrics ──────────────────────────────────────────
        sps = global_step / (time.time() - start_time)
        ev  = explained_variance(b_val, b_ret)

        mean_kl      = float(np.mean(approx_kls))   if approx_kls   else float("nan")
        mean_entropy = float(np.mean(entropies))     if entropies    else float("nan")
        mean_pg      = float(np.mean(pg_losses))     if pg_losses    else float("nan")
        mean_vf      = float(np.mean(v_losses))      if v_losses     else float("nan")
        mean_clip    = float(np.mean(clip_fracs))    if clip_fracs   else float("nan")
        mean_gnorm   = float(np.mean(grad_norms))    if grad_norms   else float("nan")

        frontier_track = builder._sampler.frontier_track
        wandb.log({
            "global_step":              global_step,
            "ppo/policy_loss":          mean_pg,
            "ppo/value_loss":           mean_vf,
            "ppo/entropy":              mean_entropy,
            "ppo/approx_kl":            mean_kl,
            "ppo/clip_fraction":        mean_clip,
            "ppo/explained_variance":   ev,
            "ppo/learning_rate":        lr_now,
            "ppo/entropy_coef":         ent_now,
            "ppo/grad_norm":            mean_gnorm,
            "ppo/update":               update_num,
            "ppo/early_stopped":        int(early_stopped),
            "system/steps_per_sec":     sps,
            "system/elapsed_hours":     (time.time() - start_time) / 3600,
        }, step=global_step)

        # ── Console output every update (live-updating line) ─────────────────
        if device.type == "cuda":
            gpu_mem_used = torch.cuda.memory_allocated(device) / 1024**3
            gpu_mem_tot  = torch.cuda.get_device_properties(device).total_memory / 1024**3
            try:
                gpu_util = torch.cuda.utilization(device)
                gpu_str  = f"  gpu={gpu_util:>3}%  mem={gpu_mem_used:.1f}/{gpu_mem_tot:.0f}GB"
            except Exception:
                gpu_str  = f"  mem={gpu_mem_used:.1f}/{gpu_mem_tot:.0f}GB"
        else:
            gpu_str = ""
        print(
            f"\r{global_step:>10,}  "
            f"lvl={frontier_track.level:>2}  "
            f"{frontier_track.name:<20}  "
            f"{sps:>6.0f}sps  "
            f"kl={mean_kl:.4f}  "
            f"ent={mean_entropy:.3f}  "
            f"pg={mean_pg:+.4f}  "
            f"vf={mean_vf:.4f}"
            f"{gpu_str}",
            end="", flush=True,
        )

        # ── Inference video ────────────────────────────────────────────────────
        if global_step >= next_video:
            print(f"\n  [VIDEO] Rendering inference videos for {builder.current_level + 1} track(s)…")
            log_inference_videos(model, builder, device, global_step)
            print(f"  Videos logged to W&B.\n")
            next_video += args.video_interval

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step >= next_ckpt:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"ppo_step{global_step:08d}_lvl{builder.current_level:02d}.pt",
            )
            save_checkpoint(
                ckpt_path, model, optimizer, global_step,
                builder.current_level, args,
                reward_window, episode_num,
                builder._sampler._idx,
                list(builder._sampler._rewards),
                sampler_crashes=list(builder._sampler._crashes),
                sampler_laps=list(builder._sampler._laps),
                wandb_run_id=run.id,
            )
            wandb.save(ckpt_path)
            print(f"\n\n  [CKPT] {ckpt_path}")
            if args.keep_checkpoints > 0:
                prune_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
            next_ckpt += args.checkpoint_interval

    # ─────────────────────────────────────────────────────────────────────────
    # End
    # ─────────────────────────────────────────────────────────────────────────
    final = os.path.join(args.checkpoint_dir, "ppo_final.pt")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_checkpoint(
        final, model, optimizer, global_step,
        builder.current_level, args,
        reward_window, episode_num,
        builder._sampler._idx,
        list(builder._sampler._rewards),
        sampler_crashes=list(builder._sampler._crashes),
        sampler_laps=list(builder._sampler._laps),
        wandb_run_id=run.id,
    )
    wandb.save(final)

    if vec_env is not None:
        vec_env.close()

    elapsed = time.time() - start_time
    print(f"\n{'-'*82}")
    print(f"Training complete  |  {global_step:,} steps  |  {elapsed/3600:.2f} h")
    print(f"Final model: {final}")
    print(f"W&B run:     {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
