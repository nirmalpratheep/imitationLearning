"""
train_torchrl.py — TorchRL PPO training for the curriculum car racer.

Replaces train_sb3.py with a torchrl-based PPO. All hyperparameters are
transferred exactly from train_sb3.py (SB3 PPO defaults + our overrides):

  learning_rate      3e-4            (Adam, eps=1e-5)  — SB3 default
  rollout frames     2048 / update   (n_steps × n_envs in SB3)
  batch size         64              (minibatch size for PPO updates)
  n_epochs           10              (passes over rollout data)
  gamma              0.99
  gae_lambda         0.95
  clip_epsilon       0.2
  vf_coef            0.5
  ent_coef           0.0             (SB3 default)
  max_grad_norm      0.5
  normalize_adv      True
  target_kl          None            (no early stop)
  log_std_init       -1.0            (initial std ≈ 0.37; SB3 DiagGaussian — no clamp)
  actor mean bias    [0]=0.3         (gentle forward accel)
  ortho init         gain=0.01 (actor) / 1.0 (critic)
  features extractor RaceEncoder     (ImpalaCNN + MLP → 288 dims)
  net_arch           empty           (direct linear heads, no extra MLP)
  share features     True            (encoder params shared across heads)

W&B metrics use identical keys to train_sb3.py.

Usage
─────
  uv run python train_torchrl.py
  uv run python train_torchrl.py --num-envs 8 --total-steps 10_000_000
  uv run python train_torchrl.py --resume checkpoints/ppo_torchrl_step500000.pt
"""

import argparse
import math
import os
import random
import re
import statistics
import sys
import time
from collections import deque

# Ensure project root (parent of training/) is on sys.path so env/ and game/ are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import wandb

# Headless pygame — must come before any game/env import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# UTF-8 stdout so box-drawing glyphs inside tensordict/torchrl banners don't
# explode on Windows cp1252 when wandb wraps stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
import multiprocessing as mp
from torchrl.envs import Compose, GymWrapper, ParallelEnv, StepCounter, TransformedEnv
from torchrl.envs.gym_like import BaseInfoDictReader
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import IndependentNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from env import CurriculumBuilder, DriveAction
from env.encoder import RaceEncoder
from env.gym_env import RaceGymEnv
from game.rl_splits import TRAIN, difficulty_of


# ─────────────────────────────────────────────────────────────────────────────
# Args  (same flags/defaults as train_sb3.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("W&B")
    g.add_argument("--wandb-project",  default="curriculum-car-racer")
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-id",       default=None)
    g.add_argument("--wandb-offline",  action="store_true")

    g = p.add_argument_group("Training budget")
    g.add_argument("--total-steps",   type=int, default=5_000_000)
    g.add_argument("--rollout-steps", type=int, default=2048,
                   help="Total frames per PPO update (across all envs)")
    g.add_argument("--num-envs",      type=int, default=4)
    g.add_argument("--batch-size",    type=int, default=64)
    g.add_argument("--ppo-epochs",    type=int, default=10)

    g = p.add_argument_group("PPO (SB3 defaults)")
    g.add_argument("--lr",            type=float, default=3e-4)
    g.add_argument("--gamma",         type=float, default=0.99)
    g.add_argument("--gae-lambda",    type=float, default=0.95)
    g.add_argument("--clip-eps",      type=float, default=0.2)
    g.add_argument("--vf-coef",       type=float, default=0.5)
    g.add_argument("--ent-coef",      type=float, default=0.0)
    g.add_argument("--max-grad-norm", type=float, default=0.5)
    g.add_argument("--target-kl",     type=float, default=0.1,
                   help="Stop PPO epochs early when approx_kl exceeds this")

    g = p.add_argument_group("Curriculum")
    g.add_argument("--threshold",    type=float, default=30.0)
    g.add_argument("--window",       type=int,   default=50)
    g.add_argument("--replay-frac",  type=float, default=0.3)

    g = p.add_argument_group("Checkpointing")
    g.add_argument("--checkpoint-interval", type=int, default=500_000)
    g.add_argument("--checkpoint-dir",      default="checkpoints")
    g.add_argument("--keep-checkpoints",    type=int, default=5)
    g.add_argument("--resume",              default=None)

    g = p.add_argument_group("Misc")
    g.add_argument("--seed",           type=int, default=42)
    g.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--compile",        action="store_true")
    g.add_argument("--video-interval", type=int, default=25_000)
    g.add_argument("--video-dir",      default="inference_videos")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Shared encoder + actor/critic heads
#
# RaceEncoder (ImpalaCNN + scalar MLP → 288) is shared between the actor and
# critic, matching SB3's share_features_extractor=True. The actor and critic
# each run the encoder once per forward (torchrl's PPO evaluates them
# separately); parameter sharing gives identical gradients to SB3.
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_batch_dims(image: torch.Tensor, scalars: torch.Tensor):
    """
    Collapse all leading batch dimensions into one so Conv2d gets a 4D tensor.
    Returns (img_flat, sca_flat, lead_shape) where lead_shape is used to
    restore the original batch structure on outputs.

    RaceEncoder expects image (B, 3, 64, 64) and scalars (B, 9). During PPO
    loss/GAE, torchrl hands us (N, T, 3, 64, 64) / (N, T, 9); during rollout
    collection it's (N, 3, 64, 64) / (N, 9). Flatten uniformly.
    """
    lead_shape = image.shape[:-3]       # everything except C,H,W
    img_flat = image.reshape(-1, *image.shape[-3:])
    sca_flat = scalars.reshape(-1, scalars.shape[-1])
    return img_flat, sca_flat, lead_shape


class _ActorNet(nn.Module):
    """Image + scalars → (loc, scale) for IndependentNormal."""

    def __init__(self, encoder: RaceEncoder, log_std_init: float = -1.0):
        super().__init__()
        self.encoder = encoder
        self.mean    = nn.Linear(encoder.out_features, 2)
        # log_std is a free parameter (not state-conditioned), matching SB3's
        # DiagGaussianDistribution. Unbounded — SB3 does not clamp log_std.
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))

        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        # Gentle forward accel so the car stays moving while exploring.
        with torch.no_grad():
            self.mean.bias[0] = 0.3

    def forward(self, image: torch.Tensor, scalars: torch.Tensor):
        img_f, sca_f, lead = _flatten_batch_dims(image, scalars)
        feat  = self.encoder(img_f, sca_f)
        loc   = self.mean(feat).reshape(*lead, 2)
        scale = self.log_std.exp().expand_as(loc)
        return loc, scale


class _CriticNet(nn.Module):
    """Image + scalars → value (1,)."""

    def __init__(self, encoder: RaceEncoder):
        super().__init__()
        self.encoder = encoder
        self.value   = nn.Linear(encoder.out_features, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, image: torch.Tensor, scalars: torch.Tensor):
        img_f, sca_f, lead = _flatten_batch_dims(image, scalars)
        v = self.value(self.encoder(img_f, sca_f))
        return v.reshape(*lead, 1)


def _sb3_ortho_init(module: nn.Module, gain: float) -> None:
    """Mirror SB3's ActorCriticPolicy.init_weights(gain) via module.apply:
    orthogonal-init every Conv2d/Linear weight with the given gain and zero
    biases. SB3 applies gain=sqrt(2) to the features extractor."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_policy_and_value(device: torch.device):
    """Build actor + critic modules with a shared RaceEncoder (shared params)."""
    encoder = RaceEncoder()
    _sb3_ortho_init(encoder, gain=math.sqrt(2))  # SB3 ortho_init=True on features extractor
    actor_net  = _ActorNet(encoder)     # head overrides encoder-wide init on its own layers
    critic_net = _CriticNet(encoder)

    actor_tdm = TensorDictModule(
        actor_net,
        in_keys=["image", "scalars"],
        out_keys=["loc", "scale"],
    )
    policy_module = ProbabilisticActor(
        module             = actor_tdm,
        in_keys            = ["loc", "scale"],
        out_keys           = ["action"],
        distribution_class = IndependentNormal,
        return_log_prob    = True,
    ).to(device)

    value_module = ValueOperator(
        module   = critic_net,
        in_keys  = ["image", "scalars"],
        out_keys = ["state_value"],
    ).to(device)

    return policy_module, value_module, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

class _EpisodeStatsReader(BaseInfoDictReader):
    """
    BaseInfoDictReader subclass so set_info_dict_reader() registers the keys in
    GymWrapper.observation_spec — required for ParallelEnv to allocate shared
    memory and transfer these values from subprocess to main process.
    """
    info_spec = Composite(
        episode_laps    = Unbounded((), dtype=torch.float32),
        episode_crashes = Unbounded((), dtype=torch.float32),
        on_track_pct    = Unbounded((), dtype=torch.float32),
        track_level     = Unbounded((), dtype=torch.float32),
    )

    def __call__(self, info, td):
        td["episode_laps"]    = torch.tensor(info.get("episode_laps",    0),   dtype=torch.float32)
        td["episode_crashes"] = torch.tensor(info.get("episode_crashes", 0),   dtype=torch.float32)
        td["on_track_pct"]    = torch.tensor(info.get("on_track_pct",    0.0), dtype=torch.float32)
        td["track_level"]     = torch.tensor(info.get("track_level",     0),   dtype=torch.float32)

    def reset(self, tensordict_reset=None):
        pass


def make_vec_env(num_envs: int, max_steps: int, laps_target: int,
                 replay_frac: float, device, shared_level: mp.Value):
    """
    ParallelEnv of GymWrapper(RaceGymEnv) — each env runs in its own subprocess
    for parallel CPU stepping. Frontier level is shared via a multiprocessing.Value
    so curriculum advances in the main process propagate instantly to all workers.
    """
    def _factory():
        gym_env = RaceGymEnv(
            sampler        = None,
            frontier_level = 0,
            replay_frac    = replay_frac,
            max_steps      = max_steps,
            laps_target    = laps_target,
            shared_level   = shared_level,
        )
        wrapped = GymWrapper(gym_env, device="cpu")
        wrapped.set_info_dict_reader(_EpisodeStatsReader())
        return wrapped

    base = ParallelEnv(num_envs, _factory, mp_start_method="fork")
    return TransformedEnv(base, Compose(StepCounter(), RewardSum()))


# ─────────────────────────────────────────────────────────────────────────────
# Inference video (frontier track only — same as train_sb3.py)
# ─────────────────────────────────────────────────────────────────────────────

def _game_frame(race_env) -> np.ndarray:
    import pygame
    from game.oval_racer import draw_car, draw_headlights

    ce   = race_env._env
    surf = ce.track.surface.copy()
    draw_headlights(surf, ce._x, ce._y, ce._angle)
    draw_car(surf, ce._x, ce._y, ce._angle)
    small = pygame.transform.scale(surf, (450, 300))
    return pygame.surfarray.array3d(small).transpose(1, 0, 2).copy()


@torch.no_grad()
def log_inference_videos(
    policy_module,
    builder: CurriculumBuilder,
    device: torch.device,
    global_step: int,
    video_dir: str = "inference_videos",
    frame_skip: int = 2,
) -> None:
    import imageio.v3 as iio
    from env.environment import RaceEnvironment
    from game.rl_splits import _ensure_pygame

    _ensure_pygame()
    os.makedirs(video_dir, exist_ok=True)

    policy_module.eval()
    video_logs = {}

    frontier_track = TRAIN[builder.current_level]
    for track in [frontier_track]:
        track.build()
        env      = RaceEnvironment(track, max_steps=3000, laps_target=1, use_image=True)
        raw_obs  = env.reset()
        frames   = [_game_frame(env)]
        step     = 0

        while not raw_obs.done:
            img = (torch.from_numpy(raw_obs.image.copy())
                   .float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device))
            scalars = torch.tensor(raw_obs.scalars, dtype=torch.float32,
                                   device=device).unsqueeze(0)

            td = TensorDict({"image": img, "scalars": scalars}, batch_size=[1])
            with set_exploration_type(ExplorationType.MEAN):
                td = policy_module(td)
            action = td.get("action")[0].clamp(-1.0, 1.0).cpu().numpy()

            raw_obs = env.step(DriveAction(
                accel=float(action[0]), steer=float(action[1])
            ))
            step += 1
            if step % frame_skip == 0:
                frames.append(_game_frame(env))

        video      = np.stack(frames, axis=0)  # (T, 300, 450, 3) uint8
        track_slug = track.name.replace(" ", "_")
        filename   = f"step{global_step:08d}_track{track.level:02d}_{track_slug}.mp4"
        iio.imwrite(os.path.join(video_dir, filename),
                    video, fps=20, codec="libx264", plugin="pyav")

        key = f"inference/track_{track.level:02d}_{track_slug}"
        video_logs[key] = wandb.Video(video, fps=20, format="mp4")

    wandb.log({**video_logs, "global_step": global_step}, step=global_step)
    policy_module.train()
    print(f"  [VIDEO] Saved frontier track video to {video_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Episode iteration over a collected rollout
# ─────────────────────────────────────────────────────────────────────────────

def _iter_episodes(td):
    """
    Yield (env_idx, step_idx, episode_reward, episode_length, info_dict) for
    every terminal step in the collected rollout.

    td has shape (N, T) with the standard torchrl layout:
      td["next","done"]          → (N, T, 1)
      td["next","episode_reward"]→ (N, T, 1)   (from RewardSum)
      td["next","step_count"]    → (N, T, 1)   (from StepCounter)
      td["next","episode_laps"]  → (N, T)      (from RaceGymEnv info, if present)
    RaceGymEnv writes episode_laps/episode_crashes/on_track_pct only on done.
    """
    next_td = td.get("next")
    dones   = next_td.get("done").squeeze(-1)                # (N, T)  bool
    ep_r    = next_td.get("episode_reward").squeeze(-1)      # (N, T)  float
    ep_l    = next_td.get("step_count").squeeze(-1)          # (N, T)  int

    # Info fields from RaceGymEnv (only populated on done steps). If they were
    # never observed, these keys will not exist — tolerate that.
    def _get(key, default):
        if key in next_td.keys():
            v = next_td.get(key)
            return v.squeeze(-1) if v.dim() > dones.dim() else v
        return torch.full_like(ep_r, float(default))

    ep_crashes   = _get("episode_crashes", 0).to(torch.float32)
    ep_laps_info = _get("episode_laps",    0).to(torch.float32)
    on_track     = _get("on_track_pct",    0.0).to(torch.float32)
    track_level  = _get("track_level",     0).to(torch.float32)
    track_name   = next_td.get("track_name", None)  # may be bytes/str tensor

    N, T = dones.shape
    for n in range(N):
        for t in range(T):
            if not bool(dones[n, t]):
                continue
            yield {
                "env_idx":      n,
                "step_idx":     t,
                "ep_reward":    float(ep_r[n, t]),
                "ep_length":    int(ep_l[n, t]),
                "ep_crashes":   int(ep_crashes[n, t]),
                "ep_laps":      int(ep_laps_info[n, t]),
                "on_track_pct": float(on_track[n, t]),
                "track_level":  int(track_level[n, t]),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, policy_module, value_module, optimizer,
                    global_step, builder, args, reward_window, episode_num,
                    wandb_run_id):
    torch.save({
        "step":             global_step,
        "curriculum_level": builder.current_level,
        "policy":           policy_module.state_dict(),
        "value":            value_module.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "args":             vars(args),
        "reward_window":    list(reward_window),
        "episode_num":      episode_num,
        "sampler_idx":      builder._sampler._idx,
        "sampler_rewards":  list(builder._sampler._rewards),
        "sampler_crashes":  list(builder._sampler._crashes),
        "sampler_laps":     list(builder._sampler._laps),
        "wandb_run_id":     wandb_run_id,
    }, path)


def prune_checkpoints(checkpoint_dir: str, keep: int):
    if keep <= 0:
        return
    import glob as _glob
    pts = sorted(_glob.glob(os.path.join(checkpoint_dir, "ppo_torchrl_step*.pt")))
    for old in pts[:-keep]:
        os.remove(old)
        print(f"  [PRUNE] Removed {os.path.basename(old)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cudnn.deterministic    = False
        torch.set_float32_matmul_precision("high")  # TF32 on A10 tensor cores

    # ── Auto-detect latest checkpoint when resuming a W&B run ────────────────
    if args.wandb_id and not args.resume:
        import glob as _glob
        ckpts = sorted(_glob.glob(os.path.join(args.checkpoint_dir, "ppo_torchrl_step*.pt")))
        if ckpts:
            args.resume = ckpts[-1]
            print(f"  [RESUME] Auto-detected checkpoint: {args.resume}")

    ckpt = None
    if args.resume:
        print(f"\n  [RESUME] Loading {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        print(f"  [RESUME] From step {ckpt['step']:,}  lvl {ckpt['curriculum_level']}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_kwargs = dict(
        project          = args.wandb_project,
        name             = args.wandb_run_name,
        config           = vars(args),
        mode             = "offline" if args.wandb_offline else "online",
        sync_tensorboard = False,
    )
    if args.wandb_id:
        wandb_kwargs["id"]     = args.wandb_id
        wandb_kwargs["resume"] = "must"
    elif ckpt and ckpt.get("wandb_run_id"):
        wandb_kwargs["id"]     = ckpt["wandb_run_id"]
        wandb_kwargs["resume"] = "allow"
    run = wandb.init(**wandb_kwargs)

    wandb.define_metric("global_step")
    for prefix in ("episode", "ppo", "curriculum", "val", "system"):
        wandb.define_metric(f"{prefix}/*", step_metric="global_step")

    # ── Curriculum ────────────────────────────────────────────────────────────
    builder = CurriculumBuilder(
        threshold   = args.threshold,
        window      = args.window,
        replay_frac = args.replay_frac,
        use_image   = True,
    )
    if ckpt:
        builder._sampler._idx     = ckpt["sampler_idx"]
        builder._sampler._rewards = deque(ckpt["sampler_rewards"], maxlen=args.window)
        builder._sampler._crashes = deque(ckpt.get("sampler_crashes", []), maxlen=args.window)
        builder._sampler._laps    = deque(ckpt.get("sampler_laps",    []), maxlen=args.window)

    sampler = builder._sampler

    # ── Environment (torchrl) ─────────────────────────────────────────────────
    N = args.num_envs
    shared_level = mp.Value("i", builder.current_level)
    vec_env = make_vec_env(
        num_envs     = N,
        max_steps    = 3000,
        laps_target  = 1,
        replay_frac  = args.replay_frac,
        device       = device,
        shared_level = shared_level,
    )
    vec_env.set_seed(args.seed)

    # ── Policy + value ────────────────────────────────────────────────────────
    policy_module, value_module, encoder = build_policy_and_value(device)

    if ckpt:
        policy_module.load_state_dict(ckpt["policy"])
        value_module.load_state_dict(ckpt["value"])

    # Sanity: run once through reset so specs match
    with torch.no_grad():
        td0 = vec_env.reset().to(device)
        policy_module(td0)
        value_module(td0)

    # ── Loss + optimiser ──────────────────────────────────────────────────────
    advantage_module = GAE(
        gamma          = args.gamma,
        lmbda          = args.gae_lambda,
        value_network  = value_module,
        average_gae    = False,
    )
    loss_module = ClipPPOLoss(
        actor_network       = policy_module,
        critic_network      = value_module,
        clip_epsilon        = args.clip_eps,
        entropy_bonus       = True,           # always compute entropy term
        entropy_coeff       = args.ent_coef,  # 0.0 ⇒ SB3 default (no bonus)
        critic_coeff        = args.vf_coef,
        loss_critic_type    = "l2",
        normalize_advantage = True,
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr, eps=1e-5)
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if args.compile:
        try:
            policy_module = torch.compile(policy_module, mode="default")
            print("torch.compile enabled on policy (default)")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    total_params = (
        sum(p.numel() for p in policy_module.parameters() if p.requires_grad)
        + sum(p.numel() for p in value_module.parameters()  if p.requires_grad
              # exclude shared encoder params (already counted in policy)
              and not any(p is q for q in encoder.parameters()))
    )

    # ── Collector ─────────────────────────────────────────────────────────────
    collector = Collector(
        vec_env,
        policy_module,
        frames_per_batch = args.rollout_steps,
        total_frames     = args.total_steps,
        device           = device,
        storing_device   = device,
        reset_at_each_iter = False,
    )

    # ── Replay buffer for PPO minibatches ─────────────────────────────────────
    replay = ReplayBuffer(
        storage = LazyTensorStorage(args.rollout_steps, device=device),
        sampler = SamplerWithoutReplacement(),
        batch_size = args.batch_size,
    )

    # ── Counters ──────────────────────────────────────────────────────────────
    global_step   = ckpt["step"]         if ckpt else 0
    episode_num   = ckpt["episode_num"]  if ckpt else 0
    reward_window = deque(ckpt["reward_window"] if ckpt else [], maxlen=args.window)
    update_num    = 0
    start_time    = time.time()

    LOG_INTERVAL  = 25_000
    next_log      = LOG_INTERVAL
    while next_log <= global_step:
        next_log += LOG_INTERVAL
    # Accumulators for the current log window
    _log_ep_rewards:  list[float] = []
    _log_ep_lengths:  list[int]   = []

    if args.checkpoint_interval > 0:
        next_ckpt = args.checkpoint_interval
        while next_ckpt <= global_step:
            next_ckpt += args.checkpoint_interval
    else:
        next_ckpt = float("inf")

    if args.video_interval > 0:
        next_video = args.video_interval
        while next_video <= global_step:
            next_video += args.video_interval
    else:
        next_video = float("inf")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nModel: {total_params:,} parameters  |  Device: {device}  |  Envs: {N}")
    print(f"Rollout: {args.rollout_steps} frames per update  (batch={args.batch_size}, epochs={args.ppo_epochs})")
    print(f"PPO:     lr={args.lr}  gamma={args.gamma}  lambda={args.gae_lambda}  clip={args.clip_eps}")
    print(f"         vf={args.vf_coef}  ent={args.ent_coef}  grad={args.max_grad_norm}")
    print(f"Curriculum: threshold={args.threshold}  window={args.window}  replay={args.replay_frac}")
    print(f"Frontier  : track {sampler.frontier_track.level} '{sampler.frontier_track.name}'")
    print(f"W&B       : {run.url}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    print("Starting training loop...", flush=True)
    for td in collector:
        rollout_frames = td.numel()
        global_step   += rollout_frames
        update_num    += 1

        # ── Episode bookkeeping + curriculum advance ─────────────────────────
        for ep in _iter_episodes(td):
            episode_num += 1
            reward_window.append(ep["ep_reward"])
            _log_ep_rewards.append(ep["ep_reward"])
            _log_ep_lengths.append(ep["ep_length"])
            rolling_mean = statistics.mean(reward_window) if reward_window else 0.0

            frontier   = sampler.frontier_track
            threshold  = args.threshold * frontier.complexity
            advanced   = builder.record(ep["ep_reward"], ep["ep_crashes"], ep["ep_laps"])
            is_replay  = (ep["track_level"] != frontier.level)

            wandb.log({
                "global_step":              global_step,
                "episode/reward":           ep["ep_reward"],
                "episode/length":           ep["ep_length"],
                "episode/laps":             ep["ep_laps"],
                "episode/crashes":          ep["ep_crashes"],
                "episode/on_track_pct":     ep["on_track_pct"],
                "episode/number":           episode_num,
                "curriculum/level":         builder.current_level,
                "curriculum/track_level":   ep["track_level"],
                "curriculum/tier":          difficulty_of(frontier),
                "curriculum/rolling_mean":  rolling_mean,
                "curriculum/threshold":     threshold,
                "curriculum/is_replay":     int(is_replay),
            }, step=global_step)

            if advanced:
                shared_level.value = builder.current_level  # propagate to all worker envs
                new_frontier = sampler.frontier_track
                print(
                    f"\n  >>> CURRICULUM ADVANCE -> "
                    f"Track {new_frontier.level} '{new_frontier.name}'  "
                    f"[lvl {builder.current_level}/{len(TRAIN)-1}]  "
                    f"rolling_mean={rolling_mean:.2f}  threshold={threshold:.2f}\n",
                    flush=True,
                )
                wandb.log({
                    "global_step":                  global_step,
                    "curriculum/level":             builder.current_level,
                    "curriculum/advanced_to_level": new_frontier.level,
                    "curriculum/advanced_to_name":  new_frontier.name,
                    "curriculum/advanced_to_tier":  difficulty_of(new_frontier),
                }, step=global_step)

        # ── Compute GAE advantages & targets, then flatten for PPO ───────────
        with torch.no_grad():
            advantage_module(td)

        data_flat = td.reshape(-1)
        replay.extend(data_flat)

        # ── PPO update: n_epochs × minibatches ───────────────────────────────
        pg_losses, v_losses, ent_losses = [], [], []
        approx_kls, clip_fracs, grad_norms = [], [], []

        for epoch in range(args.ppo_epochs):
            for _ in range(args.rollout_steps // args.batch_size):
                mb = replay.sample()
                loss_vals = loss_module(mb)

                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals.get("loss_critic",  torch.tensor(0.0, device=device))
                    + loss_vals.get("loss_entropy", torch.tensor(0.0, device=device))
                )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), args.max_grad_norm
                )
                optimizer.step()

                pg_losses.append(loss_vals["loss_objective"].detach().item())
                if "loss_critic" in loss_vals:
                    v_losses.append(loss_vals["loss_critic"].detach().item())
                if "loss_entropy" in loss_vals:
                    ent_losses.append(loss_vals["loss_entropy"].detach().item())
                if "kl_approx" in loss_vals:
                    approx_kls.append(loss_vals["kl_approx"].detach().item())
                if "clip_fraction" in loss_vals:
                    clip_fracs.append(loss_vals["clip_fraction"].detach().item())
                grad_norms.append(float(grad_norm))

            # After each epoch, re-shuffle for next epoch
            replay.empty()
            replay.extend(data_flat)

            # Early stop if policy has moved too far from rollout data
            if args.target_kl is not None and approx_kls:
                epoch_kl = float(np.mean(approx_kls))
                if epoch_kl > args.target_kl:
                    break

        replay.empty()

        # ── Explained variance ────────────────────────────────────────────────
        with torch.no_grad():
            values  = td.get("state_value").reshape(-1)
            returns = td.get(("next", "value_target")).reshape(-1) \
                if ("next", "value_target") in td.keys(include_nested=True) \
                else td.get("value_target").reshape(-1)
            var_y = torch.var(returns)
            ev = float("nan") if var_y == 0 else float(
                1.0 - torch.var(returns - values) / var_y
            )

        # ── Log PPO + system metrics ──────────────────────────────────────────
        sps = global_step / max(time.time() - start_time, 1e-6)
        _mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        wandb.log({
            "global_step":             global_step,
            "ppo/policy_loss":         _mean(pg_losses),
            "ppo/value_loss":          _mean(v_losses),
            "ppo/entropy":             _mean(ent_losses),
            "ppo/approx_kl":           _mean(approx_kls),
            "ppo/clip_fraction":       _mean(clip_fracs),
            "ppo/explained_variance":  ev,
            "ppo/learning_rate":       args.lr,
            "ppo/entropy_coef":        args.ent_coef,
            "ppo/grad_norm":           _mean(grad_norms),
            "ppo/update":              update_num,
            "system/steps_per_sec":    sps,
            "system/elapsed_hours":    (time.time() - start_time) / 3600,
        }, step=global_step)

        # ── Periodic SB3-style summary ────────────────────────────────────────
        if global_step >= next_log:
            ep_rew_mean = float(np.mean(_log_ep_rewards)) if _log_ep_rewards else float("nan")
            ep_len_mean = float(np.mean(_log_ep_lengths)) if _log_ep_lengths else float("nan")
            sps_now     = global_step / max(time.time() - start_time, 1e-6)
            frontier    = sampler.frontier_track
            win_clean   = sum(1 for l, c in zip(sampler._laps, sampler._crashes)
                              if l >= 1 and c == 0)

            def _fmt(v, fmt=".3g"):
                return ("-" if v != v else format(v, fmt))  # nan → "-"

            rows = [
                ("rollout/",             ""),
                ("   ep_len_mean",        _fmt(ep_len_mean, ".1f")),
                ("   ep_rew_mean",        _fmt(ep_rew_mean, ".3f")),
                ("   episodes",           str(episode_num)),
                ("curriculum/",          ""),
                ("   level",              f"{builder.current_level}/{len(TRAIN)-1}"),
                ("   frontier_track",     f"{frontier.level} '{frontier.name}'"),
                ("   rolling_mean",       _fmt(rolling_mean, ".2f")),
                ("   clean_wins",         f"{win_clean}/{args.window}"),
                ("time/",                ""),
                ("   fps",                _fmt(sps_now, ".0f")),
                ("   iterations",         str(update_num)),
                ("   total_timesteps",    f"{global_step:,}"),
                ("train/",               ""),
                ("   approx_kl",          _fmt(_mean(approx_kls))),
                ("   clip_fraction",      _fmt(_mean(clip_fracs))),
                ("   entropy_loss",       _fmt(_mean(ent_losses))),
                ("   explained_variance", _fmt(ev)),
                ("   learning_rate",      _fmt(args.lr)),
                ("   policy_grad_loss",   _fmt(_mean(pg_losses))),
                ("   value_loss",         _fmt(_mean(v_losses))),
                ("   grad_norm",          _fmt(_mean(grad_norms))),
            ]
            col_w = max(len(k) for k, _ in rows) + 2
            val_w = max((len(v) for _, v in rows if v), default=6) + 2
            sep   = "-" * (col_w + val_w + 5)
            print(sep)
            for k, v in rows:
                if v:
                    print(f"| {k:<{col_w}} | {v:>{val_w}} |")
                else:
                    print(f"| {k:<{col_w+val_w+3}} |")
            print(sep, flush=True)

            _log_ep_rewards.clear()
            _log_ep_lengths.clear()
            next_log += LOG_INTERVAL

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step >= next_ckpt:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"ppo_torchrl_step{global_step:08d}_lvl{builder.current_level:02d}.pt",
            )
            save_checkpoint(
                ckpt_path, policy_module, value_module, optimizer,
                global_step, builder, args,
                reward_window, episode_num, run.id,
            )
            wandb.save(ckpt_path)
            print(f"\n  [CKPT] {ckpt_path}")
            prune_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
            next_ckpt += args.checkpoint_interval

        # ── Video ─────────────────────────────────────────────────────────────
        if global_step >= next_video:
            try:
                log_inference_videos(
                    policy_module = policy_module,
                    builder       = builder,
                    device        = device,
                    global_step   = global_step,
                    video_dir     = args.video_dir,
                )
            except Exception as e:
                print(f"  [VIDEO] Warning: failed to render video: {e}")
            next_video += args.video_interval

        # ── Update collector's policy weights (in case of compile) ───────────
        collector.update_policy_weights_()

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final = os.path.join(args.checkpoint_dir, "ppo_torchrl_final.pt")
    save_checkpoint(
        final, policy_module, value_module, optimizer,
        global_step, builder, args,
        reward_window, episode_num, run.id,
    )
    wandb.save(final)

    collector.shutdown()
    elapsed = time.time() - start_time
    print(f"\n{'-'*80}")
    print(f"Training complete  |  {global_step:,} steps  |  {elapsed/3600:.2f} h")
    print(f"Final model: {final}")
    print(f"W&B run:     {run.url}")
    run.finish()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
