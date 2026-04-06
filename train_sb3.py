"""
train_sb3.py — SB3 PPO training for the curriculum car racer.

Replaces the custom CleanRL-style train.py with Stable-Baselines3 PPO.
Uses a custom ImpalaCNN+MLP features extractor (RaceEncoder) and a
curriculum callback that advances the track frontier exactly as before.

W&B metrics logged (same keys as train.py where possible)
─────────────────────────────────────────────────────────
  episode/reward          total reward per episode
  episode/length          steps in the episode
  episode/laps            laps completed
  episode/crashes         on→off track transitions
  episode/on_track_pct    % of steps spent on track

  ppo/policy_loss         clipped surrogate loss
  ppo/value_loss          MSE value loss
  ppo/entropy             policy entropy
  ppo/approx_kl           Schulman KL approximation
  ppo/clip_fraction       fraction of ratios clipped
  ppo/explained_variance  how well value fn predicts returns
  ppo/learning_rate       current LR
  ppo/entropy_coef        current entropy coefficient
  ppo/grad_norm           gradient norm before clipping

  curriculum/level        0-based frontier index within TRAIN
  curriculum/track_level  TrackDef.level (1-20)
  curriculum/track_name   TrackDef.name
  curriculum/rolling_mean mean reward over last `window` episodes
  curriculum/threshold    effective threshold for the current track

  system/steps_per_sec    throughput
  global_step             current env step count

Usage
─────
  uv run python train_sb3.py
  uv run python train_sb3.py --num-envs 8 --total-steps 10_000_000
  uv run python train_sb3.py --resume checkpoints/ppo_sb3_step500000.zip
"""

import argparse
import math
import os
import statistics
import time
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import wandb

# Headless pygame — must come before any game/env import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import CurriculumBuilder, DriveAction          # noqa: E402
from env.encoder import RaceEncoder                     # noqa: E402
from env.gym_env import RaceGymEnv                      # noqa: E402
from game.rl_splits import CurriculumSampler, TRAIN, VAL, difficulty_of  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("W&B")
    g.add_argument("--wandb-project",  default="curriculum-car-racer")
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-offline",  action="store_true")

    g = p.add_argument_group("Training budget")
    g.add_argument("--total-steps",   type=int,   default=5_000_000)
    g.add_argument("--rollout-steps", type=int,   default=2048,
                   help="Steps per env per PPO update")
    g.add_argument("--num-envs",      type=int,   default=4)

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
                   help="Entropy coef linearly annealed to this value")
    g.add_argument("--max-grad-norm",  type=float, default=0.5)
    g.add_argument("--target-kl",      type=float, default=0.02)

    g = p.add_argument_group("Curriculum")
    g.add_argument("--threshold",    type=float, default=30.0)
    g.add_argument("--window",       type=int,   default=50)
    g.add_argument("--replay-frac",  type=float, default=0.3)
    g.add_argument("--val-episodes", type=int,   default=10)

    g = p.add_argument_group("Checkpointing")
    g.add_argument("--checkpoint-interval", type=int, default=500_000)
    g.add_argument("--checkpoint-dir",      default="checkpoints")
    g.add_argument("--keep-checkpoints",    type=int, default=5)
    g.add_argument("--resume",              default=None,
                   help="Path to a .zip SB3 checkpoint to resume from")

    g = p.add_argument_group("Misc")
    g.add_argument("--seed",           type=int, default=42)
    g.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--subproc",        action="store_true",
                   help="Use SubprocVecEnv (parallel subprocesses). "
                        "Note: curriculum state is managed in the main process.")
    g.add_argument("--video-interval", type=int, default=25_000,
                   help="Log inference videos to W&B every N global steps (0 = disabled)")
    g.add_argument("--video-dir",      default="inference_videos")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Custom features extractor (SB3 interface over RaceEncoder)
# ─────────────────────────────────────────────────────────────────────────────

class RaceNetExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor that wraps RaceEncoder.

    Input : Dict obs with keys 'image' (B,3,64,64) and 'scalars' (B,9)
    Output: (B, 288) feature vector for actor/critic heads.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 288):
        super().__init__(observation_space, features_dim)
        self.encoder = RaceEncoder()  # ImpalaCNN + scalar MLP → 288-d

        # Bias the initial policy toward gentle forward acceleration (same as
        # train.py).  We can't easily access actor_mean.bias here, but SB3
        # handles ortho-init; the bias fix is applied in the callback setup.

    def forward(self, observations: dict) -> torch.Tensor:
        img     = observations["image"]     # (B, 3, 64, 64) float32 in [0,1]
        scalars = observations["scalars"]   # (B, 9)
        return self.encoder(img, scalars)


# ─────────────────────────────────────────────────────────────────────────────
# Inference video helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topdown_frame(race_env) -> np.ndarray:
    """Render a 450×300 top-down view with car position + heading overlay."""
    import pygame
    surf = race_env._env.track.surface.copy()
    x   = int(race_env._env._x)
    y   = int(race_env._env._y)
    ang = race_env._env._angle
    pygame.draw.circle(surf, (220, 50, 50), (x, y), 8)
    rad = math.radians(ang)
    tip = (int(x + 16 * math.cos(rad)), int(y + 16 * math.sin(rad)))
    pygame.draw.line(surf, (255, 220, 0), (x, y), tip, 3)
    small = pygame.transform.scale(surf, (450, 300))
    return pygame.surfarray.array3d(small).transpose(1, 0, 2).copy()


@torch.no_grad()
def log_inference_videos(
    sb3_model,
    builder:    CurriculumBuilder,
    device:     torch.device,
    global_step: int,
    video_dir:  str = "inference_videos",
    max_steps:  int = 2000,
    frame_skip: int = 4,
) -> None:
    """
    Run one greedy episode on every TRAIN track up to the current frontier,
    render top-down frames, save locally as MP4, and log to W&B.

    Parameters
    ----------
    sb3_model   : SB3 PPO model (policy used for predict())
    builder     : CurriculumBuilder (for current_level + track list)
    global_step : current env step count (used for filenames + W&B x-axis)
    video_dir   : local directory for MP4 files (auto-created)
    max_steps   : cap each episode at this many steps
    frame_skip  : only record every Nth frame to keep file sizes manageable
    """
    import imageio.v3 as iio
    from game.rl_splits import _ensure_pygame
    from env.environment import RaceEnvironment

    _ensure_pygame()
    os.makedirs(video_dir, exist_ok=True)

    sb3_model.policy.set_training_mode(False)
    n_tracks = builder.current_level + 1
    video_logs = {}

    for track in TRAIN[:n_tracks]:
        track.build()
        env = RaceEnvironment(track, max_steps=max_steps, laps_target=2, use_image=True)
        raw_obs = env.reset()

        frames = [_topdown_frame(env)]
        step   = 0

        while not raw_obs.done and step < max_steps:
            # Build the dict observation the policy expects
            img     = raw_obs.image.transpose(2, 0, 1).astype(np.float32) / 255.0
            scalars = np.array(raw_obs.scalars, dtype=np.float32)
            obs_dict = {
                "image":   img[None],      # (1, 3, 64, 64)
                "scalars": scalars[None],  # (1, 9)
            }
            action, _ = sb3_model.predict(obs_dict, deterministic=True)
            accel = float(np.clip(action[0, 0], -1.0, 1.0))
            steer = float(np.clip(action[0, 1], -1.0, 1.0))
            raw_obs = env.step(DriveAction(accel=accel, steer=steer))

            step += 1
            if step % frame_skip == 0:
                frames.append(_topdown_frame(env))

        video = np.stack(frames, axis=0)  # (T, H, W, C) uint8

        track_slug = track.name.replace(" ", "_")
        filename   = f"step{global_step:08d}_track{track.level:02d}_{track_slug}.mp4"
        local_path = os.path.join(video_dir, filename)
        iio.imwrite(local_path, video, fps=20, codec="libx264", plugin="pyav")

        key = f"inference/track_{track.level:02d}_{track_slug}"
        video_logs[key] = wandb.Video(video, fps=20, format="mp4")

    wandb.log({**video_logs, "global_step": global_step}, step=global_step)
    sb3_model.policy.set_training_mode(True)
    print(f"  [VIDEO] Saved {n_tracks} video(s) to {video_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum + W&B callback
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumWandbCallback(BaseCallback):
    """
    Handles:
    - Curriculum advancement based on rolling mean episode reward
    - W&B logging of episode / curriculum / PPO / system metrics
    - Periodic checkpointing with pruning
    """

    def __init__(
        self,
        builder:             CurriculumBuilder,
        args,
        wandb_run_id:        str,
        start_time:          float,
        total_steps:         int,
        resume_global_step:  int = 0,
        verbose:             int = 0,
    ):
        super().__init__(verbose)
        self._builder     = builder
        self._args        = args
        self._run_id      = wandb_run_id
        self._start_time  = start_time
        self._total_steps = total_steps
        self._episode_count = 0
        self._reward_window = deque(maxlen=args.window)
        self._next_ckpt = args.checkpoint_interval
        self._global_step_offset = resume_global_step  # SB3 resets num_timesteps on resume

        # Video interval tracking
        vi = getattr(args, "video_interval", 25_000)
        if vi > 0:
            self._next_video = vi
            while self._next_video <= resume_global_step:
                self._next_video += vi
        else:
            self._next_video = float("inf")

        # Skip checkpoint thresholds already passed on resume
        if args.checkpoint_interval > 0:
            while self._next_ckpt <= resume_global_step:
                self._next_ckpt += args.checkpoint_interval
        else:
            self._next_ckpt = float("inf")

        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Called after every env step in the rollout ───────────────────────────

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            ep_reward  = info.get("episode_reward",  0.0)
            ep_crashes = info.get("episode_crashes",  0)
            ep_laps    = info.get("episode_laps",     0)
            on_track   = info.get("on_track_pct",    0.0)
            ep_length  = info.get("episode", {}).get("l", 0)
            track_lvl  = info.get("track_level", 0)
            track_name = info.get("track_name", "")

            self._episode_count += 1
            self._reward_window.append(ep_reward)

            rolling_mean = statistics.mean(self._reward_window) if self._reward_window else 0.0
            sampler      = self._builder._sampler
            frontier     = sampler.frontier_track
            threshold    = self._args.threshold * frontier.complexity

            # Curriculum advancement
            advanced = self._builder.record(ep_reward, ep_crashes, ep_laps)
            if advanced:
                new_frontier = sampler.frontier_track
                print(
                    f"\n\n  > CURRICULUM ADVANCE ->  "
                    f"Track {new_frontier.level} '{new_frontier.name}'  "
                    f"[level {self._builder.current_level}/{len(TRAIN)-1}]\n"
                    f"    rolling_mean={rolling_mean:.2f}  threshold={threshold:.2f}\n"
                )
                wandb.log({
                    "global_step":                  self.num_timesteps,
                    "curriculum/level":             self._builder.current_level,
                    "curriculum/advanced_to_level": new_frontier.level,
                    "curriculum/advanced_to_name":  new_frontier.name,
                    "curriculum/advanced_to_tier":  difficulty_of(new_frontier),
                }, step=self.num_timesteps)

            # Episode metrics
            wandb.log({
                "global_step":              self.num_timesteps,
                "episode/reward":           ep_reward,
                "episode/length":           ep_length,
                "episode/laps":             ep_laps,
                "episode/crashes":          ep_crashes,
                "episode/on_track_pct":     on_track,
                "episode/number":           self._episode_count,
                "curriculum/level":         self._builder.current_level,
                "curriculum/track_level":   track_lvl,
                "curriculum/track_name":    track_name,
                "curriculum/tier":          difficulty_of(frontier),
                "curriculum/rolling_mean":  rolling_mean,
                "curriculum/threshold":     threshold,
            }, step=self.num_timesteps)

        return True  # continue training

    # ── Called after each full rollout + PPO update ──────────────────────────

    def _on_rollout_end(self) -> None:
        # Anneal entropy coefficient linearly
        anneal_ent_coef(
            self.model,
            self._args.ent_coef_start,
            self._args.ent_coef_end,
            self._total_steps,
        )
        # SB3 logs PPO metrics internally; mirror them with our key names + W&B step
        logger = self.model.logger
        if hasattr(logger, "name_to_value"):
            lv = logger.name_to_value
            sps = self.num_timesteps / max(time.time() - self._start_time, 1e-6)

            # SB3 PPO logs under "train/..."
            wandb.log({
                "global_step":             self.num_timesteps,
                "ppo/policy_loss":         lv.get("train/policy_gradient_loss", float("nan")),
                "ppo/value_loss":          lv.get("train/value_loss",           float("nan")),
                "ppo/entropy":             lv.get("train/entropy_loss",         float("nan")),
                "ppo/approx_kl":           lv.get("train/approx_kl",            float("nan")),
                "ppo/clip_fraction":       lv.get("train/clip_fraction",        float("nan")),
                "ppo/explained_variance":  lv.get("train/explained_variance",   float("nan")),
                "ppo/learning_rate":       lv.get("train/learning_rate",        float("nan")),
                "ppo/grad_norm":           lv.get("train/std",                  float("nan")),
                "ppo/entropy_coef":        self.model.ent_coef,
                "system/steps_per_sec":    sps,
                "system/elapsed_hours":    (time.time() - self._start_time) / 3600,
            }, step=self.num_timesteps)

        # Checkpoint
        if self.num_timesteps >= self._next_ckpt:
            lvl = self._builder.current_level
            ckpt_path = os.path.join(
                self._args.checkpoint_dir,
                f"ppo_sb3_step{self.num_timesteps:08d}_lvl{lvl:02d}",
            )
            self.model.save(ckpt_path)
            wandb.save(ckpt_path + ".zip")
            print(f"\n  [CKPT] {ckpt_path}.zip")
            self._prune_checkpoints()
            self._next_ckpt += self._args.checkpoint_interval

        # Inference videos
        if self.num_timesteps >= self._next_video:
            vi = getattr(self._args, "video_interval", 25_000)
            vd = getattr(self._args, "video_dir", "inference_videos")
            print(f"\n  [VIDEO] Rendering inference videos for "
                  f"{self._builder.current_level + 1} track(s)…")
            try:
                log_inference_videos(
                    sb3_model   = self.model,
                    builder     = self._builder,
                    device      = torch.device(self._args.device),
                    global_step = self.num_timesteps,
                    video_dir   = vd,
                )
            except Exception as e:
                print(f"  [VIDEO] Warning: failed to render videos: {e}")
            self._next_video += vi

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_checkpoints(self):
        keep = self._args.keep_checkpoints
        if keep <= 0:
            return
        import glob as _glob
        pts = sorted(_glob.glob(
            os.path.join(self._args.checkpoint_dir, "ppo_sb3_step*.zip")
        ))
        for old in pts[:-keep]:
            os.remove(old)
            print(f"  [PRUNE] Removed {os.path.basename(old)}")


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (linear decay from lr to lr_min)
# ─────────────────────────────────────────────────────────────────────────────

def make_lr_schedule(lr_start: float, lr_min: float):
    """Returns an SB3 learning-rate schedule function (progress_remaining 1→0)."""
    def _schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 at start, 0.0 at end
        return lr_min + (lr_start - lr_min) * progress_remaining
    return _schedule


def anneal_ent_coef(model, ent_start: float, ent_end: float, total_steps: int) -> None:
    """Update model.ent_coef in-place based on current training progress."""
    progress = model.num_timesteps / max(total_steps, 1)  # 0 → 1
    model.ent_coef = float(ent_start + (ent_end - ent_start) * progress)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_kwargs = dict(
        project = args.wandb_project,
        name    = args.wandb_run_name,
        config  = vars(args),
        mode    = "offline" if args.wandb_offline else "online",
        sync_tensorboard = False,
    )
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
    sampler = builder._sampler

    # ── Vec env ───────────────────────────────────────────────────────────────
    def _make_env():
        """Factory that closes over the shared sampler."""
        return RaceGymEnv(sampler, max_steps=3000, laps_target=1)

    N = args.num_envs
    if args.subproc:
        # SubprocVecEnv: each worker gets its own copy of sampler (forked).
        # Curriculum advancement in the main process won't propagate, but the
        # rolling window is still computed correctly in the callback.
        vec_env = SubprocVecEnv([_make_env for _ in range(N)])
    else:
        vec_env = DummyVecEnv([_make_env for _ in range(N)])

    # ── Policy kwargs (custom extractor) ─────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = RaceNetExtractor,
        features_extractor_kwargs = {"features_dim": 288},
        # No extra MLP layers between encoder and actor/critic heads
        net_arch                  = dict(pi=[], vf=[]),
        activation_fn             = nn.ReLU,
        share_features_extractor  = True,
        ortho_init                = True,
        log_std_init              = -1.0,   # initial std ≈ 0.37, same as train.py
    )

    # ── n_steps must give integer batch_size divisor ─────────────────────────
    # SB3: batch = n_steps * n_envs.  Set n_steps so that batch = rollout_steps.
    # (rollout_steps is the total transitions per update in train.py)
    n_steps = max(args.rollout_steps // N, 64)

    # ── Resume or create model ────────────────────────────────────────────────
    resume_step = 0
    if args.resume:
        print(f"\n  [RESUME] Loading {args.resume}")
        model = PPO.load(
            args.resume,
            env          = vec_env,
            device       = args.device,
            # Override only the mutable training params (not architecture)
            custom_objects = {
                "learning_rate": make_lr_schedule(args.lr, args.lr_min),
                "clip_range":    args.clip_eps,
                "ent_coef":      make_ent_schedule(args.ent_coef_start, args.ent_coef_end),
                "n_steps":       n_steps,
                "n_epochs":      args.ppo_epochs,
                "batch_size":    args.minibatch_size,
            },
        )
        # SB3 resets num_timesteps on load; we track the offset for logging
        import re
        m = re.search(r"step(\d+)", os.path.basename(args.resume))
        if m:
            resume_step = int(m.group(1))
        print(f"  [RESUME] Continuing from step {resume_step:,}\n")
    else:
        model = PPO(
            policy          = "MultiInputPolicy",
            env             = vec_env,
            learning_rate   = make_lr_schedule(args.lr, args.lr_min),
            n_steps         = n_steps,
            batch_size      = args.minibatch_size,
            n_epochs        = args.ppo_epochs,
            gamma           = args.gamma,
            gae_lambda      = args.gae_lambda,
            clip_range      = args.clip_eps,
            vf_coef         = args.vf_coef,
            ent_coef        = args.ent_coef_start,  # annealed in callback
            max_grad_norm   = args.max_grad_norm,
            target_kl       = args.target_kl,
            policy_kwargs   = policy_kwargs,
            device          = args.device,
            seed            = args.seed,
            verbose         = 1,
        )

    # Bias actor mean toward gentle forward acceleration (same as train.py)
    with torch.no_grad():
        actor_mean = model.policy.mlp_extractor.policy_net
        # Walk to the final Linear layer of the policy head
        if hasattr(model.policy, "action_net"):
            model.policy.action_net.bias[0] = 0.3

    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} parameters  |  Device: {args.device}  |  Envs: {N}")
    print(f"Batch : {n_steps}×{N}={n_steps*N} per update  |  Minibatch: {args.minibatch_size}  "
          f"|  Epochs: {args.ppo_epochs}")
    print(f"Curriculum: threshold={args.threshold}  window={args.window}  "
          f"replay_frac={args.replay_frac}")
    print(f"Frontier  : {sampler.frontier_track.level} '{sampler.frontier_track.name}'")
    print(f"W&B       : {run.url}\n")

    # ── Callback ──────────────────────────────────────────────────────────────
    start_time = time.time()
    callback = CurriculumWandbCallback(
        builder            = builder,
        args               = args,
        wandb_run_id       = run.id,
        start_time         = start_time,
        total_steps        = args.total_steps,
        resume_global_step = resume_step,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    remaining_steps = args.total_steps - resume_step
    if remaining_steps <= 0:
        print("  [INFO] Already at total_steps — nothing to train.")
    else:
        model.learn(
            total_timesteps  = remaining_steps,
            callback         = callback,
            reset_num_timesteps = (args.resume is None),
            progress_bar     = False,
        )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final = os.path.join(args.checkpoint_dir, "ppo_sb3_final")
    model.save(final)
    wandb.save(final + ".zip")

    vec_env.close()
    elapsed = time.time() - start_time
    print(f"\n{'─'*80}")
    print(f"Training complete  |  {args.total_steps:,} steps  |  {elapsed/3600:.2f} h")
    print(f"Final model: {final}.zip")
    print(f"W&B run:     {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
