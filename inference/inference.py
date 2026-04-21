"""
inference.py — Run greedy eval on all tracks and save one video per track.

Usage:
    uv run python inference/inference.py --checkpoint checkpoints/ppo_torchrl_final.pt
    uv run python inference/inference.py --checkpoint checkpoints/ppo_torchrl_final.pt --video-dir inference_videos --device cuda
"""

import argparse
import os
import sys

# Project root and training/ on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "training"))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from env import DriveAction
from env.environment import RaceEnvironment
from game.rl_splits import TRAIN, _ensure_pygame
from train_torchrl import build_policy_and_value, _game_frame


@torch.no_grad()
def run_track(policy_module, track, device, video_dir, frame_skip=2):
    """Greedy episode on track — returns (laps, crashes, video_path)."""
    import imageio.v3 as iio

    _ensure_pygame()
    track.build()
    env     = RaceEnvironment(track, max_steps=3000, laps_target=1, use_image=True)
    raw_obs = env.reset()
    frames  = [_game_frame(env)]
    step    = 0

    while not raw_obs.done:
        img = (torch.from_numpy(raw_obs.image.copy())
               .float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device))
        scalars = torch.tensor(raw_obs.scalars, dtype=torch.float32,
                               device=device).unsqueeze(0)
        td = TensorDict({"image": img, "scalars": scalars}, batch_size=[1])
        with set_exploration_type(ExplorationType.MEAN):
            td = policy_module(td)
        action  = td["action"][0].clamp(-1.0, 1.0).cpu().numpy()
        raw_obs = env.step(DriveAction(accel=float(action[0]), steer=float(action[1])))
        step += 1
        if step % frame_skip == 0:
            frames.append(_game_frame(env))

    ce = env._env
    laps, crashes = ce._laps, ce._crash_count

    os.makedirs(video_dir, exist_ok=True)
    slug      = track.name.replace(" ", "_")
    video_path = os.path.join(video_dir, f"track{track.level:02d}_{slug}.mp4")
    iio.imwrite(video_path, np.stack(frames), fps=20, codec="libx264", plugin="pyav")

    return laps, crashes, video_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--video-dir",  default="inference_videos", help="Directory to save videos")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame-skip", type=int, default=2, help="Record every N frames")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    policy_module, _, _ = build_policy_and_value(device)
    sd = ckpt["policy"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    policy_module.load_state_dict(sd)
    policy_module.eval()

    step_str = f"{ckpt['step']:,}" if "step" in ckpt else "?"
    print(f"\nCheckpoint : {args.checkpoint}  (step {step_str})")
    print(f"Device     : {device}")
    print(f"Tracks     : {len(TRAIN)}")
    print(f"Video dir  : {args.video_dir}\n")

    all_pass = True
    for track in TRAIN:
        laps, crashes, video_path = run_track(
            policy_module, track, device, args.video_dir, args.frame_skip
        )
        ok     = laps >= 1 and crashes == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  track {track.level:02d}  {track.name:<22}  {status}"
              f"  laps={laps}  crashes={crashes}  {video_path}")

    print()
    print("=" * 50)
    if all_pass:
        print("  ALL TRACKS PASSED")
    else:
        print("  SOME TRACKS FAILED — see above")
    print("=" * 50)


if __name__ == "__main__":
    main()
