#!/usr/bin/env python
"""
Quick sanity check: render one episode with a random policy and save to MP4.

Usage:
    python test_video.py                     # random agent, track 1
    python test_video.py --checkpoint ckpt.pt  # trained model
    python test_video.py --track 5           # pick a track level
"""

import argparse
import math
import os

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"   # headless pygame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p.add_argument("--track", type=int, default=1, help="Track level (1-20)")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--output", default="test_inference.mp4")
    args = p.parse_args()

    import pygame
    import torch
    from game.rl_splits import _ensure_pygame, TRAIN, CarEnv
    from game.tracks import TRACKS

    _ensure_pygame()

    # Find the requested track
    track = None
    for t in TRACKS:
        if t.level == args.track:
            track = t
            break
    if track is None:
        print(f"Track level {args.track} not found. Available: {[t.level for t in TRACKS]}")
        return

    track.build()

    # Set up model or random policy
    use_model = args.checkpoint is not None
    if use_model:
        from train import PPOActorCritic, obs_to_tensors
        from env.environment import RaceEnvironment
        from env.models import DriveAction

        device = torch.device("cpu")
        env = RaceEnvironment(track, max_steps=args.max_steps, laps_target=2, use_image=True)
        obs = env.reset()

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Handle torch.compile _orig_mod. prefix in state_dict keys
        state_dict = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model = PPOActorCritic().to(device)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        env_raw = CarEnv(track, max_steps=args.max_steps, laps_target=2)
        env_raw.reset()

    # Render frames
    def topdown_frame(car_env):
        surf = car_env.track.surface.copy()
        x, y = int(car_env._x), int(car_env._y)
        pygame.draw.circle(surf, (220, 50, 50), (x, y), 8)
        rad = math.radians(car_env._angle)
        tip = (int(x + 16 * math.cos(rad)), int(y + 16 * math.sin(rad)))
        pygame.draw.line(surf, (255, 220, 0), (x, y), tip, 3)
        small = pygame.transform.scale(surf, (450, 300))
        return pygame.surfarray.array3d(small).transpose(1, 0, 2).copy()

    frames = []
    frame_skip = 4
    step = 0

    if use_model:
        car = env._env
        frames.append(topdown_frame(car))
        with torch.no_grad():
            while not obs.done:
                img, sca = obs_to_tensors(obs, device)
                act, _, _, _ = model.get_action_and_value(
                    img.unsqueeze(0), sca.unsqueeze(0))
                obs = env.step(DriveAction(
                    accel=act.clamp(-1, 1)[0, 0].item(),
                    steer=act.clamp(-1, 1)[0, 1].item(),
                ))
                step += 1
                if step % frame_skip == 0:
                    frames.append(topdown_frame(car))
    else:
        frames.append(topdown_frame(env_raw))
        done = False
        while not done:
            # Random policy: gentle forward + slight random steer
            action = [0.5, np.random.uniform(-0.3, 0.3)]
            _, _, done, _ = env_raw.step(action)
            step += 1
            if step % frame_skip == 0:
                frames.append(topdown_frame(env_raw))

    print(f"Captured {len(frames)} frames over {step} steps")

    # Write MP4
    video = np.stack(frames, axis=0)  # (T, H, W, C)

    import imageio.v3 as iio
    # imageio-ffmpeg bundles its own ffmpeg binary
    iio.imwrite(args.output, video, fps=20, codec="libx264",
                plugin="pyav")

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
