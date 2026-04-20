"""
push_to_hub.py — Upload the curriculum-car-racer environment to HuggingFace Hub.

Usage:
    uv run python training/push_to_hub.py
    uv run python training/push_to_hub.py --repo-id your-username/curriculum-car-racer
    uv run python training/push_to_hub.py --repo-id your-username/curriculum-car-racer --private
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


MODEL_CARD = """\
---
tags:
  - openenv
  - reinforcement-learning
  - curriculum-learning
  - pygame
  - torchrl
license: mit
---

# Curriculum Car Racer — OpenEnv Environment

A Pygame-based car racing environment with curriculum learning, egocentric vision,
and 20 procedural tracks of increasing difficulty.

## Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `image` | (3, 64, 64) float32 | Egocentric 64×64 headlight view |
| `scalars` | (9,) float32 | speed, angular velocity, 5 raycasts, waypoint sin/cos |

## Action Space

`Box(-1.0, 1.0, (2,))` — `[accel, steer]`

## Curriculum

10 training tracks across 5 difficulty tiers (Easy → Single-lane).
The agent advances when its rolling mean reward exceeds a per-track threshold.

## Quick Start

```bash
git clone https://huggingface.co/datasets/{repo_id}
cd curriculum-car-racer
uv sync
bash training/cmd
```

## Serve as OpenEnv API

```bash
uvicorn env.server.app:app --host 0.0.0.0 --port 8000
# or
docker build -t curriculum-car-racer .
docker run -p 8000:8000 curriculum-car-racer
```

## Training

See [training/README.md](training/README.md) for full documentation.
"""


def push(repo_id: str, private: bool = False):
    api = HfApi()

    print(f"Creating/accessing repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Files to upload
    uploads = [
        ("openenv.yaml",  "openenv.yaml"),
        ("pyproject.toml", "pyproject.toml"),
        ("README.md",     "README.md"),
    ]

    # Folders to upload recursively
    folders = ["env", "game", "training", "doc"]

    print("Uploading individual files...")
    for local, remote in uploads:
        path = os.path.join(root, local)
        if os.path.exists(path):
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  ✓ {remote}")

    print("Uploading folders...")
    for folder in folders:
        folder_path = os.path.join(root, folder)
        if os.path.exists(folder_path):
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=["__pycache__", "*.pyc", ".venv", "*.pt", "*.mp4"],
            )
            print(f"  ✓ {folder}/")

    print("Writing model card...")
    card = MODEL_CARD.replace("{repo_id}", repo_id)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\nDone! https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="nirmalpratheep/curriculum-car-racer",
                   help="HuggingFace repo id (username/repo-name)")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    push(args.repo_id, args.private)
