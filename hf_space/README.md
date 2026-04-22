---
title: Car Racing Agent
emoji: 🏎️
colorFrom: orange
colorTo: yellow
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - reinforcement-learning
  - curriculum-learning
  - torchrl
  - ppo
  - openenv
  - pygame
short_description: A PPO agent trained from scratch on 10 curriculum tracks.
---

# Car Racing Agent

Live demo of a PPO car-racing agent trained from scratch across a 10-track
curriculum — zero crashes on the full curriculum.

- **Blog post:** <https://huggingface.co/blog/NirmalPratheep/curriculum-car-racer>
- **Source / training code:** <https://github.com/NirmalPratheep/curriculum-car-racer>
- **OpenEnv Student Challenge 2026** submission.

## How to use

1. Pick a track from the dropdown (Track 01 → Track 10).
2. Click **Reset** to spawn the agent.
3. Click **Step ×20** to advance physics a chunk at a time, or
   **Auto-Drive** to run a full lap and get a replay video.

The orange-bordered thumbnail is the actual 64×64 egocentric image the
CNN receives every step — always rotated so the car faces up.
