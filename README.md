# Imitation Learning — Curriculum Car Racer

Pygame car racing game + OpenEnv RL environment + PPO curriculum training pipeline.

## Structure

```
game/    Pygame simulation, 16 tracks, CarEnv gym wrapper, train/val/test splits
env/     OpenEnv environment, headlight image obs, ImpalaCNN encoder, curriculum builder
doc/     Full documentation
main.py  Entry point
```

## Quick Start

```bash
# Play the game
uv run python main.py          # track 1
uv run python main.py 5        # track 5

# Run track tests (headless)
uv run python -m game.test_tracks
```

## Documentation

| File | Contents |
|------|----------|
| [game/README.md](game/README.md) | Game controls, tracks, physics, RL interface |
| [env/README.md](env/README.md) | Observation space, action space, encoder, curriculum API |
| [doc/architecture.md](doc/architecture.md) | System overview, data flow, module dependencies |
| [doc/observation.md](doc/observation.md) | Why image obs, egocentric rendering pipeline, design rationale |
| [doc/encoder.md](doc/encoder.md) | Nature CNN vs ImpalaCNN, hybrid ViT option |
| [doc/curriculum.md](doc/curriculum.md) | 16-track splits, CurriculumSampler, training phases |
| [doc/training.md](doc/training.md) | PPO setup, hyperparameters, headless training, failure modes |
| [doc/headlight_agent_spec.md](doc/headlight_agent_spec.md) | Rule-based headlight agent specification |
| [doc/vit_rl_agent_spec.md](doc/vit_rl_agent_spec.md) | ViT+RL agent specification and generalisation analysis |
