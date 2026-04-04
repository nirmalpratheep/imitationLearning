# System Architecture

## Overview

```
imitationLearning/
  main.py              Entry point
  game/                Simulation + RL gym wrapper
  env/                 OpenEnv environment, encoder, curriculum
  doc/                 Documentation
```

## Data Flow

```
                        ┌─────────────────────────────────┐
                        │           game/                  │
                        │                                  │
                        │  tracks.py  ──►  TrackDef        │
                        │                  │               │
                        │  curriculum_     │ .build()      │
                        │  game.py         │               │
                        │  (human play)    ▼               │
                        │             track.surface        │
                        │             track.mask           │
                        │                  │               │
                        │  rl_splits.py    │               │
                        │  CarEnv  ◄───────┘               │
                        │    │  step([accel, steer])        │
                        │    │  → [x,y,sin,cos,spd,on,gate]│
                        └────┼────────────────────────────-┘
                             │
                        ┌────▼────────────────────────────-┐
                        │           env/                   │
                        │                                  │
                        │  RaceEnvironment                 │
                        │    │  wraps CarEnv               │
                        │    │  renders headlight image    │
                        │    │  → RaceObservation          │
                        │    │    .image  (64,64,3)        │
                        │    │    .speed / .on_track       │
                        │    │    .sin_angle / .cos_angle  │
                        │                                  │
                        │  CurriculumBuilder               │
                        │    TRAIN/VAL/TEST splits         │
                        │    CurriculumSampler             │
                        │    → RaceEnvironment per episode │
                        │                                  │
                        │  RaceEncoder (PyTorch)           │
                        │    ImpalaCNN(image) → 256-d      │
                        │    MLP(scalars)     →  32-d      │
                        │    cat              → 288-d      │
                        │    → PPO actor/critic heads      │
                        └──────────────────────────────────┘
```

## Module Dependencies

```
main.py
  └── game.curriculum_game

game.curriculum_game
  ├── game.oval_racer     (SCREEN_W/H, draw_headlights, draw_car)
  └── game.tracks         (TRACKS)

game.rl_splits
  └── game.tracks

env.environment
  ├── game.rl_splits      (CarEnv)
  ├── game.tracks         (TrackDef, SCREEN_W, SCREEN_H)
  └── game.oval_racer     (draw_headlights)

env.curriculum
  └── game.rl_splits      (TRAIN, VAL, TEST, CurriculumSampler)

env.encoder               (pure PyTorch, no game dependency)
```

## Key Classes

| Class | File | Role |
|-------|------|------|
| `TrackDef` | `game/tracks.py` | Holds waypoints, width, mask, start pos. `build()` renders the surface. |
| `CarEnv` | `game/rl_splits.py` | Gym-style wrapper. Owns physics, lap detection, episode termination. |
| `CurriculumSampler` | `game/rl_splits.py` | Manages frontier track, rolling reward window, replay schedule. |
| `RaceEnvironment` | `env/environment.py` | OpenEnv `Environment` subclass. Renders headlight image each step. |
| `CurriculumBuilder` | `env/curriculum.py` | Bridges `CurriculumSampler` with `RaceEnvironment`. One call = one episode env. |
| `ImpalaCNN` | `env/encoder.py` | Image encoder: 64×64×3 → 256-d. |
| `RaceEncoder` | `env/encoder.py` | Fuses image + scalars → 288-d feature vector. |
