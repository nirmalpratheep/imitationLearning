# Curriculum Training

## Track Splits

16 tracks organised into 4 difficulty tiers, 4 tracks per tier:

```
Tier A — Easy ovals          tracks  1, 2, 3, 4
Tier B — Rectangular shapes  tracks  5, 6, 7, 8
Tier C — Hairpins/chicanes   tracks  9,10,11,12
Tier D — Complex polygons    tracks 13,14,15,16
```

Stratified split: 2 train + 1 val + 1 test per tier.

| Split | Tracks | Role |
|-------|--------|------|
| **TRAIN** (8) | 1,2, 5,6, 9,10, 13,14 | Curriculum progression. Ordered easy→hard within each tier. |
| **VAL** (4) | 3, 7, 11, 15 | Performance gating. Never trained on. Used to confirm the agent generalises before advancing. |
| **TEST** (4) | 4, 8, 12, 16 | Held-out. Not seen until final evaluation. Hardest track in each tier. |

**Rationale**: Val and test tracks are within-tier but harder than the easiest
two train tracks. This tests within-tier generalisation without making val/test
systematically harder than anything the agent has seen.

---

## CurriculumSampler

Located in `game/rl_splits.py`. Manages which train track to sample each episode.

### Strategy: performance-gated + anti-forgetting replay

```
70% of episodes → current frontier track (the one being learned now)
30% of episodes → random track from already-mastered ones  (replay)
```

Advance to the next track when rolling mean reward over `window` episodes
exceeds `threshold`.

```python
from game.rl_splits import CurriculumSampler, TRAIN

sampler = CurriculumSampler(TRAIN, threshold=30.0, window=50, replay_frac=0.3)

track = sampler.sample()          # returns TrackDef
sampler.record(episode_reward)    # add to rolling window
if sampler.should_advance():
    sampler.advance()             # move to next track
```

### Why replay?

Without replay the agent forgets earlier tracks as it trains on harder ones.
When evaluated on TRAIN track 1 after mastering track 14, performance collapses.
30% replay keeps all mastered behaviours active at negligible extra cost.

---

## CurriculumBuilder

Located in `env/curriculum.py`. Wraps `CurriculumSampler` to produce
`RaceEnvironment` instances ready for use with the full image observation.

```python
from env import CurriculumBuilder

builder = CurriculumBuilder(
    threshold=30.0,
    window=50,
    replay_frac=0.3,
    max_steps=3000,
    laps_target=3,
    use_image=True,
)

# --- Training ---
env     = builder.next_env()           # samples frontier or replay track
obs     = env.reset()
while not obs.done:
    obs = env.step(action)
    total_reward += obs.reward
advanced = builder.record(total_reward)

# --- Validation (after each frontier advance) ---
if advanced:
    for env in builder.val_envs():     # 4 VAL tracks
        ...

# --- Final evaluation ---
for env in builder.test_envs():        # 4 TEST tracks (run once at the end)
    ...

print(builder.status)
# "Frontier: track 5 'Rounded Rectangle' [3/8]  rolling_mean=34.1  threshold=30.0"
```

---

## Recommended Training Phases

### Phase 1 — Survival (tracks 1–2, ~500 k steps)

Goal: stay on track at all.

- `threshold = 20.0` (easy to pass — just don't crash constantly)
- Reward: only `on_track` and `off_track` signals
- Expected: agent learns to steer away from white lines

### Phase 2 — Speed (tracks 1–6, ~1.5 M steps)

Goal: complete laps, go faster.

- `threshold = 30.0`
- Add lap completion bonus and speed reward
- Expected: agent learns a racing line on ovals and rectangles

### Phase 3 — Precision (tracks 1–16, ~3 M steps)

Goal: handle hairpins and complex shapes.

- `threshold = 40.0`
- Same reward, but tighter tracks demand precise corner entry
- Reduce entropy coefficient from 0.01 → 0.001 to exploit learned policy

### Phase 4 — Zero-shot evaluation on TEST tracks

Run `builder.test_envs()` — agent has never seen these tracks.
Freeze policy weights. Compare lap completion rate, mean lap time.

---

## Anti-forgetting: What Replay Actually Does

```
Episode   Sampled track    Why
──────────────────────────────────────────────────────
1–50      Track 1          Learning phase, frontier = 1
51–100    Track 1          Advancing...
101       Track 2          Frontier advances to 2
102       Track 1          30% replay — keeps track-1 skill alive
103       Track 2          Back to frontier
...
N         Track 14         Frontier at 14; replaying tracks 1–13 at 30%
```

Without replay, catastrophic forgetting causes the agent to fail track 1 by the
time it reaches track 14. With 30% replay, performance on mastered tracks
degrades by less than 10% in practice.

---

## Difficulty Progression

```
Track  Name                Width  MaxSpd  Key challenge
─────────────────────────────────────────────────────────────
 1     Wide Oval           115    3.0     Nothing — just drive
 2     Standard Oval        85    3.5     Slightly narrower
 3     Narrow Oval          58    3.5     VAL — precision needed
 4     Superspeedway        85    4.5     TEST — high speed, elliptical
 5     Rounded Rectangle    90    3.5     First corners
 6     Stadium Oval         80    4.0     Tight end-caps
 7     Tight Rectangle      65    3.5     VAL — sharp 90° corners
 8     Small Oval           60    3.2     TEST — small radius
 9     Hairpin Track        75    3.5     First hairpin
10     Chicane Track        70    3.5     Chicane section
11     Double Hairpin       70    3.5     VAL — two hairpins
12     Asymmetric Track     70    3.8     TEST — asymmetric arcs
13     L-Shape Circuit      72    4.0     First polygon
14     T-Notch Circuit      58    4.0     T junction
15     Complex Circuit      65    4.5     VAL — multi-feature
16     Master Challenge     50    4.5     TEST — narrowest + fastest
```
