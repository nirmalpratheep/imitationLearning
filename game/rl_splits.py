"""
rl_splits.py — Train / Val / Test splits for curriculum RL training.

Split design
------------
16 tracks fall into 4 difficulty groups (4 tracks each):

  Group A — Easy ovals          : tracks 1-4
  Group B — Rectangular shapes  : tracks 5-8
  Group C — Single special feat : tracks 9-12
  Group D — Complex polygons    : tracks 13-16

Stratified split: 2 train + 1 val + 1 test per group.

  TRAIN  (8)  : [1,2, 5,6, 9,10, 13,14]  — curriculum progression
  VAL    (4)  : [3,   7,   11,   15   ]  — performance gating, NOT trained on
  TEST   (4)  : [4,   8,   12,   16   ]  — held out entirely, eval at the end

Rationale
---------
* Val tracks are within-group but slightly harder than the first two train
  tracks in each group → checks generalisation within a difficulty tier.
* Test tracks are the hardest in each group (narrowest / tightest) → measures
  whether the agent can transfer to unseen geometry at each tier.
* Having representatives of every tier in every split avoids a mismatch where
  val/test are systematically harder than anything seen during training.

Usage
-----
    from rl_splits import TRAIN, VAL, TEST, make_env, CurriculumSampler, Evaluator

    # Simple fixed-schedule training
    for track in TRAIN:               # already ordered easy→hard
        env = make_env(track)
        ...train for N episodes...

    # Performance-gated curriculum
    sampler = CurriculumSampler(TRAIN)
    while sampler.current_level < len(TRAIN):
        env = make_env(sampler.sample())
        reward = run_episode(env, agent)
        sampler.record(reward)
        if sampler.should_advance():
            sampler.advance()

    # Evaluate
    ev = Evaluator(n_episodes=20)
    val_metrics  = ev.run(agent, VAL)
    test_metrics = ev.run(agent, TEST)
"""

import os
import math
import random
import statistics
from collections import deque

# ── Lazy pygame initialisation (avoids import-time display requirement) ──────
_pygame_ready = False

def _ensure_pygame():
    global _pygame_ready
    if not _pygame_ready:
        import pygame
        if not pygame.get_init():
            pygame.init()
        _pygame_ready = True


# ── Track splits ─────────────────────────────────────────────────────────────

def _get_splits():
    from .tracks import TRACKS          # TRACKS is 0-indexed, levels are 1-indexed
    by_level = {t.level: t for t in TRACKS}

    train_levels = [1, 2,  5, 6,  9, 10,  13, 14]   # 2 per group, easy→hard
    val_levels   = [3,     7,     11,     15      ]   # 1 per group, medium
    test_levels  = [4,     8,     12,     16      ]   # 1 per group, hardest

    train = [by_level[l] for l in train_levels]
    val   = [by_level[l] for l in val_levels  ]
    test  = [by_level[l] for l in test_levels ]
    return train, val, test


TRAIN, VAL, TEST = _get_splits()

# Convenience: all tracks in curriculum order (for inspection / logging)
ALL_ORDERED = sorted(TRAIN + VAL + TEST, key=lambda t: t.level)


# ── Difficulty metadata ───────────────────────────────────────────────────────

DIFFICULTY = {
    "A-easy":        {"tracks": [1, 2, 3, 4],   "description": "Full ovals"},
    "B-medium-easy": {"tracks": [5, 6, 7, 8],   "description": "Rectangular shapes"},
    "C-medium-hard": {"tracks": [9,10,11,12],   "description": "Hairpins & chicanes"},
    "D-hard":        {"tracks": [13,14,15,16],  "description": "Complex polygons"},
}


def difficulty_of(track):
    """Return the difficulty tier label for a track."""
    for tier, info in DIFFICULTY.items():
        if track.level in info["tracks"]:
            return tier
    return "unknown"


# ── Environment factory ───────────────────────────────────────────────────────

class CarEnv:
    """
    Minimal gym-style wrapper around TrackDef + Car physics.

    Observation  (7 floats, all in roughly [-1, 1]):
        [x/W, y/H, sin(angle), cos(angle), speed/max_speed, on_track, gate_side_norm]

    Action  (2 floats, each clamped to [-1, 1]):
        [accel, steer]
          accel  > 0 → accelerate,  < 0 → brake
          steer  > 0 → right,        < 0 → left

    Reward (all terms scaled by track.complexity so harder tracks give bigger signals):

        Per step
          + speed/max_speed * 0.01            tiny forward pulse
          - 0.5  * C   per step off track     discourages leaving road
          - 5.0  * C   on crash event         penalises each on→off transition

        On lap completion
          + 50 * time_ratio * dist_ratio * C
            time_ratio = par_time / actual_lap_time   clamped [0.5, 2.0]
                         >1 means faster than par (reward scales up)
            dist_ratio  = optimal_dist / actual_dist  clamped [0.5, 1.5]
                         >1 means shorter path than centerline (reward scales up)

        Terminal
          - 100 * C   if car leaves screen bounds (episode ends)

        complexity C = (115 / track.width) * (track.max_speed / 3.0)
                       Track 1 → 1.0 | Track 16 → 3.45

    Done conditions:
        * car leaves screen
        * max_steps exceeded
        * laps_target laps completed
    """

    # Physics (same as curriculum_game.py)
    ACCEL       = 0.13
    BRAKE_DECEL = 0.22
    FRICTION    = 0.038
    STEER_DEG   = 2.7

    def __init__(self, track, max_steps=3000, laps_target=3):
        _ensure_pygame()
        self.track = track
        self.max_steps   = max_steps
        self.laps_target = laps_target
        track.build()

        self._x = self._y = self._angle = self._speed = 0.0
        self._prev_side  = 0.0
        self._laps       = 0
        self._step       = 0
        # lap metrics (reset each lap)
        self._lap_start_step = 0
        self._lap_dist       = 0.0
        self._lap_prev_x     = 0.0
        self._lap_prev_y     = 0.0
        # crash tracking (on_track → off_track transitions)
        self._was_on_track   = True
        self._crash_count    = 0

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def obs_size(self):
        return 7

    @property
    def action_size(self):
        return 2

    @property
    def laps(self):
        return self._laps

    def reset(self):
        self._x     = float(self.track.start_pos[0])
        self._y     = float(self.track.start_pos[1])
        self._angle = float(self.track.start_angle)
        self._speed = 0.0
        self._prev_side      = self.track.gate_side(self._x, self._y)
        self._laps           = 0
        self._step           = 0
        self._lap_start_step = 0
        self._lap_dist       = 0.0
        self._lap_prev_x     = self._x
        self._lap_prev_y     = self._y
        self._was_on_track   = True
        self._crash_count    = 0
        return self._obs()

    def step(self, action):
        accel = float(max(-1.0, min(1.0, action[0])))
        steer = float(max(-1.0, min(1.0, action[1])))

        self._update_physics(accel, steer)
        self._step += 1

        # Accumulate lap distance
        dx = self._x - self._lap_prev_x
        dy = self._y - self._lap_prev_y
        self._lap_dist  += math.hypot(dx, dy)
        self._lap_prev_x = self._x
        self._lap_prev_y = self._y

        on        = self.track.on_track(self._x, self._y)
        curr_side = self.track.gate_side(self._x, self._y)
        C         = self.track.complexity

        # ── Reward ───────────────────────────────────────────────────────────

        # 1. Tiny forward pulse — keeps agent from stalling
        reward = self._speed / self.track.max_speed * 0.01

        # 2. Off-track per-step penalty
        if not on:
            reward -= 0.5 * C

        # 3. Crash event penalty (on_track → off_track transition)
        if self._was_on_track and not on:
            reward -= 5.0 * C
            self._crash_count += 1
        self._was_on_track = on

        # 4. Lap completion — main signal
        lap_done = (self._prev_side < -5.0 and curr_side >= 0.0
                    and abs(self._speed) > 0.3)
        if lap_done:
            self._laps += 1

            lap_steps = max(1, self._step - self._lap_start_step)
            lap_dist  = max(1.0, self._lap_dist)

            # faster than par → time_ratio > 1.0 (reward scales up)
            time_ratio = min(2.0, max(0.5,
                self.track.par_time_steps / lap_steps))

            # shorter path than centerline → dist_ratio > 1.0 (reward scales up)
            dist_ratio = min(1.5, max(0.5,
                self.track.optimal_dist / lap_dist))

            reward += 50.0 * time_ratio * dist_ratio * C

            # Reset lap tracking for next lap
            self._lap_start_step = self._step
            self._lap_dist       = 0.0
            self._lap_prev_x     = self._x
            self._lap_prev_y     = self._y

        self._prev_side = curr_side

        # 5. Terminal out-of-bounds
        out_of_bounds = not (0 <= self._x < 900 and 0 <= self._y < 600)
        if out_of_bounds:
            reward -= 100.0 * C

        done = (out_of_bounds
                or self._step >= self.max_steps
                or self._laps >= self.laps_target)

        return self._obs(), reward, done, {
            "lap":          self._laps,
            "on_track":     on,
            "step":         self._step,
            "crashes":      self._crash_count,
            "lap_dist":     self._lap_dist,
            "out_of_bounds": out_of_bounds,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_physics(self, accel, steer):
        ms = self.track.max_speed
        ratio = min(abs(self._speed) / ms, 1.0) if ms > 0 else 1.0
        self._angle += steer * self.STEER_DEG * max(0.3, ratio)

        if accel > 0:
            self._speed = min(self._speed + self.ACCEL * accel, ms)
        elif accel < 0:
            self._speed = max(self._speed + self.BRAKE_DECEL * accel,
                              -ms * 0.4)
        if self._speed > 0:
            self._speed = max(0.0, self._speed - self.FRICTION)
        elif self._speed < 0:
            self._speed = min(0.0, self._speed + self.FRICTION)

        if not self.track.on_track(self._x, self._y):
            self._speed *= 0.80

        rad = math.radians(self._angle)
        self._x += self._speed * math.cos(rad)
        self._y += self._speed * math.sin(rad)

    def _obs(self):
        t  = self.track
        gs = self.track.gate_side(self._x, self._y)
        return [
            self._x / 900.0,
            self._y / 600.0,
            math.sin(math.radians(self._angle)),
            math.cos(math.radians(self._angle)),
            self._speed / t.max_speed,
            float(t.on_track(self._x, self._y)),
            max(-1.0, min(1.0, gs / 500.0)),   # gate distance, normalised
        ]


def make_env(track, **kwargs):
    """Factory: return a fresh CarEnv for the given TrackDef."""
    return CarEnv(track, **kwargs)


# ── Curriculum sampler ────────────────────────────────────────────────────────

class CurriculumSampler:
    """
    Manages which train track to sample next.

    Strategy: performance-gated with anti-forgetting replay.
      * 70% of episodes → current frontier track
      * 30% of episodes → random track from already-mastered ones
    Advance to the next track when the rolling mean reward over
    `window` episodes exceeds `threshold`.

    Parameters
    ----------
    tracks      : ordered list of TrackDef (easy → hard)
    threshold   : mean episode reward required to advance
    window      : rolling window size for reward averaging
    replay_frac : fraction of episodes sampled from mastered tracks
    """

    def __init__(self, tracks, threshold=30.0, window=50, replay_frac=0.3):
        self.tracks       = tracks
        self.threshold    = threshold
        self.window       = window
        self.replay_frac  = replay_frac
        self._idx         = 0              # current frontier index
        self._rewards     = deque(maxlen=window)

    @property
    def current_level(self):
        return self._idx                   # 0-based index into self.tracks

    @property
    def current_track(self):
        return self.tracks[self._idx]

    @property
    def mastered(self):
        return self.tracks[:self._idx]

    @property
    def frontier_track(self):
        return self.tracks[self._idx]

    def sample(self):
        """Return the TrackDef to use for the next episode."""
        if self._idx > 0 and random.random() < self.replay_frac:
            return random.choice(self.mastered)
        return self.frontier_track

    def record(self, episode_reward):
        """Call after each episode with the total episode reward."""
        self._rewards.append(episode_reward)

    def should_advance(self):
        """True if the agent has hit the threshold on the frontier track."""
        if self._idx >= len(self.tracks) - 1:
            return False
        if len(self._rewards) < self.window:
            return False
        return statistics.mean(self._rewards) >= self.threshold

    def advance(self):
        """Move to the next track. Clears the rolling reward buffer."""
        if self._idx < len(self.tracks) - 1:
            self._idx += 1
            self._rewards.clear()
            return True
        return False

    def status(self):
        mean = statistics.mean(self._rewards) if self._rewards else float("nan")
        t = self.frontier_track
        return (f"Frontier: track {t.level} '{t.name}'  "
                f"[{self._idx+1}/{len(self.tracks)}]  "
                f"rolling_mean={mean:.2f}  threshold={self.threshold:.2f}")


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs a fixed number of greedy episodes on a list of tracks
    and returns per-track and aggregate metrics.

    agent_fn : callable(obs) → action   (e.g. your policy's greedy forward pass)
    """

    def __init__(self, n_episodes=20, max_steps=3000, laps_target=3):
        self.n_episodes  = n_episodes
        self.max_steps   = max_steps
        self.laps_target = laps_target

    def run(self, agent_fn, tracks):
        """
        Returns dict:
            {
              "per_track": [ { "level", "name", "tier", "mean_reward",
                               "mean_laps", "completion_rate" }, ... ],
              "mean_reward":      float,
              "mean_laps":        float,
              "completion_rate":  float,   # fraction of episodes with ≥1 lap
            }
        """
        per_track = []
        all_rewards, all_laps, all_complete = [], [], []

        for track in tracks:
            ep_rewards, ep_laps = [], []

            for _ in range(self.n_episodes):
                env  = make_env(track, max_steps=self.max_steps,
                                laps_target=self.laps_target)
                obs  = env.reset()
                done = False
                total_r = 0.0

                while not done:
                    action = agent_fn(obs)
                    obs, r, done, _ = env.step(action)
                    total_r += r

                ep_rewards.append(total_r)
                ep_laps.append(env.laps)

            completion = sum(1 for l in ep_laps if l >= 1) / self.n_episodes

            per_track.append({
                "level":           track.level,
                "name":            track.name,
                "tier":            difficulty_of(track),
                "mean_reward":     statistics.mean(ep_rewards),
                "std_reward":      statistics.stdev(ep_rewards) if len(ep_rewards) > 1 else 0.0,
                "mean_laps":       statistics.mean(ep_laps),
                "completion_rate": completion,
            })

            all_rewards.extend(ep_rewards)
            all_laps.extend(ep_laps)
            all_complete.extend([l >= 1 for l in ep_laps])

        return {
            "per_track":       per_track,
            "mean_reward":     statistics.mean(all_rewards),
            "mean_laps":       statistics.mean(all_laps),
            "completion_rate": sum(all_complete) / len(all_complete),
        }

    @staticmethod
    def print_report(metrics, title="Evaluation"):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  {'Lvl':<4} {'Name':<24} {'Tier':<16} "
              f"{'Reward':>8} {'Laps':>6} {'Done%':>6}")
        print(f"  {'-'*66}")
        for r in metrics["per_track"]:
            print(f"  {r['level']:<4} {r['name']:<24} {r['tier']:<16} "
                  f"{r['mean_reward']:>8.1f} {r['mean_laps']:>6.2f} "
                  f"{r['completion_rate']*100:>5.0f}%")
        print(f"  {'-'*66}")
        print(f"  {'AGGREGATE':<44} "
              f"{metrics['mean_reward']:>8.1f} {metrics['mean_laps']:>6.2f} "
              f"{metrics['completion_rate']*100:>5.0f}%")
        print(f"{'='*60}\n")


# ── Split summary (run as script) ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n16-Track Curriculum Splits")
    print("=" * 60)

    for split_name, split_tracks in [("TRAIN", TRAIN), ("VAL", VAL), ("TEST", TEST)]:
        print(f"\n{split_name}  ({len(split_tracks)} tracks)")
        print(f"  {'Lvl':<4} {'Name':<24} {'Tier':<16} {'Width':>6} {'MaxSpd':>7}")
        print(f"  {'-'*58}")
        for t in split_tracks:
            print(f"  {t.level:<4} {t.name:<24} {difficulty_of(t):<16} "
                  f"{t.width:>6} {t.max_speed:>7.1f}")

    print("\nSplit rationale:")
    print("  TRAIN  - 2 tracks per difficulty tier, ordered easy->hard for curriculum")
    print("  VAL    - 1 track per tier (within-tier generalisation check)")
    print("  TEST   - 1 track per tier (held out entirely; final evaluation only)")
