"""
rl_splits.py — Train / Val / Test splits for curriculum RL training.

Split design
------------
20 tracks fall into 5 difficulty groups:

  Group A — Easy ovals          : tracks 1-4
  Group B — Rectangular shapes  : tracks 5-8
  Group C — Single special feat : tracks 9-12
  Group D — Complex polygons    : tracks 13-16
  Group E — Single-lane (1 car) : tracks 17-20

Stratified split: 2 train + 1 val + 1 test per group.

  TRAIN  (10) : [1,2, 5,6, 9,10, 13,14, 17,18]  — curriculum progression
  VAL    (5)  : [3,   7,   11,   15,    19   ]   — performance gating, NOT trained on
  TEST   (5)  : [4,   8,   12,   16,    20   ]   — held out entirely, eval at the end

Rationale
---------
* Val tracks are within-group but slightly harder than the first two train
  tracks in each group → checks generalisation within a difficulty tier.
* Test tracks are the hardest in each group (narrowest / tightest) → measures
  whether the agent can transfer to unseen geometry at each tier.
* Having representatives of every tier in every split avoids a mismatch where
  val/test are systematically harder than anything seen during training.
* Group E (single-lane) is useful for multi-agent experiments where bottlenecks
  force cars to negotiate single-file passage.

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

import numpy as np

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

    train_levels = [1, 2,  5, 6,  9, 10,  13, 14,  17, 18]  # 2 per group, easy→hard
    val_levels   = [3,     7,     11,     15,      19     ]  # 1 per group, medium
    test_levels  = [4,     8,     12,     16,      20     ]  # 1 per group, hardest

    train = [by_level[l] for l in train_levels]
    val   = [by_level[l] for l in val_levels  ]
    test  = [by_level[l] for l in test_levels ]
    return train, val, test


TRAIN, VAL, TEST = _get_splits()

# Convenience: all tracks in curriculum order (for inspection / logging)
ALL_ORDERED = sorted(TRAIN + VAL + TEST, key=lambda t: t.level)


# ── Difficulty metadata ───────────────────────────────────────────────────────

DIFFICULTY = {
    "A-easy":        {"tracks": [1, 2, 3, 4],     "description": "Full ovals"},
    "B-medium-easy": {"tracks": [5, 6, 7, 8],     "description": "Rectangular shapes"},
    "C-medium-hard": {"tracks": [9,10,11,12],     "description": "Hairpins & chicanes"},
    "D-hard":        {"tracks": [13,14,15,16],    "description": "Complex polygons"},
    "E-single-lane": {"tracks": [17,18,19,20],   "description": "Single-lane (one car wide)"},
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

    Observation  (7 floats):
        [angular_velocity, speed/max_speed, ray×5]

        The agent learns track direction purely from the egocentric image (CNN) and
        ray distances — no privileged map information in the observation.

    Action  (2 floats, each clamped to [-1, 1]):
        [accel, steer]
          accel  > 0 → accelerate,  < 0 → brake
          steer  > 0 → right,        < 0 → left

    Reward — simple:

        Per step (on track)
          + speed/max_speed * 0.20 - 0.01   speed reward minus time penalty.
              Zero speed earns -0.01/step — the agent must move to avoid a slow
              bleed; break-even at 5% max speed.

        Per step (off track)
          - 0.1 + toward_track * 0.15       base penalty + recovery gradient.
              toward_track = velocity · direction_to_nearest_centreline_wp.
              Steering back toward track reduces the penalty; steering further
              away increases it — gives the agent a left/right signal off-track.

        Event
          - 1.0   on crash (on→off transition)
          + progress_per_wp * advance        dense forward-progress signal from
                                             track centreline (agent does NOT see
                                             waypoints — reward only)
          + 100   on lap completion

        Terminal
          - 100   if car leaves screen bounds

        Complexity (track.complexity) scales the curriculum threshold only.

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

    # Dense progress reward: one full lap of forward waypoint advances ≈ +15 total.
    PROGRESS_SCALE = 15.0

    def __init__(self, track, max_steps=3000, laps_target=3):
        _ensure_pygame()
        self.track = track
        self.max_steps   = max_steps
        self.laps_target = laps_target
        track.build()

        # Pre-compute waypoint arrays (numpy) for fast nearest-wp lookup.
        # Waypoints are centreline points generated by TrackDef.build().
        # Used only for the internal progress reward — NOT exposed in observations.
        wps = track.waypoints
        self._n_wps = len(wps)
        self._wp_x = np.array([w[0] for w in wps], dtype=np.float32)
        self._wp_y = np.array([w[1] for w in wps], dtype=np.float32)
        self._progress_per_wp = self.PROGRESS_SCALE / self._n_wps

        self._x = self._y = self._angle = self._speed = 0.0
        self._prev_side   = 0.0
        self._laps        = 0
        self._step        = 0
        self._angle_delta = 0.0
        self._wp_idx      = 0      # nearest centreline waypoint index
        self._lap_dist    = 0.0
        self._lap_prev_x  = 0.0
        self._lap_prev_y  = 0.0
        self._was_on_track = True
        self._crash_count  = 0

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def obs_size(self):
        # angular_velocity, speed, ray×5
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
        self._speed = self.track.max_speed * 0.2
        self._angle_delta  = 0.0
        self._prev_side    = self.track.gate_side(self._x, self._y)
        self._laps         = 0
        self._step         = 0
        self._wp_idx       = self._nearest_wp(self._x, self._y)
        self._lap_dist     = 0.0
        self._lap_prev_x   = self._x
        self._lap_prev_y   = self._y
        self._was_on_track = True
        self._crash_count  = 0
        return self._obs()


    def step(self, action):
        accel = float(max(-1.0, min(1.0, action[0])))
        steer = float(max(-1.0, min(1.0, action[1])))

        prev_angle = self._angle
        self._update_physics(accel, steer)
        self._angle_delta = self._angle - prev_angle
        self._step += 1

        on        = self.track.on_track(self._x, self._y)
        curr_side = self.track.gate_side(self._x, self._y)

        # Lap distance (logging only)
        dx = self._x - self._lap_prev_x
        dy = self._y - self._lap_prev_y
        if on:
            self._lap_dist += math.hypot(dx, dy)
        self._lap_prev_x = self._x
        self._lap_prev_y = self._y

        # ── Reward ───────────────────────────────────────────────────────────
        speed_frac = self._speed / self.track.max_speed

        # 1. On-track: speed reward with time penalty — agent must keep moving.
        #    Zero speed earns -0.01/step, full speed earns +0.19/step.
        #    Off-track: base penalty + recovery gradient.
        #    The recovery term rewards steering *toward* the centreline so the
        #    agent has a signal distinguishing left vs right when off-track.
        if on:
            reward = speed_frac * 0.20 - 0.01
        else:
            # Base off-track penalty
            reward = -0.1
            # Recovery gradient: dot-product of velocity with direction to nearest wp.
            # Positive when heading back toward track, negative when moving further away.
            wp_dx = float(self._wp_x[self._wp_idx]) - self._x
            wp_dy = float(self._wp_y[self._wp_idx]) - self._y
            wp_dist = math.hypot(wp_dx, wp_dy) + 1e-6
            car_vx = self._speed * math.cos(math.radians(self._angle))
            car_vy = self._speed * math.sin(math.radians(self._angle))
            toward_track = (car_vx * wp_dx / wp_dist + car_vy * wp_dy / wp_dist)
            reward += toward_track * 0.15   # ±0.15 depending on direction

        # 2. Crash penalty (on→off transition)
        if self._was_on_track and not on:
            reward -= 1.0
            self._crash_count += 1
        self._was_on_track = on

        # 3. Waypoint progress — dense forward-progress signal.
        #    Agent does NOT see waypoints; this shapes the gradient toward
        #    completing the track, not just going fast in any direction.
        new_wp = self._nearest_wp(self._x, self._y)
        if on:
            diff = new_wp - self._wp_idx
            n = self._n_wps
            if diff > n // 2:
                diff -= n
            elif diff < -n // 2:
                diff += n
            reward += diff * self._progress_per_wp
        self._wp_idx = new_wp

        # 4. Lap completion
        lap_done = (self._prev_side < -5.0 and curr_side >= 0.0
                    and self._speed > 0.3)
        if lap_done:
            self._laps    += 1
            reward        += 100.0
            self._lap_dist = 0.0
            self._lap_prev_x = self._x
            self._lap_prev_y = self._y

        self._prev_side = curr_side

        # 5. Out-of-bounds terminal
        out_of_bounds = not (0 <= self._x < 900 and 0 <= self._y < 600)
        if out_of_bounds:
            reward -= 100.0

        done = (out_of_bounds
                or self._step >= self.max_steps
                or self._laps >= self.laps_target)

        return self._obs(), reward, done, {
            "lap":           self._laps,
            "on_track":      on,
            "step":          self._step,
            "crashes":       self._crash_count,
            "lap_dist":      self._lap_dist,
            "out_of_bounds": out_of_bounds,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _nearest_wp(self, x, y):
        """Return index of the nearest centreline waypoint to (x, y)."""
        dx = self._wp_x - x
        dy = self._wp_y - y
        return int(np.argmin(dx * dx + dy * dy))

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

    # Ray angles relative to heading (degrees). Covers lateral + diagonal + forward.
    _RAY_ANGLES = [-90, -45, 0, 45, 90]
    _RAY_MAX    = 120   # max ray length in px (normalise distances to 0..1)
    _RAY_STEP   = 2     # step size in px

    def _raycast(self):
        """
        Cast 5 rays from the car at fixed angles relative to heading.
        Returns list of 5 floats in [0, 1]:
            1.0 = boundary is MAX px away (clear road)
            0.0 = boundary is right at the car (on the edge / off track)
        Left/right rays give lateral clearance; diagonal/front give lookahead.
        """
        results = []
        for rel_deg in self._RAY_ANGLES:
            abs_rad = math.radians(self._angle + rel_deg)
            dx = math.cos(abs_rad) * self._RAY_STEP
            dy = math.sin(abs_rad) * self._RAY_STEP
            px, py = self._x, self._y
            dist = 0.0
            while dist < self._RAY_MAX:
                px += dx
                py += dy
                dist += self._RAY_STEP
                if not self.track.on_track(px, py):
                    break
            results.append(dist / self._RAY_MAX)
        return results

    def _obs(self):
        t    = self.track
        rays = self._raycast()   # 5 floats: left, front-left, front, front-right, right
        ang_vel = self._angle_delta / self.STEER_DEG   # ≈ [-1, 1]
        return [
            ang_vel,
            self._speed / t.max_speed,
            *rays,
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
        self._crashes     = deque(maxlen=window)   # crashes per episode
        self._laps        = deque(maxlen=window)   # laps completed per episode

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

    def record(self, episode_reward, episode_crashes=0, episode_laps=0):
        """Call after each episode with the total reward, crash count, and lap count."""
        self._rewards.append(episode_reward)
        self._crashes.append(episode_crashes)
        self._laps.append(episode_laps)

    def should_advance(self):
        """
        True when every episode in the window completed ≥1 lap with zero crashes.

        This directly measures mastery: the agent can reliably drive a full lap
        cleanly, regardless of reward scale or reward function changes.

        The reward threshold acts as a secondary guard against a slow creep
        that technically completes a lap but at negligible speed.
        """
        if self._idx >= len(self.tracks) - 1:
            return False
        if len(self._rewards) < self.window:
            return False
        all_lapped    = all(l >= 1 for l in self._laps)
        all_clean     = all(c == 0 for c in self._crashes)
        effective     = self.threshold * self.frontier_track.complexity
        reward_ok     = statistics.mean(self._rewards) >= effective
        return all_lapped and all_clean and reward_ok

    def advance(self):
        """Move to the next track. Clears all rolling buffers."""
        if self._idx < len(self.tracks) - 1:
            self._idx += 1
            self._rewards.clear()
            self._crashes.clear()
            self._laps.clear()
            return True
        return False

    @property
    def rolling_crashes(self):
        """Mean crashes per episode over the current window."""
        return statistics.mean(self._crashes) if self._crashes else float("nan")

    @property
    def rolling_laps(self):
        """Mean laps per episode over the current window."""
        return statistics.mean(self._laps) if self._laps else float("nan")

    def status(self):
        mean     = statistics.mean(self._rewards) if self._rewards else float("nan")
        crashes  = statistics.mean(self._crashes) if self._crashes else float("nan")
        t        = self.frontier_track
        effective = self.threshold * t.complexity
        crash_free = all(c == 0 for c in self._crashes) if self._crashes else False
        return (f"Frontier: track {t.level} '{t.name}'  "
                f"[{self._idx+1}/{len(self.tracks)}]  "
                f"rolling_mean={mean:.2f}  threshold={effective:.2f}  "
                f"crashes/ep={crashes:.2f}  crash_free={crash_free}")


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
    print("\n20-Track Curriculum Splits")
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
