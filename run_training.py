#!/usr/bin/env python
"""
run_training.py — Fully automated training pipeline.

Launches train.py, monitors W&B metrics every 60s, and auto-restarts with
fixes when the agent is stuck. Stops when all 10 curriculum tracks are
mastered or the step budget is exhausted.

Usage:
    uv run python run_training.py
    uv run python run_training.py --total-steps 5_000_000
    uv run python run_training.py --dry-run   # print commands without running
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────

TOTAL_TRACKS       = 10      # curriculum has 10 training tracks (indices 0-9)
POLL_INTERVAL      = 60      # seconds between W&B polls
STALL_STEPS        = 200_000 # if no curriculum advance for this many steps → intervene
METRIC_CHECK_EVERY = 50_000  # check health thresholds at these boundaries

# Base training command (no --resume, no overrides yet)
# Tuned for fast, clean laps (minimum distance + minimum time):
#   - gamma 0.997: longer horizon values the +50 lap bonus more strongly
#   - ent-coef-start 0.005: low entropy so the clamped log_std can tighten early,
#     producing precise steering rather than noisy exploration
#   - rollout-steps 4096: each rollout covers 2+ full laps, giving PPO complete
#     episodes to learn from (avoids truncated-return bias)
#   - threshold 18: slightly easier gate so the agent advances quickly and spends
#     more budget on hard tracks where distance/time matter most
#   - window 20: faster advancement decisions (still requires 20 consecutive
#     clean-lap episodes)
BASE_CMD = [
    sys.executable, "train_sb3.py",
    "--total-steps",         "10_000_000",
    "--num-envs",            "8",
    "--subproc",                            # parallel env stepping
    "--compile",                            # torch.compile policy
    "--rollout-steps",       "4096",
    "--ppo-epochs",          "4",
    "--minibatch-size",      "512",
    "--lr",                  "3e-4",
    "--lr-min",              "1e-5",
    "--gamma",               "0.997",
    "--gae-lambda",          "0.95",
    "--clip-eps",            "0.2",
    "--vf-coef",             "0.5",
    "--ent-coef-start",      "0.005",
    "--ent-coef-end",        "0.001",
    "--max-grad-norm",       "0.5",
    "--target-kl",           "0.015",
    "--threshold",           "80.0",
    "--window",              "20",
    "--replay-frac",         "0.3",
    "--checkpoint-interval", "250_000",
    "--keep-checkpoints",    "5",
    "--video-interval",      "25_000",
]

# Health thresholds: step → (min_reward, min_on_track_pct, min_explained_var, max_grad_norm)
# Reward scale: lap = +100, on-track ~0.2/step max, crash = -1, off-track = -0.2/step
THRESHOLDS = {
      50_000: ( -300,  60, 0.40, 30),
     100_000: ( -100,  75, 0.60, 15),
     200_000: (    0,  85, 0.80, 10),
     500_000: (   50,  90, 0.85,  8),
   1_000_000: (  100,  93, 0.90,  6),
   2_000_000: (  200,  95, 0.93,  5),
   3_000_000: (  300,  96, 0.93,  4),
   5_000_000: (  400,  97, 0.95,  4),
}

# Escalating fixes for curriculum stalls (tried in order)
STALL_FIXES = [
    # attempt 1: longer rollouts + slightly more exploration
    {"--ent-coef-start": "0.01", "--rollout-steps": "4096"},
    # attempt 2: lower threshold + more exploration
    {"--threshold": "15.0", "--ent-coef-start": "0.015"},
    # attempt 3: lower threshold + more replay + slower LR
    {"--threshold": "12.0", "--replay-frac": "0.4", "--lr": "1e-4"},
    # attempt 4: very low threshold + moderate exploration
    {"--threshold": "10.0", "--ent-coef-start": "0.02", "--ent-coef-end": "0.003",
     "--rollout-steps": "4096"},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def diagnose_from_checkpoint(ckpt_path):
    """
    Load a checkpoint, run one greedy episode on track 1, and diagnose
    the agent's behavior.  Returns a dict with:
        on_track_pct, laps, crashes, total_reward, wp_progress, diagnosis, fix
    """
    import torch
    # Import lazily to avoid circular imports / heavy loading at start
    from train import PPOActorCritic, obs_to_tensors
    from env.environment import RaceEnvironment
    from env.models import DriveAction
    from game.rl_splits import _ensure_pygame, TRAIN

    _ensure_pygame()

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    model = PPOActorCritic().to(device)
    model.load_state_dict(state)
    model.eval()

    level = ckpt.get("curriculum_level", 0)
    track = TRAIN[min(level, len(TRAIN) - 1)]
    track.build()
    env = RaceEnvironment(track, max_steps=3000, laps_target=3, use_image=True)
    obs = env.reset()

    total_r = 0.0
    steps = 0
    on_steps = 0

    with torch.no_grad():
        while not obs.done:
            img, sca = obs_to_tensors(obs, device)
            act, _, _, _ = model.get_action_and_value(
                img.unsqueeze(0), sca.unsqueeze(0))
            obs = env.step(DriveAction(
                accel=act.clamp(-1, 1)[0, 0].item(),
                steer=act.clamp(-1, 1)[0, 1].item(),
            ))
            total_r += obs.reward
            steps += 1
            if env._env.track.on_track(env._env._x, env._env._y):
                on_steps += 1

    ce = env._env
    on_pct = 100 * on_steps / max(1, steps)
    wp = ce._wp_idx

    result = {
        "track_level": track.level,
        "steps": steps,
        "on_track_pct": on_pct,
        "laps": ce._laps,
        "crashes": ce._crash_count,
        "total_reward": total_r,
        "wp_progress": wp,
    }

    # Diagnose
    if on_pct < 30:
        result["diagnosis"] = "NOT_STEERING"
        result["detail"] = (f"Car off-track {100-on_pct:.0f}% of the time. "
                            f"Not learning to steer.")
        result["fix"] = "Reduce initial speed or accel bias; increase on-track reward"
    elif ce._laps == 0 and wp < 20:
        result["diagnosis"] = "NOT_PROGRESSING"
        result["detail"] = (f"On-track {on_pct:.0f}% but only reached waypoint {wp}. "
                            f"Not making forward progress.")
        result["fix"] = "Increase waypoint progress reward (PROGRESS_SCALE)"
    elif ce._laps == 0 and wp >= 20:
        result["diagnosis"] = "ALMOST_LAPPING"
        result["detail"] = (f"Reached waypoint {wp}/{ce._n_wps} but didn't complete a lap. "
                            f"Close to breakthrough.")
        result["fix"] = "Continue training; optionally lower threshold"
    elif ce._laps >= 1 and ce._crash_count > 0:
        result["diagnosis"] = "LAPPING_WITH_CRASHES"
        result["detail"] = (f"Completed {ce._laps} lap(s) with {ce._crash_count} crash(es). "
                            f"Needs cleaner driving.")
        result["fix"] = "Continue training; crashes should decrease naturally"
    elif ce._laps >= 1 and ce._crash_count == 0:
        result["diagnosis"] = "MASTERING"
        result["detail"] = (f"Clean {ce._laps} lap(s), reward={total_r:.1f}. "
                            f"Should advance soon.")
        result["fix"] = "No fix needed; check threshold if not advancing"
    else:
        result["diagnosis"] = "UNKNOWN"
        result["detail"] = f"on_track={on_pct:.0f}%, laps={ce._laps}, crashes={ce._crash_count}"
        result["fix"] = "Manual inspection needed"

    return result


class _Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        self._stdout = sys.__stdout__
        self._file = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    @property
    def encoding(self):
        return self._stdout.encoding

    def reconfigure(self, **kwargs):
        pass


LOG_FILE = Path("run_training.log")
sys.stdout = _Tee(LOG_FILE)
sys.stderr = _Tee(LOG_FILE)


def find_summary():
    """Find the latest W&B summary JSON."""
    matches = sorted(glob.glob("wandb/run-*/files/wandb-summary.json"))
    return matches[-1] if matches else None


def read_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def latest_checkpoint():
    # SB3 checkpoints (.zip) take priority; fall back to legacy .pt files
    pts = sorted(glob.glob("checkpoints/ppo_sb3_step*.zip"))
    if pts:
        return pts[-1]
    pts = sorted(glob.glob("checkpoints/ppo_step*.pt"))
    return pts[-1] if pts else None


def kill_proc(proc):
    """Gracefully terminate training process."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def build_cmd(overrides=None, resume_path=None):
    """Build the train.py command with optional overrides and resume."""
    cmd = list(BASE_CMD)

    if resume_path:
        cmd.extend(["--resume", resume_path])

    if overrides:
        # Remove existing keys that we're overriding
        for key in overrides:
            # Find and remove existing key-value pair
            try:
                idx = cmd.index(key)
                cmd.pop(idx)      # remove key
                if idx < len(cmd) and not cmd[idx].startswith("--"):
                    cmd.pop(idx)  # remove value
            except ValueError:
                pass
            # Add override
            cmd.extend([key, overrides[key]])

    return cmd


def check_health(summary, step):
    """Check metrics against thresholds. Returns (ok, failures, fix_overrides)."""
    # Find the applicable threshold boundary
    applicable = None
    for boundary in sorted(THRESHOLDS.keys()):
        if boundary <= step:
            applicable = boundary
    if applicable is None:
        return True, [], {}

    min_reward, min_on_track, min_ev, max_grad = THRESHOLDS[applicable]

    reward   = summary.get("episode/reward",         float("nan"))
    on_track = summary.get("episode/on_track_pct",   float("nan"))
    ev       = summary.get("ppo/explained_variance",  float("nan"))
    grad     = summary.get("ppo/grad_norm",           float("nan"))
    kl       = summary.get("ppo/approx_kl",           float("nan"))
    stopped  = summary.get("ppo/early_stopped",       0)
    pl       = summary.get("ppo/policy_loss",         None)

    failures = []
    fixes = {}

    # Critical
    if pl != pl or pl is None:
        failures.append("CRITICAL: policy_loss is NaN")
        return False, failures, {}  # restart from scratch

    if stopped == 1 and (kl != kl or kl == 0.0):
        failures.append("CRITICAL: policy frozen (early_stopped=1, kl=0)")
        fixes.update({"--lr": "1e-4", "--max-grad-norm": "2.0"})

    # Soft failures
    if reward == reward and reward < min_reward:
        failures.append(f"reward {reward:.1f} < {min_reward}")
        fixes["--ent-coef-start"] = "0.02"

    if on_track == on_track and on_track < min_on_track:
        failures.append(f"on_track {on_track:.1f}% < {min_on_track}%")

    if grad == grad and grad > max_grad:
        failures.append(f"grad_norm {grad:.1f} > {max_grad}")
        fixes.update({"--max-grad-norm": "2.0", "--lr": "1e-4"})

    if kl == kl and kl > 0.03:
        failures.append(f"approx_kl {kl:.4f} > 0.03")
        fixes["--lr"] = "1e-4"

    if not failures:
        return True, [], {}
    return False, failures, fixes


def fmt(summary):
    s = summary
    return (
        f"step={s.get('global_step',0):>10,}  "
        f"lvl={s.get('curriculum/level',0)}  "
        f"reward={s.get('episode/reward', float('nan')):>8.1f}  "
        f"on_track={s.get('episode/on_track_pct', float('nan')):>5.1f}%  "
        f"ev={s.get('ppo/explained_variance', float('nan')):>6.3f}  "
        f"kl={s.get('ppo/approx_kl', float('nan')):>7.4f}  "
        f"laps={s.get('episode/laps', 0):.1f}  "
        f"crashes={s.get('episode/crashes', 0):.1f}"
    )


# ── Main Loop ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--fresh", action="store_true",
                   help="Start fresh (ignore existing checkpoints)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--stall-steps", type=int, default=STALL_STEPS,
                   help="Steps without curriculum advance before intervention")
    args = p.parse_args()

    # Update total steps in BASE_CMD
    idx = BASE_CMD.index("--total-steps")
    BASE_CMD[idx + 1] = str(args.total_steps)

    print("=" * 80)
    print("  AUTOMATED TRAINING PIPELINE")
    print(f"  Target: master all {TOTAL_TRACKS} curriculum tracks")
    print(f"  Budget: {args.total_steps:,} steps")
    print(f"  Stall threshold: {args.stall_steps:,} steps without advancement")
    print(f"  Log: {LOG_FILE}")
    print("=" * 80)

    proc = None
    stall_fix_idx = 0                # which STALL_FIXES entry to try next
    last_advance_step = 0            # step when curriculum last advanced
    last_level = 0                   # last known curriculum level
    checked_boundaries = set()
    consecutive_health_fails = 0
    run_count = 0

    try:
        while True:
            # ── Launch training ──────────────────────────────────────────
            ckpt = None if (run_count == 0 and args.fresh) else latest_checkpoint()
            overrides = None

            if ckpt is None:
                # Fresh start
                cmd = build_cmd()
            else:
                # Resume from latest checkpoint
                if stall_fix_idx > 0 and stall_fix_idx <= len(STALL_FIXES):
                    overrides = STALL_FIXES[stall_fix_idx - 1]
                cmd = build_cmd(overrides=overrides, resume_path=ckpt)

            run_count += 1
            print(f"\n{'─' * 80}")
            print(f"  [RUN {run_count}] {'(dry-run) ' if args.dry_run else ''}Launching training")
            if overrides:
                print(f"  [FIX] Applying: {overrides}")
            if ckpt:
                print(f"  [RESUME] From: {ckpt}")
            print(f"  CMD: {' '.join(cmd)}")
            print(f"{'─' * 80}\n")

            if args.dry_run:
                print("  (dry-run mode — exiting)")
                return

            proc = subprocess.Popen(
                cmd,
                stdout=sys.__stdout__,
                stderr=sys.__stderr__,
            )

            # ── Monitor loop ─────────────────────────────────────────────
            restart_needed = False
            restart_reason = ""

            while proc.poll() is None:
                time.sleep(POLL_INTERVAL)

                summary_path = find_summary()
                if not summary_path:
                    continue

                s = read_summary(summary_path)
                step  = s.get("global_step", 0)
                level = s.get("curriculum/level", 0)

                print(fmt(s))

                # ── Victory check ────────────────────────────────────────
                if level >= TOTAL_TRACKS - 1:
                    # Check if the window is full (agent is actually mastering the last track)
                    rolling = s.get("curriculum/rolling_mean", 0)
                    threshold = s.get("curriculum/threshold", 999)
                    if rolling >= threshold:
                        print("\n" + "=" * 80)
                        print("  ALL TRACKS MASTERED!")
                        print(f"  Final level: {level + 1}/{TOTAL_TRACKS}")
                        print(f"  Steps: {step:,}")
                        print("=" * 80)
                        kill_proc(proc)
                        return

                # ── Curriculum advancement tracking ──────────────────────
                if level > last_level:
                    print(f"\n  >>> ADVANCED to level {level} (track {level + 1}) at step {step:,}")
                    last_level = level
                    last_advance_step = step
                    stall_fix_idx = 0  # reset fix escalation
                    consecutive_health_fails = 0

                # ── Stall detection with video diagnosis ──────────────────
                steps_since_advance = step - last_advance_step
                if step > 0 and steps_since_advance >= args.stall_steps:
                    ckpt_for_diag = latest_checkpoint()
                    if ckpt_for_diag:
                        print(f"\n  [DIAG] Stall detected at level {level}. "
                              f"Running diagnosis on {ckpt_for_diag}...")
                        try:
                            diag = diagnose_from_checkpoint(ckpt_for_diag)
                            print(f"  [DIAG] {diag['diagnosis']}: {diag['detail']}")
                            print(f"  [DIAG] Suggested fix: {diag['fix']}")
                        except Exception as e:
                            print(f"  [DIAG] Failed: {e}")
                            diag = {"diagnosis": "UNKNOWN"}

                    if stall_fix_idx < len(STALL_FIXES):
                        restart_reason = (
                            f"Curriculum stalled at level {level} for "
                            f"{steps_since_advance:,} steps "
                            f"({diag.get('diagnosis', '?')}). "
                            f"Applying fix {stall_fix_idx + 1}/{len(STALL_FIXES)}."
                        )
                        stall_fix_idx += 1
                        last_advance_step = step  # reset stall timer
                        restart_needed = True
                        break
                    else:
                        print(f"\n  [WARN] All {len(STALL_FIXES)} stall fixes exhausted. "
                              f"Continuing without intervention.")
                        last_advance_step = step  # reset to avoid repeated warnings

                # ── Health check at 50K boundaries ───────────────────────
                for boundary in sorted(THRESHOLDS.keys()):
                    if boundary <= step and boundary not in checked_boundaries:
                        checked_boundaries.add(boundary)
                        ok, failures, fixes = check_health(s, step)
                        if ok:
                            print(f"  [HEALTH] PASS at {boundary // 1000}K")
                            consecutive_health_fails = 0
                        else:
                            print(f"  [HEALTH] ISSUES at {boundary // 1000}K: "
                                  f"{'; '.join(failures)}")
                            consecutive_health_fails += 1

                            # Only restart on 2 consecutive hard failures
                            hard = [f for f in failures if "CRITICAL" in f]
                            if hard:
                                restart_reason = f"Critical failure: {hard[0]}"
                                restart_needed = True
                                break
                            elif consecutive_health_fails >= 3 and fixes:
                                restart_reason = (
                                    f"3 consecutive health failures. "
                                    f"Applying: {fixes}"
                                )
                                # Merge health fixes into stall fixes
                                restart_needed = True
                                break

                # ── Budget check ─────────────────────────────────────────
                if step >= args.total_steps:
                    print(f"\n  [BUDGET] Reached {step:,} / {args.total_steps:,} steps.")
                    kill_proc(proc)
                    break

                if restart_needed:
                    break

            # ── Handle restart ───────────────────────────────────────────
            if restart_needed:
                print(f"\n  [RESTART] {restart_reason}")
                kill_proc(proc)
                time.sleep(5)  # let W&B flush
                continue

            # ── Training ended naturally ─────────────────────────────────
            exit_code = proc.wait() if proc.poll() is None else proc.returncode
            print(f"\n  [DONE] train.py exited with code {exit_code}")

            if exit_code != 0 and latest_checkpoint():
                print("  [RETRY] Non-zero exit — restarting from last checkpoint")
                time.sleep(5)
                continue

            break

    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Shutting down...")
        kill_proc(proc)
    finally:
        # Print final summary
        ckpt = latest_checkpoint()
        summary_path = find_summary()
        s = read_summary(summary_path) if summary_path else {}
        print(f"\n{'=' * 80}")
        print(f"  FINAL STATUS")
        print(f"  Level: {s.get('curriculum/level', '?')}/{TOTAL_TRACKS}")
        print(f"  Steps: {s.get('global_step', '?'):,}" if isinstance(s.get('global_step'), int) else "  Steps: ?")
        print(f"  Runs:  {run_count}")
        print(f"  Checkpoint: {ckpt or 'none'}")
        print(f"  Log: {LOG_FILE}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
