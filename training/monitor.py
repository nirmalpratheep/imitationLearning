"""
monitor.py — Automatic 50k-step quality checks for PPO training.

Runs alongside train.py. Reads the latest W&B summary every 60 seconds,
fires a check at each 50k step boundary, and prints PASS/FAIL with the
exact fix command to run if a metric is out of range.

Usage:
    python monitor.py                        # auto-detects latest wandb run
    python monitor.py --run-id yopa1a8k      # monitor a specific run
    python monitor.py --interval 30          # poll every 30 s (default 60)
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path


class _Tee:
    """Write to both stdout and a log file simultaneously."""
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
        pass  # already line-buffered via buffering=1


LOG_FILE = Path("monitor.log")
sys.stdout = _Tee(LOG_FILE)

# ── Thresholds at each 50k boundary ──────────────────────────────────────────
# (min_reward, min_on_track_pct, min_explained_var, max_grad_norm)
THRESHOLDS = {
    # (min_reward, min_on_track_pct, min_explained_var, max_grad_norm)
    # Reward expectations account for waypoint progress (+5/lap) and speed (+0.05/step)
      50_000: (-500,  60, 0.50, 30),
     100_000: (-200,  75, 0.70, 15),
     150_000: ( -50,  80, 0.80, 12),
     200_000: (   0,  85, 0.85, 10),
     250_000: (  10,  88, 0.85, 10),
     300_000: (  20,  90, 0.88,  8),
     400_000: (  30,  92, 0.90,  8),
     500_000: (  40,  93, 0.90,  8),
     750_000: (  60,  94, 0.92,  7),
   1_000_000: (  80,  95, 0.93,  6),
   1_500_000: ( 100,  95, 0.93,  5),
   2_000_000: ( 130,  96, 0.95,  5),
   2_500_000: ( 150,  96, 0.95,  5),
   3_000_000: ( 170,  96, 0.95,  4),
   4_000_000: ( 190,  97, 0.96,  4),
   5_000_000: ( 200,  97, 0.96,  4),
   7_500_000: ( 200,  97, 0.96,  4),
  10_000_000: ( 200,  97, 0.96,  4),
}

CHECKPOINT_INTERVAL = 250_000  # must match --checkpoint-interval used in train.py


def find_summary(run_id=None):
    base = Path("wandb")
    if run_id:
        pattern = str(base / f"run-*{run_id}" / "files" / "wandb-summary.json")
    else:
        pattern = str(base / "run-*" / "files" / "wandb-summary.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def read_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def next_threshold(step):
    """Return the next 50k boundary at or above current step."""
    boundaries = sorted(THRESHOLDS.keys())
    for b in boundaries:
        if b >= step:
            return b
    return None


def check(s, boundary, prev_ev_failed=False):
    """
    Evaluate metrics against thresholds for a given boundary.
    EV is only a hard failure if it failed the PREVIOUS check too (avoids
    false alarms during rapid-improvement phases where critic lags policy).
    Returns (passed: bool, failures: list[str], fix_cmd: str | None).
    """
    min_reward, min_on_track, min_ev, max_grad = THRESHOLDS[boundary]

    reward   = s.get("episode/reward",          float("nan"))
    on_track = s.get("episode/on_track_pct",    float("nan"))
    ev       = s.get("ppo/explained_variance",  float("nan"))
    kl       = s.get("ppo/approx_kl",           float("nan"))
    grad     = s.get("ppo/grad_norm",            float("nan"))
    stopped  = s.get("ppo/early_stopped",        0)
    pl       = s.get("ppo/policy_loss",          None)
    step     = s.get("global_step",              0)

    failures = []
    fix_parts = []

    # Hard failures — kill immediately
    if pl != pl or pl is None:  # NaN check
        failures.append(f"  CRITICAL  ppo/policy_loss is NaN — no learning, likely log-prob bug")
        return False, failures, None  # restart from scratch, no resume

    if stopped == 1 and (kl != kl or kl == 0.0):
        failures.append(f"  CRITICAL  early_stopped=1 every update + kl=0/NaN — policy frozen")
        fix_parts.append("--lr 1e-4 --max-grad-norm 2.0")

    # Soft failures — resume with fix
    if reward < min_reward:
        failures.append(f"  FAIL  episode/reward {reward:.1f} < {min_reward}")
        fix_parts.append("--ent-coef-start 0.02")

    if on_track < min_on_track:
        failures.append(f"  FAIL  on_track_pct {on_track:.1f}% < {min_on_track}%")
        fix_parts.append("--ent-coef-start 0.03")

    if ev < min_ev:
        if prev_ev_failed:
            failures.append(f"  FAIL  explained_variance {ev:.3f} < {min_ev} (2nd consecutive check)")
            fix_parts.append("--vf-coef 1.0")
        else:
            # First time EV is low — warn but don't trigger fix (critic lags during fast improvement)
            failures.append(f"  WARN  explained_variance {ev:.3f} < {min_ev} (watching, will act if still low next check)")

    if grad > max_grad:
        failures.append(f"  WARN  grad_norm {grad:.1f} > {max_grad}")
        fix_parts.append("--max-grad-norm 2.0 --lr 1e-4")

    if kl == kl and kl > 0.02:
        failures.append(f"  WARN  approx_kl {kl:.4f} > 0.02")
        fix_parts.append("--lr 1e-4")

    if not failures:
        return True, [], None

    # Build resume command from latest checkpoint
    ckpts = sorted(glob.glob("checkpoints/*.pt"))
    # Deduplicate flags: last flag wins for repeated keys (e.g. two --ent-coef-start)
    seen_keys = {}
    for part in fix_parts:
        tokens = part.strip().split()
        i = 0
        while i < len(tokens):
            if tokens[i].startswith("--") and i + 1 < len(tokens) and not tokens[i+1].startswith("--"):
                seen_keys[tokens[i]] = tokens[i+1]
                i += 2
            elif tokens[i].startswith("--"):
                seen_keys[tokens[i]] = None
                i += 1
            else:
                i += 1
    deduped = " ".join(
        f"{k} {v}" if v is not None else k
        for k, v in seen_keys.items()
    )

    if ckpts:
        latest = ckpts[-1]
        fix_cmd = (f"python train.py --resume {latest} --num-envs 8 --compile "
                   f"--checkpoint-interval {CHECKPOINT_INTERVAL} {deduped}")
    else:
        fix_cmd = (f"python train.py --num-envs 8 --compile "
                   f"--checkpoint-interval {CHECKPOINT_INTERVAL} {deduped}")

    return False, failures, fix_cmd


def fmt_metrics(s):
    return (
        f"  step={s.get('global_step',0):>8,}  "
        f"reward={s.get('episode/reward', float('nan')):>8.1f}  "
        f"on_track={s.get('episode/on_track_pct', float('nan')):>5.1f}%  "
        f"ev={s.get('ppo/explained_variance', float('nan')):>6.3f}  "
        f"kl={s.get('ppo/approx_kl', float('nan')):>7.4f}  "
        f"grad={s.get('ppo/grad_norm', float('nan')):>6.1f}  "
        f"stopped={int(s.get('ppo/early_stopped', 0))}  "
        f"lvl={s.get('curriculum/level', 0)}  "
        f"sps={s.get('system/steps_per_sec', 0):.0f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id",   default=None, help="W&B run ID to monitor")
    p.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = p.parse_args()

    print(f"Monitor started — polling every {args.interval}s")
    print(f"{'Step':>10}  {'Reward':>8}  {'OnTrack':>8}  {'EV':>6}  "
          f"{'KL':>7}  {'GradN':>6}  {'Stop':>4}  {'Lvl':>3}  {'SPS':>5}")
    print("-" * 90)

    checked_boundaries = set()
    failed_ev_boundaries = set()   # boundaries where EV failed last check
    last_step = -1

    while True:
        summary_path = find_summary(args.run_id)
        if not summary_path:
            print("  [waiting for wandb run to start...]")
            time.sleep(args.interval)
            continue

        s = read_summary(summary_path)
        step = s.get("global_step", 0)

        if step != last_step:
            print(fmt_metrics(s))
            last_step = step

        # Find the highest 50k boundary we have crossed but not yet checked
        for boundary in sorted(THRESHOLDS.keys()):
            if boundary <= step and boundary not in checked_boundaries:
                checked_boundaries.add(boundary)
                prev_ev_failed = boundary in failed_ev_boundaries
                passed, failures, fix_cmd = check(s, boundary, prev_ev_failed)

                # Track EV failures for next check
                ev = s.get("ppo/explained_variance", 1.0)
                min_ev = THRESHOLDS[boundary][2]
                if ev < min_ev:
                    failed_ev_boundaries.add(boundary)
                else:
                    failed_ev_boundaries.discard(boundary)

                # Only hard-fail if there are actionable failures (not just EV warnings)
                hard_failures = [f for f in failures if "FAIL" in f or "CRITICAL" in f]
                sep = "=" * 70
                if not hard_failures:
                    print(f"\n{sep}")
                    print(f"  PASS  {boundary//1000}k check at step {step:,}")
                    for f in failures:  # print any WARNs
                        print(f)
                    print(f"{sep}\n")
                else:
                    print(f"\n{sep}")
                    print(f"  FAIL  {boundary//1000}k check at step {step:,}")
                    for f in failures:
                        print(f)
                    if fix_cmd:
                        print(f"\n  Fix — kill train.py then run:")
                        print(f"    {fix_cmd}")
                    else:
                        print(f"\n  Fix — restart from scratch (no checkpoint to resume):")
                        print(f"    python train.py --num-envs 8 --compile "
                              f"--checkpoint-interval {CHECKPOINT_INTERVAL} --lr 1e-4 --max-grad-norm 2.0")
                    print(f"{sep}\n")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
