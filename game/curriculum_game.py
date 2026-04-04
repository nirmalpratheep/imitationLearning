"""
Curriculum Car Racer — One-lap challenge.

Rules:
  * Complete one full lap without touching the fence (white border).
  * Touching the fence OR pressing R = OUT -> restart from start, attempt +1.
  * Trace path and distance covered reset on every restart.
  * Game ends when the finish line (= start line) is crossed cleanly.

Controls:
  Arrow keys  drive
  N / P       next / prev track
  1-9         jump to track 1-9
  R           manual restart (counts as an attempt)
  ESC         quit
"""

import math
import pygame

from .oval_racer import SCREEN_W, SCREEN_H, draw_headlights, draw_car
from .tracks import TRACKS

FPS = 60

ACCEL       = 0.13
BRAKE_DECEL = 0.22
FRICTION    = 0.038
STEER_DEG   = 2.7

C_YELLOW = (255, 215,   0)
C_HUD    = (230, 230, 230)
C_GREEN  = ( 50, 220,  80)
C_BLUE   = ( 60, 140, 255)

RACING = "racing"
DONE   = "done"

PATH_SAMPLE_EVERY = 2   # record a path point every N frames


# ── Car ──────────────────────────────────────────────────────────────────────

class Car:
    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        self.x     = float(self.track.start_pos[0])
        self.y     = float(self.track.start_pos[1])
        self.angle = float(self.track.start_angle)
        self.speed = 0.0

    def update(self, accel, steer):
        ms    = self.track.max_speed
        ratio = min(abs(self.speed) / ms, 1.0) if ms > 0 else 0.0
        self.angle += steer * STEER_DEG * max(0.3, ratio)
        if accel > 0:
            self.speed = min(self.speed + ACCEL, ms)
        elif accel < 0:
            self.speed = max(self.speed - BRAKE_DECEL, -ms * 0.4)
        if self.speed > 0:
            self.speed = max(0.0, self.speed - FRICTION)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + FRICTION)
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_path(surf, pts, color, width=2):
    if len(pts) >= 2:
        ipts = [(int(x), int(y)) for x, y in pts]
        pygame.draw.lines(surf, color, False, ipts, width)


def draw_hud(surf, track, car, race, fonts):
    _, small = fonts

    lt = race.lap_elapsed()
    text = (
        f"Lv{track.level}: {track.name}"
        f"   Spd {abs(car.speed)*FPS:4.1f}"
        f"   Attempt {race.attempts}"
        f"   Lap {lt:.2f}s"
        f"   Total {race.total_elapsed():.2f}s"
        f"   Dist {race.current_distance:.0f}px"
        f"   Max {race._max_spd:.1f}"
        f"   |  Arrows=drive  N/P=track  1-9=jump  R=restart  ESC=quit"
    )

    rendered = small.render(text, True, C_HUD)
    bar_h = rendered.get_height() + 4
    bar = pygame.Surface((SCREEN_W, bar_h), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 200))
    surf.blit(bar, (0, 0))
    surf.blit(rendered, (6, 2))


def draw_summary(surf, race, fonts):
    """Blocking summary overlay shown when finish line is crossed."""
    font, small = fonts
    big = pygame.font.SysFont("consolas", 38, bold=True)
    med = pygame.font.SysFont("consolas", 22, bold=True)

    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    surf.blit(overlay, (0, 0))

    cx = SCREEN_W // 2

    def centre(text, color, fnt, y):
        s = fnt.render(text, True, color)
        surf.blit(s, (cx - s.get_width() // 2, y))

    centre("FINISH!", C_GREEN, big, 130)

    pygame.draw.line(surf, (100, 100, 100), (cx - 220, 183), (cx + 220, 183), 1)

    rows = [
        ("Lap time",    f"{race.lap_time:.2f} s"),
        ("Total time",  f"{race.total_time:.2f} s"),
        ("Distance",    f"{race.lap_dist:.0f} px"),
        ("Max speed",   f"{race.lap_max_spd:.1f} px/s"),
        ("Avg speed",   f"{race.lap_avg_spd:.1f} px/s"),
        ("Attempts",    str(race.attempts)),
    ]
    label_x = cx - 130
    value_x = cx + 140
    for i, (label, value) in enumerate(rows):
        y = 196 + i * 28
        surf.blit(med.render(label, True, (170, 170, 170)), (label_x, y))
        surf.blit(med.render(value, True, C_HUD),
                  (value_x - med.size(value)[0], y))

    pygame.draw.line(surf, (100, 100, 100), (cx - 220, 368), (cx + 220, 368), 1)
    verdict = "Perfect run - no restarts!" if race.attempts == 1 else \
              f"Finished in {race.attempts} attempts"
    centre(verdict, C_YELLOW, font, 378)
    centre("R = retry   N/P = change track   ESC = quit",
           (150, 150, 150), small, 418)


# ── Race state ────────────────────────────────────────────────────────────────

class RaceState:
    def __init__(self, track):
        self.track       = track
        self.car         = Car(track)
        self.state       = RACING
        self.attempts    = 1

        self._lap_timer_started   = False  # starts on first key press per attempt
        self._total_timer_started = False  # starts on first key press ever
        self.total_start = None
        self.lap_start   = None

        self.lap_time    = 0.0   # locked on finish
        self.lap_dist    = 0.0   # locked on finish
        self.total_time  = 0.0   # locked on finish
        self.lap_max_spd = 0.0   # locked on finish (px/s)
        self.lap_avg_spd = 0.0   # locked on finish (px/s)

        self.prev_side   = track.gate_side(self.car.x, self.car.y)
        self._lap_armed  = False  # True once car is clearly past the gate

        # Path trace + speed — cleared on every reset
        self.current_path     = []
        self.current_distance = 0.0   # px covered this attempt
        self._max_spd         = 0.0   # peak speed this attempt (px/s)
        self._spd_sum         = 0.0   # for rolling average
        self._spd_count       = 0
        self._frame           = 0
        self._prev_x          = self.car.x
        self._prev_y          = self.car.y

    # ── helpers ──────────────────────────────────────────────────────────────

    def lap_elapsed(self):
        if not self._lap_timer_started:
            return 0.0
        return (pygame.time.get_ticks() - self.lap_start) / 1000.0

    def total_elapsed(self):
        if self.state == DONE:
            return self.total_time
        if not self._total_timer_started:
            return 0.0
        return (pygame.time.get_ticks() - self.total_start) / 1000.0

    def _record(self):
        """Accumulate distance + speed stats every frame; record path every N frames."""
        dx = self.car.x - self._prev_x
        dy = self.car.y - self._prev_y
        self.current_distance += math.hypot(dx, dy)
        self._prev_x, self._prev_y = self.car.x, self.car.y

        pps = abs(self.car.speed) * FPS
        if pps > self._max_spd:
            self._max_spd = pps
        self._spd_sum   += pps
        self._spd_count += 1

        self._frame += 1
        if self._frame % PATH_SAMPLE_EVERY == 0:
            self.current_path.append((self.car.x, self.car.y))

    def _reset_attempt(self):
        """Clear trace, distance, speed stats, and reset car to start."""
        self.current_path     = []
        self.current_distance = 0.0
        self._max_spd         = 0.0
        self._spd_sum         = 0.0
        self._spd_count       = 0
        self._frame           = 0
        self.car.reset()
        self._prev_x  = self.car.x
        self._prev_y  = self.car.y
        self.attempts   += 1
        self.lap_start          = None
        self._lap_timer_started = False
        self.prev_side   = self.track.gate_side(self.car.x, self.car.y)
        self._lap_armed  = False

    def manual_reset(self):
        self._reset_attempt()

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, accel, steer):
        if self.state == DONE:
            return

        if (accel != 0 or steer != 0) and not self._lap_timer_started:
            now = pygame.time.get_ticks()
            self.lap_start = now
            self._lap_timer_started = True
            if not self._total_timer_started:
                self.total_start = now
                self._total_timer_started = True

        self.car.update(accel, steer)
        self._record()

        # Fence hit → OUT, clear trace
        if not self.track.on_track(self.car.x, self.car.y):
            self._reset_attempt()
            return

        # Finish line detection (same line as start).
        # Phase 1 — arm: wait until car is 50 px ahead of gate going forward.
        # Phase 2 — trigger: detect when gate_side crosses from negative → positive.
        # This avoids the < -5 threshold bug where fast cars skip the window.
        curr_side = self.track.gate_side(self.car.x, self.car.y)
        if not self._lap_armed and curr_side > 50:
            self._lap_armed = True
        if self._lap_armed and self.prev_side < 0 and curr_side >= 0 and self.car.speed > 0.3:
            self.lap_time    = self.lap_elapsed()
            self.lap_dist    = self.current_distance
            self.total_time  = self.total_elapsed()
            self.lap_max_spd = self._max_spd
            self.lap_avg_spd = (self._spd_sum / self._spd_count
                                if self._spd_count else 0.0)
            self.state       = DONE
        self.prev_side = curr_side

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, surf, fonts):
        surf.blit(self.track.surface, (0, 0))

        # Current attempt path in blue (cleared after every reset)
        _draw_path(surf, self.current_path, C_BLUE, width=2)

        draw_headlights(surf, self.car.x, self.car.y, self.car.angle)
        draw_car(surf, self.car.x, self.car.y, self.car.angle)

        if self.state == RACING:
            draw_hud(surf, self.track, self.car, self, fonts)
        else:
            draw_summary(surf, self, fonts)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(start_track=1):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()
    fonts  = (pygame.font.SysFont("consolas", 20, bold=True),
              pygame.font.SysFont("consolas", 14))

    track_idx = max(0, min(start_track - 1, len(TRACKS) - 1))

    def new_race(idx):
        t = TRACKS[idx]
        t.build()
        pygame.display.set_caption(f"Curriculum Racer  Lv{t.level}: {t.name}")
        return RaceState(t)

    race = new_race(track_idx)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # After finish: full retry (attempt counter resets to 1)
                    # While racing: counts as an attempt
                    if race.state == DONE:
                        race = new_race(track_idx)
                    else:
                        race.manual_reset()

                elif event.key == pygame.K_n:
                    track_idx = (track_idx + 1) % len(TRACKS)
                    race = new_race(track_idx)

                elif event.key == pygame.K_p:
                    track_idx = (track_idx - 1) % len(TRACKS)
                    race = new_race(track_idx)

                else:
                    for ki, key in enumerate([
                        pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                        pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9
                    ]):
                        if event.key == key and ki < len(TRACKS):
                            track_idx = ki
                            race = new_race(track_idx)
                            break

        if race.state == RACING:
            keys  = pygame.key.get_pressed()
            accel = (1 if keys[pygame.K_UP]    else 0) - (1 if keys[pygame.K_DOWN]  else 0)
            steer = (1 if keys[pygame.K_RIGHT]  else 0) - (1 if keys[pygame.K_LEFT]  else 0)
            race.step(accel, steer)

        race.draw(screen, fonts)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    import sys
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run(start_track=level)
