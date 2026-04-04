"""
Oval Car Racer
Controls: Arrow keys to drive, R to reset, ESC to quit.
"""

import math
import pygame

# ── Screen ──────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 900, 600
FPS = 60

# ── Oval geometry ────────────────────────────────────────────────────────────
CX, CY   = SCREEN_W // 2, SCREEN_H // 2
OUTER_RX, OUTER_RY = 380, 240
INNER_RX, INNER_RY = 290, 155
MID_RY   = (OUTER_RY + INNER_RY) // 2   # 197

START_X  = float(CX)
START_Y  = float(CY + MID_RY)

# ── Colours ─────────────────────────────────────────────────────────────────
C_GRASS  = ( 45, 110,  45)
C_TRACK  = ( 52,  52,  52)
C_WHITE  = (255, 255, 255)
C_YELLOW = (255, 215,   0)
C_CAR    = (220,  50,  50)
C_WIND   = (160, 210, 255)
C_HUD    = (230, 230, 230)
C_WARN   = (255,  70,  70)

# ── Car physics ──────────────────────────────────────────────────────────────
MAX_SPEED   = 4.5
ACCEL       = 0.13
BRAKE_DECEL = 0.22
FRICTION    = 0.038
STEER_DEG   = 2.7


# ────────────────────────────────────────────────────────────────────────────
# Track geometry
# ────────────────────────────────────────────────────────────────────────────

def _in_ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def on_track(x, y):
    return (_in_ellipse(x, y, CX, CY, OUTER_RX, OUTER_RY) and
            not _in_ellipse(x, y, CX, CY, INNER_RX, INNER_RY))


def _rect(cx, cy, rx, ry):
    return pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2)


# ────────────────────────────────────────────────────────────────────────────
# Drawing
# ────────────────────────────────────────────────────────────────────────────


def build_track_surface():
    surf = pygame.Surface((SCREEN_W, SCREEN_H))
    surf.fill(C_GRASS)

    # Tarmac
    pygame.draw.ellipse(surf, C_TRACK, _rect(CX, CY, OUTER_RX, OUTER_RY))
    pygame.draw.ellipse(surf, C_GRASS, _rect(CX, CY, INNER_RX, INNER_RY))

    # White borders
    bw = 3
    pygame.draw.ellipse(surf, C_WHITE, _rect(CX, CY, OUTER_RX, OUTER_RY), bw)
    pygame.draw.ellipse(surf, C_WHITE, _rect(CX, CY, INNER_RX, INNER_RY), bw)

    # Finish line — vertical white line at bottom of track
    line_y = CY + MID_RY
    track_w = OUTER_RX - INNER_RX
    line_x  = CX
    pygame.draw.line(surf, C_WHITE, (line_x, line_y - track_w // 2), (line_x, line_y + track_w // 2), 3)

    return surf


def draw_headlights(surf, x, y, angle_deg):
    CONE_LEN  = 60           # pixels ahead
    HALF_ANG  = 30           # half of 60-degree spread
    STEPS     = 12           # arc smoothness
    # Build cone polygon: origin + arc points
    pts = [(x, y)]
    for i in range(STEPS + 1):
        a = math.radians(angle_deg - HALF_ANG + (2 * HALF_ANG) * i / STEPS)
        pts.append((x + math.cos(a) * CONE_LEN,
                    y + math.sin(a) * CONE_LEN))

    # Draw on alpha surface so it blends with track
    cone = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    pygame.draw.polygon(cone, (255, 255, 180, 55), pts)   # soft yellow fill
    pygame.draw.lines(cone, (255, 255, 180, 120), False, pts[1:], 1)  # edge glow
    surf.blit(cone, (0, 0))


def draw_car(surf, x, y, angle_deg):
    w, h = 26, 12
    img  = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(img, C_CAR,  (0, 0, w, h),         border_radius=3)
    pygame.draw.rect(img, C_WIND, (w - 9, 2, 7, h - 4), border_radius=2)
    pygame.draw.rect(img, (255, 200, 0), (w - 3, 3, 3, h - 6))
    rot = pygame.transform.rotate(img, -angle_deg)
    surf.blit(rot, rot.get_rect(center=(int(x), int(y))))


def draw_hud(surf, speed, lap, best, last, off_track, lap_done):
    font  = pygame.font.SysFont("consolas", 20, bold=True)
    small = pygame.font.SysFont("consolas", 15)

    panel = pygame.Surface((240, 85), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    surf.blit(panel, (10, 10))

    def put(text, color, row):
        surf.blit(font.render(text, True, color), (18, 16 + row * 28))

    put(f"Speed : {abs(speed) * 65:5.1f} km/h", C_HUD, 0)
    put(f"Lap   : {lap}", C_HUD, 1)
    best_s = f"{best:.2f}s" if best < 1e8 else "--"
    last_s = f"{last:.2f}s" if last < 1e8 else "--"
    put(f"Last:{last_s}  Best:{best_s}", C_HUD, 2)

    if off_track:
        msg = font.render("! OFF TRACK !", True, C_WARN)
        surf.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, 12))

    if lap_done:
        msg = font.render("LAP COMPLETE!", True, C_YELLOW)
        surf.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, 52))

    hint = small.render("Arrows = drive    R = reset    ESC = quit", True, (150, 150, 150))
    surf.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 22))


# ────────────────────────────────────────────────────────────────────────────
# Car
# ────────────────────────────────────────────────────────────────────────────

class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x     = START_X
        self.y     = START_Y
        self.angle = 180.0
        self.speed = 0.0

    def update(self, accel, steer):
        speed_ratio = min(abs(self.speed) / MAX_SPEED, 1.0)
        self.angle += steer * STEER_DEG * max(0.3, speed_ratio)

        if accel > 0:
            self.speed = min(self.speed + ACCEL, MAX_SPEED)
        elif accel < 0:
            self.speed = max(self.speed - BRAKE_DECEL, -MAX_SPEED * 0.4)

        if self.speed > 0:
            self.speed = max(0.0, self.speed - FRICTION)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + FRICTION)

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

        if not on_track(self.x, self.y):
            self.speed *= 0.80


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Oval Car Racer")
    clock  = pygame.time.Clock()

    track_surf = build_track_surface()
    car        = Car()

    lap       = 0
    best_time = float("inf")
    last_time = float("inf")
    lap_start = pygame.time.get_ticks()
    flash     = 0
    prev_y    = car.y

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    car.reset()
                    prev_y    = car.y
                    lap_start = pygame.time.get_ticks()
                    flash     = 0

        keys  = pygame.key.get_pressed()
        accel = (1 if keys[pygame.K_UP]    else 0) - (1 if keys[pygame.K_DOWN]  else 0)
        steer = (1 if keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_LEFT] else 0)

        car.update(accel, steer)

        # Lap: car crosses start/finish line (y ~ START_Y) moving left, near CX
        near_x    = abs(car.x - CX) < (OUTER_RX - INNER_RX) // 2 + 10
        crossed   = prev_y < START_Y <= car.y   # crossed going downward
        if near_x and crossed and car.speed > 0.5:
            lap      += 1
            elapsed   = (pygame.time.get_ticks() - lap_start) / 1000.0
            last_time = elapsed
            best_time = min(best_time, elapsed)
            lap_start = pygame.time.get_ticks()
            flash     = FPS * 2

        prev_y = car.y
        if flash > 0:
            flash -= 1

        screen.blit(track_surf, (0, 0))
        draw_headlights(screen, car.x, car.y, car.angle)
        draw_car(screen, car.x, car.y, car.angle)
        draw_hud(screen, car.speed, lap, best_time, last_time,
                 not on_track(car.x, car.y), flash > 0)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
