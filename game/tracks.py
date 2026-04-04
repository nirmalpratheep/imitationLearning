"""
tracks.py — Track definitions for the curriculum car racer.

Angle convention (pygame y-down):
  0°  = right (+x)
  90° = down  (+y)
  180° = left  (-x)
  270° = up    (-y)
"""

import math
import pygame

SCREEN_W, SCREEN_H = 900, 600

# Colours
C_GRASS = (45, 110, 45)
C_TRACK = (52, 52, 52)
C_WHITE = (255, 255, 255)


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def _arc(cx, cy, rx, ry, a0_deg, a1_deg, n=24):
    """Return n+1 points along an elliptical arc from a0_deg to a1_deg."""
    pts = []
    for i in range(n + 1):
        t = a0_deg + (a1_deg - a0_deg) * i / n
        rad = math.radians(t)
        x = cx + rx * math.cos(rad)
        y = cy + ry * math.sin(rad)
        pts.append((x, y))
    return pts


def _full_ellipse(cx, cy, rx, ry, n=80, start_deg=90):
    """Return n+1 points of a full ellipse starting at start_deg."""
    return _arc(cx, cy, rx, ry, start_deg, start_deg + 360, n)


def _ipts(pts):
    """Convert float point list to integer tuples."""
    return [(int(round(x)), int(round(y))) for x, y in pts]


# ────────────────────────────────────────────────────────────────────────────
# TrackDef
# ────────────────────────────────────────────────────────────────────────────

class TrackDef:
    def __init__(self, level, name, waypoints, width, start_pos, start_angle, max_speed):
        self.level = level
        self.name = name
        self.waypoints = waypoints      # list of (x,y) floats
        self.width = width
        self.start_pos = start_pos      # (x, y) floats
        self.start_angle = start_angle  # degrees
        self.max_speed = max_speed

        self.surface = None
        self.mask = None
        self.hud_corner = (8, 8)  # default; updated after build()

        # Unit vector in start_angle direction (for gate_side)
        rad = math.radians(start_angle)
        self._gate_dx = math.cos(rad)
        self._gate_dy = math.sin(rad)

        # ── Reward metadata (computed once here, used by CarEnv) ─────────────
        # Perimeter of the waypoint polygon = approximate track centerline length
        self.optimal_dist = sum(
            math.hypot(waypoints[(i + 1) % len(waypoints)][0] - waypoints[i][0],
                       waypoints[(i + 1) % len(waypoints)][1] - waypoints[i][1])
            for i in range(len(waypoints))
        )

        # Expected lap time (frames) at 70 % of max speed — accounts for corners
        self.par_time_steps = self.optimal_dist / (max_speed * 0.70)

        # Difficulty multiplier: narrow + fast = harder
        # Track 1 (width=115, spd=3.0) → 1.0 | Track 16 (width=50, spd=4.5) → 3.45
        _BASE_WIDTH = 115.0
        _BASE_SPEED = 3.0
        self.complexity = (_BASE_WIDTH / width) * (max_speed / _BASE_SPEED)

    def _best_hud_corner(self, panel_w, panel_h, margin=8):
        """Return (x, y) of the screen corner with fewest track pixels under the HUD panel."""
        corners = [
            (margin, margin),
            (SCREEN_W - panel_w - margin, margin),
            (margin, SCREEN_H - panel_h - margin),
            (SCREEN_W - panel_w - margin, SCREEN_H - panel_h - margin),
        ]
        best_pos, best_count = corners[0], float('inf')
        for cx, cy in corners:
            count = sum(
                1
                for px in range(cx, cx + panel_w, 6)
                for py in range(cy, cy + panel_h, 6)
                if self.mask.get_at((px, py))[0] > 128
            )
            if count < best_count:
                best_count, best_pos = count, (cx, cy)
        return best_pos

    def build(self):
        """Draw the track onto self.surface and build self.mask."""
        BORDER = 6   # white border thickness on each edge (pixels)

        surf = pygame.Surface((SCREEN_W, SCREEN_H))
        surf.fill(C_GRASS)

        ipts_list = _ipts(self.waypoints)
        r      = self.width // 2
        r_out  = r + BORDER

        # 1. White outer strip — sets both edge borders
        pygame.draw.lines(surf, C_WHITE, True, ipts_list, self.width + BORDER * 2)
        for pt in ipts_list:
            pygame.draw.circle(surf, C_WHITE, pt, r_out)

        # 2. Grey tarmac — covers centre, leaves white only at the two edges
        pygame.draw.lines(surf, C_TRACK, True, ipts_list, self.width)
        for pt in ipts_list:
            pygame.draw.circle(surf, C_TRACK, pt, r)

        # 3. Checkered start / finish line across the full road width
        self._draw_start_finish(surf)

        self.surface = surf

        # Mask: covers the full road width (including border) so on_track
        # returns True all the way to the white edge lines.
        mask_surf = pygame.Surface((SCREEN_W, SCREEN_H))
        mask_surf.fill((0, 0, 0))
        pygame.draw.lines(mask_surf, C_WHITE, True, ipts_list, self.width + BORDER * 2)
        for pt in ipts_list:
            pygame.draw.circle(mask_surf, C_WHITE, pt, r_out)
        self.mask = mask_surf
        self.hud_corner = self._best_hud_corner(330, 175)

    def _draw_start_finish(self, surf):
        """
        Checkered black/white flag pattern across the track at start_pos,
        perpendicular to the driving direction.  2 rows × N columns of 10 px cells.
        """
        CELL = 10
        ROWS = 2
        sx, sy = self.start_pos

        # Unit vectors: across the track (perp) and along the track (along)
        perp_rad  = math.radians(self.start_angle + 90)
        along_rad = math.radians(self.start_angle)
        perp  = (math.cos(perp_rad),  math.sin(perp_rad))
        along = (math.cos(along_rad), math.sin(along_rad))

        n_cols = self.width // CELL + 4   # slightly wider than road
        half   = n_cols / 2.0

        for row in range(ROWS):
            v = (row - ROWS / 2.0 + 0.5) * CELL   # offset along driving dir
            for col in range(-int(half) - 1, int(half) + 2):
                u = col * CELL                      # offset across track
                color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
                # Four corners of this cell in screen space
                pts = []
                for du, dv in [(-CELL/2, -CELL/2), (CELL/2, -CELL/2),
                                (CELL/2,  CELL/2),  (-CELL/2, CELL/2)]:
                    px = sx + (u + du) * perp[0] + (v + dv) * along[0]
                    py = sy + (u + du) * perp[1] + (v + dv) * along[1]
                    pts.append((int(px), int(py)))
                pygame.draw.polygon(surf, color, pts)

    def on_track(self, x, y):
        """Return True if pixel (x, y) is on the track mask."""
        if self.mask is None:
            return False
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= SCREEN_W or iy >= SCREEN_H:
            return False
        color = self.mask.get_at((ix, iy))
        # White = on track
        return color[0] > 128

    def gate_side(self, x, y):
        """
        Dot product of (pos - start_pos) with start direction unit vector.
        Positive = ahead of gate, negative = behind gate.
        """
        dx = x - self.start_pos[0]
        dy = y - self.start_pos[1]
        return dx * self._gate_dx + dy * self._gate_dy


# ────────────────────────────────────────────────────────────────────────────
# Track builders
# ────────────────────────────────────────────────────────────────────────────

def _build_all_tracks():
    tracks = []

    # ── GROUP 1: Full ellipses ───────────────────────────────────────────────

    # 1. Wide Oval
    wp = _full_ellipse(450, 300, 370, 215, n=80, start_deg=90)
    tracks.append(TrackDef(
        level=1, name="Wide Oval",
        waypoints=wp, width=115,
        start_pos=(450, 515), start_angle=180, max_speed=3.0
    ))

    # 2. Standard Oval
    wp = _full_ellipse(450, 300, 330, 195, n=80, start_deg=90)
    tracks.append(TrackDef(
        level=2, name="Standard Oval",
        waypoints=wp, width=85,
        start_pos=(450, 495), start_angle=180, max_speed=3.5
    ))

    # 3. Narrow Oval
    wp = _full_ellipse(450, 300, 320, 185, n=80, start_deg=90)
    tracks.append(TrackDef(
        level=3, name="Narrow Oval",
        waypoints=wp, width=58,
        start_pos=(450, 485), start_angle=180, max_speed=3.5
    ))

    # 4. Superspeedway
    wp = _full_ellipse(450, 300, 395, 160, n=80, start_deg=90)
    tracks.append(TrackDef(
        level=4, name="Superspeedway",
        waypoints=wp, width=85,
        start_pos=(450, 460), start_angle=180, max_speed=4.5
    ))

    # ── GROUP 2: Rounded rectangles ─────────────────────────────────────────

    # 5. Rounded Rectangle
    # TL corner at (250,230), TR at (650,230), BR at (650,370), BL at (250,370), r=130
    # BUT with r=130, bottom of BR arc = 370+130=500, BL bottom = 370+130=500
    # arcs: TL 180→270, TR 270→360, BR 0→90, BL 90→180
    tl_arc = _arc(250, 230, 130, 130, 180, 270, 24)  # (120,230)→(250,100) wait...
    # TL center (250,230): 180° → (250-130,230)=(120,230), 270° → (250,230-130)=(250,100)
    tr_arc = _arc(650, 230, 130, 130, 270, 360, 24)  # (650,100)→(780,230)
    br_arc = _arc(650, 370, 130, 130, 0, 90, 24)    # (780,370)→(650,500)
    bl_arc = _arc(250, 370, 130, 130, 90, 180, 24)  # (250,500)→(120,370)
    wp = tl_arc + tr_arc + br_arc + bl_arc
    tracks.append(TrackDef(
        level=5, name="Rounded Rectangle",
        waypoints=wp, width=90,
        start_pos=(450, 500), start_angle=180, max_speed=3.5
    ))

    # 6. Stadium Oval
    left_arc = _arc(200, 300, 120, 120, 90, 270, 24)   # (200,420)→(200,180)
    right_arc = _arc(700, 300, 120, 120, 270, 450, 24)  # (700,180)→(700,420)
    wp = left_arc + right_arc
    tracks.append(TrackDef(
        level=6, name="Stadium Oval",
        waypoints=wp, width=80,
        start_pos=(450, 420), start_angle=180, max_speed=4.0
    ))

    # 7. Tight Rectangle
    # TL=(185,195), TR=(715,195), BR=(715,405), BL=(185,405), r=65
    tl_arc = _arc(185, 195, 65, 65, 180, 270, 24)
    tr_arc = _arc(715, 195, 65, 65, 270, 360, 24)
    br_arc = _arc(715, 405, 65, 65, 0, 90, 24)
    bl_arc = _arc(185, 405, 65, 65, 90, 180, 24)
    wp = tl_arc + tr_arc + br_arc + bl_arc
    tracks.append(TrackDef(
        level=7, name="Tight Rectangle",
        waypoints=wp, width=65,
        start_pos=(450, 470), start_angle=180, max_speed=3.5
    ))

    # 8. Small Oval
    wp = _full_ellipse(450, 300, 265, 165, n=80, start_deg=90)
    tracks.append(TrackDef(
        level=8, name="Small Oval",
        waypoints=wp, width=60,
        start_pos=(450, 465), start_angle=180, max_speed=3.2
    ))

    # ── GROUP 3: Two half-arcs ───────────────────────────────────────────────

    # 9. Hairpin Track
    # arc1: right gentle: (700,440)→(700,160) through (820,300)
    arc1 = _arc(700, 300, 120, 140, 90, -90, 24)
    # arc2: left tight: (220,160)→(220,440) through (140,300)
    arc2 = _arc(220, 300, 80, 140, 270, 90, 24)
    wp = arc1 + arc2
    tracks.append(TrackDef(
        level=9, name="Hairpin Track",
        waypoints=wp, width=75,
        start_pos=(460, 440), start_angle=0.0, max_speed=3.5
    ))

    # 10. Chicane Track
    # Rounded rect with chicane on bottom
    tl_arc = _arc(250, 240, 100, 100, 180, 270, 24)
    tr_arc = _arc(650, 240, 100, 100, 270, 360, 24)
    br_arc = _arc(650, 360, 100, 100, 0, 90, 24)    # ends at (650,460)
    bl_arc = _arc(250, 360, 100, 100, 90, 180, 24)  # starts at (250,460)
    # Chicane inserted between br_arc end and bl_arc start
    chicane = [(650, 460), (575, 460), (545, 498), (450, 498), (355, 498), (325, 460), (250, 460)]
    wp = tl_arc + tr_arc + br_arc + chicane + bl_arc
    tracks.append(TrackDef(
        level=10, name="Chicane Track",
        waypoints=wp, width=70,
        start_pos=(450, 498), start_angle=180, max_speed=3.5
    ))

    # 11. Double Hairpin
    arc1 = _arc(670, 300, 80, 140, 90, -90, 24)   # (670,440)→(670,160)
    arc2 = _arc(230, 300, 80, 140, 270, 90, 24)   # (230,160)→(230,440)
    wp = arc1 + arc2
    tracks.append(TrackDef(
        level=11, name="Double Hairpin",
        waypoints=wp, width=70,
        start_pos=(450, 440), start_angle=0.0, max_speed=3.5
    ))

    # 12. Asymmetric Track
    arc1 = _arc(700, 300, 140, 140, 90, -90, 24)  # right wide (700,440)→(700,160)
    arc2 = _arc(200, 300, 80, 140, 270, 90, 24)   # left tight (200,160)→(200,440)
    wp = arc1 + arc2
    tracks.append(TrackDef(
        level=12, name="Asymmetric Track",
        waypoints=wp, width=70,
        start_pos=(450, 440), start_angle=0.0, max_speed=3.8
    ))

    # ── GROUP 4: Polygon tracks ──────────────────────────────────────────────

    # 13. L-Shape Circuit
    wp = [
        (750, 510), (150, 510),
        (150, 140), (410, 140),
        (410, 310), (590, 310),
        (590, 140), (750, 140),
    ]
    tracks.append(TrackDef(
        level=13, name="L-Shape Circuit",
        waypoints=wp, width=72,
        start_pos=(450, 510), start_angle=180, max_speed=4.0
    ))

    # 14. T-Notch Circuit
    wp = [
        (780, 530), (120, 530), (120, 150), (360, 150),
        (360, 310), (540, 310), (540, 150), (780, 150),
    ]
    tracks.append(TrackDef(
        level=14, name="T-Notch Circuit",
        waypoints=wp, width=58,
        start_pos=(450, 530), start_angle=180, max_speed=4.0
    ))

    # 15. Complex Circuit
    wp = [
        (750, 490), (580, 490), (545, 450),
        (190, 450), (125, 380), (125, 195), (195, 135),
        (375, 135), (415, 185), (415, 315), (475, 350),
        (650, 350), (700, 295), (700, 160), (760, 130),
        (825, 150), (825, 430),
    ]
    tracks.append(TrackDef(
        level=15, name="Complex Circuit",
        waypoints=wp, width=65,
        start_pos=(660, 490), start_angle=180, max_speed=4.5
    ))

    # 16. Master Challenge (same waypoints as track 15, narrower)
    tracks.append(TrackDef(
        level=16, name="Master Challenge",
        waypoints=list(wp), width=50,
        start_pos=(660, 490), start_angle=180, max_speed=4.5
    ))

    return tracks


TRACKS = _build_all_tracks()
