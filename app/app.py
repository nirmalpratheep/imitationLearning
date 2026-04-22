"""
app.py — Car Racing Agent · Gradio Demo
OpenEnv Student Challenge 2026 · NirmalPratheep
"""

import os, sys, tempfile, uuid, traceback

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Local layout: app.py lives in app/, repo root is parent.
# HF Space layout: app.py sits at root alongside game/, env/, training/.
_ROOT    = _APP_DIR if os.path.isdir(os.path.join(_APP_DIR, "game")) else os.path.dirname(_APP_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "training"))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from game.rl_splits import TRAIN, _ensure_pygame
from env.environment import RaceEnvironment
from env.models import DriveAction
from train_torchrl import build_policy_and_value

_ensure_pygame()
import pygame
from game.oval_racer import draw_car, draw_headlights

# ── Constants ─────────────────────────────────────────────────────────────────
STEP_CHUNK        = 20
MAX_STEPS         = 3000
LAPS_TARGET       = 1
FRAME_W, FRAME_H  = 540, 360
HEADLIGHT_PX      = 192
DEVICE            = torch.device("cpu")

_CKPT_CANDIDATES  = [
    os.path.join(_APP_DIR, "ppo_torchrl_final.pt"),
    os.path.join(_ROOT,    "ppo_torchrl_final.pt"),
    os.path.join(_ROOT,    "checkpoints", "ppo_torchrl_final.pt"),
]
CKPT_PATH = next((p for p in _CKPT_CANDIDATES if os.path.isfile(p)), None)
POLICY    = None


def _load_policy():
    global POLICY
    if POLICY is not None:
        return POLICY
    if CKPT_PATH is None:
        raise RuntimeError("Checkpoint not found. Put ppo_torchrl_final.pt in app/ or checkpoints/")
    policy, _, _ = build_policy_and_value(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd   = ckpt.get("policy", ckpt.get("model", {}))
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    policy.load_state_dict(sd)
    policy.eval()
    POLICY = policy
    print(f"✓ Policy loaded: {CKPT_PATH}")
    return POLICY


# ── Rendering ─────────────────────────────────────────────────────────────────

def _surf_to_pil(surf, size) -> Image.Image:
    small = pygame.transform.scale(surf, size)
    arr   = pygame.surfarray.array3d(small).transpose(1, 0, 2)
    return Image.fromarray(arr.astype(np.uint8))


def _game_frame(race_env, trail=None) -> Image.Image:
    ce   = race_env._env
    surf = ce.track.surface.copy()
    # Draw path trail before car so car renders on top
    if trail and len(trail) > 1:
        scale_x = FRAME_W / 900
        scale_y = FRAME_H / 600
        for i, (px, py) in enumerate(trail):
            alpha = max(60, int(255 * i / len(trail)))
            r = max(2, int(4 * i / len(trail)))
            color = (255, int(140 * i / len(trail)), 0)  # orange fade-in
            pygame.draw.circle(surf, color, (int(px), int(py)), r)
    draw_headlights(surf, ce._x, ce._y, ce._angle)
    draw_car(surf, ce._x, ce._y, ce._angle)
    return _surf_to_pil(surf, (FRAME_W, FRAME_H))


def _headlight_frame(race_env) -> Image.Image:
    img64 = race_env._render_headlight_image()
    return Image.fromarray(img64).resize((HEADLIGHT_PX, HEADLIGHT_PX), Image.NEAREST)


def _placeholder(w, h, text, bg=(15, 17, 26), fg=(80, 90, 120)) -> Image.Image:
    img  = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=fg, font=font)
    return img


def _placeholder_main():
    return _placeholder(FRAME_W, FRAME_H, "← Click  Reset  to load the track")

def _placeholder_pov():
    return _placeholder(HEADLIGHT_PX, HEADLIGHT_PX, "POV", fg=(100, 110, 140))


def _agent_action(obs):
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    policy  = _load_policy()
    img     = (torch.from_numpy(obs.image.copy())
               .float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE))
    scalars = torch.tensor(obs.scalars, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    td      = TensorDict({"image": img, "scalars": scalars}, batch_size=[1])
    with set_exploration_type(ExplorationType.MEAN):
        td = policy(td)
    a = td["action"][0].detach().clamp(-1.0, 1.0).cpu().numpy()
    return float(a[0]), float(a[1])


# ── Status HTML ───────────────────────────────────────────────────────────────
TRACK_CHOICES = [f"Track {t.level:02d} — {t.name}" for t in TRAIN]

def _get_track(label: str):
    lvl = int(label.split(" ")[1])
    return next((t for t in TRAIN if t.level == lvl), TRAIN[0])


def _stat_card(label, value, cls=""):
    return (f"<div class='sc'><div class='sl'>{label}</div>"
            f"<div class='sv {cls}'>{value}</div></div>")


def _status_html(state, error=None):
    if error:
        return (f"<div class='stat-row'>"
                f"<div class='sc err'><div class='sl'>ERROR</div>"
                f"<div class='sv' style='font-size:.7rem;color:#ff6b6b;word-break:break-all'>{error}</div></div>"
                f"</div>")
    if state is None:
        return (f"<div class='stat-row'>"
                + _stat_card("STATUS", "IDLE", "idle")
                + _stat_card("SPEED", "—")
                + _stat_card("LAPS", "—")
                + _stat_card("STEPS", "—")
                + "</div>")
    ce    = state["env"]._env
    laps  = ce._laps
    speed = ce._speed
    step  = state["step"]
    done  = state["done"]
    spd   = f"{int(abs(speed)/ce.track.max_speed*100)}%"
    if done and laps >= LAPS_TARGET:
        st, cls = "✅ PASS", "pass"
    elif done:
        st, cls = "💥 CRASH", "fail"
    else:
        st, cls = "▶ RUNNING", "run"
    return (f"<div class='stat-row'>"
            + _stat_card("STATUS", st, cls)
            + _stat_card("SPEED", spd)
            + _stat_card("LAPS", f"{laps}/{LAPS_TARGET}")
            + _stat_card("STEPS", f"{step}")
            + "</div>")


# ── Callbacks ─────────────────────────────────────────────────────────────────

def reset(track_label):
    try:
        _ensure_pygame()
        track = _get_track(track_label)
        track.build()
        env   = RaceEnvironment(track, max_steps=MAX_STEPS, laps_target=LAPS_TARGET, use_image=True)
        obs   = env.reset()
        state = {"env": env, "obs": obs, "step": 0, "done": False, "trail": []}
        return state, _game_frame(env), _headlight_frame(env), _status_html(state), None
    except Exception as e:
        traceback.print_exc()
        return None, _placeholder_main(), _placeholder_pov(), _status_html(None, str(e)), None


def step_agent(state):
    """Generator: yields a frame every 4 physics steps so the image streams smoothly."""
    try:
        if state is None:
            yield (state, _placeholder_main(), _placeholder_pov(),
                   _status_html(None, "No session — press Reset first"), None)
            return
        if state["done"]:
            yield (state, _game_frame(state["env"], state.get("trail")),
                   _headlight_frame(state["env"]), _status_html(state), None)
            return

        env   = state["env"]
        obs   = state["obs"]
        trail = state.setdefault("trail", [])
        YIELD_EVERY = 4   # stream a new frame every N physics steps

        for i in range(STEP_CHUNK):
            if state["done"]:
                break
            accel, steer  = _agent_action(obs)
            obs            = env.step(DriveAction(accel=accel, steer=steer))
            state["step"] += 1
            trail.append((env._env._x, env._env._y))
            if obs.done:
                state["done"] = True
            # stream intermediate frames
            if (i + 1) % YIELD_EVERY == 0 or state["done"]:
                yield (state,
                       _game_frame(env, trail),
                       _headlight_frame(env),
                       _status_html(state),
                       None)

        state["obs"] = obs
        yield state, _game_frame(env, trail), _headlight_frame(env), _status_html(state), None

    except Exception as e:
        traceback.print_exc()
        yield state, _placeholder_main(), _placeholder_pov(), _status_html(state, str(e)), None


def auto_drive(track_label):
    """Generator: streams live frames while driving, then yields the final MP4."""
    try:
        import imageio.v3 as iio
        _ensure_pygame()
        track = _get_track(track_label)
        track.build()
        env   = RaceEnvironment(track, max_steps=MAX_STEPS, laps_target=LAPS_TARGET, use_image=True)
        obs   = env.reset()
        trail  = []
        frames = [np.array(_game_frame(env))]
        n      = 0
        YIELD_EVERY = 3   # stream a UI update every N steps

        # Show initial frame
        state = {"env": env, "obs": obs, "step": 0, "done": False, "trail": trail}
        yield state, _game_frame(env, trail), _headlight_frame(env), _status_html(state), None

        while not obs.done:
            accel, steer = _agent_action(obs)
            obs           = env.step(DriveAction(accel=accel, steer=steer))
            n            += 1
            trail.append((env._env._x, env._env._y))

            frame_img = _game_frame(env, trail)
            frames.append(np.array(frame_img))

            if n % YIELD_EVERY == 0:
                state["step"] = n
                yield (state, frame_img, _headlight_frame(env), _status_html(state), None)

        # Final state
        ce      = env._env
        laps    = ce._laps
        crashes = getattr(ce, "_crash_count", 0)
        result  = "✅ PASS" if laps >= LAPS_TARGET and crashes == 0 else "💥 FAIL"

        vpath = os.path.join(tempfile.gettempdir(), f"race_{uuid.uuid4().hex[:8]}.mp4")
        iio.imwrite(vpath, np.stack(frames), fps=20, codec="libx264", plugin="pyav")

        state = {"env": env, "obs": obs, "step": n, "done": True, "trail": trail}
        extra = (f"<div class='stat-row'>"
                 + _stat_card("RESULT", result, "pass" if "PASS" in result else "fail")
                 + _stat_card("LAPS", str(laps))
                 + _stat_card("STEPS", str(n))
                 + _stat_card("FRAMES", str(len(frames)))
                 + "</div>")
        yield state, _game_frame(env, trail), _headlight_frame(env), extra, vpath

    except Exception as e:
        traceback.print_exc()
        yield None, _placeholder_main(), _placeholder_pov(), _status_html(None, str(e)), None


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* ─────────────────────────────────────────────────────────────
   Car Racing Agent · premium light theme
   palette: warm orange accent on neutral stone background
   ───────────────────────────────────────────────────────────── */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background:
        radial-gradient(1200px 600px at 10% -10%, #ffedd5 0%, transparent 50%),
        radial-gradient(900px 500px at 110% 10%, #fef3c7 0%, transparent 45%),
        linear-gradient(180deg, #fafafa 0%, #f3f4f6 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 13px !important;
    color: #1f2937 !important;
    min-height: 100vh;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* strip default Gradio borders — we'll add our own where needed */
.gr-box, .gr-form, .gr-panel,
div[class*="component-"], div[data-testid] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════════════════════
   BANNER — hero strip
   ═══════════════════════════════════════════════════════════════ */
.banner {
    position: relative;
    background:
        linear-gradient(135deg, #ffffff 0%, #fff7ed 100%);
    border-bottom: 1px solid rgba(249, 115, 22, 0.15);
    padding: 22px 32px 20px;
    margin-bottom: 20px;
    overflow: hidden;
}
.banner::before {
    content: ""; position: absolute; inset: 0;
    background:
        radial-gradient(500px 200px at 15% 120%, rgba(249,115,22,.18), transparent 70%),
        radial-gradient(400px 180px at 90% -30%, rgba(251,191,36,.22), transparent 70%);
    pointer-events: none;
}
.banner-inner { max-width: 1280px; margin: 0 auto; position: relative; z-index: 1; }
.banner h1 {
    font-size: 1.85rem; font-weight: 900; margin: 0 0 6px;
    letter-spacing: -0.02em;
    background: linear-gradient(92deg, #9a3412 0%, #ea580c 40%, #f97316 65%, #fbbf24 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: inline-block;
    filter: drop-shadow(0 1px 0 rgba(255,255,255,.6));
}
.banner .sub {
    color: #57534e; font-size: 0.83rem; margin: 0 0 12px;
    font-weight: 500; line-height: 1.5;
}
.banner .sub strong { font-weight: 700; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.badge {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid #fed7aa;
    color: #9a3412;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: .05em;
    backdrop-filter: blur(6px);
    box-shadow: 0 1px 2px rgba(120, 53, 15, .04);
    transition: transform .12s ease, box-shadow .12s ease;
}
.badge:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(120, 53, 15, .10); }
.badge.green {
    background: rgba(240, 253, 244, .9); border-color: #86efac; color: #166534;
}
.badge.hf-badge {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-color: #fbbf24; color: #78350f;
    text-decoration: none; cursor: pointer;
    font-weight: 800;
}
.badge.hf-badge:hover { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); }

/* ═══════════════════════════════════════════════════════════════
   STAT CARDS
   ═══════════════════════════════════════════════════════════════ */
.stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
.sc {
    flex: 1; min-width: 64px;
    background: #ffffff;
    border: 1px solid #e7e5e4 !important;
    border-radius: 10px !important;
    padding: 10px 8px;
    text-align: center;
    box-shadow:
        0 1px 2px rgba(17, 24, 39, .04),
        0 0 0 1px rgba(255, 255, 255, .6) inset;
    transition: border-color .15s ease, box-shadow .15s ease;
}
.sc:hover {
    border-color: #fdba74 !important;
    box-shadow: 0 4px 12px rgba(249, 115, 22, .10);
}
.sl {
    font-size: 0.55rem; color: #a8a29e;
    letter-spacing: .14em; text-transform: uppercase;
    margin-bottom: 5px; font-weight: 800;
}
.sv {
    font-size: 1.1rem; font-weight: 800;
    color: #1c1917;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    letter-spacing: -0.01em;
}
.sv.pass { color: #15803d; text-shadow: 0 0 18px rgba(34, 197, 94, .25); }
.sv.fail { color: #dc2626; }
.sv.run  { color: #2563eb; }
.sv.idle { color: #a8a29e; }
.sc.err  {
    flex: 100%;
    border-color: #fecaca !important;
    background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
}
.sc.err .sv { color: #b91c1c; font-size: .72rem; word-break: break-all; }

/* ═══════════════════════════════════════════════════════════════
   IMAGES — track view + agent POV
   ═══════════════════════════════════════════════════════════════ */
.main-view, .agent-pov {
    position: relative;
    border: 1px solid #e7e5e4 !important;
    border-radius: 14px !important;
    overflow: hidden;
    background: #ffffff;
    box-shadow:
        0 10px 30px -12px rgba(17, 24, 39, .12),
        0 0 0 1px rgba(255, 255, 255, .8) inset;
    transition: box-shadow .2s ease, transform .2s ease;
}
.main-view:hover {
    box-shadow:
        0 18px 44px -14px rgba(17, 24, 39, .18),
        0 0 0 1px rgba(255, 255, 255, .8) inset;
}
.main-view img,
.agent-pov img {
    width: 100% !important; display: block !important;
    border: none !important; border-radius: 0 !important;
}
.agent-pov {
    border-color: #fb923c !important;
    box-shadow:
        0 8px 24px -10px rgba(249, 115, 22, .35),
        0 0 0 1px rgba(255, 237, 213, .8) inset;
}
.agent-pov::after {
    content: "CNN INPUT";
    position: absolute; top: 8px; right: 8px;
    background: rgba(249, 115, 22, .95);
    color: #fff; font-size: 0.55rem;
    letter-spacing: .16em; font-weight: 800;
    padding: 2px 7px; border-radius: 999px;
    box-shadow: 0 2px 6px rgba(249, 115, 22, .35);
    pointer-events: none;
}
.agent-pov img { image-rendering: pixelated !important; }

/* ═══════════════════════════════════════════════════════════════
   PANEL TITLES (small orange labels above each box)
   ═══════════════════════════════════════════════════════════════ */
.panel-title {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.6rem; font-weight: 800;
    letter-spacing: .18em; text-transform: uppercase;
    color: #ea580c;
    padding: 3px 10px;
    background: linear-gradient(90deg, #fff7ed 0%, rgba(255,255,255,0) 100%);
    border-left: 3px solid #fb923c;
    border-radius: 2px;
}

/* ═══════════════════════════════════════════════════════════════
   HELP BOX
   ═══════════════════════════════════════════════════════════════ */
.help-box {
    background:
        linear-gradient(180deg, #ffffff 0%, #fafaf9 100%);
    border: 1px solid #e7e5e4 !important;
    border-radius: 14px !important;
    padding: 14px 16px;
    font-size: 0.78rem; color: #44403c; line-height: 1.65;
    box-shadow:
        0 4px 14px -6px rgba(17, 24, 39, .08),
        0 0 0 1px rgba(255, 255, 255, .6) inset;
}
.help-box strong { color: #1c1917; font-weight: 700; }
.help-box .help-title {
    font-size: .7rem; font-weight: 800; letter-spacing: .14em;
    color: #ea580c; text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.help-box .step { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 9px; }
.help-box .num {
    background: linear-gradient(135deg, #fb923c 0%, #f97316 100%);
    color: #fff;
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 800;
    flex-shrink: 0; margin-top: 1px;
    box-shadow: 0 2px 6px rgba(249, 115, 22, .35);
}
.help-box .footer-note {
    margin-top: 10px; padding-top: 10px;
    border-top: 1px dashed #e7e5e4;
    font-size: 0.72rem; color: #78716c;
}
.help-box .footer-note .accent { color: #ea580c; font-weight: 700; }

/* ═══════════════════════════════════════════════════════════════
   VIDEO PANEL
   ═══════════════════════════════════════════════════════════════ */
.video-panel {
    border: 1px solid #e7e5e4 !important;
    border-radius: 14px !important;
    overflow: hidden;
    background: #0c0a09;
    box-shadow:
        0 10px 30px -12px rgba(17, 24, 39, .15),
        0 0 0 1px rgba(255, 255, 255, .8) inset;
}
.video-panel video { width: 100% !important; display: block !important; }

/* ═══════════════════════════════════════════════════════════════
   BUTTONS — premium tactile feel
   ═══════════════════════════════════════════════════════════════ */
button.gr-button, button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: .01em !important;
    border-radius: 10px !important;
    transition: transform .08s ease, box-shadow .15s ease, filter .15s ease !important;
    border: 1px solid transparent !important;
}
button.gr-button:hover { transform: translateY(-1px); filter: brightness(1.04); }
button.gr-button:active { transform: translateY(0); }

/* primary = orange */
button.primary, button[variant="primary"], .gr-button-primary {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    border-color: #c2410c !important;
    box-shadow:
        0 4px 12px -2px rgba(249, 115, 22, .35),
        0 0 0 1px rgba(255, 255, 255, .25) inset !important;
}
button.primary:hover { box-shadow: 0 6px 18px -2px rgba(249, 115, 22, .45) !important; }

/* secondary = neutral white */
button.secondary, button[variant="secondary"], .gr-button-secondary {
    background: #ffffff !important;
    color: #1c1917 !important;
    border-color: #e7e5e4 !important;
    box-shadow: 0 1px 2px rgba(17, 24, 39, .05) !important;
}
button.secondary:hover { border-color: #fb923c !important; color: #ea580c !important; }

/* stop = red */
button.stop, button[variant="stop"], .gr-button-stop {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    color: #ffffff !important;
    border-color: #15803d !important;
    box-shadow:
        0 4px 12px -2px rgba(22, 163, 74, .35),
        0 0 0 1px rgba(255, 255, 255, .25) inset !important;
}
button.stop:hover { box-shadow: 0 6px 18px -2px rgba(22, 163, 74, .45) !important; }

/* ═══════════════════════════════════════════════════════════════
   LABELS & INPUTS
   ═══════════════════════════════════════════════════════════════ */
label, .gr-input-label, span[data-testid="block-label"] {
    color: #44403c !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
}
select, input, textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    background: #ffffff !important;
    border: 1px solid #e7e5e4 !important;
    border-radius: 10px !important;
    color: #1c1917 !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
select:focus, input:focus, textarea:focus {
    outline: none !important;
    border-color: #fb923c !important;
    box-shadow: 0 0 0 3px rgba(251, 146, 60, .18) !important;
}

/* ═══════════════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════════════ */
.footer-strip {
    margin-top: 24px; padding: 14px 32px 18px;
    border-top: 1px solid #e7e5e4;
    color: #78716c; font-size: 0.72rem;
    text-align: center;
    background: linear-gradient(180deg, rgba(255,255,255,0) 0%, #fafaf9 100%);
}
.footer-strip a {
    color: #ea580c; text-decoration: none; font-weight: 700;
    border-bottom: 1px dashed #fdba74;
}
.footer-strip a:hover { color: #c2410c; border-bottom-color: #ea580c; }
"""

BANNER_HTML = """
<div class="banner">
  <div class="banner-inner">
    <h1>🏎️ Car Racing Agent</h1>
    <p class="sub">
      A PPO agent trained from scratch · 10 tracks · egocentric vision ·
      <strong style="color:#15803d">10 / 10 tracks — zero crashes</strong>
      &nbsp;·&nbsp; OpenEnv Student Challenge 2026
    </p>
    <div class="badges">
      <span class="badge">OpenEnv API</span>
      <span class="badge">TorchRL PPO</span>
      <span class="badge">Curriculum Learning</span>
      <span class="badge">ImpalaCNN</span>
      <span class="badge">~1.3 M Steps</span>
      <span class="badge green">10 / 10 ✅</span>
      <a class="badge hf-badge" href="https://huggingface.co/spaces/nirmalpratheep/curriculum-car-racer" target="_blank" rel="noopener">📝 Blog Post</a>
    </div>
  </div>
</div>
"""

HELP_HTML = f"""
<div class="help-box">
  <div class="help-title">🎮 How to use</div>
  <div class="step">
    <div class="num">1</div>
    <div>Pick any of the <strong>10 curriculum tracks</strong> — from the easy Wide Oval up to the Hairpin and Chicane.</div>
  </div>
  <div class="step">
    <div class="num">2</div>
    <div>Click <strong>Reset</strong> to spawn the agent at the start line. The track view and agent POV appear immediately.</div>
  </div>
  <div class="step">
    <div class="num">3</div>
    <div>Click <strong>Step ×{STEP_CHUNK}</strong> to advance {STEP_CHUNK} physics steps at a time — watch the path trail build up.</div>
  </div>
  <div class="step">
    <div class="num">4</div>
    <div>Click <strong>Auto-Drive</strong> to run a full lap and generate a <strong>replay video</strong> below.</div>
  </div>
  <div class="footer-note">
    The <span class="accent">orange-bordered image</span> is the actual 64×64 input the neural network receives —
    always rotated so the car faces upward.
  </div>
</div>
"""

FOOTER_HTML = """
<div class="footer-strip">
  Car Racing Agent · trained with
  <a href="https://github.com/pytorch/rl" target="_blank">TorchRL</a> PPO ·
  served via <a href="https://openenv.dev" target="_blank">OpenEnv</a> ·
  <a href="https://huggingface.co/spaces/nirmalpratheep/Car-Racing-Agent" target="_blank">🤗 HF Space</a> ·
  <a href="https://huggingface.co/blog/NirmalPratheep/curriculum-car-racer" target="_blank">📝 Blog</a> ·
  <a href="https://github.com/NirmalPratheep/curriculum-car-racer" target="_blank">GitHub</a>
</div>
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Car Racing Agent — OpenEnv Demo") as demo:

    gr.HTML(BANNER_HTML)

    session_state = gr.State(None)

    with gr.Row(equal_height=False):

        # ── Left column ───────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML("<div class='panel-title' style='margin:4px 0 6px'>Select Track</div>")
            track_dd = gr.Dropdown(
                choices=TRACK_CHOICES, value=TRACK_CHOICES[0],
                label="", interactive=True,
            )
            gr.HTML("<div style='height:10px'></div>")
            reset_btn = gr.Button("🎬  Reset", variant="secondary")
            step_btn  = gr.Button(f"⏩  Step ×{STEP_CHUNK}", variant="primary")
            auto_btn  = gr.Button("🏁  Auto-Drive Full Lap", variant="stop")
            gr.HTML("<div style='height:10px'></div>")
            status_out = gr.HTML(_status_html(None))
            gr.HTML("<div style='height:6px'></div>")
            gr.HTML(HELP_HTML)

        # ── Right column ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML("<div class='panel-title' style='margin:4px 0 6px'>Live Track View</div>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=300):
                    frame_img = gr.Image(
                        label="Top-Down Track", type="pil",
                        height=360, interactive=False,
                        elem_classes=["main-view"],
                        value=_placeholder_main(),
                    )
                with gr.Column(scale=1, min_width=160):
                    headlight_img = gr.Image(
                        label="Agent POV — CNN Input (64×64)",
                        type="pil", height=200, interactive=False,
                        elem_classes=["agent-pov"],
                        value=_placeholder_pov(),
                    )
            gr.HTML("<div style='height:10px'></div>")
            gr.HTML("<div class='panel-title' style='margin:4px 0 6px'>Auto-Drive Replay</div>")
            video_out = gr.Video(
                label="", height=300, show_label=False,
                elem_classes=["video-panel"],
            )

    gr.HTML(FOOTER_HTML)

    # ── Wire callbacks ────────────────────────────────────────────────────
    step_event = step_btn.click(
        fn=step_agent,
        inputs=[session_state],
        outputs=[session_state, frame_img, headlight_img, status_out, video_out],
    )
    auto_event = auto_btn.click(
        fn=auto_drive,
        inputs=[track_dd],
        outputs=[session_state, frame_img, headlight_img, status_out, video_out],
    )
    # Reset cancels any running auto-drive or step generator, then resets.
    reset_btn.click(
        fn=reset,
        inputs=[track_dd],
        outputs=[session_state, frame_img, headlight_img, status_out, video_out],
        cancels=[auto_event, step_event],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CSS,
        theme=gr.themes.Soft(primary_hue="orange"),
    )
