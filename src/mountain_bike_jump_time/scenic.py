"""Deterministic, asset-free Pygame scenery and episode playback.

Animation is presentation only: the discrete takeoff/landing cells are authoritative. No
random numbers are consumed from the environment and no display is opened for RGB/GIF
rendering. Coordinates below use a 960 x 540 design canvas.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RenderConfig:
    """Visual settings, independent of the environment's physics and RNG.

    ``theme='auto'`` and ``seed=None`` derive a stable look from the latent configuration.
    Use an explicit seed to vary scenery for the same track. ``show_debug`` adds the agent's
    visibility window and jump trace.
    """

    width: int = 960
    height: int = 540
    fps: int = 30
    theme: str = "auto"
    seed: int | None = None
    show_hud: bool = True
    show_debug: bool = False
    step_seconds: float = 0.32
    hold_seconds: float = 0.9

    def __post_init__(self):
        if self.width < 320 or self.height < 180:
            raise ValueError("Render dimensions must be at least 320 x 180")
        if not 1 <= self.fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if self.theme not in ("auto", "alpine", "desert", "dusk"):
            raise ValueError("theme must be auto, alpine, desert or dusk")
        if not math.isfinite(self.step_seconds) or self.step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")
        if not math.isfinite(self.hold_seconds) or self.hold_seconds < 0:
            raise ValueError("hold_seconds must be finite and nonnegative")


# sky top/bottom, distant peaks, near peaks, trees, rock, rock light, trail, accent
PALETTES = {
    "alpine": (
        (26, 65, 86),
        (177, 214, 201),
        (104, 151, 159),
        (60, 113, 124),
        (24, 76, 78),
        (43, 65, 67),
        (65, 86, 81),
        (192, 199, 160),
        (188, 235, 149),
    ),
    "desert": (
        (101, 66, 100),
        (250, 187, 131),
        (182, 118, 117),
        (156, 87, 81),
        (82, 74, 74),
        (94, 61, 66),
        (143, 84, 74),
        (248, 190, 132),
        (255, 215, 137),
    ),
    "dusk": (
        (21, 28, 65),
        (156, 133, 169),
        (99, 99, 145),
        (62, 67, 111),
        (29, 44, 78),
        (34, 42, 63),
        (53, 62, 84),
        (160, 170, 194),
        (147, 222, 226),
    ),
}


def _mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _smooth(t):
    return t * t * (3 - 2 * t)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float  # elevation in design pixels, positive upwards
    angle: float
    speed: float
    phase: str
    progress: float = 0.0


class ScenicRenderer:
    """Reusable scene.

    ``frame(t)`` is pure with respect to simulation state.
    """

    W, H = 960, 540
    CELL, RISE = 66.0, 15.0
    RADIUS = 11

    def __init__(
        self,
        latent,
        config,
        slope_per_cell,
        trajectory,
        jump_time,
        landing_position,
        reward_components,
        options=None,
        episode_done=None,
    ):
        import pygame

        self.pg = pygame
        self.options = options or RenderConfig()
        self.latent, self.config = latent, config
        self.trajectory = trajectory or [{"position": 0, "speed": latent.initial_speed}]
        self.jump_time, self.landing = jump_time, landing_position
        self.reward = reward_components
        self.done = (
            (jump_time is not None or bool(reward_components.is_missed))
            if episode_done is None
            else episode_done
        )
        key = repr((latent, tuple(int(s) for s in slope_per_cell))).encode()
        self.seed = (
            self.options.seed
            if self.options.seed is not None
            else int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        )
        self.rng = np.random.default_rng(self.seed)
        self.theme = (
            self.options.theme if self.options.theme != "auto" else tuple(PALETTES)[self.seed % 3]
        )
        self.palette = PALETTES[self.theme]
        # Positive slope is uphill in env._advance_one_step; use the same sign.
        self.elevation = np.r_[0, np.cumsum(slope_per_cell[:-1])] * self.RISE
        self.gap = latent.pre_gap_steps - 0.5
        self.target = self.gap + latent.gap_length
        self.target_end = self.target + latent.platform_length
        self.post_end = self.target_end + latent.post_gap_length
        self.ride = self.trajectory[: jump_time + 2] if jump_time is not None else self.trajectory
        # The jump log duplicates the takeoff position, not an extra riding step.
        if jump_time is not None and len(self.ride) > 1:
            self.ride = self.ride[:-1]
        self.ride_duration = max(0, len(self.ride) - 1) * self.options.step_seconds
        self.x0 = float(self.ride[-1]["position"])
        self.speed0 = float(self.ride[-1]["speed"])
        self.flight_duration = 0.0
        if self.jump_time is not None and self.landing is not None:
            self.flight_duration = min(
                1.65, max(0.7, abs(self.landing - self.x0) / max(1, self.speed0) * 0.48)
            )
        self.fall_duration = 0.65 if self.done and self._falls() else 0.0
        self.event_time = self.ride_duration + self.flight_duration
        self.duration = (
            self.event_time + self.fall_duration + (self.options.hold_seconds if self.done else 0)
        )
        self.canvas = pygame.Surface((self.W, self.H))
        self.sky = self._make_sky()
        # Generate once. All animation later is a deterministic function of t.
        self.ridges = []
        for layer in range(3):
            xs = np.arange(-600, 2401, 100 if layer < 2 else 65)
            ys = self.rng.uniform(0, 1, len(xs))
            self.ridges.append((xs, ys))
        self.trees = [
            (float(x), float(self.rng.uniform(35, 90))) for x in np.arange(-500, 2500, 45)
        ]
        self.clouds = [
            (
                float(self.rng.uniform(-100, 1050)),
                float(self.rng.uniform(100, 215)),
                float(self.rng.uniform(0.5, 1.2)),
            )
            for _ in range(6)
        ]
        self.stars = self.rng.uniform((0, 40), (960, 245), (65, 2))
        self.rocks = [
            (float(x), float(self.rng.uniform(4, 11)))
            for x in np.arange(-1, config.track_length + 1, 0.8)
        ]
        pygame.font.init()
        self.fonts = {
            size: pygame.font.SysFont("DejaVu Sans,Arial", size, bold=(size in (13, 22, 28)))
            for size in (11, 13, 16, 22, 28)
        }
        self.window = None
        self.clock = None
        self.closed = False
        self._owned_display = False

    def ground(self, x):
        return float(np.interp(x, np.arange(len(self.elevation)), self.elevation))

    def _falls(self):
        x = self.landing if self.landing is not None else self.x0
        return (
            self.jump_time is None
            or self.gap <= x < self.target
            or self.target_end <= x < self.post_end
            or x >= self.config.track_length
        )

    def pose(self, t):
        if t < self.ride_duration:
            index = min(int(t / self.options.step_seconds), len(self.ride) - 2)
            u = t / self.options.step_seconds - index
            a, b = self.ride[index : index + 2]
            x = a["position"] + (b["position"] - a["position"]) * u
            # Do not draw the wheels riding across the empty gap on a no-jump fall.
            if self.done and self.jump_time is None:
                x = min(x, self.gap)
            angle = math.atan2(self.ground(x + 0.2) - self.ground(x - 0.2), 0.4 * self.CELL)
            return Pose(x, self.ground(x), angle, float(a["speed"]), "ride", u)
        if self.flight_duration and t < self.event_time:
            u = (t - self.ride_duration) / self.flight_duration
            x = self.x0 + (self.landing - self.x0) * u
            rise = min(155, max(64, abs(self.landing - self.x0) * 19))
            y = (
                (1 - u) * self.ground(self.x0)
                + u * self.ground(self.landing)
                + 4 * rise * u * (1 - u)
            )
            return Pose(x, y, 0.32 * math.cos(u * math.pi), self.speed0, "air", u)
        x = float(self.landing) if self.landing is not None else self.x0
        if self.done and self._falls():
            if self.jump_time is None:
                x = self.gap
            u = min(1, max(0, (t - self.event_time) / self.fall_duration))
            return Pose(
                x + 0.45 * u,
                self.ground(x) - 190 * u * u,
                -0.95 * u,
                self.speed0,
                "fall",
                u,
            )
        angle = math.atan2(self.ground(x + 0.2) - self.ground(x - 0.2), 0.4 * self.CELL)
        return Pose(
            x,
            self.ground(x),
            angle,
            self.speed0,
            "land" if self.done else "ride",
            min(1, max(0, (t - self.event_time) * 4)),
        )

    def _make_sky(self):
        pg = self.pg
        sky = pg.Surface((self.W, self.H))
        for y in range(self.H):
            pg.draw.line(
                sky,
                _mix(self.palette[0], self.palette[1], y / self.H),
                (0, y),
                (self.W, y),
            )
        glow = pg.Surface((self.W, self.H), pg.SRCALPHA)
        sun = (745, 152)
        for r in range(112, 32, -4):
            pg.draw.circle(glow, (255, 223, 180, int((112 - r) * 0.3)), sun, r)
        sky.blit(glow, (0, 0))
        disc = pg.Surface((82, 82), pg.SRCALPHA)
        pg.draw.circle(
            disc,
            (242, 231, 199) if self.theme != "dusk" else (228, 227, 242),
            (41, 41),
            31,
        )
        if self.theme == "dusk":
            pg.draw.circle(disc, (0, 0, 0, 0), (29, 32), 29)
        sky.blit(disc, (sun[0] - 41, sun[1] - 41))
        return sky

    def _text(self, text, xy, size=13, color=(233, 240, 236)):
        self.canvas.blit(self.fonts[size].render(text, True, color), xy)

    def _panel(self, rect, alpha=210):
        s = self.pg.Surface((rect[2], rect[3]), self.pg.SRCALPHA)
        self.pg.draw.rect(s, (14, 28, 38, alpha), s.get_rect(), border_radius=12)
        self.canvas.blit(s, rect[:2])

    def _background(self, camera, t):
        pg, surf, pal = self.pg, self.canvas, self.palette
        surf.blit(self.sky, (0, 0))
        if self.theme == "dusk":
            for i, (x, y) in enumerate(self.stars):
                pg.draw.circle(
                    surf,
                    _mix(pal[1], (240, 242, 255), 0.55 + 0.35 * math.sin(t + i)),
                    (int(x), int(y)),
                    1,
                )
        clouds = pg.Surface((self.W, self.H), pg.SRCALPHA)
        for x, y, size in self.clouds:
            x = (x - camera * 0.045 + t * 4 + 180) % 1300 - 180
            for dx, dy, w, h in ((0, 0, 100, 18), (25, -9, 48, 22), (50, -4, 55, 22)):
                pg.draw.ellipse(
                    clouds,
                    (245, 235, 224, 36),
                    (x + dx * size, y + dy * size, w * size, h * size),
                )
        surf.blit(clouds, (0, 0))
        for layer, (xs, noise) in enumerate(self.ridges):
            parallax = (0.10, 0.22, 0.38)[layer]
            baseline = (326, 365, 400)[layer]
            amplitude = (180, 136, 90)[layer]
            color = (pal[2], pal[3], _mix(pal[3], pal[4], 0.55))[layer]
            spacing = float(xs[1] - xs[0])
            first = math.floor((camera * parallax - 100) / spacing)
            last = math.ceil((camera * parallax + self.W + 100) / spacing)
            # Tile a seeded sequence in world space, including on very long tracks.
            points = [
                (
                    int(i * spacing - camera * parallax),
                    int(baseline - noise[i % len(noise)] * amplitude),
                )
                for i in range(first, last + 1)
            ]
            pg.draw.polygon(surf, color, [(points[0][0], self.H), *points, (points[-1][0], self.H)])
            if layer == 0 and self.theme == "alpine":
                for i in range(1, len(points) - 1):
                    a, b, c = points[i - 1 : i + 2]
                    if b[1] < a[1] and b[1] < c[1]:
                        pg.draw.polygon(
                            surf,
                            (188, 214, 209),
                            [
                                b,
                                (
                                    b[0] + int((c[0] - b[0]) * 0.3),
                                    b[1] + int((c[1] - b[1]) * 0.3),
                                ),
                                (b[0] - 5, b[1] + 12),
                                (
                                    b[0] + int((a[0] - b[0]) * 0.35),
                                    b[1] + int((a[1] - b[1]) * 0.35),
                                ),
                            ],
                        )
        for x, h in self.trees:
            x = int((x - camera * 0.48 + 100) % (len(self.trees) * 45) - 100)
            if -100 < x < 1060:
                self._tree(x, 417, h, pal[4])
        # Small distant birds give the sky life without distracting from the jump.
        for i in range(3):
            x, y = (t * 12 + i * 32 + 280) % 1100 - camera * 0.04, 185 + i * 9
            wing = 3 + math.sin(t * 5 + i) * 3
            pg.draw.lines(surf, pal[3], False, [(x - 6, y - wing), (x, y), (x + 6, y - wing)], 1)

    def _tree(self, x, y, h, color):
        pg, surf = self.pg, self.canvas
        if self.theme == "desert":
            pg.draw.line(surf, color, (x, y), (x, y - h * 0.7), 5)
            pg.draw.lines(
                surf,
                color,
                False,
                [
                    (x, y - h * 0.35),
                    (x - h * 0.16, y - h * 0.35),
                    (x - h * 0.16, y - h * 0.52),
                ],
                4,
            )
            pg.draw.lines(
                surf,
                color,
                False,
                [
                    (x, y - h * 0.5),
                    (x + h * 0.15, y - h * 0.5),
                    (x + h * 0.15, y - h * 0.65),
                ],
                4,
            )
        else:
            pg.draw.line(surf, color, (x, y), (x, y - h), 3)
            for u in (0.38, 0.58, 0.78):
                half = h * u * 0.29
                pg.draw.polygon(
                    surf,
                    color,
                    [(x, y - h), (x - half, y - h + h * u), (x + half, y - h + h * u)],
                )

    def _terrain(self, camera, base, t):
        pg, surf, pal = self.pg, self.canvas, self.palette

        def screen(x):
            return (int(x * self.CELL - camera), int(base - self.ground(x)))

        pieces = [
            (-3, self.gap, False),
            (self.target, self.target_end, True),
            (self.post_end, self.config.track_length + 3, False),
        ]
        for start, end, target in pieces:
            if end <= start:
                continue
            xs = sorted(
                set([start, end] + [float(x) for x in range(math.ceil(start), math.ceil(end))])
            )
            top = [screen(x) for x in xs]
            polygon = [*top, (top[-1][0], 620), (top[0][0], 620)]
            pg.draw.polygon(surf, pal[5], polygon)
            # Subsurface ribbon follows the actual track, leaving both chasms empty.
            pg.draw.polygon(surf, pal[6], [*top, *[(x, y + 24) for x, y in reversed(top)]])
            for depth in (52, 96, 149):
                pg.draw.lines(
                    surf,
                    _mix(pal[5], pal[6], 0.35),
                    False,
                    [(x, y + depth + int(7 * math.sin(x * 0.018))) for x, y in top],
                    2,
                )
            for x, radius in self.rocks:
                if start + 0.25 < x < end - 0.25:
                    sx, sy = screen(x)
                    pg.draw.polygon(
                        surf,
                        _mix(pal[5], pal[6], 0.7),
                        [
                            (sx, sy + 39),
                            (sx + radius, sy + 45),
                            (sx + radius * 0.5, sy + 49),
                            (sx - radius, sy + 47),
                        ],
                    )
                    if not target:
                        for dx in (-3, 1, 4):
                            pg.draw.line(
                                surf,
                                pal[7],
                                (sx + dx, sy - 3),
                                (sx + dx + math.sin(t * 1.7 + x) * 2, sy - 8 - abs(dx)),
                                1,
                            )
            pg.draw.lines(surf, pal[7], False, top, 7)
            pg.draw.lines(
                surf,
                pal[8] if target else _mix(pal[7], (255, 255, 230), 0.3),
                False,
                [(x, y - 2) for x, y in top],
                2,
            )
            if target:
                # Short illuminated target marks; no fake bridge over the gaps.
                for x in np.arange(start + 0.12, end, 0.22):
                    a, b = screen(x), screen(min(x + 0.10, end))
                    pg.draw.line(surf, pal[8], (a[0], a[1] - 6), (b[0], b[1] - 6), 3)
                for x in (start + 0.08, end - 0.08):
                    sx, sy = screen(x)
                    pg.draw.line(surf, (218, 226, 215), (sx, sy), (sx, sy - 62), 2)
                    wave = math.sin(t * 4 + x) * 4
                    pg.draw.polygon(
                        surf,
                        pal[8],
                        [(sx, sy - 62), (sx + 25, sy - 57 + wave), (sx, sy - 46)],
                    )
                center = screen((start + end) / 2)
                self._text("LANDING", (center[0] - 28, center[1] + 17), 11, pal[8])
        # Cliff edge chevrons at the takeoff lip.
        sx, sy = screen(self.gap - 0.28)
        pg.draw.line(surf, pal[7], (sx, sy - 1), (sx, sy - 35), 2)
        pg.draw.polygon(
            surf,
            pal[8],
            [
                (sx - 9, sy - 35),
                (sx + 10, sy - 35),
                (sx + 17, sy - 28),
                (sx + 10, sy - 21),
                (sx - 9, sy - 21),
            ],
        )
        pg.draw.lines(
            surf,
            pal[5],
            False,
            [(sx + 1, sy - 32), (sx + 6, sy - 28), (sx + 1, sy - 24)],
            2,
        )

    def _bike(self, pose, camera, base, t):
        pg, surf = self.pg, self.canvas
        x, y = pose.x * self.CELL - camera, base - pose.y - self.RADIUS
        # Compress the rider and suspension briefly on contact.
        compression = (
            0
            if pose.phase not in ("ride", "land")
            else (
                2 * math.sin(t * 18)
                if pose.phase == "ride"
                else 6 * math.sin(pose.progress * math.pi)
            )
        )
        angle = pose.angle

        def p(dx, dy):
            return (
                round(x + dx * math.cos(angle) + dy * math.sin(angle)),
                round(y - dx * math.sin(angle) + dy * math.cos(angle)),
            )

        shadow = pg.Surface((100, 24), pg.SRCALPHA)
        altitude = max(0, pose.y - self.ground(pose.x))
        pg.draw.ellipse(
            shadow,
            (9, 20, 27, max(10, int(65 - altitude * 0.3))),
            (10, 5, 72 - max(0, int(altitude * 0.15)), 9),
        )
        if pose.phase != "fall":
            surf.blit(shadow, (x - 50, base - self.ground(pose.x) - 3))
        rear, front = (-24, 0), (24, 0)
        wheel_turn = pose.x * self.CELL / self.RADIUS
        for wheel in (rear, front):
            cx, cy = p(*wheel)
            pg.draw.circle(surf, (16, 27, 35), (cx, cy), 13)
            pg.draw.circle(surf, (185, 203, 208), (cx, cy), 10, 2)
            pg.draw.circle(surf, (51, 65, 76), (cx, cy), 7, 1)
            for a in range(0, 360, 90):
                theta = math.radians(a) + wheel_turn
                pg.draw.aaline(
                    surf,
                    (118, 147, 157),
                    (cx, cy),
                    (cx + 9 * math.cos(theta), cy + 9 * math.sin(theta)),
                )
            pg.draw.circle(surf, (222, 232, 227), (cx, cy), 2)
        crank, saddle, head = (
            (-3, -2),
            (-11, -23 + compression * 0.3),
            (14, -24 + compression * 0.3),
        )
        for a, b in (
            (rear, saddle),
            (saddle, crank),
            (crank, rear),
            (saddle, head),
            (head, crank),
            (head, front),
        ):
            pg.draw.line(surf, (105, 232, 212), p(*a), p(*b), 4)
            pg.draw.aaline(surf, (190, 255, 230), p(*a), p(*b))
        pg.draw.line(surf, (216, 224, 211), p(17, -15), p(*front), 3)
        pg.draw.line(
            surf,
            (21, 36, 44),
            p(-18, -26 + compression * 0.3),
            p(-5, -26 + compression * 0.3),
            4,
        )
        pg.draw.lines(surf, (223, 233, 226), False, [p(*head), p(16, -32), p(23, -32)], 3)
        pedal_phase = t * 12 if pose.phase == "ride" else 0.7
        pedal = (
            crank[0] + 8 * math.cos(pedal_phase),
            crank[1] + 5 * math.sin(pedal_phase),
        )
        hip = (-13, -38 + compression)
        knee = (1, -24 + compression)
        shoulder = (5, -52 + compression)
        pg.draw.lines(surf, (21, 35, 53), False, [p(*hip), p(*knee), p(*pedal)], 7)
        pg.draw.line(
            surf,
            (234, 237, 219),
            p(pedal[0] - 3, pedal[1]),
            p(pedal[0] + 7, pedal[1]),
            4,
        )
        pg.draw.line(surf, (255, 163, 105), p(*hip), p(*shoulder), 11)
        pg.draw.line(surf, (255, 197, 140), p(-11, -39 + compression), p(4, -50 + compression), 3)
        pg.draw.lines(
            surf,
            (247, 185, 139),
            False,
            [p(*shoulder), p(14, -39 + compression), p(21, -32)],
            5,
        )
        pg.draw.circle(surf, (233, 190, 152), p(8, -62 + compression), 7)
        pg.draw.circle(surf, (237, 244, 222), p(6, -66 + compression), 9)
        pg.draw.line(surf, (53, 82, 89), p(6, -66 + compression), p(17, -63 + compression), 4)
        pg.draw.line(surf, (18, 32, 44), p(15, -60 + compression), p(10, -54 + compression), 3)

    def _particles(self, pose, camera, base, t):
        pg, pal = self.pg, self.palette
        layer = pg.Surface((self.W, self.H), pg.SRCALPHA)
        if pose.phase == "ride" and t < self.ride_duration:
            for i in range(13):
                u = (t * 1.8 + i / 13) % 1
                x = pose.x * self.CELL - camera - 29 - u * 74
                y = base - self.ground(pose.x) - 2 - u * 15 + math.sin(i * 3) * 5
                pg.draw.circle(
                    layer,
                    (*pal[7], int((1 - u) * 90)),
                    (int(x), int(y)),
                    max(1, int(4 * (1 - u))),
                )
        age = t - self.event_time
        if self.done and 0 <= age <= 0.7 and not self._falls():
            for i in range(20):
                theta = (i / 20) * math.pi
                x = pose.x * self.CELL - camera + math.cos(theta) * age * 110
                y = base - self.ground(pose.x) - math.sin(theta) * age * 110 + age * age * 130
                pg.draw.circle(layer, (*pal[7], int(130 * (1 - age / 0.7))), (int(x), int(y)), 3)
        self.canvas.blit(layer, (0, 0))

    def _hud(self, pose, t):
        pal, pg, surf = self.palette, self.pg, self.canvas
        self._panel((24, 22, 283, 66))
        self._text("MOUNTAIN BIKE  /  JUMP TIME", (40, 34), 13, pal[8])
        self._text(
            {
                "alpine": "01  /  ALPINE RIDGELINE",
                "desert": "02  /  CANYON COUNTRY",
                "dusk": "03  /  BLUE HOUR",
            }[self.theme],
            (40, 58),
            11,
        )
        self._panel((760, 22, 176, 66))
        self._text(f"{pose.speed:.0f}", (777, 29), 28)
        self._text("CELLS / STEP", (814, 43), 11)
        self._text(
            {
                "ride": "ON THE TRAIL",
                "air": "AIRBORNE",
                "land": "TOUCHDOWN",
                "fall": "MISSED THE GAP",
            }[pose.phase],
            (777, 68),
            11,
            pal[8],
        )
        self._panel((24, 478, 912, 42))
        self._text("TRAIL", (40, 493), 11)
        x, y, width = 103, 499, 575
        pg.draw.line(surf, (96, 116, 121), (x, y), (x + width, y), 3)
        for a, b, color in (
            (self.gap, self.target, (35, 45, 55)),
            (self.target, self.target_end, pal[8]),
            (self.target_end, self.post_end, (35, 45, 55)),
        ):
            pg.draw.line(
                surf,
                color,
                (x + width * max(0, a) / self.config.track_length, y),
                (
                    x + width * min(self.config.track_length, b) / self.config.track_length,
                    y,
                ),
                6,
            )
        pg.draw.circle(
            surf,
            (244, 243, 224),
            (int(x + width * np.clip(pose.x / self.config.track_length, 0, 1)), y),
            5,
        )
        if self.done and t >= self.event_time:
            self._text(f"RETURN  {self.reward.total:+.2f}", (720, 491), 13, pal[8])
        else:
            self._text(f"CELL  {pose.x:04.1f} / {self.config.track_length}", (720, 491), 13)
        if self.done and t >= self.event_time + self.fall_duration:
            self._panel((345, 106, 270, 66), 230)
            landed = self.jump_time is not None and not self.reward.is_missed
            self._text(
                "CLEAN LANDING" if landed else "MISSED LANDING",
                (362, 116),
                22,
                pal[8] if landed else (255, 168, 132),
            )
            self._text(
                f'Error {self.reward.landing_error:.1f} cells  /  Jump {self.jump_time if self.jump_time is not None else "none"}',
                (362, 147),
                11,
            )

    def frame(self, t):
        t = max(0, min(float(t), self.duration))
        pose = self.pose(t)
        # Keep rider left of center; bounded world camera preserves landscape depth.
        camera = max(-150.0, pose.x * self.CELL - 325)
        # Vertical tracking is tied to ground (not bike height), so jumps read clearly.
        base = 359 + self.ground(pose.x)
        self._background(camera, t)
        self._terrain(camera, base, t)
        if self.options.show_debug:
            anchor = (
                self.x0
                if t >= self.ride_duration and self.jump_time is not None
                else math.floor(pose.x)
            )
            x = (anchor + 0.5) * self.CELL - camera
            right = (
                min(
                    anchor + self.config.visibility_k + 0.5,
                    self.config.track_length - 0.5,
                )
                * self.CELL
                - camera
            )
            overlay = self.pg.Surface((self.W, self.H), self.pg.SRCALPHA)
            self.pg.draw.rect(overlay, (246, 220, 115, 30), (x, 185, max(0, right - x), 270))
            self.canvas.blit(overlay, (0, 0))
            self._text("AGENT VISIBILITY", (max(24, int(x) + 8), 195), 11)
            if self.flight_duration and t >= self.ride_duration:
                points = []
                for moment in np.linspace(self.ride_duration, min(t, self.event_time - 1e-6), 40):
                    p = self.pose(moment)
                    points.append((p.x * self.CELL - camera, base - p.y - self.RADIUS))
                if len(points) > 1:
                    self.pg.draw.aalines(self.canvas, self.palette[8], False, points)
        self._particles(pose, camera, base, t)
        self._bike(pose, camera, base, t)
        if self.options.show_hud:
            self._hud(pose, t)
        output = (
            self.canvas
            if (self.options.width, self.options.height) == (self.W, self.H)
            else self.pg.transform.smoothscale(
                self.canvas, (self.options.width, self.options.height)
            )
        )
        return np.transpose(self.pg.surfarray.array3d(output), (1, 0, 2)).copy()

    def show(self, rgb):
        pg = self.pg
        if self.closed:
            return False
        if self.window is None:
            self._owned_display = not pg.display.get_init()
            pg.display.init()
            self.window = pg.display.set_mode((self.options.width, self.options.height))
            pg.display.set_caption("Mountain Bike / Jump Time")
            self.clock = pg.time.Clock()
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.closed = True
                return False
        self.window.blit(pg.surfarray.make_surface(np.transpose(rgb, (1, 0, 2))), (0, 0))
        pg.display.flip()
        self.clock.tick(self.options.fps)
        return True

    def close(self):
        if self._owned_display:
            self.pg.display.quit()
        self.window = None
        self.closed = True


def _gif_palette(rgb):
    """Keep small but important rider/HUD colors through GIF quantization."""
    from PIL import Image

    palette = Image.fromarray(rgb).quantize(colors=192)
    colors = list(palette.getpalette())
    colors.extend([0] * (768 - len(colors)))
    accents = [
        (105, 232, 212),
        (190, 255, 230),
        (255, 163, 105),
        (255, 197, 140),
        (247, 185, 139),
        (233, 190, 152),
        (237, 244, 222),
        (233, 240, 236),
        (255, 168, 132),
        (188, 235, 149),
        (255, 215, 137),
        (147, 222, 226),
        (185, 203, 208),
        (21, 35, 53),
        (16, 27, 35),
        (53, 82, 89),
    ]
    for i, color in enumerate(accents, 192):
        colors[i * 3 : i * 3 + 3] = color
    palette.putpalette(colors)
    return palette


def render_scenic_episode(
    *,
    latent,
    config,
    slope_per_cell,
    trajectory,
    jump_time,
    landing_position,
    reward_components,
    mode="rgb_array",
    save_path=None,
    render_config=None,
    episode_done=None,
):
    """Render the final frame, play an episode, or export a palette-compressed GIF.

    GIF frames are quantized immediately instead of retaining full RGB frames. PNG exports
    and RGB calls render only the final frame. Human playback supports Escape/window close;
    display initialization is never needed for file export.
    """
    if mode not in ("rgb_array", "human"):
        raise ValueError("mode must be rgb_array or human")
    suffix = Path(save_path).suffix.lower() if save_path else None
    if suffix not in (None, ".gif", ".png"):
        raise ValueError("Pygame exports support .gif and .png")
    scene = ScenicRenderer(
        latent,
        config,
        slope_per_cell,
        trajectory,
        jump_time,
        landing_position,
        reward_components,
        render_config,
        episode_done,
    )
    frames = []
    try:
        animated = mode == "human" or suffix == ".gif"
        times = (
            np.linspace(
                0,
                scene.duration,
                max(2, math.ceil(scene.duration * scene.options.fps) + 1),
            )
            if animated
            else [scene.duration]
        )
        for t in times:
            rgb = scene.frame(t)
            if suffix == ".gif":
                from PIL import Image

                # Fixed palette shared by all frames avoids palette flicker.
                if not frames:
                    palette = _gif_palette(scene.frame(scene.duration / 2))
                frames.append(
                    Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE)
                )
            if mode == "human" and not scene.show(rgb):
                break
        if save_path:
            from PIL import Image

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            if suffix == ".gif":
                # GIF stores centiseconds; distribute rounding error across frames.
                durations = [
                    max(
                        10,
                        10
                        * (
                            round((i + 1) * 100 / scene.options.fps)
                            - round(i * 100 / scene.options.fps)
                        ),
                    )
                    for i in range(len(frames))
                ]
                frames[0].save(
                    save_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=0,
                    optimize=False,
                    disposal=2,
                )
            else:
                Image.fromarray(rgb).save(save_path)
        return None if mode == "human" else rgb
    finally:
        scene.close()
