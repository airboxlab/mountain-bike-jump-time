"""2D visualization of a rollout.

Two rendering backends are supported, selected via the ``renderer`` argument
of :func:`render_episode`:

* ``"matplotlib"`` (default) — static 2D figure drawn with matplotlib. Writes
  a PNG when ``save_path`` is given and returns the frame as an
  ``(H, W, 3)`` uint8 array.
* ``"pygame"`` — animated 2D rendering of the bike traveling along the
  track. When ``save_path`` is given the animation is written as an
  animated GIF; ``mode="human"`` opens a pygame window and plays the
  animation; ``mode="rgb_array"`` returns the final frame as an
  ``(H, W, 3)`` uint8 array.

Both backends draw the terrain (with slope shading), gap, target platform,
the bike's per-step position, the visibility window at the moment of the
jump, the jump trajectory (parabolic arc) and the reward decomposition.
They are intentionally headless-friendly: callers may pass ``save_path``
to write the output to disk without ever opening a window.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from mountain_bike_jump_time.env import EnvConfig, LatentConfig, RewardComponents


VALID_RENDERERS = ("matplotlib", "pygame")


def render_episode(
    latent: LatentConfig,
    config: EnvConfig,
    slope_per_cell: np.ndarray,
    trajectory: list[dict[str, Any]],
    jump_time: int | None,
    landing_position: int | None,
    reward_components: RewardComponents,
    mode: str = "rgb_array",
    save_path: str | None = None,
    renderer: str = "matplotlib",
):
    """Render the rollout using the selected backend.

    :param latent: latent configuration of the episode
    :param config: environment configuration
    :param slope_per_cell: array of slopes per cell (length == config.track_length)
    :param trajectory: list of per-step dictionaries, each with keys ``position`` and ``speed``.
    :param jump_time: index of the step where the jump was initiated, or ``None`` if no jump was attempted.
    :param landing_position: position where the bike landed, or ``None`` if no jump was attempted.
    :param reward_components: reward decomposition for the episode.
    :param mode: ``"rgb_array"`` returns the rendered frame as an ``(H, W, 3)`` uint8
        array; ``"human"`` opens an interactive window (matplotlib: a
        figure window; pygame: an animated window).
    :param save_path: Optional path; when provided, the rendered output is written
        to disk (PNG for the matplotlib backend, animated GIF for the
        pygame backend).
    :param renderer: ``"matplotlib"`` for a static 2D figure (default) or
        ``"pygame"`` for an animated 2D rendering.
    """
    if renderer not in VALID_RENDERERS:
        raise ValueError(f"Unknown renderer {renderer!r}; expected one of {VALID_RENDERERS}.")
    if renderer == "pygame":
        return _render_episode_pygame(
            latent=latent,
            config=config,
            slope_per_cell=slope_per_cell,
            trajectory=trajectory,
            jump_time=jump_time,
            landing_position=landing_position,
            reward_components=reward_components,
            mode=mode,
            save_path=save_path,
        )
    return _render_episode_matplotlib(
        latent=latent,
        config=config,
        slope_per_cell=slope_per_cell,
        trajectory=trajectory,
        jump_time=jump_time,
        landing_position=landing_position,
        reward_components=reward_components,
        mode=mode,
        save_path=save_path,
    )


def _render_episode_matplotlib(
    latent: LatentConfig,
    config: EnvConfig,
    slope_per_cell: np.ndarray,
    trajectory: list[dict[str, Any]],
    jump_time: int | None,
    landing_position: int | None,
    reward_components: RewardComponents,
    mode: str = "rgb_array",
    save_path: str | None = None,
):
    """Static matplotlib rendering of a rollout (see :func:`render_episode`)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, config.track_length - 0.5)
    ax.set_ylim(-2.0, 3.0)
    ax.set_xlabel("position (cell)")
    ax.set_ylabel("elevation (a.u.)")
    ax.set_title("Mountain-bike jump-time rollout")

    # Build elevation profile by integrating the slope (downhill -> elevation
    # decreases as you advance, since slope == +1 means uphill in our model).
    elevation = np.zeros(config.track_length, dtype=float)
    for i in range(1, config.track_length):
        elevation[i] = elevation[i - 1] - slope_per_cell[i - 1]
    xs = np.arange(config.track_length)
    ax.plot(xs, elevation, color="saddlebrown", linewidth=2, label="terrain")
    ax.fill_between(xs, elevation, elevation.min() - 1, color="peru", alpha=0.3)

    # Gap & platform.
    gap_start = latent.pre_gap_steps
    gap_end = gap_start + latent.gap_length
    plat_start = gap_end
    plat_end = plat_start + latent.platform_length
    post_gap_start = plat_end
    ax.add_patch(
        mpatches.Rectangle(
            (gap_start - 0.5, elevation.min() - 1),
            latent.gap_length,
            elevation.max() - elevation.min() + 3,
            color="red",
            alpha=0.15,
            label="gap",
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (post_gap_start - 0.5, elevation.min() - 1),
            latent.post_gap_length,
            elevation.max() - elevation.min() + 3,
            color="red",
            alpha=0.15,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (plat_start - 0.5, elevation[min(plat_start, len(elevation) - 1)] - 0.05),
            latent.platform_length,
            0.25,
            color="green",
            alpha=0.6,
            label="platform",
        )
    )

    # Bike per-step positions.
    bike_x = [s["position"] for s in trajectory if s["position"] < config.track_length]
    bike_y = [
        elevation[min(s["position"], config.track_length - 1)] + 0.2
        for s in trajectory
        if s["position"] < config.track_length
    ]
    ax.plot(bike_x, bike_y, "o-", color="black", markersize=4, label="bike path")

    # Jump arc.
    if jump_time is not None and landing_position is not None:
        jump_step = trajectory[jump_time + 1] if jump_time + 1 < len(trajectory) else None
        # The step *causing* the jump was logged at index jump_time + 1 (since
        # reset() pre-logs index 0); the position recorded there is the
        # take-off cell.
        if jump_step is not None:
            x0 = jump_step["position"]
        else:
            x0 = trajectory[-1]["position"]
        y0 = elevation[min(x0, config.track_length - 1)] + 0.2
        x1 = landing_position
        y1 = (
            elevation[min(max(x1, 0), config.track_length - 1)] + 0.2
            if 0 <= x1 < config.track_length
            else y0
        )
        arc_x = np.linspace(x0, x1, 30)
        # parabolic arc, peak above the midpoint
        peak = max(y0, y1) + 1.0
        t_param = (arc_x - x0) / max(x1 - x0, 1)
        arc_y = (
            (1 - t_param) * y0 + t_param * y1 + 4 * t_param * (1 - t_param) * (peak - (y0 + y1) / 2)
        )
        ax.plot(arc_x, arc_y, "--", color="blue", label="jump arc")
        ax.plot([x1], [y1], "x", color="blue", markersize=10, label="landing")

        # Visibility window at jump time.
        k = config.visibility_k
        if k > 0:
            vis_lo = x0 + 1 - 0.5
            vis_hi = min(x0 + k, config.track_length) - 0.5
            ax.add_patch(
                mpatches.Rectangle(
                    (vis_lo, ax.get_ylim()[0]),
                    max(vis_hi - vis_lo, 0),
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    facecolor="yellow",
                    alpha=0.1,
                    label="visible at jump",
                )
            )

    # Reward decomposition text.
    text = (
        f"jump_time={jump_time}  landing={landing_position}\n"
        f"landing_err={reward_components.landing_error:.2f}  "
        f"fall={reward_components.is_missed:.0f}\n"
        f"return={reward_components.total:.2f}"
    )
    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100)

    if mode == "human":  # pragma: no cover - interactive
        plt.show()
        plt.close(fig)
        return None

    # rgb_array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Pygame backend (animated 2D rendering)
# ---------------------------------------------------------------------------


# Window/canvas constants for the pygame renderer.
_PG_WIDTH = 900
_PG_HEIGHT = 360
_PG_MARGIN_X = 40
_PG_MARGIN_TOP = 40
_PG_MARGIN_BOTTOM = 60
_PG_FRAME_DELAY_MS = 200  # ~5 FPS, gentle pacing for human viewing
_PG_BG = (245, 245, 250)
_PG_TERRAIN = (139, 69, 19)
_PG_TERRAIN_FILL = (205, 133, 63)
_PG_GAP = (220, 70, 70)
_PG_PLATFORM = (60, 160, 80)
_PG_BIKE = (20, 20, 20)
_PG_BIKE_TRAIL = (90, 90, 90)
_PG_ARC = (40, 90, 220)
_PG_VISIBILITY = (255, 220, 90)
_PG_TEXT = (20, 20, 20)


def _render_episode_pygame(
    latent: LatentConfig,
    config: EnvConfig,
    slope_per_cell: np.ndarray,
    trajectory: list[dict[str, Any]],
    jump_time: int | None,
    landing_position: int | None,
    reward_components: RewardComponents,
    mode: str = "rgb_array",
    save_path: str | None = None,
):
    """Animated pygame rendering of a rollout (see :func:`render_episode`).

    Generates one frame per step in ``trajectory`` and, after the jump, a
    handful of additional frames sliding the bike along the parabolic
    jump arc. Frames are blitted to an off-screen surface so the renderer
    works headlessly (e.g. on CI) without an attached display.
    """
    # Lazy import so the matplotlib backend remains usable when pygame is
    # not installed in the active environment.
    headless = mode != "human"
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # noqa: E402

    pygame.display.init()
    pygame.font.init()
    try:
        if mode == "human":  # pragma: no cover - interactive
            surface = pygame.display.set_mode((_PG_WIDTH, _PG_HEIGHT))
            pygame.display.set_caption("Mountain-bike jump-time rollout")
        else:
            # Create the display surface so font / image APIs work, but draw
            # onto an off-screen Surface to keep the headless path clean.
            pygame.display.set_mode((1, 1))
            surface = pygame.Surface((_PG_WIDTH, _PG_HEIGHT))

        elevation = _elevation_from_slopes(slope_per_cell, config.track_length)
        layout = _PygameLayout(elevation=elevation, config=config)
        font = pygame.font.SysFont("DejaVuSans,Arial", 14)

        frames: list[np.ndarray] = []
        clock = pygame.time.Clock() if mode == "human" else None

        ride_steps = [s for s in trajectory if s["position"] < config.track_length]
        # Animated jump arc points (post take-off slide across the gap).
        arc_points = _jump_arc_points(
            ride_steps=ride_steps,
            trajectory=trajectory,
            jump_time=jump_time,
            landing_position=landing_position,
            elevation=elevation,
            track_length=config.track_length,
            num_points=24,
            take_off_speed=_take_off_speed(trajectory=trajectory, jump_time=jump_time),
        )

        # One animation frame per ride step, then one per arc point.
        ride_frame_count = max(len(ride_steps), 1)
        total_frames = ride_frame_count + len(arc_points)
        for f in range(total_frames):
            _pg_draw_static(
                surface=surface,
                pygame=pygame,
                layout=layout,
                latent=latent,
                config=config,
                elevation=elevation,
            )
            # Bike trail up to current frame.
            visible_ride = ride_steps[: min(f + 1, ride_frame_count)]
            _pg_draw_bike_trail(surface, pygame, layout, visible_ride, elevation)

            # Current bike position (either still riding or mid-air).
            if f < ride_frame_count:
                pos = visible_ride[-1]["position"] if visible_ride else 0
                bike_xy = layout.cell_to_xy(pos, y_offset=-12)
            else:
                bike_xy = arc_points[f - ride_frame_count]

            # Show jump arc fully once airborne.
            if f >= ride_frame_count and arc_points:
                _pg_draw_arc(surface, pygame, arc_points, up_to=f - ride_frame_count + 1)

            # Visibility window: shown at each step, anchored to the bike's
            # current cell during the ride (the cells it can "see" ahead) and
            # frozen at the take-off cell once airborne.
            if config.visibility_k > 0:
                if f < ride_frame_count:
                    anchor_cell = visible_ride[-1]["position"] if visible_ride else 0
                else:
                    anchor_cell = _take_off_cell(trajectory=trajectory, jump_time=jump_time)
                _pg_draw_visibility(
                    surface=surface,
                    pygame=pygame,
                    layout=layout,
                    config=config,
                    anchor_cell=anchor_cell,
                )

            _pg_draw_bike(surface, pygame, bike_xy)
            # Once the bike has finished moving (on its final frame), overlay
            # a green tick (landed on platform) or a red cross (fell / missed).
            if f == total_frames - 1:
                _pg_draw_outcome_marker(
                    surface=surface,
                    pygame=pygame,
                    bike_xy=bike_xy,
                    reward_components=reward_components,
                )
            _pg_draw_legend(surface, pygame, font)
            _pg_draw_reward_text(
                surface=surface,
                pygame=pygame,
                font=font,
                jump_time=jump_time,
                landing_position=landing_position,
                reward_components=reward_components,
            )

            if mode == "human":  # pragma: no cover - interactive
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                clock.tick(1000 // max(_PG_FRAME_DELAY_MS, 1))
            else:
                frames.append(_pg_surface_to_rgb(pygame, surface))

        if save_path is not None and frames:
            _save_gif(frames, save_path, duration_ms=_PG_FRAME_DELAY_MS)

        if mode == "human":  # pragma: no cover - interactive
            return None
        return frames[-1] if frames else None
    finally:
        pygame.display.quit()
        pygame.font.quit()


class _PygameLayout:
    """Maps logical (cell, elevation) coordinates to pixel coordinates."""

    def __init__(self, elevation: np.ndarray, config: EnvConfig):
        self.config = config
        self.elevation = elevation
        self.draw_w = _PG_WIDTH - 2 * _PG_MARGIN_X
        self.draw_h = _PG_HEIGHT - _PG_MARGIN_TOP - _PG_MARGIN_BOTTOM
        self.cell_step = self.draw_w / max(config.track_length - 1, 1)
        self.elev_min = float(elevation.min()) - 1.0
        self.elev_max = float(elevation.max()) + 2.0
        self.elev_range = max(self.elev_max - self.elev_min, 1e-6)

    def cell_to_x(self, cell: float) -> int:
        return int(_PG_MARGIN_X + cell * self.cell_step)

    def elev_to_y(self, elev: float) -> int:
        # Pygame Y axis points down; flip so larger elevations sit higher.
        norm = (elev - self.elev_min) / self.elev_range
        return int(_PG_MARGIN_TOP + (1.0 - norm) * self.draw_h)

    def cell_to_xy(self, cell: int, y_offset: int = 0) -> tuple[int, int]:
        cell_clipped = min(max(cell, 0), self.config.track_length - 1)
        return (
            self.cell_to_x(cell_clipped),
            self.elev_to_y(self.elevation[cell_clipped]) + y_offset,
        )


def _elevation_from_slopes(slope_per_cell: np.ndarray, track_length: int) -> np.ndarray:
    elevation = np.zeros(track_length, dtype=float)
    for i in range(1, track_length):
        elevation[i] = elevation[i - 1] - slope_per_cell[i - 1]
    return elevation


def _take_off_cell(trajectory: list[dict[str, Any]], jump_time: int | None) -> int | None:
    if jump_time is None:
        return None
    if jump_time + 1 < len(trajectory):
        return int(trajectory[jump_time + 1]["position"])
    return int(trajectory[-1]["position"])


def _take_off_speed(trajectory: list[dict[str, Any]], jump_time: int | None) -> int:
    if jump_time is None:
        return 1
    if jump_time + 1 < len(trajectory):
        return int(trajectory[jump_time + 1]["speed"])
    return int(trajectory[-1]["speed"])


def _jump_arc_points(
    ride_steps: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    jump_time: int | None,
    landing_position: int | None,
    elevation: np.ndarray,
    track_length: int,
    num_points: int,
    take_off_speed: int = 1,
) -> list[tuple[int, int]]:
    if jump_time is None or landing_position is None or not ride_steps:
        return []
    x0 = _take_off_cell(trajectory=trajectory, jump_time=jump_time)
    if x0 is None:
        return []
    x1 = landing_position
    y0 = float(elevation[min(max(x0, 0), track_length - 1)]) + 0.2
    if 0 <= x1 < track_length:
        y1 = float(elevation[x1]) + 0.2
    else:
        # Landing past the track: extrapolate the terrain slope at the take-off
        # cell so the arc lands at a physically plausible height.
        y1 = y0
    # Physics-flavoured parabola: model the bike as a projectile with constant
    # horizontal velocity equal to its take-off ``speed`` and an initial
    # vertical velocity chosen so that the projectile range matches the
    # discrete landing offset. With horizontal velocity ``v`` and range ``R``
    # (both in cells), time-of-flight is ``T = R / v`` and the apex height
    # under gravity ``g`` is ``g * T^2 / 8``. Picking ``g = 1`` cell/step^2
    # gives a peak height in cells that scales with the *square* of the
    # range and inversely with speed — slow jumps loft higher, fast jumps
    # stay flatter, both lengthen the arc visually with longer ranges.
    horizontal_distance = float(x1 - x0)
    v = float(max(take_off_speed, 1))
    g = 1.0
    time_of_flight = abs(horizontal_distance) / v
    peak_offset = (g * time_of_flight * time_of_flight) / 8.0
    # Clip extremes so the arc stays visible without dominating the canvas.
    peak_offset = float(np.clip(peak_offset, 0.6, 4.0))
    peak = max(y0, y1) + peak_offset
    ts = np.linspace(0.0, 1.0, num_points)
    xs_cell = x0 + ts * (x1 - x0)
    ys = (1 - ts) * y0 + ts * y1 + 4 * ts * (1 - ts) * (peak - (y0 + y1) / 2)
    # Build a temporary layout to convert; but callers want pixel coords so we
    # convert here using elevation-based math (mirrors _PygameLayout but we
    # don't have it here -> compute against caller-provided arrays).
    elev_min = float(elevation.min()) - 1.0
    elev_max = float(elevation.max()) + 2.0
    elev_range = max(elev_max - elev_min, 1e-6)
    draw_w = _PG_WIDTH - 2 * _PG_MARGIN_X
    draw_h = _PG_HEIGHT - _PG_MARGIN_TOP - _PG_MARGIN_BOTTOM
    cell_step = draw_w / max(track_length - 1, 1)
    pts: list[tuple[int, int]] = []
    for cx, cy in zip(xs_cell, ys):
        px = int(_PG_MARGIN_X + cx * cell_step)
        py = int(_PG_MARGIN_TOP + (1.0 - (cy - elev_min) / elev_range) * draw_h)
        pts.append((px, py - 12))
    return pts


def _pg_draw_static(
    surface,
    pygame,
    layout: _PygameLayout,
    latent: LatentConfig,
    config: EnvConfig,
    elevation: np.ndarray,
) -> None:
    surface.fill(_PG_BG)
    # Terrain polyline & fill below.
    points = [
        (layout.cell_to_x(i), layout.elev_to_y(elevation[i])) for i in range(config.track_length)
    ]
    base_y = _PG_MARGIN_TOP + layout.draw_h
    fill_poly = [
        (_PG_MARGIN_X, base_y),
        *points,
        (layout.cell_to_x(config.track_length - 1), base_y),
    ]
    pygame.draw.polygon(surface, _PG_TERRAIN_FILL, fill_poly)
    pygame.draw.lines(surface, _PG_TERRAIN, False, points, 2)

    # Gaps (vertical bands): both the pre-platform and post-platform gaps.
    for gap_start, gap_len in (
        (latent.pre_gap_steps, latent.gap_length),
        (latent.pre_gap_steps + latent.gap_length + latent.platform_length, latent.post_gap_length),
    ):
        if gap_len <= 0:
            continue
        gap_end = gap_start + gap_len
        gap_left = layout.cell_to_x(gap_start) - int(layout.cell_step // 2)
        gap_right = layout.cell_to_x(gap_end) - int(layout.cell_step // 2)
        gap_rect = pygame.Rect(
            gap_left,
            _PG_MARGIN_TOP,
            max(gap_right - gap_left, 2),
            layout.draw_h,
        )
        gap_surf = pygame.Surface((gap_rect.width, gap_rect.height), pygame.SRCALPHA)
        gap_surf.fill((*_PG_GAP, 60))
        surface.blit(gap_surf, gap_rect.topleft)

    # Platform (horizontal bar at terrain height of plat_start).
    plat_start = latent.pre_gap_steps + latent.gap_length
    plat_end = min(plat_start + latent.platform_length, config.track_length)
    if plat_start < config.track_length:
        plat_y = layout.elev_to_y(elevation[plat_start]) - 4
        plat_x0 = layout.cell_to_x(plat_start)
        plat_x1 = layout.cell_to_x(plat_end - 1) if plat_end - 1 >= plat_start else plat_x0
        pygame.draw.rect(
            surface,
            _PG_PLATFORM,
            pygame.Rect(plat_x0, plat_y, max(plat_x1 - plat_x0, 4), 6),
        )


def _pg_draw_bike_trail(
    surface,
    pygame,
    layout: _PygameLayout,
    ride_steps: list[dict[str, Any]],
    elevation: np.ndarray,
) -> None:
    if len(ride_steps) < 1:
        return
    # The bike can move more than one cell per step (``position += speed``),
    # so connecting consecutive trajectory entries with a straight line cuts
    # under any terrain peak in between. Walk through every integer cell the
    # bike rolled over and anchor the trail to that cell's elevation so it
    # follows the terrain profile.
    track_length = elevation.shape[0]
    cells: list[int] = []
    prev = int(ride_steps[0]["position"])
    cells.append(prev)
    for s in ride_steps[1:]:
        cur = int(s["position"])
        if cur == prev:
            continue
        step = 1 if cur > prev else -1
        c = prev + step
        while c != cur + step:
            cells.append(c)
            c += step
        prev = cur
    # Build the polyline by hugging the terrain at each visited cell.
    pts: list[tuple[int, int]] = []
    for c in cells:
        c_clamped = min(max(c, 0), track_length - 1)
        pts.append(
            (
                layout.cell_to_x(c_clamped),
                layout.elev_to_y(float(elevation[c_clamped])) - 12,
            )
        )
    if len(pts) >= 2:
        pygame.draw.lines(surface, _PG_BIKE_TRAIL, False, pts, 2)
    # Mark the actual per-step stops so the discrete dynamics stay visible.
    for s in ride_steps:
        c = min(max(int(s["position"]), 0), track_length - 1)
        p = (
            layout.cell_to_x(c),
            layout.elev_to_y(float(elevation[c])) - 12,
        )
        pygame.draw.circle(surface, _PG_BIKE_TRAIL, p, 3)


def _pg_draw_arc(surface, pygame, arc_points: list[tuple[int, int]], up_to: int) -> None:
    visible = arc_points[: max(up_to, 2)]
    if len(visible) >= 2:
        pygame.draw.lines(surface, _PG_ARC, False, visible, 2)


def _pg_draw_bike(surface, pygame, bike_xy: tuple[int, int]) -> None:
    """Draw a small side-view bicycle centered on ``bike_xy``.

    ``bike_xy`` is treated as the bike's body reference point (roughly the
    bottom-bracket / center of the frame). Two wheels sit below it, with a
    triangular frame, seat, handlebar and a simple stick rider above.
    """
    cx, cy = int(bike_xy[0]), int(bike_xy[1])

    # Geometry (in pixels). Tuned to keep the bike legible at the renderer's
    # resolution while staying close in footprint to the previous marker.
    wheel_r = 5
    wheel_dx = 9  # half wheelbase
    wheel_y = cy + 4  # wheels sit slightly below the reference point

    rear_wheel = (cx - wheel_dx, wheel_y)
    front_wheel = (cx + wheel_dx, wheel_y)

    # Frame attachment points.
    bb = (cx, wheel_y)  # bottom bracket (between wheels)
    seat_top = (cx - 3, cy - 6)
    handlebar = (cx + wheel_dx - 1, cy - 7)

    # Wheels (tire + hub).
    for wheel in (rear_wheel, front_wheel):
        pygame.draw.circle(surface, _PG_BIKE, wheel, wheel_r, 1)
        pygame.draw.circle(surface, _PG_BIKE, wheel, 1)

    # Frame: seat tube, top tube, down tube, chain stay, fork.
    pygame.draw.line(surface, _PG_BIKE, bb, seat_top, 2)  # seat tube
    pygame.draw.line(surface, _PG_BIKE, seat_top, handlebar, 2)  # top tube
    pygame.draw.line(surface, _PG_BIKE, bb, handlebar, 2)  # down tube
    pygame.draw.line(surface, _PG_BIKE, rear_wheel, seat_top, 2)  # seat stay
    pygame.draw.line(surface, _PG_BIKE, handlebar, front_wheel, 2)  # fork

    # Seat and handlebar.
    pygame.draw.line(
        surface, _PG_BIKE, (seat_top[0] - 3, seat_top[1]), (seat_top[0] + 3, seat_top[1]), 2
    )
    pygame.draw.line(
        surface,
        _PG_BIKE,
        (handlebar[0] - 2, handlebar[1] - 2),
        (handlebar[0] + 3, handlebar[1] - 2),
        2,
    )

    # Simple stick rider above the frame.
    hip = (cx - 1, cy - 7)
    shoulder = (cx + 2, cy - 13)
    head_center = (cx + 3, cy - 17)
    pygame.draw.line(surface, _PG_BIKE, hip, shoulder, 2)  # torso
    pygame.draw.line(surface, _PG_BIKE, shoulder, handlebar, 2)  # arm
    pygame.draw.circle(surface, _PG_BIKE, head_center, 3, 1)  # head


_PG_OUTCOME_OK = (30, 160, 70)
_PG_OUTCOME_FAIL = (200, 40, 40)


def _pg_draw_outcome_marker(
    surface,
    pygame,
    bike_xy: tuple[int, int],
    reward_components: RewardComponents,
) -> None:
    """Draw a green tick (landed on platform) or red cross (fell) above the bike."""
    cx, cy = int(bike_xy[0]), int(bike_xy[1])
    # Anchor the marker well above the bike (which is ~22px tall here) so it
    # doesn't overlap the rider's head.
    mx, my = cx, cy - 34
    size = 9
    if float(reward_components.is_missed) <= 0.0:
        # Green tick: two strokes forming a check mark.
        pygame.draw.lines(
            surface,
            _PG_OUTCOME_OK,
            False,
            [(mx - size, my), (mx - 2, my + size - 2), (mx + size, my - size + 1)],
            3,
        )
    else:
        # Red cross: two diagonals.
        pygame.draw.line(
            surface,
            _PG_OUTCOME_FAIL,
            (mx - size, my - size),
            (mx + size, my + size),
            3,
        )
        pygame.draw.line(
            surface,
            _PG_OUTCOME_FAIL,
            (mx - size, my + size),
            (mx + size, my - size),
            3,
        )


def _pg_draw_visibility(
    surface,
    pygame,
    layout: _PygameLayout,
    config: EnvConfig,
    anchor_cell: int | None,
) -> None:
    if anchor_cell is None:
        return
    vis_lo_cell = anchor_cell + 1
    vis_hi_cell = min(anchor_cell + config.visibility_k, config.track_length)
    if vis_hi_cell <= vis_lo_cell:
        return
    x0 = layout.cell_to_x(vis_lo_cell) - int(layout.cell_step // 2)
    x1 = layout.cell_to_x(vis_hi_cell - 1) + int(layout.cell_step // 2)
    rect = pygame.Rect(x0, _PG_MARGIN_TOP, max(x1 - x0, 2), layout.draw_h)
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((*_PG_VISIBILITY, 50))
    surface.blit(overlay, rect.topleft)


def _pg_draw_legend(surface, pygame, font) -> None:
    items = [
        ("terrain", _PG_TERRAIN),
        ("gap", _PG_GAP),
        ("platform", _PG_PLATFORM),
        ("bike", _PG_BIKE),
        ("jump arc", _PG_ARC),
        ("visible", _PG_VISIBILITY),
    ]
    x = _PG_WIDTH - _PG_MARGIN_X - 110
    y = _PG_MARGIN_TOP
    for label, color in items:
        pygame.draw.rect(surface, color, pygame.Rect(x, y, 10, 10))
        text = font.render(label, True, _PG_TEXT)
        surface.blit(text, (x + 16, y - 2))
        y += 16


def _pg_draw_reward_text(
    surface,
    pygame,
    font,
    jump_time: int | None,
    landing_position: int | None,
    reward_components: RewardComponents,
) -> None:
    lines = [
        f"jump_time={jump_time}  landing={landing_position}",
        (
            f"landing_err={reward_components.landing_error:.2f}  "
            f"fall={reward_components.is_missed:.0f}"
        ),
        f"return={reward_components.total:.2f}",
    ]
    x = _PG_MARGIN_X
    y = _PG_HEIGHT - _PG_MARGIN_BOTTOM + 6
    for line in lines:
        text = font.render(line, True, _PG_TEXT)
        surface.blit(text, (x, y))
        y += 16


def _pg_surface_to_rgb(pygame, surface) -> np.ndarray:
    arr = pygame.surfarray.array3d(surface)
    # pygame returns shape (W, H, 3); convert to (H, W, 3).
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8, copy=False)


def _save_gif(frames: list[np.ndarray], save_path: str, duration_ms: int) -> None:
    """Persist ``frames`` as an animated GIF using Pillow."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    if not images:
        return
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
