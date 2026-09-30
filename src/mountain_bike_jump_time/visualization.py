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

The scenic backend adds procedural landscapes and smooth animation; diagnostic
visibility and trajectory overlays are opt-in with RenderConfig(show_debug=True).
Matplotlib retains the static terrain, trajectory and reward diagnostic view.
They are intentionally headless-friendly: callers may pass ``save_path``
to write the output to disk without ever opening a window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from mountain_bike_jump_time.env import EnvConfig, LatentConfig, RewardComponents
    from mountain_bike_jump_time.scenic import RenderConfig


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
    render_config: RenderConfig | None = None,
    episode_done: bool | None = None,
):
    """Render the rollout using the selected backend.

    :param latent: latent configuration of the episode
    :param config: environment configuration
    :param slope_per_cell: array of slopes per cell (length == config.track_length)
    :param trajectory: list of per-step dictionaries, each with keys ``position`` and
        ``speed``.
    :param jump_time: index of the step where the jump was initiated, or ``None`` if no jump
        was attempted.
    :param landing_position: position where the bike landed, or ``None`` if no jump was
        attempted.
    :param reward_components: reward decomposition for the episode.
    :param mode:``"rgb_array"`` returns the rendered frame as an ``(H, W, 3)`` uint8 array;
        ``"human"`` opens an interactive window (matplotlib: a figure window; pygame: an
        animated window).
    :param save_path: Optional path; when provided, the rendered output is written to disk
        (PNG for the matplotlib backend, animated GIF for the pygame backend).
    :param render_config: Optional RenderConfig for scenery, output size and animation.
    :param episode_done: Explicit terminal flag for partial episode rendering.
    :param renderer:``"matplotlib"`` for a static 2D figure (default) or ``"pygame"`` for an
        animated 2D rendering.
    """
    if renderer not in VALID_RENDERERS:
        raise ValueError(f"Unknown renderer {renderer!r}; expected one of {VALID_RENDERERS}.")
    if renderer == "pygame":
        from mountain_bike_jump_time.scenic import render_scenic_episode

        return render_scenic_episode(
            latent=latent,
            config=config,
            slope_per_cell=slope_per_cell,
            trajectory=trajectory,
            jump_time=jump_time,
            landing_position=landing_position,
            reward_components=reward_components,
            mode=mode,
            save_path=save_path,
            render_config=render_config,
            episode_done=episode_done,
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
    # increases on uphill cells (slope == +1 in the environment).
    elevation = np.zeros(config.track_length, dtype=float)
    for i in range(1, config.track_length):
        elevation[i] = elevation[i - 1] + slope_per_cell[i - 1]
    ax.set_ylim(float(elevation.min()) - 2, float(elevation.max()) + 3)
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
