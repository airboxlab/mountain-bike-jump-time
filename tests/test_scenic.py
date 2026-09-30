"""Renderer contracts: simulation isolation, export, lifecycle and edge cases."""

import copy
import os

import numpy as np
import pytest
from PIL import Image

from mountain_bike_jump_time import (
    EnvConfig,
    LatentConfig,
    MountainBikeJumpEnv,
    RenderConfig,
    render_episode,
)
from mountain_bike_jump_time.demo import example_episode
from mountain_bike_jump_time.scenic import ScenicRenderer


def scene_for(env, **options):
    return ScenicRenderer(
        env.latent,
        env.config,
        env._slope_per_cell,
        env.trajectory,
        env._jump_time,
        env._landing_position,
        env.reward_components,
        RenderConfig(**options),
        env._done,
    )


@pytest.mark.parametrize("seed", [0, 2, 7, 12])
def test_rendering_preserves_dynamics_and_rng(seed):
    plain = MountainBikeJumpEnv()
    rendered = MountainBikeJumpEnv(render_mode="rgb_array")
    plain.reset(seed=seed)
    rendered.reset(seed=seed)
    for step in range(20):
        rng_before = copy.deepcopy(rendered.np_random.bit_generator.state)
        first, second = rendered.render(), rendered.render()
        np.testing.assert_array_equal(first, second)
        assert first.shape == (540, 960, 3)
        assert first.dtype == np.uint8
        assert rendered.np_random.bit_generator.state == rng_before
        action = int(step == 5)
        a, b = plain.step(action), rendered.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        if a[2] or a[3]:
            rendered.render()
            break
    # Even the next sampled episode must be unaffected by rendering.
    np.testing.assert_array_equal(plain.reset()[0], rendered.reset()[0])
    plain.close()
    rendered.close()


def test_visual_variation_is_repeatable_and_independent_of_physics():
    env = example_episode(0)
    images = [
        scene_for(env, theme=theme, seed=42).frame(0.4) for theme in ("alpine", "desert", "dusk")
    ]
    assert all(not np.array_equal(images[0], other) for other in images[1:])
    np.testing.assert_array_equal(images[0], scene_for(env, theme="alpine", seed=42).frame(0.4))
    assert not np.array_equal(images[0], scene_for(env, theme="alpine", seed=43).frame(0.4))


def test_interpolation_and_landing_use_recorded_cells():
    env = example_episode(0)
    scene = scene_for(env)
    before = scene.pose(scene.options.step_seconds / 2)
    assert env.trajectory[0]["position"] < before.x < env.trajectory[1]["position"]
    takeoff = scene.pose(scene.ride_duration)
    landed = scene.pose(scene.event_time)
    assert takeoff.x == env.trajectory[env._jump_time + 1]["position"]
    assert landed.x == env._landing_position
    assert landed.y == scene.ground(env._landing_position)
    assert scene.pose(scene.ride_duration + scene.flight_duration * 0.5).phase == "air"


def test_positive_slope_is_drawn_uphill():
    env = MountainBikeJumpEnv()
    env.reset(options={"latent": LatentConfig(1, (1, 0), 12, 1, 3, 2)})
    scene = scene_for(env)
    assert scene.ground(2) > scene.ground(1)


def test_partial_episode_does_not_show_a_terminal_outcome():
    env = MountainBikeJumpEnv(render_mode="rgb_array")
    env.reset(seed=0)
    scene = scene_for(env)
    assert not scene.done
    assert scene.pose(scene.duration).phase == "ride"
    assert scene.duration == 0
    env.close()


@pytest.mark.parametrize(
    "seed,jump_step,never_jump",
    [(0, None, False), (7, None, False), (0, 0, False), (0, None, True)],
)
def test_success_miss_early_jump_and_no_jump(seed, jump_step, never_jump):
    env = example_episode(seed, jump_step, never_jump)
    scene = scene_for(env, width=480, height=270, show_debug=True)
    for t in (0, scene.duration / 2, scene.duration):
        rgb = scene.frame(t)
        assert rgb.shape == (270, 480, 3)
    if never_jump:
        assert scene.pose(scene.duration).phase == "fall"
    if seed == 0 and jump_step is None and not never_jump:
        assert scene.pose(scene.duration).phase == "land"


def test_overshoot_and_long_tracks_render():
    cfg = EnvConfig(track_length=100)
    env = MountainBikeJumpEnv(cfg)
    env.reset(options={"latent": LatentConfig(4, (0,), 90, 2, 3, 2)})
    scene = ScenicRenderer(
        env.latent,
        cfg,
        env._slope_per_cell,
        [
            {"position": 89, "speed": 4, "action": None},
            {"position": 89, "speed": 4, "action": 1},
        ],
        0,
        108,
        env.reward_components,
        RenderConfig(),
        True,
    )
    assert scene.pose(scene.duration).phase == "fall"
    assert scene.frame(scene.duration).shape == (540, 960, 3)


def test_exports_and_headless_display_isolation(tmp_path, monkeypatch):
    import pygame

    pygame.display.quit()
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    env = example_episode(0)
    args = dict(
        latent=env.latent,
        config=env.config,
        slope_per_cell=env._slope_per_cell,
        trajectory=env.trajectory,
        jump_time=env._jump_time,
        landing_position=env._landing_position,
        reward_components=env.reward_components,
        renderer="pygame",
        render_config=RenderConfig(width=480, height=270, fps=15),
    )
    rgb = render_episode(**args, save_path=str(tmp_path / "preview.gif"))
    with Image.open(tmp_path / "preview.gif") as gif:
        assert gif.n_frames > 20
        assert gif.size == (480, 270)
        times = []
        for i in range(gif.n_frames):
            gif.seek(i)
            times.append(gif.info["duration"])
        assert abs(sum(times) / 1000 - scene_for(env).duration) < 0.2
    png = tmp_path / "preview.png"
    render_episode(**args, save_path=str(png))
    with Image.open(png) as image:
        np.testing.assert_array_equal(rgb, np.asarray(image))
    assert "SDL_VIDEODRIVER" not in os.environ
    assert not pygame.display.get_init()
    with pytest.raises(ValueError, match="support"):
        render_episode(**args, save_path=str(tmp_path / "preview.mp4"))


def test_human_window_reused_and_closed(monkeypatch):
    import pygame

    pygame.display.quit()
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = MountainBikeJumpEnv(
        render_mode="human",
        render_config=RenderConfig(fps=60, step_seconds=0.01, hold_seconds=0),
    )
    env.reset(seed=0)
    env.render()
    window = pygame.display.get_surface()
    assert window is not None
    env.step(0)
    env.render()
    assert pygame.display.get_surface() is window
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    env.render()
    assert env._render_scene.closed
    env.close()
    assert not pygame.display.get_init()
    # A subsequent fresh window remains possible after close.
    env.reset(seed=1)
    env.render()
    assert pygame.display.get_surface() is not None
    env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theme": "invalid"},
        {"fps": 0},
        {"width": 10},
        {"step_seconds": 0},
        {"hold_seconds": -1},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        RenderConfig(**kwargs)
