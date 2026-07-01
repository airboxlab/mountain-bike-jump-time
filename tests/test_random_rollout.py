"""Random-policy rollout test that exercises the visualization."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mountain_bike_jump_time import EnvConfig, MountainBikeJumpEnv, render_episode


def _run_random_episode(env: MountainBikeJumpEnv, rng: np.random.Generator):
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    while not done:
        action = int(rng.integers(0, 2))
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info


def _rollout_for_rendering(seed: int = 0) -> MountainBikeJumpEnv:
    cfg = EnvConfig()
    env = MountainBikeJumpEnv(cfg)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        info = _run_random_episode(env, rng)
        assert "reward_components" in info
    return env


def test_random_policy_rollout_runs_and_visualization_produces_image(tmp_path):
    env = _rollout_for_rendering()

    # Render the most recent episode and check the output PNG exists & is
    # a proper RGB array.
    out_path = tmp_path / "rollout.png"
    img = render_episode(
        latent=env.latent,
        config=env.config,
        slope_per_cell=env._slope_per_cell,
        trajectory=env.trajectory,
        jump_time=env._jump_time,
        landing_position=env._landing_position,
        reward_components=env.reward_components,
        mode="rgb_array",
        save_path=str(out_path),
    )
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_pygame_renderer_produces_animated_gif(tmp_path):
    pytest.importorskip("pygame")
    PIL_image = pytest.importorskip("PIL.Image")

    env = _rollout_for_rendering(seed=1)
    out_path = tmp_path / "rollout.gif"
    img = render_episode(
        latent=env.latent,
        config=env.config,
        slope_per_cell=env._slope_per_cell,
        trajectory=env.trajectory,
        jump_time=env._jump_time,
        landing_position=env._landing_position,
        reward_components=env.reward_components,
        mode="rgb_array",
        save_path=str(out_path),
        renderer="pygame",
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8

    # Confirm we actually wrote an animated GIF (more than 1 frame).
    with PIL_image.open(out_path) as gif:
        assert gif.format == "GIF"
        assert getattr(gif, "n_frames", 1) > 1


def test_render_episode_rejects_unknown_renderer():
    env = _rollout_for_rendering(seed=2)
    with pytest.raises(ValueError, match="Unknown renderer"):
        render_episode(
            latent=env.latent,
            config=env.config,
            slope_per_cell=env._slope_per_cell,
            trajectory=env.trajectory,
            jump_time=env._jump_time,
            landing_position=env._landing_position,
            reward_components=env.reward_components,
            renderer="opengl",
        )
