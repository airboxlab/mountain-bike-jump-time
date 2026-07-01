"""Unit tests for the mountain-bike jump-time environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import mountain_bike_jump_time  # noqa: F401  (registers the env)
from mountain_bike_jump_time import EnvConfig, LatentConfig, MountainBikeJumpEnv
from mountain_bike_jump_time.env import RewardComponents

# ---------------------------------------------------------------- spaces


def test_gymnasium_registration_and_spaces():
    env = gym.make("MountainBikeJump-v0")
    assert env.action_space.n == 2
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    env.close()


def test_observation_shape_matches_visibility_k():
    cfg = EnvConfig(visibility_k_ratio=0.5)  # 50% of track length
    env = MountainBikeJumpEnv(cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 + 2 * 0.5 * cfg.track_length,)
    assert env.observation_space.shape == obs.shape


def test_invalid_action_raises():
    env = MountainBikeJumpEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(2)


def test_step_after_done_raises():
    env = MountainBikeJumpEnv()
    env.reset(seed=0)
    env.step(1)  # jump -> terminated
    with pytest.raises(RuntimeError):
        env.step(0)


# ----------------------------------------------------------- determinism


def test_seed_reproducibility():
    env1 = MountainBikeJumpEnv()
    env2 = MountainBikeJumpEnv()
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)
    assert env1.latent == env2.latent


def test_passing_latent_via_options_overrides_sampling():
    env = MountainBikeJumpEnv()
    latent = LatentConfig(
        initial_speed=1,
        slope_segments=(0, 0),
        pre_gap_steps=12,
        gap_length=1,
        platform_length=3,
        post_gap_length=1,
    )
    env.reset(seed=0, options={"latent": latent})
    assert env.latent == latent


# --------------------------------------------------------------- rewards


def test_jump_too_early_lands_far_from_platform():
    """Jumping at t=0 from position 0 should miss the platform and be penalised."""
    cfg = EnvConfig()
    env = MountainBikeJumpEnv(cfg)
    latent = LatentConfig(
        initial_speed=1,
        slope_segments=(0, 0),
        pre_gap_steps=13,
        gap_length=1,
        platform_length=3,
        post_gap_length=1,
    )
    env.reset(seed=0, options={"latent": latent})
    _, reward, term, trunc, info = env.step(1)
    assert term and not trunc
    comp: RewardComponents = info["reward_components"]
    # The bike jumps from position 0 with speed 1 → lands very far from
    # the platform, so it's a fall with a non-trivial landing error.
    assert comp.is_missed == 1.0
    assert comp.landing_error > 0
    assert reward < 0


def test_never_jumping_gives_fall_penalty():
    """A policy that never jumps on a track with a gap always falls."""
    cfg = EnvConfig()
    env = MountainBikeJumpEnv(cfg)
    env.reset(seed=0)
    done = False
    last_info = {}
    while not done:
        _, _, term, trunc, last_info = env.step(0)
        done = term or trunc
    comp: RewardComponents = last_info["reward_components"]
    assert comp.is_missed == 1.0


def test_reward_components_sum_matches_total():
    cfg = EnvConfig()
    env = MountainBikeJumpEnv(cfg)
    env.reset(seed=3)
    _, reward, _, _, info = env.step(1)
    comp: RewardComponents = info["reward_components"]
    expected = -(cfg.c_fall * comp.is_missed + cfg.c_landing * comp.landing_error)
    assert reward == pytest.approx(expected)
    assert comp.total == pytest.approx(expected)


# --------------------------------------------------------- OPE machinery


def test_enumerate_latents_yields_valid_probability_distribution():
    env = MountainBikeJumpEnv()
    pairs = list(env.enumerate_latents())
    assert len(pairs) > 0
    total = sum(p for _, p in pairs)
    assert total == pytest.approx(1.0)
    # All latents must be physically valid.
    cfg = env.config
    for latent, _ in pairs:
        total_len = (
            latent.pre_gap_steps
            + latent.gap_length
            + latent.platform_length
            + latent.post_gap_length
        )
        assert total_len <= cfg.track_length
        assert latent.pre_gap_steps >= 1
        assert latent.platform_length == cfg.platform_length
