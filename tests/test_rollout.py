import numpy as np
import pytest

from mountain_bike_jump_time.env import LatentConfig, MountainBikeJumpEnv
from mountain_bike_jump_time.rollout import EpisodeData, return_for, snapshot_episode


def test_return_for_matches_live_rollout():
    """``return_for`` (pure simulator) must agree with a live rollout."""
    env = MountainBikeJumpEnv()
    latent = LatentConfig(
        initial_speed=2,
        slope_segments=(0, -1),
        pre_gap_steps=5,
        gap_length=1,
        platform_length=2,
        post_gap_length=2,
    )
    # Live rollout: do nothing for 2 steps then jump.
    env.reset(seed=0, options={"latent": latent})
    env.step(0)
    env.step(0)
    _, live_reward, _, _, _ = env.step(1)

    # Switch time u=2.
    oracle_reward = return_for(latent, switch_time=2)
    assert live_reward == pytest.approx(oracle_reward)


def test_policy_value_by_enumeration_is_finite():
    """We can compute V(pi) for a trivial 'jump at u=2' policy."""
    env = MountainBikeJumpEnv()
    value = sum(
        prob * return_for(latent, switch_time=2) for latent, prob in env.enumerate_latents()
    )
    assert np.isfinite(value)
    assert value <= 0.0  # all reward components are non-positive


def test_episode_data_from_snapshot():
    """We can compute episode data from a snapshot of the environment."""
    env = MountainBikeJumpEnv()
    latent = LatentConfig(
        initial_speed=2,
        slope_segments=(0, -1),
        pre_gap_steps=5,
        gap_length=1,
        platform_length=2,
        post_gap_length=2,
    )
    env.reset(seed=0, options={"latent": latent})
    env.step(0)
    env.step(0)
    env.step(1)

    snapshot = snapshot_episode(env)
    episode_data = EpisodeData.from_snapshot(snapshot)
    assert episode_data.latent == latent
    assert episode_data.action_probs is None  # we didn't record action probabilities in this test
    assert episode_data.actions == [None, 0, 0, 1]
    assert len(episode_data.observations) == 4  # initial obs + 3 steps
    assert episode_data.episode_length == 4
    assert episode_data.episode_return == sum(step["reward"] for step in snapshot["trajectory"])
