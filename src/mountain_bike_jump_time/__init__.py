"""Mountain-bike jump-time toy environment for OPE evaluation.

Public API:
    MountainBikeJumpEnv: the Farama Gymnasium environment.
    LatentConfig: dataclass describing a fully-resolved latent episode.
    EnvConfig: dataclass with the environment's configurable parameters.

The environment is also registered with Gymnasium under the id
``MountainBikeJump-v0`` upon importing this package.
"""

from gymnasium.envs.registration import register

from mountain_bike_jump_time.env import (
    EnvConfig,
    LatentConfig,
    MountainBikeJumpEnv,
    RewardComponents,
)
from mountain_bike_jump_time.visualization import render_episode

__all__ = [
    "EnvConfig",
    "LatentConfig",
    "MountainBikeJumpEnv",
    "RewardComponents",
    "render_episode",
]

register(
    id="MountainBikeJump-v0",
    entry_point="mountain_bike_jump_time.env:MountainBikeJumpEnv",
)
