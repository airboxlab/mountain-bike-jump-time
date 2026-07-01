import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.core.rl_module import RLModule

from mountain_bike_jump_time.env import EnvConfig, LatentConfig, MountainBikeJumpEnv
from mountain_bike_jump_time.train_ppo import VisualizationConfig
from mountain_bike_jump_time.visualization import render_episode


@dataclass
class EpisodeData:
    """Data captured for a single episode rollout, suitable for OPE or visualization."""

    # episode latent configuration and environment config
    latent: LatentConfig
    # actions taken by the agent at each time step
    actions: list[int]
    # rewards received at each time step
    rewards: list[float]
    # observations received at each time step after taking the action
    observations: list[np.ndarray]
    # total return (undiscounted sum of rewards) for the episode
    episode_return: float
    # action probabilities (logits) at each time step
    action_probs: list[np.ndarray] | None = None

    @property
    def episode_length(self) -> int:
        """Return the length of the episode (number of time steps)."""
        return len(self.actions)

    @property
    def switch_time(self) -> int | None:
        """Return the time step at which the agent jumped, or None if never jumped."""
        for t, action in enumerate(self.actions):
            if action == 1:
                return t
        return None

    @staticmethod
    def from_snapshot(snapshot: dict[str, Any]) -> "EpisodeData":
        """Construct an EpisodeData from a snapshot captured by :func:`snapshot_episode`."""
        latent = snapshot["latent"]
        trajectory = snapshot.get("trajectory", [])
        return EpisodeData(
            latent=latent,
            actions=[step["action"] for step in trajectory],
            rewards=[step["reward"] for step in trajectory],
            observations=[step["obs"] for step in trajectory],
            episode_return=sum(step["reward"] for step in trajectory),
        )


@dataclass
class EvalStats:
    """Summary statistics for a set of evaluation episodes."""

    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    mean_length: float
    jump_rate: float
    num_episodes: int

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the evaluation statistics."""
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_length": self.mean_length,
            "jump_rate": self.jump_rate,
            "num_episodes": self.num_episodes,
        }


def evaluate_policy_episodes(
    algo: Algorithm,
    *,
    env_config: EnvConfig | None = None,
    num_episodes: int = 50,
    seed: int = 12345,
) -> dict[str, Any]:
    """Roll out the trained policy *greedily* and return both summary statistics and trajectories
    data for each episode that can be used for off-policy evaluation (OPE) or visualization.

    Uses ``algo.get_module()`` to query the trained ``RLModule`` directly so
    this works even when there are no remote env-runners.

    :param algo: A trained RLlib algorithm with a greedy policy.
    :param env_config: Optional environment configuration to use for the
    :param num_episodes: Number of greedy evaluation episodes to run.
    :param seed: Random seed for reproducibility.
    :return: A dictionary containing:
        - statistics: mean/std/min/max returns, mean episode length, and jump rate (fraction of episodes where the agent jumped).
        - episodes_data: a list of :class:`EpisodeData` for each episode, containing the latent configuration,
          action probabilities, actions, rewards, observations, total return, and episode length.
    """
    rl_module: RLModule | None = algo.get_module()
    if rl_module is None:
        raise RuntimeError("algo.get_module() returned None, cannot evaluate policy.")

    env_config = env_config or EnvConfig()
    env = MountainBikeJumpEnv(env_config)
    rng = np.random.default_rng(seed)

    returns: list[float] = []
    lengths: list[int] = []
    jump_rate = 0
    episodes_data: list[EpisodeData] = []

    for _ in range(num_episodes):
        # track the episode data for OPE or visualization
        logits: list[np.ndarray] = []

        # sample a new latent configuration for each episode
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = truncated = False
        ep_return = 0.0
        ep_len = 0
        jumped = False

        while not (terminated or truncated):
            act_logits = greedy_action_logits(rl_module, obs)
            logits.append(act_logits)
            action = greedy_action(act_logits)

            jumped = jumped or (action == 1)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        snapshot = snapshot_episode(env)
        episode_data = EpisodeData.from_snapshot(snapshot)
        episode_data.action_probs = logits
        episodes_data.append(episode_data)
        returns.append(ep_return)
        lengths.append(ep_len)
        jump_rate += int(jumped)

    eval_stats = EvalStats(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        mean_length=float(np.mean(lengths)),
        jump_rate=jump_rate / num_episodes,
        num_episodes=num_episodes,
    )

    return {
        "eval_stats": eval_stats,
        "episodes_data": episodes_data,
    }


def exact_policy_value(
    rl_module: RLModule,
    *,
    env_config: EnvConfig | None = None,
    visualization: VisualizationConfig | None = None,
    verbose: bool = False,
) -> float:
    """Exact greedy-policy value via enumeration of the latent space.

    The environment is fully enumerable in its latent randomness, so we can
    compute ``V(π) = Σ_ω p(ω) · G(π, ω)`` without any Monte-Carlo noise.
    We use deterministic rollouts to compute the return for that latent.

    :param rl_module: A trained RLlib RLModule with a greedy policy.
    :param env_config: Optional environment configuration to use for the
        exact evaluation. If ``None``, the default ``EnvConfig()`` is used.
    :param visualization: Optional :class:`VisualizationConfig` controlling
        whether/where per-episode PNG rollouts are written. When ``None``
        (the default), no figures are produced.
    :param verbose: If ``True``, advertise the resolved visualization output
        directory to stdout.
    :return: The exact greedy policy value V(π).
    """
    env = MountainBikeJumpEnv(env_config or EnvConfig())

    viz_enabled = visualization is not None and visualization.enabled
    out_dir: Path | None = None
    if viz_enabled:
        out_dir = resolve_output_dir(visualization.output_dir, verbose=verbose)

    best: tuple[float, dict[str, Any]] | None = None
    worst: tuple[float, dict[str, Any]] | None = None

    total_prob = 0.0
    total_omega_prob = 0.0
    for idx, (latent, prob) in enumerate(env.enumerate_latents()):
        obs, _ = env.reset(options={"latent": latent})
        terminated = truncated = False
        ep_return = 0.0
        while not (terminated or truncated):
            action = greedy_action(greedy_action_logits(rl_module, obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
        total_prob += prob * ep_return
        total_omega_prob += prob

        if viz_enabled:
            need_snapshot = (
                visualization.save_all or visualization.save_best or visualization.save_worst
            )
            snapshot = snapshot_episode(env) if need_snapshot else None
            ext = visualization.file_extension
            if visualization.save_all:
                render_snapshot(
                    snapshot,
                    out_dir / f"episode_{idx:04d}_return_{ep_return:+.3f}.{ext}",
                    renderer=visualization.renderer,
                )
            if visualization.save_best and (best is None or ep_return > best[0]):
                best = (ep_return, snapshot)
            if visualization.save_worst and (worst is None or ep_return < worst[0]):
                worst = (ep_return, snapshot)

    if viz_enabled:
        ext = visualization.file_extension
        if visualization.save_best and best is not None:
            render_snapshot(
                best[1],
                out_dir / f"best_return_{best[0]:+.3f}.{ext}",
                renderer=visualization.renderer,
            )
        if visualization.save_worst and worst is not None:
            render_snapshot(
                worst[1],
                out_dir / f"worst_return_{worst[0]:+.3f}.{ext}",
                renderer=visualization.renderer,
            )

    # Sanity check: Σ_ω p(ω) should sum to 1.0
    if not np.isclose(total_omega_prob, 1.0):
        raise RuntimeError(f"latent probabilities sum to {total_omega_prob:.6f}, expected 1.0")
    return float(total_prob)


def greedy_action_logits(rl_module: RLModule, obs: np.ndarray) -> np.ndarray:
    """Run a single inference step on the RLModule and return the action logits.

    :param rl_module: The RLModule to query.
    :param obs: The observation to feed into the RLModule.
    :return: The action logits (shape: [2]).
    """
    batch = {"obs": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)}
    with torch.no_grad():
        out = rl_module.forward_inference(batch)
    logits = out["action_dist_inputs"]
    return logits.squeeze(0).cpu().numpy()


def greedy_action(logits: np.ndarray) -> int:
    """Transform action logits into a greedy action (0 or 1) by taking the argmax.

    :param logits: The action logits (shape: [2]).
    :return: The greedy action (0 or 1).
    """
    return int(np.argmax(logits))


def render_snapshot(
    snapshot: dict[str, Any], save_path: Path, renderer: str = "matplotlib"
) -> None:
    """Render a snapshot captured by :func:`_snapshot_episode` to ``save_path``."""
    render_episode(
        latent=snapshot["latent"],
        config=snapshot["config"],
        slope_per_cell=snapshot["slope_per_cell"],
        trajectory=snapshot["trajectory"],
        jump_time=snapshot["jump_time"],
        landing_position=snapshot["landing_position"],
        reward_components=snapshot["reward_components"],
        mode="rgb_array",
        save_path=str(save_path),
        renderer=renderer,
    )


def snapshot_episode(env: MountainBikeJumpEnv) -> dict[str, Any]:
    """Capture everything needed to re-render an episode after the env moves on."""
    return {
        "latent": env.latent,
        "config": env.config,
        "slope_per_cell": env._slope_per_cell.copy(),
        "trajectory": env.trajectory,
        "jump_time": env._jump_time,
        "landing_position": env._landing_position,
        "reward_components": env.reward_components,
    }


def resolve_output_dir(output_dir: str | None, verbose: bool) -> Path:
    """Resolve the visualization output directory, creating a temp dir if needed.

    :param output_dir: User-supplied directory, or ``None`` for a fresh temp dir.
    :param verbose: If ``True``, print the directory path to stdout so the user
        knows where to look for the PNGs.
    :return: A path to an existing, writable directory.
    """
    if output_dir is None:
        out = Path(tempfile.mkdtemp(prefix="mbjt_viz_"))
        if verbose:
            print(f"[viz] No --viz-output-dir given; saving visualizations to {out}")
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[viz] Saving visualizations to {out}")
    return out


def return_for(latent: LatentConfig, switch_time: int | None) -> float:
    """Deterministic return ``G(u, omega)`` for a given switch time.

    ``switch_time=None`` means *never jump* over the whole horizon
    (matches a behavior policy that always outputs ``0``).

    This method is independent of the live episode state and is intended
    for OPE oracle computations / unit tests.

    :param latent: the latent configuration to simulate
    :param switch_time: the time step at which to jump, or None to never jump
    :return: the total return (sum of rewards) for this rollout
    """

    # initialize the environment with latent configuration
    env: MountainBikeJumpEnv = MountainBikeJumpEnv()
    env.reset(seed=0, options={"latent": latent})

    # simulate the environment step by step
    cfg = env.config
    slopes = env.materialize_slope(latent.slope_segments, cfg.track_length)
    position = 0
    speed = latent.initial_speed

    # Replay the rollout up to switch_time (or until termination).
    for t in range(cfg.track_length):

        # If we reach the switch time, we trigger the jump and compute the reward.
        if switch_time is not None and t == switch_time:
            env._position = position
            env._speed = speed
            env._latent = latent
            env._slope_per_cell = slopes
            landing = env.compute_landing_position(position, speed)
            comp = env.compute_reward(latent, jump_time=t, landing_position=landing)
            return comp.total

        # Otherwise, we continue the simulation without jumping.
        slope_here = int(slopes[min(position, cfg.track_length - 1)])
        position += speed
        speed = int(np.clip(speed - slope_here, cfg.min_speed, cfg.max_speed))

        # Did we reach the first gap without jumping? The bike cannot
        # roll over the gap on the ground — any advance to the first
        # gap's start cell (or past it) is a fall.
        gs, _ = env.first_gap_range(latent)
        if position >= gs:
            env._position = position
            comp = env.compute_reward(latent, jump_time=None, landing_position=None)
            return comp.total

    # we should never reach here in a well-formed environment, but if we do, we compute the reward for falling.
    env._position = position
    comp = env.compute_reward(latent, jump_time=None, landing_position=None)
    return comp.total
