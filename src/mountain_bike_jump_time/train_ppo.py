"""Ray RLlib PPO training + evaluation example for ``MountainBikeJump-v0``.

This module provides a small, self-contained example of training a PPO policy
on the toy mountain-bike jump-time environment using Ray RLlib's new API stack
(``EnvRunner`` / ``RLModule``) and then evaluating the learned policy both
via RLlib's built-in evaluation loop and by computing the *exact* policy
value through enumeration of the finite discrete latent space.

Usage
-----
As a CLI:

.. code-block:: bash

    python -m mountain_bike_jump_time.train_ppo --iterations 20

Per-episode rollouts of the latent-space enumeration can optionally be
saved as figures via ``--viz-all`` (every episode), ``--viz-best``
(highest-return episode) and ``--viz-worst`` (lowest-return episode).
Two visualization backends are available, selected with
``--viz-renderer``: ``matplotlib`` (default) writes a static PNG per
episode, ``pygame`` writes an animated GIF per episode. The output
directory defaults to a fresh temp dir whose path is printed at runtime;
pass ``--viz-output-dir`` to override it.

As a library:

.. code-block:: python

    from mountain_bike_jump_time.train_ppo import train_and_evaluate
    result = train_and_evaluate(iterations=20)
    print(result["mean_eval_return"])
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig

import mountain_bike_jump_time  # noqa: F401  -- registers MountainBikeJump-v0
from mountain_bike_jump_time import EnvConfig

ENV_NAME = "MountainBikeJump-v0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def build_ppo_config(
    *,
    num_env_runners: int = 0,
    train_batch_size: int = 1024,
    minibatch_size: int = 128,
    num_epochs: int = 5,
    lr: float = 3e-4,
    gamma: float = 0.99,
    use_lstm: bool = False,
    seed: int | None = 0,
    env_config: EnvConfig | None = None,
) -> PPOConfig:
    """Build a :class:`PPOConfig` suited to the tiny jump-timing environment.

    Defaults are intentionally lightweight so this example runs in a few
    seconds on a CPU-only laptop while still showing learning signal.

    :param num_env_runners: Number of remote ``EnvRunner`` actors. ``0`` runs everything in the
        local process which is fastest for this tiny environment.
    :param train_batch_size: PPO training batch size.
    :param minibatch_size: PPO minibatch size.
    :param num_epochs: PPO number of epochs per training iteration.
    :param lr: PPO learning rate.
    :param gamma: PPO discount factor.
    :param use_lstm: If ``True``, use an LSTM in the policy network.
    :param seed: Random seed for reproducibility.
    :param env_config: Optional :class:`EnvConfig` forwarded to the gym
        environment constructor (e.g. to change ``track_length``).
    :return: A configured :class:`PPOConfig` instance ready to build an algorithm.
    """
    env_kwargs: dict[str, Any] = {}
    if env_config is not None:
        env_kwargs["config"] = env_config
    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_kwargs)
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=gamma,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
            model={
                "use_lstm": use_lstm,
            },
        )
        .evaluation(
            evaluation_interval=None,  # we trigger evaluation manually
        )
        .debugging(seed=seed)
        .learners(num_learners=0)
        # torch+cpu is used as dependency.
        # If you have a GPU, you'll need to:
        # - install torch with GPU support (e.g. update poetry dependencies)
        # - set num_gpus=1 here
        .resources(num_gpus=0)
    )
    return config


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
@dataclass
class VisualizationConfig:
    """Configuration for saving per-episode rollout visualizations.

    :param save_all: If ``True``, save a rollout for every episode enumerated
        from the latent space.
    :param save_best: If ``True``, save a rollout for the latent episode with
        the highest greedy return.
    :param save_worst: If ``True``, save a rollout for the latent episode with
        the lowest greedy return.
    :param output_dir: Directory where outputs are written. If ``None``, a
        fresh temporary directory is created (and advertised to the user).
    :param renderer: Visualization backend; ``"matplotlib"`` (default) writes
        static PNG figures, ``"pygame"`` writes animated GIFs.
    """

    save_all: bool = False
    save_best: bool = False
    save_worst: bool = False
    output_dir: str | None = None
    renderer: str = "matplotlib"

    @property
    def enabled(self) -> bool:
        return self.save_all or self.save_best or self.save_worst

    @property
    def file_extension(self) -> str:
        return "gif" if self.renderer == "pygame" else "png"


# ---------------------------------------------------------------------------
# Train + evaluate driver
# ---------------------------------------------------------------------------
def train_and_evaluate(
    *,
    iterations: int = 20,
    num_env_runners: int = 0,
    eval_episodes: int = 50,
    train_batch_size: int = 1024,
    minibatch_size: int = 128,
    num_epochs: int = 5,
    use_lstm: bool = False,
    seed: int | None = 0,
    verbose: bool = True,
    visualization: VisualizationConfig | None = None,
    env_config: EnvConfig | None = None,
) -> dict[str, Any]:
    """Train a PPO policy and return train/eval metrics.

    :param iterations: Number of ``algo.train()`` iterations.
    :param num_env_runners: Number of remote ``EnvRunner`` actors. ``0`` runs everything in the
        local process which is fastest for this tiny environment.
    :param eval_episodes: Number of greedy evaluation episodes used for the empirical estimate.
    :param train_batch_size: PPO training batch size.
    :param minibatch_size: PPO minibatch size.
    :param num_epochs: PPO number of epochs per training iteration.
    :param use_lstm: If ``True``, use an LSTM in the policy network.
    :param seed: Random seed for reproducibility.
    :param verbose: If ``True``, print training and evaluation progress to stdout.
    :param visualization: Optional :class:`VisualizationConfig` enabling rollout
        visualizations of the latent-space enumeration. ``None`` (default)
        disables visualization entirely.
    :param env_config: Optional :class:`EnvConfig` used by both the
        training environment and the exact policy-value enumeration.
    :return: A dictionary containing training history, evaluation statistics, and the exact policy value.
    """
    shutdown_after = not ray.is_initialized()
    if shutdown_after:
        ray.init(include_dashboard=False, logging_level=logging.ERROR)

    try:
        config = build_ppo_config(
            num_env_runners=num_env_runners,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            use_lstm=use_lstm,
            seed=seed,
            env_config=env_config,
        )
        algo = config.build_algo()

        train_history: list[dict[str, float]] = []
        try:
            for i in range(iterations):
                result = algo.train()
                env_runners = result.get("env_runners", {})
                ep_return = env_runners.get("episode_return_mean")
                ep_len = env_runners.get("episode_len_mean")
                train_history.append(
                    {
                        "iteration": i + 1,
                        "episode_return_mean": (
                            float(ep_return) if ep_return is not None else float("nan")
                        ),
                        "episode_len_mean": (float(ep_len) if ep_len is not None else float("nan")),
                    }
                )
                if verbose:
                    print(
                        f"[iter {i + 1:>3d}/{iterations}] "
                        f"episode_return_mean={ep_return} "
                        f"episode_len_mean={ep_len}"
                    )

            from mountain_bike_jump_time.rollout import (
                evaluate_policy_episodes,
                exact_policy_value,
            )

            rollout_data = evaluate_policy_episodes(
                algo, env_config=env_config, num_episodes=eval_episodes, seed=(seed or 0) + 1
            )

            rl_module = algo.get_module()
            if rl_module is None:
                raise RuntimeError("algo.get_module() returned None, cannot evaluate policy.")
            exact_value = exact_policy_value(
                rl_module,
                env_config=env_config,
                visualization=visualization,
                verbose=verbose,
            )
        finally:
            algo.stop()

        from mountain_bike_jump_time.rollout import EvalStats

        eval_stats: EvalStats = rollout_data["eval_stats"]

        if verbose:
            print("\n=== Evaluation (greedy rollouts) ===")
            for k, v in eval_stats.as_dict().items():
                print(f"  {k}: {v}")
            print(f"\nExact greedy policy value V(π) = {exact_value:.4f}")

        return {
            "train_history": train_history,
            "eval_stats": eval_stats,
            "exact_policy_value": exact_value,
            "mean_eval_return": eval_stats.mean_return,
            "episodes_data": rollout_data["episodes_data"],
        }
    finally:
        if shutdown_after:
            ray.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mountain_bike_jump_time.train_ppo",
        description=(
            "Train a Ray RLlib PPO policy on MountainBikeJump-v0 and "
            "evaluate it both empirically (greedy rollouts) and exactly "
            "(latent-space enumeration)."
        ),
    )
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--num-env-runners", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--use-lstm", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--track-length",
        type=int,
        default=None,
        help=(
            "Override the env ``track_length``. The gap + platform + post-gap "
            "obstacle is always placed near the end of the track, so a longer "
            "track simply means a longer pre-gap ride."
        ),
    )
    p.add_argument(
        "--viz-all",
        action="store_true",
        help="Save a rollout for every episode of the latent-space enumeration.",
    )
    p.add_argument(
        "--viz-best",
        action="store_true",
        help="Save a rollout for the latent episode with the highest greedy return.",
    )
    p.add_argument(
        "--viz-worst",
        action="store_true",
        help="Save a rollout for the latent episode with the lowest greedy return.",
    )
    p.add_argument(
        "--viz-output-dir",
        type=str,
        default=None,
        help=(
            "Directory where visualization files are written. "
            "Defaults to a fresh temporary directory whose path is printed at runtime."
        ),
    )
    p.add_argument(
        "--viz-renderer",
        type=str,
        choices=("matplotlib", "pygame"),
        default="matplotlib",
        help=(
            "Visualization backend: 'matplotlib' for static PNG figures "
            "(default), 'pygame' for animated GIFs."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    visualization = VisualizationConfig(
        save_all=args.viz_all,
        save_best=args.viz_best,
        save_worst=args.viz_worst,
        output_dir=args.viz_output_dir,
        renderer=args.viz_renderer,
    )
    env_config: EnvConfig | None = None
    if args.track_length is not None:
        env_config = EnvConfig(track_length=args.track_length)
    train_and_evaluate(
        iterations=args.iterations,
        num_env_runners=args.num_env_runners,
        eval_episodes=args.eval_episodes,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        use_lstm=args.use_lstm,
        seed=args.seed,
        verbose=not args.quiet,
        visualization=visualization,
        env_config=env_config,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
