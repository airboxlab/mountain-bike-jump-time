"""Off-Policy Evaluation (OPE) example: switch-time IPS estimator.

This module trains two PPO policies with different training parameter sets
(a *behavior* policy that logs data and a *target* policy that we want to
evaluate) and then compares the value of the target policy estimated from
behavior-policy rollouts via an Inverse-Propensity-Scoring (IPS) estimator
against the *exact* target-policy value obtained by enumerating the finite
discrete latent space.

The environment has a special *switch-time* structure: the agent plays
``0`` (continue) until it plays a single irreversible ``1`` (jump) that
ends the episode. For a full episode with switch time
:math:`u \\in \\{0, 1, \\dots, H-1\\}` (or :math:`u = \\text{None}` if the
policy never jumped), the trajectory-level policy probability factorizes
as

.. math::

    \\pi^z(u \\mid s_0)
    = \\left(\\prod_{t=0}^{u-1} \\pi(a_t = 0 \\mid s_t)\\right)
      \\pi(a_u = 1 \\mid s_u)

(and :math:`\\pi^z(\\text{None} \\mid s_0) = \\prod_{t=0}^{H-1} \\pi(a_t = 0 \\mid s_t)`
when the whole horizon is played with action ``0``).

The IPS estimator is then

.. math::

    V_{IPS} = \\frac{1}{n} \\sum_{i=1}^n
              \\frac{\\pi_e^z(u^i \\mid s_0^i)}{\\pi_b^z(u^i \\mid s_0^i)} G^i

where :math:`\\pi_b` is the behavior policy, :math:`\\pi_e` is the target
policy, :math:`u^i` is the switch time observed on the :math:`i`-th
behavior-policy rollout and :math:`G^i` is that rollout's return.

Usage
-----
As a CLI:

.. code-block:: bash

    python -m mountain_bike_jump_time.example.ope_ips \\
        --behavior-iterations 5 --target-iterations 20 --ope-episodes 200

As a library:

.. code-block:: python

    from mountain_bike_jump_time.example.ope_ips import run_ope_example
    result = run_ope_example(ope_episodes=200)
    print(result["v_ips"], result["exact_target_value"])
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig

from mountain_bike_jump_time.env import EnvConfig, MountainBikeJumpEnv
from mountain_bike_jump_time.rollout import (
    EpisodeData,
    exact_policy_value,
    greedy_action_logits,
)

# A "policy" for OPE bookkeeping is any callable that, given an observation,
# returns a length-2 probability vector ``[P(a=0|s), P(a=1|s)]``.
PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class OPEResult:
    """Result of one OPE example run."""

    v_ips: float
    exact_target_value: float
    exact_behavior_value: float
    behavior_empirical_return: float
    num_episodes: int
    absolute_error: float = field(init=False)

    def __post_init__(self) -> None:
        self.absolute_error = abs(self.v_ips - self.exact_target_value)

    def as_dict(self) -> dict[str, float]:
        return {
            "v_ips": self.v_ips,
            "exact_target_value": self.exact_target_value,
            "exact_behavior_value": self.exact_behavior_value,
            "behavior_empirical_return": self.behavior_empirical_return,
            "num_episodes": self.num_episodes,
            "absolute_error": self.absolute_error,
        }


def switch_time_prob(
    observations: list[np.ndarray] | None,
    switch_time: int | None,
    policy: PolicyFn | None,
    probs: list[np.ndarray] | None,
) -> float:
    """Trajectory-level switch-time policy probability :math:`\\pi^z(u \\mid s_0)`.

    Evaluates ``policy`` on the given state sequence and returns the
    probability that this policy would produce the switch-time trajectory
    described by ``switch_time`` (i.e. play ``0`` on ``s_0, ..., s_{u-1}``
    and ``1`` on ``s_u``; or play ``0`` on the entire horizon when
    ``switch_time is None``).

    :param observations: The sequence of observations
        ``s_0, s_1, ..., s_L`` visited before each action was taken. Used only if ``probs`` is not provided.
    :param switch_time: Time step of the (single) jump, or ``None`` for a
        never-jump trajectory.
    :param policy: A callable ``s -> [P(0|s), P(1|s)]``.
    :param probs: Pre-computed policy probability vectors for each observation. If provided, ``policy`` is ignored.
    :return: The trajectory probability (a scalar in [0, 1]).
    """
    assert (observations is not None) != (
        probs is not None
    ), "Exactly one of observations or probs must be provided."
    assert policy is not None or probs is not None, "Either policy or probs must be provided."

    traj_length = len(observations) if observations is not None else len(probs)

    if switch_time is None:
        # Never-jump trajectory: play 0 at every visited state.
        prob = 1.0
        if probs is not None:
            for p in probs:
                prob *= float(p[0])
        else:
            for obs in observations:
                p = policy(obs)
                prob *= float(p[0])
        return prob

    if switch_time < 0 or switch_time >= traj_length:
        raise ValueError(
            f"switch_time={switch_time} is out of range "
            f"for a trajectory of length {traj_length}"
        )

    prob = 1.0
    for t in range(switch_time):
        if probs is not None:
            p = probs[t]
        else:
            p = policy(observations[t])
        prob *= float(p[0])

    if probs is not None:
        prob *= float(probs[switch_time][1])
    else:
        prob *= float(policy(observations[switch_time])[1])
    return prob


def v_ips(
    episodes: list[EpisodeData],
    target_policy: PolicyFn,
) -> float:
    """Switch-time IPS estimator of the target policy value.

    Computes

    .. math::

        V_{IPS} = \\frac{1}{n} \\sum_{i=1}^n
                  \\frac{\\pi_e^z(u^i \\mid s_0^i)}
                       {\\pi_b^z(u^i \\mid s_0^i)} G^i

    using the behavior probabilities logged in ``episodes`` and the
    provided ``target_policy`` to re-evaluate :math:`\\pi_e^z`.

    :param episodes: Behavior-policy rollouts. Must be non-empty.
    :param target_policy: Callable returning the target-policy action
        probability vector for a given observation.
    :return: The IPS estimate :math:`\\hat V_{IPS}`.
    :raises ValueError: If ``episodes`` is empty, or if any behavior
        trajectory has zero probability under the logged policy (which
        would indicate a bug — behavior probabilities are what generated
        the sample).
    """
    if not episodes:
        raise ValueError("v_ips requires at least one behavior episode.")

    total = 0.0
    for ep in episodes:
        # Behavior probability uses the logged action-probability vectors
        # (no re-run of the behavior RLModule required).
        pi_b = switch_time_prob(
            probs=ep.action_probs, switch_time=ep.switch_time, policy=None, observations=None
        )
        if pi_b <= 0.0:
            raise ValueError(
                "Behavior-policy probability of a logged trajectory is 0. "
                "This should be impossible for a stochastic behavior policy."
            )
        pi_e = switch_time_prob(
            observations=ep.observations,
            switch_time=ep.switch_time,
            policy=target_policy,
            probs=None,
        )
        total += (pi_e / pi_b) * ep.episode_return

    return total / len(episodes)


def policy_fn_from_module(rl_module: Any) -> PolicyFn:
    """Wrap an RLlib ``RLModule`` as a plain ``PolicyFn``.

    :param rl_module: A trained RLModule with ``forward_inference``.
    :return: A callable ``obs -> [P(0|obs), P(1|obs)]``.
    """

    def policy(obs: np.ndarray) -> np.ndarray:
        logits = greedy_action_logits(rl_module, obs)
        x = np.asarray(logits, dtype=np.float64)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    return policy


def sample_behavior_episodes(
    behavior_policy: PolicyFn,
    *,
    env_config: EnvConfig | None = None,
    num_episodes: int = 200,
    seed: int = 20240101,
) -> list[EpisodeData]:
    """Sample stochastic rollouts from ``behavior_policy``.

    Actions are drawn from the categorical distribution returned by
    ``behavior_policy`` at each visited state. The full action-probability
    vector is stored for later IPS re-weighting.

    :param behavior_policy: A ``PolicyFn`` (typically wrapping a trained
        behavior RLModule).
    :param env_config: Environment configuration forwarded to
        :class:`MountainBikeJumpEnv`.
    :param num_episodes: Number of behavior-policy rollouts to collect.
    :param seed: Base seed for the numpy PRNG and per-episode env resets.
    :return: A list of :class:`BehaviorEpisode`.
    """
    env = MountainBikeJumpEnv(env_config or EnvConfig())
    rng = np.random.default_rng(seed)

    episodes: list[EpisodeData] = []
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = truncated = False
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        probs: list[np.ndarray] = []
        ep_return = 0.0
        switch_time: int | None = None
        t = 0
        while not (terminated or truncated):
            p = np.asarray(behavior_policy(obs), dtype=np.float64)
            # Store the exact observation object that will be looked up in
            # v_ips via `is`-identity, so we copy defensively.
            obs_copy = np.array(obs, copy=True)
            observations.append(obs_copy)
            probs.append(p)
            action = int(rng.choice(2, p=p))
            actions.append(action)
            if action == 1 and switch_time is None:
                switch_time = t
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            ep_return += float(reward)
            t += 1

        episodes.append(
            EpisodeData(
                latent=env.latent,
                observations=observations,
                actions=actions,
                action_probs=probs,
                rewards=rewards,
                episode_return=ep_return,
            )
        )

    return episodes


# ---------------------------------------------------------------------------
# End-to-end example driver
# ---------------------------------------------------------------------------


@dataclass
class PolicyTrainingParams:
    """Training parameters for one PPO policy in the OPE example.

    Only a small subset of :func:`train_ppo.build_ppo_config` knobs is
    exposed here — the ones we vary between behavior and target policies
    to obtain two *distinct* but *overlapping* policies (a requirement
    for IPS to have finite variance).
    """

    iterations: int = 5
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_param: float = 0.2
    vf_loss_coeff: float = 1.0
    entropy_coeff: float = 0.05
    train_batch_size: int = 512
    minibatch_size: int = 128
    num_epochs: int = 3
    use_lstm: bool = False
    seed: int = 0


def get_trained_policy_module(params: PolicyTrainingParams, env_config: EnvConfig | None) -> Any:
    """Train a PPO policy with the given ``params`` and return its RLModule.

    The RLlib algorithm is stopped after extracting the module so no
    background actors leak.

    :param params: Training parameters for this policy.
    :param env_config: Optional environment configuration.
    :return: A trained RLModule (standalone copy).
    """

    import mountain_bike_jump_time  # noqa: F401 - registers MountainBikeJump-v0
    from mountain_bike_jump_time.train_ppo import ENV_NAME

    env_kwargs: dict[str, Any] = {}
    if env_config is not None:
        env_kwargs["config"] = env_config

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_kwargs)
        .framework("torch")
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1, rollout_fragment_length="auto")
        .training(
            train_batch_size=params.train_batch_size,
            minibatch_size=params.minibatch_size,
            num_epochs=params.num_epochs,
            lr=params.lr,
            gamma=params.gamma,
            lambda_=params.lambda_,
            clip_param=params.clip_param,
            entropy_coeff=params.entropy_coeff,
            vf_loss_coeff=params.vf_loss_coeff,
            model={"use_lstm": params.use_lstm},
        )
        .evaluation(evaluation_interval=None)
        .debugging(seed=params.seed)
        .learners(num_learners=0)
        .resources(num_gpus=0)
    )
    algo = config.build_algo()
    try:
        for _ in range(params.iterations):
            algo.train()
        # Extract a *standalone* copy of the RLModule so we can safely stop
        # the algorithm without losing the weights.
        module = algo.get_module()
        if module is None:
            raise RuntimeError("algo.get_module() returned None.")
        return module
    finally:
        algo.stop()


def run_ope_example(
    *,
    behavior_params: PolicyTrainingParams | None = None,
    target_params: PolicyTrainingParams | None = None,
    ope_episodes: int = 20,
    env_config: EnvConfig | None = None,
    verbose: bool = True,
    ope_seed: int = 42,
) -> OPEResult:
    """Train behavior + target policies and compare :math:`V_{IPS}` to :math:`V(\\pi_e)`.

    :param behavior_params: PPO training params for the behavior policy
        (default: a small, high-entropy run — good for logging).
    :param target_params: PPO training params for the target policy
        (default: a longer, lower-entropy run — the policy we evaluate).
    :param ope_episodes: Number of behavior-policy rollouts used for the
        IPS estimate.
    :param env_config: Optional :class:`EnvConfig` shared by both trainings
        and the exact-value enumeration.
    :param verbose: If ``True``, print progress and the final comparison.
    :param ope_seed: Base seed for behavior rollout collection.
    :return: An :class:`OPEResult` summarising the comparison.
    """

    behavior_params = behavior_params or PolicyTrainingParams(
        iterations=5, lr=3e-4, entropy_coeff=0.05, seed=0
    )
    target_params = target_params or PolicyTrainingParams(
        iterations=20, lr=3e-4, entropy_coeff=0.01, seed=1
    )

    shutdown_after = not ray.is_initialized()
    if shutdown_after:
        ray.init(include_dashboard=False, logging_level=logging.ERROR)

    try:
        if verbose:
            print(f"[ope] Training behavior policy: {behavior_params}")
        behavior_module = get_trained_policy_module(behavior_params, env_config)
        if verbose:
            print(f"[ope] Training target policy:   {target_params}")
        target_module = get_trained_policy_module(target_params, env_config)

        behavior_policy = policy_fn_from_module(behavior_module)
        target_policy = policy_fn_from_module(target_module)

        if verbose:
            print(f"[ope] Sampling {ope_episodes} behavior rollouts")
        episodes = sample_behavior_episodes(
            behavior_policy,
            env_config=env_config,
            num_episodes=ope_episodes,
            seed=ope_seed,
        )

        v_ips_estimate = v_ips(episodes, target_policy)
        empirical_behavior_return = float(np.mean([ep.episode_return for ep in episodes]))

        # Ground truth: exact value of each *stochastic* policy under the
        # switch-time factorization, obtained by enumerating both the
        # discrete latent space and every possible switch time.
        exact_target_value = exact_policy_value(target_module, env_config=env_config)
        exact_behavior_value = exact_policy_value(behavior_module, env_config=env_config)

        result = OPEResult(
            v_ips=float(v_ips_estimate),
            exact_target_value=float(exact_target_value),
            exact_behavior_value=float(exact_behavior_value),
            behavior_empirical_return=empirical_behavior_return,
            num_episodes=len(episodes),
        )

        if verbose:
            print("\n=== OPE (switch-time IPS) ===")
            for k, v in result.as_dict().items():
                print(f"  {k}: {v}")

        return result
    finally:
        if shutdown_after:
            ray.shutdown()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mountain_bike_jump_time.ope_ips",
        description=(
            "Train two PPO policies with different training parameter sets, "
            "then compare the switch-time IPS estimate of the target policy "
            "value with the exact target-policy value."
        ),
    )
    p.add_argument("--behavior-iterations", type=int, default=5)
    p.add_argument("--behavior-lr", type=float, default=3e-4)
    p.add_argument("--behavior-gamma", type=float, default=0.99)
    p.add_argument("--behavior-lambda", type=float, default=0.95)
    p.add_argument("--behavior-clip-param", type=float, default=0.2)
    p.add_argument("--behavior-vf-loss-coeff", type=float, default=1.0)
    p.add_argument("--behavior-entropy-coeff", type=float, default=0.05)
    p.add_argument("--behavior-use-lstm", action="store_true")
    p.add_argument("--behavior-seed", type=int, default=0)

    p.add_argument("--target-iterations", type=int, default=20)
    p.add_argument("--target-lr", type=float, default=3e-4)
    p.add_argument("--target-gamma", type=float, default=0.99)
    p.add_argument("--target-lambda", type=float, default=0.95)
    p.add_argument("--target-clip-param", type=float, default=0.2)
    p.add_argument("--target-vf-loss-coeff", type=float, default=1.0)
    p.add_argument("--target-entropy-coeff", type=float, default=0.01)
    p.add_argument("--target-use-lstm", action="store_true")
    p.add_argument("--target-seed", type=int, default=1)

    p.add_argument("--ope-episodes", type=int, default=200)
    p.add_argument("--ope-seed", type=int, default=20240101)
    p.add_argument("--track-length", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    env_config: EnvConfig | None = None
    if args.track_length is not None:
        env_config = EnvConfig(track_length=args.track_length)

    behavior_params = PolicyTrainingParams(
        iterations=args.behavior_iterations,
        lr=args.behavior_lr,
        gamma=args.behavior_gamma,
        lambda_=args.behavior_lambda,
        clip_param=args.behavior_clip_param,
        vf_loss_coeff=args.behavior_vf_loss_coeff,
        entropy_coeff=args.behavior_entropy_coeff,
        use_lstm=args.behavior_use_lstm,
        seed=args.behavior_seed,
    )
    target_params = PolicyTrainingParams(
        iterations=args.target_iterations,
        lr=args.target_lr,
        gamma=args.target_gamma,
        lambda_=args.target_lambda,
        clip_param=args.target_clip_param,
        vf_loss_coeff=args.target_vf_loss_coeff,
        entropy_coeff=args.target_entropy_coeff,
        use_lstm=args.target_use_lstm,
        seed=args.target_seed,
    )
    run_ope_example(
        behavior_params=behavior_params,
        target_params=target_params,
        ope_episodes=args.ope_episodes,
        env_config=env_config,
        ope_seed=args.ope_seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
