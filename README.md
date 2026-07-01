# mountain-bike-jump-time

<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="/assets/images/mountain_bike_hero_dark.png"
    >
    <img
      alt="hero"
      src="/assets/images/mountain_bike_hero.png"
      width="300"
      height="300"
    >
  </picture>
</p>

A toy [Farama Gymnasium](https://gymnasium.farama.org/) environment: a stochastic *jump-timing* problem in the
spirit of "optimal start" problem with random latent environment configuration that can be fully enumerated for true policy value computation.

A bike rides along a 1-D track and must decide *when* to jump in order to land on a small target platform surrounded by two gaps. The agent only controls the binary action `{0: continue, 1: jump}`; the first `1` is an irreversible switch action. Bike speed evolves according to the local slope.

<p align="center">
  <picture>
    <img
      alt="replay"
      src="/assets/images/episode_render_1.gif"
    >
  </picture>
</p>

This environment can be useful for **Off-Policy Evaluation (OPE)** estimators evaluation: the latent randomness is *finite and discrete*, which makes the true value of any policy `V(π) = Σ_ω p(ω) · G(π, ω)` exactly computable by enumeration, ideal for benchmarking IS / DM / hybrid OPE estimators.

Developping and validating OPE estimators on real-world data can be difficult. Main advantages of this toy env are:

- **Exact ground truth**: a toy environment allows exact computation of true policy value, making estimator bias and error directly measurable.
- **Controlled complexity**: stochasticity, partial observability, support mismatch, reward shape, and policy overlap can be varied independently.
- **Always known behavior policy**: action probabilities and logging mechanisms are fully known, avoiding ambiguity from schedules, overrides, and hidden logic.
- **Fast reproducible experiments**: many independent datasets can be generated with fixed seeds to measure bias, variance, MSE, and confidence interval behavior.
- **Clear failure analysis**: estimator failures are easier to isolate, visualize, and explain because the latent dynamics, counterfactuals, and reward components are known.

Note: this environment was developped as a way to test custom estimators developped in the Triggy project. Jump time problem was framed as close as possible to optimal start time problem.

## Install

Requires **Python 3.12** and **Poetry 1.8.4**.

```bash
poetry install
```

## Usage

```python
import gymnasium as gym
import mountain_bike_jump_time  # registers MountainBikeJump-v0

env = gym.make("MountainBikeJump-v0")
obs, info = env.reset(seed=0)

done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

print(info["reward_components"])
```

### Computing the exact value of a policy

```python
from mountain_bike_jump_time import MountainBikeJumpEnv
from mountain_bike_jump_time.rollout import return_for

env = MountainBikeJumpEnv()

# Enumerate all latent environment configurations and compute policy value
# Switch-time policy: always jump at u = 3.
value = sum(p * return_for(omega, switch_time=3)
            for omega, p in env.enumerate_latents())
```

### Visualizing a rollout

Two backends are available. The default `matplotlib` renderer produces
static PNG figures; the `pygame` renderer produces animated GIFs (and can
open an interactive window with `mode="human"`).

```python
from mountain_bike_jump_time import render_episode

# Static matplotlib figure (default).
render_episode(
    latent=env.latent,
    config=env.config,
    slope_per_cell=env._slope_per_cell,
    trajectory=env.trajectory,
    jump_time=env._jump_time,
    landing_position=env._landing_position,
    reward_components=env.reward_components,
    save_path="rollout.png",
)

# Animated pygame rendering written as a GIF.
render_episode(
    latent=env.latent,
    config=env.config,
    slope_per_cell=env._slope_per_cell,
    trajectory=env.trajectory,
    jump_time=env._jump_time,
    landing_position=env._landing_position,
    reward_components=env.reward_components,
    save_path="rollout.gif",
    renderer="pygame",
)
```

The training CLI exposes the same choice via `--viz-renderer matplotlib|pygame`.

### Training a PPO policy with Ray RLlib

A small end-to-end training + evaluation example using Ray RLlib's new API
stack (PPO on torch) is provided:

```bash
poetry run python -m mountain_bike_jump_time.train_ppo --iterations 20
```

The script trains a PPO policy on `MountainBikeJump-v0`, then evaluates it
both empirically (greedy rollouts) and *exactly* by enumerating the finite
discrete latent space:

```python
from mountain_bike_jump_time.train_ppo import train_and_evaluate

result = train_and_evaluate(iterations=20, eval_episodes=50)
print(result["eval_stats"]["mean_return"])
print(result["exact_policy_value"])  # V(π) via latent enumeration
```

## Test

```bash
poetry run pytest
```
