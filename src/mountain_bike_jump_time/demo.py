"""Preview the scenic renderer without installing or training RLlib.

python -m mountain_bike_jump_time.demo --theme alpine --output episode.gif
"""

from __future__ import annotations

import argparse

from mountain_bike_jump_time import MountainBikeJumpEnv, RenderConfig, render_episode


def example_episode(seed=0, jump_step=None, never_jump=False):
    """Choose a good jump for a demo only; this is an oracle, not a learned policy."""
    env = MountainBikeJumpEnv()
    env.reset(seed=seed)
    latent = env.latent
    if jump_step is None and not never_jump:
        best = (-float("inf"), 0)
        for candidate in range(env.config.track_length):
            env.reset(options={"latent": latent})
            for step in range(env.config.track_length):
                _, reward, terminated, truncated, _ = env.step(int(step == candidate))
                if terminated or truncated:
                    if reward > best[0]:
                        best = reward, candidate
                    break
        jump_step = best[1]
    env.reset(options={"latent": latent})
    for step in range(env.config.track_length):
        _, _, terminated, truncated, _ = env.step(int(not never_jump and step == jump_step))
        if terminated or truncated:
            return env
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("auto", "alpine", "desert", "dusk"), default="auto")
    parser.add_argument("--seed", type=int, default=0, help="Environment and visual seed")
    parser.add_argument("--output", default=None, help="Write .gif animation or .png final frame")
    parser.add_argument("--human", action="store_true", help="Play in a window; Escape closes it")
    parser.add_argument("--debug", action="store_true", help="Show agent visibility and jump trace")
    parser.add_argument("--jump-step", type=int, help="Override the demo's oracle jump timing")
    parser.add_argument("--never-jump", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    env = example_episode(args.seed, args.jump_step, args.never_jump)
    output = args.output or (None if args.human else "episode.gif")
    render_episode(
        latent=env.latent,
        config=env.config,
        slope_per_cell=env._slope_per_cell,
        trajectory=env.trajectory,
        jump_time=env._jump_time,
        landing_position=env._landing_position,
        reward_components=env.reward_components,
        renderer="pygame",
        mode="human" if args.human else "rgb_array",
        save_path=output,
        render_config=RenderConfig(
            theme=args.theme, seed=args.seed, fps=args.fps, show_debug=args.debug
        ),
        episode_done=True,
    )
    env.close()
    if output:
        print(f"Saved {output} (return {env.reward_components.total:+.2f})")


if __name__ == "__main__":
    main()
