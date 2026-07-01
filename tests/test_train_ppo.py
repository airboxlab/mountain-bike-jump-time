"""Smoke test for the Ray RLlib PPO training + evaluation example.

Keeps the run very small (2 iterations, a few episodes) so it fits in the
regular ``pytest`` budget while exercising the full code path: building the
config, training, greedy rollout evaluation, and exact policy-value
enumeration.
"""

from __future__ import annotations

import math

import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("ray.rllib")
pytest.importorskip("torch")

from mountain_bike_jump_time.train_ppo import (  # noqa: E402
    VisualizationConfig,
    build_ppo_config,
    train_and_evaluate,
)


def test_build_ppo_config_uses_env_and_torch():
    config = build_ppo_config()
    assert config.env == "MountainBikeJump-v0"
    assert config.framework_str == "torch"


def test_train_and_evaluate_smoke():
    result = train_and_evaluate(
        iterations=2,
        eval_episodes=4,
        train_batch_size=256,
        minibatch_size=64,
        num_epochs=1,
        seed=0,
        verbose=False,
    )

    assert len(result["train_history"]) == 2
    assert result["eval_stats"].num_episodes == 4
    # Returns are bounded above by 0 (rewards are negative penalties).
    assert result["eval_stats"].mean_return <= 0.0
    assert math.isfinite(result["exact_policy_value"])
    assert result["exact_policy_value"] <= 0.0


def test_train_and_evaluate_with_visualization(tmp_path):
    out_dir = tmp_path / "viz"
    result = train_and_evaluate(
        iterations=1,
        eval_episodes=1,
        train_batch_size=256,
        minibatch_size=64,
        num_epochs=1,
        seed=0,
        verbose=False,
        visualization=VisualizationConfig(
            save_all=False,  # too slow
            save_best=True,
            save_worst=True,
            output_dir=str(out_dir),
        ),
    )

    assert math.isfinite(result["exact_policy_value"])
    assert out_dir.is_dir()
    pngs = sorted(out_dir.glob("*.png"))

    assert not any(p.name.startswith("episode_") for p in pngs)
    assert any(p.name.startswith("best_") for p in pngs)
    assert any(p.name.startswith("worst_") for p in pngs)
    for p in pngs:
        assert p.stat().st_size > 0


def test_visualization_disabled_does_not_create_output(tmp_path):
    out_dir = tmp_path / "should_not_exist"
    # ``enabled`` is False when all flags are off, so no directory is created.
    result = train_and_evaluate(
        iterations=1,
        eval_episodes=2,
        train_batch_size=256,
        minibatch_size=64,
        num_epochs=1,
        seed=0,
        verbose=False,
        visualization=VisualizationConfig(output_dir=str(out_dir)),
    )
    assert math.isfinite(result["exact_policy_value"])
    assert not out_dir.exists()


def test_train_and_evaluate_with_pygame_visualization(tmp_path):
    pytest.importorskip("pygame")
    out_dir = tmp_path / "viz_pygame"
    result = train_and_evaluate(
        iterations=1,
        eval_episodes=2,
        train_batch_size=256,
        minibatch_size=64,
        num_epochs=1,
        seed=0,
        verbose=False,
        visualization=VisualizationConfig(
            save_best=True,
            save_worst=True,
            output_dir=str(out_dir),
            renderer="pygame",
        ),
    )

    assert math.isfinite(result["exact_policy_value"])
    assert out_dir.is_dir()
    gifs = sorted(out_dir.glob("*.gif"))
    assert any(p.name.startswith("best_") for p in gifs)
    assert any(p.name.startswith("worst_") for p in gifs)
    # The matplotlib backend should not have run, so no PNGs.
    assert not list(out_dir.glob("*.png"))
    for p in gifs:
        assert p.stat().st_size > 0
