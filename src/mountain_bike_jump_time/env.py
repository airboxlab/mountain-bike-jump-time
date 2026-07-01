"""Mountain-bike jump-time toy Gymnasium environment.

The agent rides a bike on a 1D track and must decide *when* to jump in order
to land on a small target platform surrounded by two gaps. The agent only
controls jump timing; speed evolves according to the local slope.

Design notes
------------
* Finite-horizon, episodic. Binary action space: ``0`` continue, ``1`` jump.
* The first ``1`` is an irreversible switch action that ends the episode.
* The latent episode configuration ``omega`` is drawn from a *finite* discrete
  set, so the true value of any policy can be computed exactly by
  enumeration via :meth:`MountainBikeJumpEnv.enumerate_latents`.
* The policy observes only a local window of length ``visibility_k`` ahead of
  the bike, plus its own position and speed. Terrain past the visible window
  is masked out (zeros) and an explicit visibility-mask channel is provided.
* The reward is shaped (early/late/landing/fall) and decomposed in the
  ``info`` dict for diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for :class:`MountainBikeJumpEnv`.

    All choice tuples define the *finite discrete* support of the latent
    randomness, which keeps the latent space enumerable.
    """

    # Episode shape
    track_length: int = 20
    visibility_k_ratio: float = 0.25  # fraction of track_length

    # Latent randomness (finite discrete supports)
    initial_speed_choices: tuple[int, ...] = (1, 2)
    # First gap (before the platform).
    gap_length_choices: tuple[int, ...] = (1, 2)
    # Second gap (after the platform). Combined with ``end_padding_choices``
    # this forces the platform to sit near the end of the track regardless
    # of ``track_length``: ``pre_gap_steps`` is derived so that
    # ``pre_gap_steps + gap + platform_length + post_gap + end_padding ==
    # track_length``.
    post_gap_length_choices: tuple[int, ...] = (1, 2)
    # Number of "ride" cells remaining after the post-platform gap. Small
    # values mean the agent must ride almost all the way to the end before
    # jumping.
    end_padding_choices: tuple[int, ...] = (0, 1, 2)
    # The target platform length is fixed across episodes (only its position
    # is randomized through the gap / padding draws above).
    platform_length: int = 3
    # The track is split into slope segments equal-width segments and
    # each one is assigned a slope drawn from ``slope_choices`` ∈ {-1, 0, +1}.
    n_slope_segments_choices: tuple[int, ...] = (2, 3, 4, 5)
    slope_choices: tuple[int, ...] = (-1, 0, 1)

    # Speed dynamics
    min_speed: int = 1
    max_speed: int = 4

    # Jump dynamics: landing offset = round(jump_speed_coef * speed - slope)
    # (downhill -> slope=-1 -> longer jump; uphill -> shorter jump).
    jump_speed_coef: float = 1.5

    # Reward shaping. Landing (or not) on the platform already provides the
    # main learning signal via ``c_fall``; ``c_landing`` is a small,
    # distance-decaying guide (the farther the bike lands from the platform
    # center, the larger the penalty).
    c_fall: float = 5.0
    c_landing: float = 0.1

    def __post_init__(self) -> None:
        if self.track_length <= 0:
            raise ValueError("track_length must be positive")
        if not (0 < self.visibility_k_ratio <= 1):
            raise ValueError("visibility_k_ratio must be in (0, 1)")
        if self.min_speed < 1:
            raise ValueError("min_speed must be >= 1")
        if self.max_speed < self.min_speed:
            raise ValueError("max_speed must be >= min_speed")
        if self.platform_length < 1:
            raise ValueError("platform_length must be >= 1")
        for s in self.slope_choices:
            if s not in (-1, 0, 1):
                raise ValueError("slope_choices entries must be in {-1, 0, +1}")
        # The largest possible (gap, post_gap, padding) draw must leave
        # room for at least one cell of pre-gap ride.
        worst_case = (
            1
            + max(self.gap_length_choices)
            + self.platform_length
            + max(self.post_gap_length_choices)
            + max(self.end_padding_choices)
        )
        if worst_case > self.track_length:
            raise ValueError(
                f"track_length={self.track_length} is too small to fit "
                f"pre-gap ride + gap + platform + post-gap + padding "
                f"(needs at least {worst_case})."
            )

    @property
    def visibility_k(self) -> int:
        """Number of cells visible ahead of the bike."""
        return max(1, int(round(self.visibility_k_ratio * self.track_length)))


@dataclass(frozen=True)
class LatentConfig:
    """Fully-resolved latent episode configuration ``omega``.

    Together with the agent's *switch time* this fully determines the return.
    """

    initial_speed: int
    slope_segments: tuple[int, ...]  # one slope per terrain segment
    pre_gap_steps: int  # cells before the first gap starts
    gap_length: int  # length of the first gap (before the platform)
    platform_length: int  # length of the target platform
    post_gap_length: int  # length of the second gap (after the platform)


@dataclass
class RewardComponents:
    """Per-episode decomposition of the (negative) reward.

    The reward signal is intentionally simple: ``is_missed`` is ``1.0``
    whenever the bike does not land on the platform (either it fell into a
    gap, or jumped but missed the platform). ``landing_error`` is a small, distance-decaying guide
    equal to the absolute distance between the (landing or final) position
    and the platform center.
    """

    landing_error: float = 0.0
    is_missed: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MountainBikeJumpEnv(gym.Env):
    """Mountain-bike jump-time toy environment.

    Observation
        A flat ``Box`` of shape ``(3 + 2 * visibility_k,)`` containing,
        in order: normalized position, normalized speed,
        slope window of length ``k`` (values in {-1, 0, +1}, zeros where
        beyond track), and visibility-mask window of length ``k``
        (``1`` if the cell exists, ``0`` otherwise).

    Action
        ``Discrete(2)``: ``0`` continue, ``1`` jump.

    Reward
        ``0`` at every intermediate step. A single terminal reward equal to
        ``- (c_fall * is_missed + c_landing * landing_error)`` is delivered
        when the episode ends. ``is_missed`` is ``1.0`` whenever the bike
        does not land on the platform (fell into a gap, or jumped but missed the platform).
        ``landing_error`` is the absolute distance from the (landing or
        final) position to the platform center; it acts as a small
        distance-decaying guide. The decomposition is also exposed through
        ``info["reward_components"]``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config: EnvConfig = config if config is not None else EnvConfig()
        self.render_mode = render_mode

        k = self.config.visibility_k
        obs_dim = 3 + 2 * k
        # Bounds: position/speed in [0, 1] (normalized), slope in [-1, 1],
        # visibility mask in [0, 1]. We use a single broad box for simplicity.
        low = np.full(obs_dim, -1.0, dtype=np.float32)
        high = np.full(obs_dim, 1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Episode state (filled by reset)
        self._latent: LatentConfig | None = None
        self._slope_per_cell: np.ndarray | None = None
        self._position: int = 0
        self._speed: int = 0
        self._t: int = 0
        self._done: bool = False
        self._jumped: bool = False
        self._jump_time: int | None = None
        self._landing_position: int | None = None
        self._reward_components: RewardComponents = RewardComponents()
        # Trajectory buffer used by the visualization helper.
        self._trajectory: list[dict[str, Any]] = []

    # ----------------------------------------------------------- properties
    @property
    def latent(self) -> LatentConfig | None:
        """The latent episode configuration of the current episode, if any."""
        return self._latent

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """Per-step trajectory log for visualization/debugging."""
        return list(self._trajectory)

    @property
    def reward_components(self) -> RewardComponents:
        """Reward decomposition of the last finished episode."""
        return self._reward_components

    # ----------------------------------------------------------- public API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "latent" in options:
            latent = options["latent"]
            if not isinstance(latent, LatentConfig):
                raise TypeError("options['latent'] must be a LatentConfig")
            self._latent = latent
        else:
            self._latent = self._sample_latent(self.np_random)

        self._slope_per_cell = self.materialize_slope(
            self._latent.slope_segments, self.config.track_length
        )
        self._position = 0
        self._speed = self._latent.initial_speed
        self._t = 0
        self._done = False
        self._jumped = False
        self._jump_time = None
        self._landing_position = None
        self._reward_components = RewardComponents()
        self._trajectory = []

        obs = self._build_observation()
        self._log_step(obs=obs, action=None, reward=0.0)
        return obs, self._build_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a terminated episode; call reset() first")
        if action not in (0, 1):
            raise ValueError(f"invalid action {action!r}; expected 0 or 1")

        reward = 0.0
        terminated = False
        truncated = False

        if action == 1:
            # Irreversible jump: resolve landing & compute terminal reward.
            self._jumped = True
            self._jump_time = self._t
            self._landing_position = self.compute_landing_position(
                position=self._position, speed=self._speed
            )
            self._reward_components = self.compute_reward(
                self._latent, jump_time=self._t, landing_position=self._landing_position
            )
            reward = self._reward_components.total
            terminated = True
        else:
            # Continue: advance physics one step.
            self._advance_one_step()
            # Fall into a gap without ever jumping?
            if self._is_in_gap(self._position):
                self._reward_components = self.compute_reward(
                    self._latent,
                    jump_time=None,
                    landing_position=None,
                )
                reward = self._reward_components.total
                terminated = True
            elif self._t >= self.config.track_length or self._position >= self.config.track_length:
                # Out of horizon / fell off the end without jumping -> late.
                self._reward_components = self.compute_reward(
                    self._latent,
                    jump_time=None,
                    landing_position=None,
                )
                reward = self._reward_components.total
                truncated = True

        self._done = terminated or truncated
        obs = self._build_observation()
        self._log_step(obs=obs, action=int(action), reward=float(reward))
        return obs, float(reward), terminated, truncated, self._build_info()

    def render(self):  # pragma: no cover - thin wrapper around visualization
        if self.render_mode is None:
            return None
        from mountain_bike_jump_time.visualization import render_episode

        return render_episode(
            latent=self._latent,
            config=self.config,
            slope_per_cell=self._slope_per_cell,
            trajectory=self._trajectory,
            jump_time=self._jump_time,
            landing_position=self._landing_position,
            reward_components=self._reward_components,
            mode=self.render_mode,
        )

    # ------------------------------------------------- enumeration / OPE API
    def enumerate_latents(self) -> Iterator[tuple[LatentConfig, float]]:
        """Yield every ``(omega, p(omega))`` pair.

        The factorized prior is uniform on each axis, so each combination has
        equal probability ``1 / |Omega|``.
        """
        cfg = self.config
        # Enumerate slope-segment sequences across every allowed segment
        # count, with the same "no two consecutive segments share the same
        # slope" constraint used by ``_sample_latent``. This keeps the
        # number of *visually distinct* terrain segments equal to
        # ``len(slope_segments)`` and matches the sampling distribution
        # (previously, enumeration produced only ``max(n_slope_segments_choices)``-
        # segment latents, inconsistent with sampling).
        slope_sequences: list[tuple[int, ...]] = []
        for n in cfg.n_slope_segments_choices:
            for seq in product(*[cfg.slope_choices] * n):
                if all(seq[i] != seq[i - 1] for i in range(1, len(seq))):
                    slope_sequences.append(seq)
        combos = list(
            product(
                cfg.initial_speed_choices,
                slope_sequences,
                cfg.gap_length_choices,
                cfg.post_gap_length_choices,
                cfg.end_padding_choices,
            )
        )
        # Build physically valid latents. ``pre_gap_steps`` is derived from
        # ``track_length`` and the other lengths so the obstacle sits near
        # the end of the track regardless of ``track_length``.
        valid: list[LatentConfig] = []
        for init_speed, slope_segs, gap_len, post_gap_len, padding in combos:
            pre_gap = cfg.track_length - padding - post_gap_len - cfg.platform_length - gap_len
            if pre_gap < 1:
                continue
            valid.append(
                LatentConfig(
                    initial_speed=init_speed,
                    slope_segments=tuple(slope_segs),
                    pre_gap_steps=pre_gap,
                    gap_length=gap_len,
                    platform_length=cfg.platform_length,
                    post_gap_length=post_gap_len,
                )
            )
        if not valid:
            raise RuntimeError(
                "EnvConfig leaves no valid latent configurations; "
                "increase track_length or shrink gap/platform/padding choices."
            )
        prob = 1.0 / len(valid)
        for latent in valid:
            yield latent, prob

    # =====================================================================
    # Internal helpers
    # =====================================================================
    def _sample_latent(self, rng: np.random.Generator) -> LatentConfig:
        cfg = self.config
        while True:
            gap_length = int(rng.choice(cfg.gap_length_choices))
            post_gap_length = int(rng.choice(cfg.post_gap_length_choices))
            padding = int(rng.choice(cfg.end_padding_choices))
            pre_gap_steps = (
                cfg.track_length - padding - post_gap_length - cfg.platform_length - gap_length
            )
            if pre_gap_steps < 1:
                continue
            n_segs = int(rng.choice(cfg.n_slope_segments_choices))
            slope_segments = self._sample_slope_segments(rng, n_segs)
            return LatentConfig(
                initial_speed=int(rng.choice(cfg.initial_speed_choices)),
                slope_segments=slope_segments,
                pre_gap_steps=pre_gap_steps,
                gap_length=gap_length,
                platform_length=cfg.platform_length,
                post_gap_length=post_gap_length,
            )

    def _sample_slope_segments(self, rng: np.random.Generator, n_segs: int) -> tuple[int, ...]:
        """Sample a slope sequence with no two consecutive equal slopes.

        Without this constraint, two adjacent segments can be drawn with the
        same slope value (e.g. ``(1, 1)``), which merges them visually into
        a single segment in the renderer. Enforcing distinct neighbours
        guarantees that ``len(slope_segments)`` matches the number of
        visually-distinct terrain segments.
        """
        choices = self.config.slope_choices
        if n_segs <= 1 or len(choices) <= 1:
            return tuple(int(rng.choice(choices)) for _ in range(max(n_segs, 0)))
        segs: list[int] = [int(rng.choice(choices))]
        for _ in range(n_segs - 1):
            others = [s for s in choices if s != segs[-1]]
            segs.append(int(rng.choice(others)))
        return tuple(segs)

    @staticmethod
    def materialize_slope(segments: tuple[int, ...], track_length: int) -> np.ndarray:
        """Materialize a slope-per-cell array from a slope-segment sequence.

        :param segments: tuple of slope values, one per segment
        :param track_length: total number of cells in the track
        :return: array of length ``track_length`` with slope values per cell
        """
        out = np.zeros(track_length, dtype=np.int8)
        seg_len = track_length // len(segments)
        for i, slope in enumerate(segments):
            start = i * seg_len
            end = track_length if i == len(segments) - 1 else (i + 1) * seg_len
            out[start:end] = slope
        return out

    # -- terrain queries ---------------------------------------------------
    def first_gap_range(self, latent: LatentConfig) -> tuple[int, int]:
        start = latent.pre_gap_steps
        return start, start + latent.gap_length

    def platform_range(self, latent: LatentConfig) -> tuple[int, int]:
        start = latent.pre_gap_steps + latent.gap_length
        return start, start + latent.platform_length

    def _is_in_gap(self, position: int) -> bool:
        """Return ``True`` if reaching ``position`` without jumping is a fall.

        The agent must jump *before* the first gap: any movement that lands
        on or past the first gap's start is treated as a fall (the bike
        cannot magically cross the gap on the ground). The post-platform
        region (gap + end padding) is symmetrically off-limits without a
        successful platform landing.
        """
        gs, _ = self.first_gap_range(self._latent)
        if position >= gs:
            return True
        return False

    # -- dynamics ----------------------------------------------------------
    def _advance_one_step(self) -> None:
        cfg = self.config
        slope_here = int(self._slope_per_cell[min(self._position, cfg.track_length - 1)])
        self._position += self._speed
        # speed update: downhill (-1) accelerates, uphill (+1) decelerates.
        self._speed = int(np.clip(self._speed - slope_here, cfg.min_speed, cfg.max_speed))
        self._t += 1

    def compute_landing_position(self, position: int, speed: int) -> int:
        cfg = self.config
        idx = min(max(position, 0), cfg.track_length - 1)
        slope = int(self._slope_per_cell[idx])
        offset = int(round(cfg.jump_speed_coef * speed - slope))
        offset = max(offset, 1)
        return position + offset

    # -- reward ------------------------------------------------------------
    def compute_reward(
        self,
        latent: LatentConfig,
        jump_time: int | None,
        landing_position: int | None,
    ) -> RewardComponents:
        """Compute the terminal reward decomposition.

        The signal is intentionally simple. The primary signal is binary:
        landing on the platform yields ``is_missed=0``; anything else
        (falling into either gap or jumping and missing) yields ``is_missed=1``. A small
        ``landing_error`` term — the absolute distance from the (landing or
        final) position to the platform center — provides a smooth guide
        that decays as the bike lands closer to the platform.
        """
        cfg = self.config
        ps, pe = self.platform_range(latent)
        platform_center = (ps + pe - 1) / 2.0
        comp = RewardComponents()

        if jump_time is None:
            # Never jumped: always a fall. The bike walked into the first gap.
            comp.is_missed = 1.0
            comp.landing_error = abs(self._position - platform_center)
        else:
            landed_on_platform = ps <= landing_position < pe
            comp.landing_error = abs(landing_position - platform_center)
            if not landed_on_platform:
                comp.is_missed = 1.0

        comp.total = -(cfg.c_fall * comp.is_missed + cfg.c_landing * comp.landing_error)
        return comp

    # -- observation -------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        cfg = self.config
        k = cfg.visibility_k
        norm_pos = self._position / max(cfg.track_length - 1, 1)
        norm_speed = self._speed / cfg.max_speed
        slope_window = np.zeros(k, dtype=np.float32)
        mask_window = np.zeros(k, dtype=np.float32)
        for i in range(k):
            cell = self._position + 1 + i  # one step ahead of bike
            if 0 <= cell < cfg.track_length:
                slope_window[i] = float(self._slope_per_cell[cell])
                mask_window[i] = 1.0
        return np.concatenate(
            [
                np.array(
                    [norm_pos, norm_speed, float(self._t) / cfg.track_length], dtype=np.float32
                ),
                slope_window,
                mask_window,
            ]
        )

    def _build_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "position": int(self._position),
            "speed": int(self._speed),
            "t": int(self._t),
            "jumped": bool(self._jumped),
            "jump_time": self._jump_time,
            "landing_position": self._landing_position,
            "latent": self._latent,
        }
        if self._done:
            info["reward_components"] = self._reward_components
        return info

    def _log_step(self, obs: np.ndarray, action: int | None, reward: float) -> None:
        self._trajectory.append(
            {
                "t": self._t,
                "position": int(self._position),
                "speed": int(self._speed),
                "action": action,
                "reward": float(reward),
                "obs": obs.copy(),
            }
        )
