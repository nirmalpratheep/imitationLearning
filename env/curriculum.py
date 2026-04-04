"""
CurriculumBuilder — bridges game/rl_splits.py with OpenEnv environments.

Splits (from rl_splits.py):
    TRAIN  8 tracks  — 2 per difficulty tier, easy → hard
    VAL    4 tracks  — 1 per tier, performance gating
    TEST   4 tracks  — 1 per tier, held-out evaluation

Usage:
    builder = CurriculumBuilder()

    # Training loop
    env = builder.next_env()
    obs = env.reset()
    while not obs.done:
        obs = env.step(DriveAction(accel=1.0, steer=0.0))
    builder.record(total_reward)   # advances frontier when ready

    # Validation
    for env in builder.val_envs():
        ...

    # Status
    print(builder.status)
"""

from typing import Iterator, List

from game.rl_splits import TRAIN, VAL, TEST, CurriculumSampler
from .environment import RaceEnvironment


class CurriculumBuilder:
    """
    Wraps CurriculumSampler to produce ready-to-use RaceEnvironment instances.

    Parameters
    ----------
    threshold    : mean episode reward needed to advance to the next track
    window       : rolling window size for reward averaging
    replay_frac  : fraction of episodes replayed from already-mastered tracks
    max_steps    : max steps per episode for every environment created
    laps_target  : lap target per episode
    use_image    : include 64×64 egocentric headlight image in observations
    """

    def __init__(
        self,
        threshold: float = 30.0,
        window: int = 50,
        replay_frac: float = 0.3,
        max_steps: int = 3000,
        laps_target: int = 3,
        use_image: bool = True,
    ):
        self._sampler = CurriculumSampler(
            TRAIN,
            threshold=threshold,
            window=window,
            replay_frac=replay_frac,
        )
        self._max_steps = max_steps
        self._laps_target = laps_target
        self._use_image = use_image

    # ── Frontier ─────────────────────────────────────────────────────────────

    def next_env(self) -> RaceEnvironment:
        """Return an environment for the next episode (respects replay schedule)."""
        track = self._sampler.sample()
        track.build()
        return RaceEnvironment(track, self._max_steps, self._laps_target, self._use_image)

    def record(self, episode_reward: float) -> bool:
        """
        Record the reward for the last episode.
        Automatically advances the frontier when the threshold is met.
        Returns True if the curriculum advanced.
        """
        self._sampler.record(episode_reward)
        if self._sampler.should_advance():
            return self._sampler.advance()
        return False

    # ── Fixed splits ─────────────────────────────────────────────────────────

    def train_envs(self) -> List[RaceEnvironment]:
        """One environment per TRAIN track (all 8, in order)."""
        return self._make_envs(TRAIN)

    def val_envs(self) -> List[RaceEnvironment]:
        """One environment per VAL track (4 tracks, one per difficulty tier)."""
        return self._make_envs(VAL)

    def test_envs(self) -> List[RaceEnvironment]:
        """One environment per TEST track (4 tracks, held-out)."""
        return self._make_envs(TEST)

    # ── Iteration helper ─────────────────────────────────────────────────────

    def iter_train(self) -> Iterator[RaceEnvironment]:
        """Yield environments one by one through the full TRAIN split in order."""
        for env in self.train_envs():
            yield env

    # ── Info ─────────────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._sampler.status()

    @property
    def current_level(self) -> int:
        """0-based index of the current frontier track within TRAIN."""
        return self._sampler.current_level

    @property
    def is_complete(self) -> bool:
        """True when all TRAIN tracks have been mastered."""
        return self._sampler.current_level >= len(TRAIN) - 1

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_envs(self, tracks) -> List[RaceEnvironment]:
        envs = []
        for track in tracks:
            track.build()
            envs.append(RaceEnvironment(track, self._max_steps, self._laps_target, self._use_image))
        return envs
