"""Episode registry: maps each episode ID to the tool backend serving that episode."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from ..app.backend import ToolBackend
from ..app.mock_app import MockApp
from ..data.scenario import Scenario


class EpisodeError(Exception):
    """A tool call that can't be matched to a valid episode (missing or bad IDs)."""


@dataclass
class Episode:
    episode_id: str
    scenario_id: str
    fault_seed: int
    backend: ToolBackend


class EpisodeRegistry:
    """Creates one episode per episode ID on first use and returns it on later calls."""

    def __init__(self, scenarios: list[Scenario]):
        self.scenarios: dict[str, Scenario] = {}
        for scenario in scenarios:
            if scenario.scenario_id in self.scenarios:
                raise ValueError(f"duplicate scenario_id {scenario.scenario_id!r}")
            self.scenarios[scenario.scenario_id] = scenario

        self.apps: dict[str, MockApp] = {}  # one read-only MockApp per scenario, shared by its episodes
        self.episodes: dict[str, Episode] = {}
        self.lock = threading.Lock()

    def get_or_create(self, episode_id: str | None, scenario_id: str | None, fault_seed: int) -> ToolBackend:
        """Return the backend for this episode, creating the episode on its first call."""
        if not episode_id:
            raise EpisodeError("missing episode ID (x-episode-id)")
        if not scenario_id:
            raise EpisodeError("missing scenario ID (x-scenario-id)")

        with self.lock:
            episode = self.episodes.get(episode_id)
            if episode is not None:
                if episode.scenario_id != scenario_id or episode.fault_seed != fault_seed:
                    raise EpisodeError(
                        f"episode {episode_id!r} was started with scenario {episode.scenario_id!r} "
                        f"and seed {episode.fault_seed}; got {scenario_id!r} and seed {fault_seed}"
                    )
                return episode.backend

            scenario = self.scenarios.get(scenario_id)
            if scenario is None:
                raise EpisodeError(f"unknown scenario ID {scenario_id!r}")

            backend = self._build_backend(scenario, episode_id, fault_seed)
            self.episodes[episode_id] = Episode(episode_id, scenario_id, fault_seed, backend)
            return backend

    def _build_backend(self, scenario: Scenario, episode_id: str, fault_seed: int) -> ToolBackend:
        """Build the tool stack for a new episode: the scenario's shared MockApp."""
        app = self.apps.get(scenario.scenario_id)
        if app is None:
            app = MockApp(scenario)
            self.apps[scenario.scenario_id] = app
        return app
