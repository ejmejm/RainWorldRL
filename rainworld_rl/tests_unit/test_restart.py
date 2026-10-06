"""Game-free tests for RainWorldEnv's automatic restart of a dead, hung or stuck game."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from rainworld_rl import launcher
from rainworld_rl import shared_memory as sm
from rainworld_rl.config import Config
from rainworld_rl.rainworld_env import RainWorldEnv

from fake_mapping import FakeMapping


@pytest.fixture
def game(monkeypatch):
    """A fake game behind a fake ``launcher.launch``: each launch starts a fresh ``FakeMapping``,
    except that the next ``fail`` launches raise ``LaunchError``."""
    game = SimpleNamespace(mapping = FakeMapping(), launches = 0, fail = 0)

    def fake_launch(config, build, restart, instance):
        assert build is False and restart is True
        game.launches += 1
        if game.fail:
            game.fail -= 1
            raise launcher.LaunchError("fake launch failed")
        game.mapping = FakeMapping()

    monkeypatch.setattr(launcher, "launch", fake_launch)
    return game


def make_env(game, **kwargs) -> RainWorldEnv:
    client = sm.SharedMemoryClient(8, 4, mapping_factory = lambda: game.mapping)
    client.slow_poll_interval = 0.001
    env = RainWorldEnv(8, 4, frame_timeout = 0.05, ready_timeout = 0.2, config = Config(), client = client, **kwargs)
    env.reset(options = {"wipe": False})
    return env


@pytest.mark.parametrize("fault, error", [("service_steps", "StepTimeoutError"), ("alive", "GameNotRunningError")])
def test_hung_or_dead_game_is_restarted(game, caplog, fault, error):
    env = make_env(game)
    setattr(game.mapping, fault, False)
    with caplog.at_level(logging.WARNING):
        *_, info = env.step(0)
    assert game.launches == 1 and info["game_restarted"] and info["ready"]
    assert error in caplog.text
    assert env.step(0)[4]["game_restarted"] is False


def stop_ready(mapping: FakeMapping) -> None:
    mapping.auto_ready = False
    mapping.set_mod_bit(sm.STATUS_READY, False)


def test_stuck_not_ready_is_restarted(game):
    env = make_env(game)
    env.stuck_timeout = 0.05
    stop_ready(game.mapping)
    assert not env.step(0)[4]["game_restarted"]
    time.sleep(0.06)
    assert env.step(0)[4]["game_restarted"] and game.launches == 1


def test_human_override_does_not_count_as_stuck(game):
    env = make_env(game)
    env.stuck_timeout = 0.1
    stop_ready(game.mapping)
    env.step(0)
    game.mapping.set_mod_bit(sm.STATUS_HUMAN_OVERRIDE, True)
    threading.Timer(0.2, game.mapping.set_mod_bit, (sm.STATUS_HUMAN_OVERRIDE, False)).start()
    assert not env.step(0)[4]["game_restarted"]   # blocked 0.2 s under F10
    assert game.launches == 0


def test_three_failed_relaunches_raise(game):
    env = make_env(game)
    game.fail = 3
    game.mapping.alive = False
    with pytest.raises(launcher.LaunchError, match = "3 attempts failed"):
        env.step(0)
    assert game.launches == 3


def test_auto_restart_off_raises(game):
    env = make_env(game, auto_restart = False)
    game.mapping.service_steps = False
    with pytest.raises(sm.StepTimeoutError):
        env.step(0)
    assert game.launches == 0
