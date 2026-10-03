"""
Smoke tests for the harness itself. These run WITHOUT the game:

* imports / sys.path wiring,
* e2e tests are collected but skipped without ``--e2e`` (via pytester subprocess),
* a ``FakeEnv`` honouring the RainWorldEnv API contract so the helpers are covered.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests import conftest as harness_conftest
from tests.harness import (
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    INFO_CONTRACT,
    NUM_ACTIONS,
    STATUS_CONNECTED,
    STATUS_IN_GAME,
    STATUS_MOD_ALIVE,
    STATUS_READY,
    assert_info_contract,
    get_client,
    infos,
    read_state,
    state_field,
    status_bit,
    step_n,
    wait_for,
)

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
E2E_DIR = TESTS_DIR / "e2e"
E2E_MODULES = sorted(p.name for p in E2E_DIR.glob("test_*.py"))


# ---------------------------------------------------------------------------
# FakeEnv: in-memory stand-in for RainWorldEnv (same public surface)
# ---------------------------------------------------------------------------
class FakeClient:
    """Mimics SharedMemoryClient.read_state(): a dict of PROTOCOL.md header fields."""

    def __init__(self, env: "FakeEnv"):
        self._env = env
        self._heartbeat = 0

    def read_state(self) -> dict:
        self._heartbeat += 1
        status = STATUS_MOD_ALIVE | STATUS_IN_GAME
        if self._env.connected:
            status |= STATUS_CONNECTED | STATUS_READY
        return {
            "status": status,
            "heartbeat": self._heartbeat,
            "step_counter": self._env._step_counter,
            "room_index": self._env._room_index,
        }


class FakeEnv:
    """Satisfies the RainWorldEnv contract: launch/connect/reset/step/close + info keys."""

    START_X = 100.0
    START_Y = 50.0
    START_ROOM = 7
    SPEED = 3.0

    def __init__(self, frame_width: int = 160, frame_height: int = 90, ticks_per_step: int = 1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ticks_per_step = ticks_per_step
        self.observation_space = SimpleNamespace(shape=(frame_height, frame_width, 3), dtype=np.uint8)
        self.action_space = SimpleNamespace(n=NUM_ACTIONS)
        self.connected = False
        self.launched = False
        self.closed = False
        self.resets = 0
        self.client = FakeClient(self)
        self._step_counter = 0
        self._cycle = 0
        self._x = self.START_X
        self._y = self.START_Y
        self._room_index = self.START_ROOM

    # -- lifecycle --------------------------------------------------------
    def launch(self, build: bool = True, restart: bool = True):
        self.launched = True
        self.connected = True

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False
        self.closed = True

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        wipe = True if options is None else options.get("wipe", True)
        self.connected = True
        if wipe:
            self.resets += 1
            self._cycle = 0
            self._x, self._y = self.START_X, self.START_Y
            self._room_index = self.START_ROOM
        return self._frame(), self._info()

    def step(self, action: int):
        if not self.connected:
            raise RuntimeError("not connected")
        if not 0 <= int(action) < NUM_ACTIONS:
            raise ValueError(f"action {action} outside Discrete({NUM_ACTIONS})")
        if action in (2, 9, 13, 17):
            self._x += self.SPEED * self.ticks_per_step
        elif action in (1, 8, 12, 16):
            self._x -= self.SPEED * self.ticks_per_step
        self._step_counter += 1
        return self._frame(), 0.0, False, False, self._info()

    # -- helpers ----------------------------------------------------------
    def _frame(self) -> np.ndarray:
        return np.full(self.observation_space.shape, self._step_counter % 256, dtype=np.uint8)

    def _info(self) -> dict:
        return {
            "player_dead": False,
            "karma": 1,
            "karma_cap": 5,
            "food": 0,
            "player_pos": (float(self._x), float(self._y)),
            "room_index": int(self._room_index),
            "cycle_number": int(self._cycle),
            "step_counter": int(self._step_counter),
            "in_game": True,
            "human_override": False,
            # protocol v3 game-state fields
            "ready": True,
            "food_max": 7,
            "food_to_hibernate": 4,
            "malnourished": False,
            "cycle_progress": float(self._step_counter) / 10000.0,
            "in_shelter": False,
            "cycle_survived": False,
            "rain": False,
            "dialog_open": False,
        }


@pytest.fixture
def fake():
    env = FakeEnv()
    env.connect()
    return env


# ---------------------------------------------------------------------------
# Imports / wiring
# ---------------------------------------------------------------------------
def test_repo_parent_on_sys_path():
    assert str(REPO_ROOT) in sys.path
    assert harness_conftest.REPO_PARENT == REPO_ROOT.parent
    assert callable(harness_conftest.step_n)


def test_rainworld_namespace_package_resolves():
    """`rainworld_rl` must resolve to this repo (namespace package via the parent dir)."""
    pkg = pytest.importorskip("rainworld_rl", reason="rainworld_rl package not importable")
    assert Path(pkg.__file__).resolve().parent == (REPO_ROOT / "rainworld_rl").resolve()


@pytest.mark.xfail(strict=False, reason="API surface check")
def test_new_api_surface_exists():
    """Checks the API contract the e2e tests are written against (informational until the rewrite lands)."""
    env_mod = importlib.import_module("rainworld_rl.rainworld_env")
    launcher = importlib.import_module("rainworld_rl.launcher")
    shm = importlib.import_module("rainworld_rl.shared_memory")
    config = importlib.import_module("rainworld_rl.config")

    for method in ("launch", "connect", "reset", "step", "close"):
        assert callable(getattr(env_mod.RainWorldEnv, method)), f"RainWorldEnv.{method} missing"
    for name in ("launch", "kill_game", "LaunchError"):
        assert hasattr(launcher, name), f"launcher.{name} missing"
    for name in ("SharedMemoryClient", "GameNotRunningError"):
        assert hasattr(shm, name), f"shared_memory.{name} missing"
    assert callable(getattr(shm.SharedMemoryClient, "read_state", None)), "SharedMemoryClient.read_state missing"
    for name in ("Config", "load_config"):
        assert hasattr(config, name), f"config.{name} missing"


# ---------------------------------------------------------------------------
# e2e collection behaviour (subprocess so the real conftest/ini are used)
# ---------------------------------------------------------------------------
def test_e2e_tests_are_skipped_without_flag(pytester: pytest.Pytester):
    result = pytester.runpytest_subprocess(str(E2E_DIR), "-q")
    outcomes = result.parseoutcomes()
    assert result.ret == 0, result.stdout.str()
    assert outcomes.get("skipped", 0) >= len(E2E_MODULES), outcomes
    for key in ("passed", "failed", "errors", "error", "xfailed", "xpassed"):
        assert outcomes.get(key, 0) == 0, outcomes
    result.stdout.fnmatch_lines(["*pass --e2e to run*"])


def test_e2e_tests_collect_with_flag(pytester: pytest.Pytester):
    """With --e2e everything collects (no import errors) but nothing runs under --collect-only."""
    result = pytester.runpytest_subprocess(str(E2E_DIR), "--e2e", "--collect-only", "-q")
    assert result.ret == 0, result.stdout.str()
    out = result.stdout.str()
    assert "error" not in out.lower(), out
    for module in E2E_MODULES:
        assert f"tests/e2e/{module}::" in out, f"{module} not collected:\n{out}"


# ---------------------------------------------------------------------------
# Helpers against FakeEnv
# ---------------------------------------------------------------------------
def test_fake_env_satisfies_contract(fake: FakeEnv):
    obs, info = fake.reset()
    assert obs.shape == (90, 160, 3) and obs.dtype == np.uint8
    assert_info_contract(info)
    assert set(info) == set(INFO_CONTRACT)

    obs, reward, terminated, truncated, info = fake.step(ACTION_NOOP)
    assert reward == 0.0 and terminated is False and truncated is False
    assert_info_contract(info)
    with pytest.raises(ValueError):
        fake.step(NUM_ACTIONS)


def test_step_n_defaults_to_noop(fake: FakeEnv):
    x0 = fake._info()["player_pos"][0]
    results = step_n(fake, 5)
    assert len(results) == 5
    assert all(isinstance(obs, np.ndarray) and isinstance(info, dict) for obs, info in results)
    assert infos(results, "step_counter") == [1, 2, 3, 4, 5]
    assert results[-1][1]["player_pos"][0] == x0


def test_step_n_with_action_fn(fake: FakeEnv):
    seen = []

    def policy(i: int) -> int:
        seen.append(i)
        return ACTION_RIGHT if i < 4 else ACTION_LEFT

    results = step_n(fake, 6, policy)
    assert seen == list(range(6))
    xs = [pos[0] for pos in infos(results, "player_pos")]
    assert xs[3] - FakeEnv.START_X == pytest.approx(4 * FakeEnv.SPEED)
    assert xs[-1] - xs[3] == pytest.approx(-2 * FakeEnv.SPEED)


def test_step_n_zero_steps(fake: FakeEnv):
    assert step_n(fake, 0) == []


def test_assert_info_contract_reports_missing_keys():
    with pytest.raises(AssertionError, match="missing contract keys"):
        assert_info_contract({"karma": 1})


def test_state_helpers_with_dict_state(fake: FakeEnv):
    state = read_state(fake)
    assert get_client(fake) is fake.client
    assert status_bit(state, "ready") and status_bit(state, "connected")
    assert status_bit(state, "mod_alive") and status_bit(state, "in_game")
    assert not status_bit(state, "player_dead") and not status_bit(state, "human_override")
    assert state_field(state, "heartbeat") == 1
    assert state_field(state, "missing", default=None) is None
    with pytest.raises(KeyError):
        state_field(state, "missing")
    with pytest.raises(KeyError):
        status_bit(state, "not_a_bit")


def test_state_helpers_prefer_decoded_fields():
    obj = SimpleNamespace(status=0, ready=True, heartbeat=42)
    assert status_bit(obj, "ready") is True  # decoded field wins over the raw byte
    assert status_bit(obj, "in_game") is False
    assert state_field(obj, "heartbeat") == 42

    fake = FakeEnv()  # not connected -> READY bit clear
    assert status_bit(fake.client.read_state(), "ready") is False


def test_get_client_falls_back_to_private_attr():
    env = SimpleNamespace(_client="private")
    assert get_client(env) == "private"
    with pytest.raises(AttributeError):
        get_client(SimpleNamespace())


def test_wait_for():
    counter = {"n": 0}

    def hit_on_third() -> bool:
        counter["n"] += 1
        return counter["n"] >= 3

    assert wait_for(hit_on_third, timeout=2.0, interval=0.001) is True
    assert wait_for(lambda: False, timeout=0.05, interval=0.01) is False
