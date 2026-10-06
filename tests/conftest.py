"""
pytest configuration for the Rain World RL tests.

* Adds ``--e2e`` / ``--no-launch`` / ``--no-build``.
* Marks everything under ``tests/e2e`` with ``e2e`` and skips it unless ``--e2e``.
* Provides the session-scoped ``game`` fixture (launch or attach once) and the
  per-test ``env`` (shared, no reset) and ``fresh_env`` (``reset()`` first) fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rainworld_rl import GameNotRunningError, RainWorldEnv, ReadyTimeoutError
from rainworld_rl.launcher import LaunchError, kill_game

E2E_DIR = Path(__file__).resolve().parent / "e2e"

# Shared env configuration. Keep frames small so steps are fast.
DEFAULT_FRAME_WIDTH = 160
DEFAULT_FRAME_HEIGHT = 90
DEFAULT_TICKS_PER_STEP = 1


# ---------------------------------------------------------------------------
# CLI options / markers
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rainworld", "Rain World end-to-end harness")
    group.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests against the real game (launches Rain World unless --no-launch).",
    )
    group.addoption(
        "--no-launch",
        action="store_true",
        default=False,
        help="Assume Rain World (with the RainWorldRL mod) is already running: connect instead of "
        "launching, and leave it running afterwards.",
    )
    group.addoption(
        "--no-build",
        action="store_true",
        default=False,
        help="Launch without rebuilding/deploying the C# mod.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_e2e = config.getoption("--e2e")
    skip_e2e = pytest.mark.skip(
        reason="end-to-end test: pass --e2e to run it against the real game "
        "(add --no-launch if Rain World is already running)"
    )
    for item in items:
        if Path(item.path).resolve().is_relative_to(E2E_DIR):
            item.add_marker(pytest.mark.e2e)
        if item.get_closest_marker("e2e") is not None and not run_e2e:
            item.add_marker(skip_e2e)


# ---------------------------------------------------------------------------
# Game / env fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def game(pytestconfig: pytest.Config):
    """
    Session-wide connected ``RainWorldEnv``.

    * default: ``env.launch(restart=True)`` -> build if possible (else deploy the packaged DLL), start, wait READY
    * ``--no-launch``: ``env.connect()`` to an already-running game

    Teardown closes the env and (unless ``--no-launch``) kills the game.
    A launch/connect failure aborts the whole session with the error message
    (``LaunchError`` carries the BepInEx log tail).
    """
    no_launch = pytestconfig.getoption("--no-launch")
    env = RainWorldEnv(
        frame_width=DEFAULT_FRAME_WIDTH,
        frame_height=DEFAULT_FRAME_HEIGHT,
        ticks_per_step=DEFAULT_TICKS_PER_STEP,
    )
    try:
        if no_launch:
            env.connect()
        else:
            env.launch(build=False if pytestconfig.getoption("--no-build") else None, restart=True)
    except GameNotRunningError as exc:
        pytest.exit(
            "--no-launch given but no running Rain World with the RainWorldRL mod was found: "
            f"{exc}",
            returncode=1,
        )
    except LaunchError as exc:
        pytest.exit(f"Rain World launch failed:\n{exc}", returncode=1)
    except ReadyTimeoutError as exc:
        pytest.exit(
            f"Rain World started but the mod never reported READY: {exc}\n"
            "(is the game stuck in a menu? is HUMAN_OVERRIDE (F10) active?)",
            returncode=1,
        )

    yield env

    try:
        env.close()
    except Exception as exc:  # pragma: no cover - best effort teardown
        print(f"[harness] env.close() raised during teardown: {exc!r}", file=sys.stderr)
    finally:
        if not no_launch:
            try:
                kill_game()
            except Exception as exc:  # pragma: no cover - best effort teardown
                print(f"[harness] kill_game() raised during teardown: {exc!r}", file=sys.stderr)


@pytest.fixture
def env(game):
    """The shared connected env, WITHOUT a reset (fast; state carries over between tests)."""
    return game


@pytest.fixture
def fresh_env(game):
    """The shared env after ``reset()`` (wipes the RL save and reloads: ~10-30 s)."""
    game.reset()
    return game


@pytest.fixture
def reattach_shared_env(game):
    """
    Use in tests that create a SECOND ``RainWorldEnv`` against the running game.

    The mapping is shared, so the second env's ``close()`` clears the CONNECTED bit
    and its frame dimensions overwrite the header. After the test this fixture
    re-attaches the shared env (re-writes its dims, re-raises CONNECTED, waits READY).
    """
    yield game
    # connect() is a no-op while the env believes it is connected, so drop the
    # connection first; close() only clears CONNECTED, it never kills the game.
    game.close()
    game.connect()
    # Sanity: the shared env must be serviced at its own dimensions again.
    obs, _r, _t, _tr, _info = game.step(0)
    assert obs.shape == tuple(game.observation_space.shape), (
        f"shared env not restored: got frame {obs.shape}, expected {game.observation_space.shape}"
    )
