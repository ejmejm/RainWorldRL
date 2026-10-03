"""
pytest configuration for the Rain World RL test harness.

* Puts the repo's parent directory on ``sys.path`` so ``rainworld_rl`` resolves
  as a namespace package (``from rainworld_rl.python.rainworld_env import ...``).
* Adds ``--e2e`` / ``--no-launch`` / ``--no-build``.
* Marks everything under ``tests/e2e`` with ``e2e`` and skips it unless ``--e2e``.
* Provides the session-scoped ``game`` fixture (launch or attach once) and the
  per-test ``env`` (shared, no reset) and ``fresh_env`` (``reset()`` first) fixtures.

The ``rainworld_rl.python`` modules are imported lazily inside fixtures so that
collection never fails while the client is being rewritten.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
REPO_PARENT = REPO_ROOT.parent  # makes `import rainworld_rl` work (namespace package)
E2E_DIR = TESTS_DIR / "e2e"

for _p in (str(REPO_ROOT), str(REPO_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest_plugins = ["pytester"]  # used by tests/unit/test_harness_smoke.py

from tests.harness import step_n  # noqa: E402,F401  (re-exported for convenience)

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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end test that launches/drives the real Rain World game "
        "(skipped unless --e2e is given)",
    )


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except ValueError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_e2e = config.getoption("--e2e")
    skip_e2e = pytest.mark.skip(
        reason="end-to-end test: pass --e2e to run it against the real game "
        "(add --no-launch if Rain World is already running)"
    )
    for item in items:
        if _is_under(Path(str(item.path)), E2E_DIR):
            item.add_marker(pytest.mark.e2e)
        if item.get_closest_marker("e2e") is not None and not run_e2e:
            item.add_marker(skip_e2e)


# ---------------------------------------------------------------------------
# Lazy import of the client API
# ---------------------------------------------------------------------------
def import_api() -> SimpleNamespace:
    """Import the rainworld_rl.python API; abort the session with a clear message if it is missing."""
    try:
        from rainworld_rl.python.rainworld_env import RainWorldEnv
        from rainworld_rl.python.launcher import launch, kill_game, LaunchError
        from rainworld_rl.python import shared_memory
        from rainworld_rl.python.shared_memory import SharedMemoryClient, GameNotRunningError
    except ImportError as exc:  # pragma: no cover - depends on the python/ rewrite
        pytest.exit(
            f"Cannot import the rainworld_rl.python API needed for e2e tests: {exc}\n"
            f"(sys.path contains {REPO_PARENT}; is the python/ rewrite complete?)",
            returncode=1,
        )

    class _NeverRaised(Exception):
        """Placeholder when the client does not define an optional exception type."""

    return SimpleNamespace(
        RainWorldEnv=RainWorldEnv,
        launch=launch,
        kill_game=kill_game,
        LaunchError=LaunchError,
        SharedMemoryClient=SharedMemoryClient,
        GameNotRunningError=GameNotRunningError,
        # Raised by connect()/launch() when the mod never reaches READY (optional in the contract).
        ReadyTimeoutError=getattr(shared_memory, "ReadyTimeoutError", _NeverRaised),
    )


@pytest.fixture(scope="session")
def e2e_options(request: pytest.FixtureRequest) -> SimpleNamespace:
    """The harness CLI options as a namespace."""
    return SimpleNamespace(
        e2e=request.config.getoption("--e2e"),
        no_launch=request.config.getoption("--no-launch"),
        no_build=request.config.getoption("--no-build"),
    )


@pytest.fixture(scope="session")
def api(e2e_options: SimpleNamespace) -> SimpleNamespace:
    """The lazily imported rainworld_rl.python API (RainWorldEnv, launch, kill_game, ...)."""
    if not e2e_options.e2e:
        pytest.skip("requires --e2e")
    return import_api()


# ---------------------------------------------------------------------------
# Game / env fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def game(e2e_options: SimpleNamespace, api: SimpleNamespace):
    """
    Session-wide connected ``RainWorldEnv``.

    * default: ``env.launch(build=not --no-build, restart=True)`` -> build, deploy, start, wait READY
    * ``--no-launch``: ``env.connect()`` to an already-running game

    Teardown closes the env and (unless ``--no-launch``) kills the game.
    A launch/connect failure aborts the whole session with the error message
    (``LaunchError`` carries the BepInEx log tail).
    """
    env = api.RainWorldEnv(
        frame_width=DEFAULT_FRAME_WIDTH,
        frame_height=DEFAULT_FRAME_HEIGHT,
        ticks_per_step=DEFAULT_TICKS_PER_STEP,
    )
    try:
        if e2e_options.no_launch:
            env.connect()
        else:
            env.launch(build=not e2e_options.no_build, restart=True)
    except api.GameNotRunningError as exc:
        pytest.exit(
            "--no-launch given but no running Rain World with the RainWorldRL mod was found: "
            f"{exc}",
            returncode=1,
        )
    except api.LaunchError as exc:
        pytest.exit(f"Rain World launch failed:\n{exc}", returncode=1)
    except api.ReadyTimeoutError as exc:
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
        if not e2e_options.no_launch:
            try:
                api.kill_game()
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
