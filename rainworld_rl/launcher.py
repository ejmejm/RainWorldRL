"""
Build, deploy and launch Rain World with the RainWorldRL mod.

This is Windows-only (the shared memory mapping is a named Windows file
mapping, and the game is launched via ``RainWorld.exe``). **Steam must already
be running**: the executable is started directly, and Rain World's Steam
integration will fail (or bounce to Steam) if the client is not up.

Pipeline used by ``launch()``::

    build()  ->  kill_game()  ->  deploy()  ->  start_game()  ->  wait_for_mod_alive()

The DLL is copied only after the game is killed because BepInEx keeps plugin
assemblies locked while the game runs.

CLI::

    python -m rainworld_rl.launcher [--no-build] [--no-restart] [--build-only]
                                           [--wait-ready] [--config PATH]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from .config import Config, REPO_ROOT, load_config
from .shared_memory import (
    GameNotRunningError,
    ReadyTimeoutError,
    SharedMemoryClient,
)


logger = logging.getLogger(__name__)

GAME_PROCESS_NAME = "RainWorld.exe"
BUILD_CONFIGURATION = "Debug"
BUILT_DLL_RELATIVE = Path("bin") / BUILD_CONFIGURATION / "net472" / "RainWorldRL.dll"
LOG_TAIL_LINES = 40

_SUBPROCESS_FLAGS = {}
if sys.platform == "win32":  # keep console-less tools from flashing windows
    _SUBPROCESS_FLAGS["creationflags"] = subprocess.CREATE_NO_WINDOW


class LaunchError(RuntimeError):
    """Building, starting or attaching to the game failed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tail_log(config: Config, lines: int = LOG_TAIL_LINES) -> str:
    """Return the last ``lines`` lines of ``BepInEx/LogOutput.log`` (or a note if missing)."""
    log_path = config.bepinex_log
    try:
        text = log_path.read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return f"(no BepInEx log at {log_path})"
    tail = text.splitlines()[-lines:]
    if not tail:
        return f"(BepInEx log at {log_path} is empty)"
    return "\n".join(tail)


def _launch_error(config: Config, message: str) -> LaunchError:
    return LaunchError(
        f"{message}\n\n--- last {LOG_TAIL_LINES} lines of {config.bepinex_log} ---\n{tail_log(config)}"
    )


def _run(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        list(cmd),
        cwd = str(cwd) if cwd else None,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = timeout,
        **_SUBPROCESS_FLAGS,
    )


# ---------------------------------------------------------------------------
# Build / deploy
# ---------------------------------------------------------------------------

def build(config: Optional[Config] = None, deploy: bool = True, configuration: str = BUILD_CONFIGURATION) -> Path:
    """
    Run ``dotnet build -c <configuration> -p:RainWorldDir=<game_dir>`` in the repo root.

    Args:
        deploy: Also copy the DLL into the plugins directory. Fails if the game
            is running (the DLL is locked); ``launch()`` therefore builds first
            and deploys after ``kill_game()``.

    Returns:
        Path of the built DLL (in ``bin/``), or the deployed DLL if ``deploy``.
    """
    config = config or load_config()
    cmd = [
        "dotnet", "build",
        "-c", configuration,
        f"-p:RainWorldDir={config.game_dir.as_posix()}",
        "-nologo",
    ]
    logger.info("Building mod: %s", " ".join(cmd))
    try:
        result = _run(cmd, cwd = REPO_ROOT, timeout = 600)
    except FileNotFoundError as e:
        raise LaunchError("`dotnet` was not found on PATH; install the .NET SDK") from e

    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip().splitlines()
        raise LaunchError("dotnet build failed:\n" + "\n".join(output[-LOG_TAIL_LINES:]))

    built = REPO_ROOT / "bin" / configuration / "net472" / "RainWorldRL.dll"
    if not built.is_file():
        raise LaunchError(f"Build reported success but {built} does not exist")
    logger.info("Build succeeded: %s", built)

    if deploy:
        return deploy_dll(config, built)
    return built


# Alias so ``launch(build = ...)`` can keep the natural keyword name.
build_mod = build


def deploy_dll(config: Optional[Config] = None, built_dll: Optional[Path] = None) -> Path:
    """Copy the built DLL into ``<game_dir>/BepInEx/plugins``."""
    config = config or load_config()
    built_dll = built_dll or (REPO_ROOT / BUILT_DLL_RELATIVE)
    if not built_dll.is_file():
        raise LaunchError(f"Built DLL not found at {built_dll}; run build() first")
    if not config.plugins_dir.is_dir():
        raise LaunchError(
            f"Plugins directory not found: {config.plugins_dir}. Is BepInEx installed and game_dir correct?"
        )
    try:
        shutil.copy2(built_dll, config.plugin_dll_path)
    except PermissionError as e:
        raise LaunchError(
            f"Could not overwrite {config.plugin_dll_path} (is the game running? use restart=True)"
        ) from e
    logger.info("Deployed %s", config.plugin_dll_path)
    return config.plugin_dll_path


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------

def is_game_running() -> bool:
    """True if a ``RainWorld.exe`` process exists."""
    if sys.platform != "win32":
        return False
    result = _run(["tasklist", "/FI", f"IMAGENAME eq {GAME_PROCESS_NAME}", "/NH", "/FO", "CSV"])
    return GAME_PROCESS_NAME.lower() in result.stdout.lower()


def kill_game(timeout: float = 15.0) -> bool:
    """
    Terminate every running ``RainWorld.exe`` and wait for it to exit.

    Returns:
        True if a process was killed, False if none was running.
    """
    if sys.platform != "win32":
        raise LaunchError("kill_game() is only supported on Windows")
    if not is_game_running():
        return False

    logger.info("Killing running %s", GAME_PROCESS_NAME)
    _run(["taskkill", "/F", "/T", "/IM", GAME_PROCESS_NAME])

    deadline = time.monotonic() + timeout
    while is_game_running():
        if time.monotonic() >= deadline:
            raise LaunchError(f"{GAME_PROCESS_NAME} did not exit within {timeout:.0f}s after taskkill")
        time.sleep(0.25)
    # Give the OS a moment to release file locks / the old mapping.
    time.sleep(0.5)
    return True


def start_game(config: Optional[Config] = None) -> subprocess.Popen:
    """
    Launch ``RainWorld.exe`` directly (not through Steam's URL handler).

    Steam must be running. The returned Popen handle is informational: the
    game is not a child we wait on, and closing Python does not close it.
    """
    config = config or load_config()
    config.validate_game_dir()
    logger.info("Starting %s", config.exe_path)
    try:
        return subprocess.Popen(
            [str(config.exe_path)],
            cwd = str(config.game_dir),
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            close_fds = True,
        )
    except OSError as e:
        raise LaunchError(f"Failed to start {config.exe_path}: {e}") from e


# ---------------------------------------------------------------------------
# Waiting on the mod
# ---------------------------------------------------------------------------

def wait_for_mod_alive(config: Optional[Config] = None, timeout: Optional[float] = None) -> None:
    """
    Block until the mod's heartbeat advances (``MOD_ALIVE`` + changing heartbeat).

    Does not set ``CONNECTED``. Raises ``LaunchError`` (with the BepInEx log
    tail) on timeout.
    """
    config = config or load_config()
    timeout = config.launch_timeout if timeout is None else timeout
    deadline = time.monotonic() + timeout
    client = SharedMemoryClient()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _launch_error(
                    config,
                    f"The RainWorldRL mod did not come alive within {timeout:.0f}s of starting the game.",
                )
            try:
                # alive_grace = 0: this loop does its own retrying and process check.
                client.wait_for_alive(timeout = min(1.0, max(0.1, remaining)), alive_grace = 0.0)
                logger.info("Mod is alive")
                return
            except GameNotRunningError:
                if not is_game_running():
                    # Give it a moment: the process may not have been spawned yet.
                    time.sleep(0.5)
                    if not is_game_running():
                        raise _launch_error(config, f"{GAME_PROCESS_NAME} is not running (it exited or failed to start).")
                time.sleep(0.5)
    finally:
        client.disconnect()


def wait_for_ready(config: Optional[Config] = None, timeout: Optional[float] = None, ready_timeout: float = 60.0) -> None:
    """
    Wait for the mod to be alive, then connect, wait for ``READY`` and disconnect.

    Mostly useful from the CLI to confirm a launch worked end-to-end. Library
    users should prefer ``RainWorldEnv.connect()``, which keeps the connection.
    """
    config = config or load_config()
    wait_for_mod_alive(config, timeout)
    client = SharedMemoryClient()
    try:
        client.connect(wait_ready = True, ready_timeout = ready_timeout)
        logger.info("Mod is READY")
    except (GameNotRunningError, ReadyTimeoutError) as e:
        raise _launch_error(config, str(e)) from e
    finally:
        client.disconnect()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def launch(
    config: Optional[Config] = None,
    build: bool = True,  # noqa: A002 - mirrors the CLI flag name
    restart: bool = True,
    timeout: Optional[float] = None,
) -> Optional[subprocess.Popen]:
    """
    Build (optional), (re)start the game and wait for the mod's heartbeat.

    Args:
        build: Run ``dotnet build`` and deploy the DLL.
        restart: Kill a running game first. If False and the game is already
            running, nothing is started; we just wait for the mod to be alive.
        timeout: Seconds to wait for the mod heartbeat (default ``config.launch_timeout``).

    Returns:
        The Popen handle if a new process was started, else None.

    Raises:
        LaunchError: with the tail of ``BepInEx/LogOutput.log`` in the message.
    """
    config = config or load_config()
    config.validate_game_dir()

    built_dll = None
    if build:
        built_dll = build_mod(config, deploy = False)

    process = None
    running = is_game_running()
    if restart or not running:
        if running:
            kill_game()
        if build:
            deploy_dll(config, built_dll)
        process = start_game(config)
    else:
        if build:
            logger.warning("Game is running and restart=False; deploying the DLL may fail if it is locked")
            deploy_dll(config, built_dll)
        logger.info("Game already running; attaching without restart")

    try:
        wait_for_mod_alive(config, timeout)
    except LaunchError:
        if process is not None and process.poll() is not None:
            raise _launch_error(config, f"{GAME_PROCESS_NAME} exited with code {process.returncode} during startup.")
        raise
    return process


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog = "python -m rainworld_rl.launcher",
        description = "Build, deploy and launch Rain World with the RainWorldRL mod (Steam must be running).",
    )
    parser.add_argument("--no-build", action = "store_true", help = "Skip dotnet build / DLL deploy")
    parser.add_argument("--no-restart", action = "store_true", help = "Do not kill an already-running game")
    parser.add_argument("--build-only", action = "store_true", help = "Build and deploy, then exit")
    parser.add_argument("--wait-ready", action = "store_true", help = "After the mod is alive, also wait for READY")
    parser.add_argument("--kill", action = "store_true", help = "Kill a running game and exit")
    parser.add_argument("--config", type = Path, default = None, help = "Path to a rainworld_rl.toml")
    parser.add_argument("--timeout", type = float, default = None, help = "Override launch_timeout (seconds)")
    parser.add_argument("-v", "--verbose", action = "store_true", help = "Debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level = logging.DEBUG if args.verbose else logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        if config.source:
            logger.info("Using config %s (game_dir=%s)", config.source, config.game_dir)
        else:
            logger.info("Using default config (game_dir=%s)", config.game_dir)

        if args.kill:
            print("Killed running game" if kill_game() else "No game running")
            return 0

        if args.build_only:
            if is_game_running():
                logger.warning("Game is running; the deployed DLL will not be reloaded until it restarts")
            path = build(config, deploy = True)
            print(f"Deployed {path}")
            return 0

        launch(config, build = not args.no_build, restart = not args.no_restart, timeout = args.timeout)
        if args.wait_ready:
            wait_for_ready(config, timeout = 1.0)
        print("Rain World is running with the RainWorldRL mod alive.")
        return 0
    except LaunchError as e:
        print(f"ERROR: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
