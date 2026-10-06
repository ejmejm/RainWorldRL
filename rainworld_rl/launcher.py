"""
Build, deploy and launch Rain World with the RainWorldRL mod.

On Windows ``RainWorld.exe`` is started directly and one instance runs per
machine. On Linux each instance runs under Wine on its own Xvfb display,
optionally inside the Apptainer image ``config.container``, with its own Wine
prefix and a /dev/shm file as the shared memory (``default_shm_path``). The
Steam client is not needed.

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
import hashlib
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Config, REPO_ROOT, load_config
from .shared_memory import (
    GameNotRunningError,
    ReadyTimeoutError,
    SharedMemoryClient,
    default_shm_path,
)


logger = logging.getLogger(__name__)

GAME_PROCESS_NAME = "RainWorld.exe"
BUILD_CONFIGURATION = "Debug"
BUILT_DLL_RELATIVE = Path("bin") / BUILD_CONFIGURATION / "net472" / "RainWorldRL.dll"
# The prebuilt mod shipped with the package, so a pip install needs no .NET SDK. build() refreshes it
# whenever the mod sources change; SOURCE_HASH records which sources it was built from.
PACKAGED_DLL = Path(__file__).resolve().parent / "mod" / "RainWorldRL.dll"
PACKAGED_HASH = PACKAGED_DLL.with_name("SOURCE_HASH")
LOG_TAIL_LINES = 40

_SUBPROCESS_FLAGS = {}
if sys.platform == "win32":  # keep console-less tools from flashing windows
    _SUBPROCESS_FLAGS["creationflags"] = subprocess.CREATE_NO_WINDOW


class LaunchError(RuntimeError):
    """Building, starting or attaching to the game failed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def game_log_path(config: Config, instance: int = 0) -> Path:
    """
    The log to show on failures: ``BepInEx/LogOutput.log`` on Windows; on Linux
    the instance's console output (it includes the BepInEx log lines), since
    instances share the game directory.
    """
    if sys.platform == "win32":
        return config.bepinex_log
    return config.wine_prefix_dir / f"{instance}.log"


def tail_log(config: Config, lines: int = LOG_TAIL_LINES, instance: int = 0) -> str:
    """Return the last ``lines`` lines of ``game_log_path()`` (or a note if missing)."""
    log_path = game_log_path(config, instance)
    try:
        text = log_path.read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return f"(no log at {log_path})"
    tail = text.splitlines()[-lines:]
    if not tail:
        return f"(log at {log_path} is empty)"
    return "\n".join(tail)


def _launch_error(config: Config, message: str, instance: int = 0) -> LaunchError:
    return LaunchError(
        f"{message}\n\n--- last {LOG_TAIL_LINES} lines of {game_log_path(config, instance)} ---\n"
        f"{tail_log(config, instance = instance)}"
    )


def _run(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        list(cmd),
        cwd = str(cwd) if cwd else None,
        env = env,
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

def mod_source_hash() -> Optional[str]:
    """sha256 of the mod sources (line endings normalised), or None outside a source checkout."""
    project = REPO_ROOT / "RainWorldRL.csproj"
    if not project.is_file():
        return None
    h = hashlib.sha256()
    for f in sorted(REPO_ROOT.glob("*.cs")) + [project]:
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _can_build() -> bool:
    return mod_source_hash() is not None and shutil.which("dotnet") is not None


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

    source_hash = mod_source_hash()
    if not PACKAGED_HASH.is_file() or PACKAGED_HASH.read_text().strip() != source_hash:
        PACKAGED_DLL.parent.mkdir(exist_ok = True)
        shutil.copy2(built, PACKAGED_DLL)
        PACKAGED_HASH.write_text(source_hash + "\n")
        logger.info("Updated the packaged DLL %s (commit it with the source change)", PACKAGED_DLL)

    if deploy:
        return deploy_dll(config, built)
    return built


# Alias so ``launch(build = ...)`` can keep the natural keyword name.
build_mod = build


def deploy_dll(config: Optional[Config] = None, built_dll: Optional[Path] = None) -> Path:
    """Copy ``built_dll`` (default: the packaged prebuilt DLL) into ``<game_dir>/BepInEx/plugins``."""
    config = config or load_config()
    built_dll = built_dll or PACKAGED_DLL
    if not built_dll.is_file():
        raise LaunchError(f"Mod DLL not found at {built_dll}; run build() first")
    if not config.plugins_dir.is_dir():
        raise LaunchError(
            f"Plugins directory not found: {config.plugins_dir}. Is BepInEx installed and game_dir correct?"
        )
    try:
        # Copy then rename: on Linux, game instances that are still running keep the old file.
        tmp = config.plugin_dll_path.with_suffix(".dll.tmp")
        shutil.copy2(built_dll, tmp)
        os.replace(tmp, config.plugin_dll_path)
    except PermissionError as e:
        raise LaunchError(
            f"Could not overwrite {config.plugin_dll_path} (is the game running? use restart=True)"
        ) from e
    logger.info("Deployed %s", config.plugin_dll_path)
    return config.plugin_dll_path


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------

def is_game_running(instance: int = 0) -> bool:
    """True if the game is running (Windows: any ``RainWorld.exe``; Linux: this instance)."""
    if sys.platform != "win32":
        return _linux_game_pgid(instance) is not None
    result = _run(["tasklist", "/FI", f"IMAGENAME eq {GAME_PROCESS_NAME}", "/NH", "/FO", "CSV"])
    return GAME_PROCESS_NAME.lower() in result.stdout.lower()


def kill_game(timeout: float = 15.0, instance: int = 0) -> bool:
    """
    Terminate the game and wait for it to exit (Windows: every ``RainWorld.exe``;
    Linux: this instance's process group).

    Returns:
        True if a process was killed, False if none was running.
    """
    if sys.platform != "win32":
        return _kill_game_linux(timeout, instance)
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


def start_game(config: Optional[Config] = None, instance: int = 0) -> subprocess.Popen:
    """
    Launch ``RainWorld.exe`` directly (Windows) or under Wine (Linux).

    The returned Popen handle is informational: the game is not a child we
    wait on, and closing Python does not close it.
    """
    config = config or load_config()
    config.validate_game_dir()
    if sys.platform != "win32":
        return _start_game_linux(config, instance)
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
# Linux: Wine + Xvfb, optionally inside Apptainer
# ---------------------------------------------------------------------------

# winhttp: BepInEx's doorstop proxy DLL must win over Wine's builtin. mscoree/mshtml:
# skip the wine-mono / wine-gecko installers (Unity brings its own Mono).
WINE_DLL_OVERRIDES = "winhttp=n,b;mscoree=;mshtml="
XVFB_SCREEN = "1366x768x24"
GAME_CPUS = 6  # cores each game instance is pinned to (see _game_cpus)

# Create a Wine prefix. If the image ships DXVK (/opt/dxvk), install it: D3D11 -> Vulkan on lavapipe
# renders ~1.5x faster than Wine's own D3D11 -> OpenGL on llvmpipe. DXVK crashes (division by zero in
# dxgi) on the display modes Wine reads from Xvfb via XRandR, so XRandR/XVidMode are turned off.
_INIT_PREFIX = r"""
wineboot -i || exit 1
if [ -d /opt/dxvk ]; then
  for dll in d3d11 dxgi d3d10core; do
    cp --remove-destination "/opt/dxvk/x64/$dll.dll" "$WINEPREFIX/drive_c/windows/system32/" || exit 1
    wine reg add 'HKCU\Software\Wine\DllOverrides' /v "$dll" /d native /f || exit 1
  done
  wine reg add 'HKCU\Software\Wine\X11 Driver' /v UseXRandR /d N /f || exit 1
  wine reg add 'HKCU\Software\Wine\X11 Driver' /v UseXVidMode /d N /f || exit 1
fi
wineserver -w
"""

# The OpenGL renderers (wsl, virtualgl) need Wine's own D3D11 -> OpenGL, not the DXVK in the prefix.
WINED3D_OVERRIDES = WINE_DLL_OVERRIDES + ";d3d11,dxgi,d3d10core=b"

WSL_LIB_DIR = "/usr/lib/wsl"  # WSL2's GPU user-space libraries (libd3d12, libdxcore)

# Start a private Xvfb (-displayfd picks a free display without races) and run the
# command line (``[vglrun ...] wine RainWorld.exe``) on it.
_RUN_ON_XVFB = r"""
coproc XVFB { exec Xvfb -displayfd 1 -nolisten tcp -screen 0 "$RAINWORLD_RL_SCREEN"; }
read -r -u "${XVFB[0]}" display || { echo "Xvfb failed to start" >&2; exit 1; }
trap 'kill $XVFB_PID 2>/dev/null' EXIT
DISPLAY=":$display" "$@"
"""


def resolve_renderer(config: Config) -> str:
    """
    ``config.renderer`` with ``auto`` resolved: ``virtualgl`` when an NVIDIA GPU
    is visible, else ``wsl`` on WSL2, else ``cpu``. AMD/Intel GPUs stay on
    ``cpu`` unless ``virtualgl`` is set explicitly (untested there).
    """
    if config.renderer != "auto":
        return config.renderer
    if Path("/dev/nvidiactl").exists():
        return "virtualgl"
    if Path("/dev/dxg").exists():
        return "wsl"
    return "cpu"


def _in_container(config: Config, cmd: List[str]) -> List[str]:
    """Wrap ``cmd`` in ``apptainer exec`` when ``config.container`` is set."""
    if config.container is None:
        return cmd
    args = [a for d in (config.game_dir, config.wine_prefix_dir) for a in ("--bind", str(d))]
    renderer = resolve_renderer(config)
    if renderer == "wsl":
        args += ["--bind", WSL_LIB_DIR, "--env", f"LD_LIBRARY_PATH={WSL_LIB_DIR}/lib"]
    if renderer == "virtualgl" and Path("/dev/nvidiactl").exists():
        args.append("--nv")  # bind the host's NVIDIA driver (its EGL library) into the container
    return ["apptainer", "exec", *args, "--pwd", str(config.game_dir), str(config.container), *cmd]


def _renderer(config: Config) -> Tuple[Dict[str, str], List[str]]:
    """
    Environment and command prefix for the renderer (``resolve_renderer``):

    * ``cpu``: on the CPU: DXVK on Mesa lavapipe when the prefix has DXVK,
      else Wine's OpenGL on llvmpipe. Works everywhere.
    * ``wsl``: WSL2's GPU through Mesa's d3d12 driver (needs /dev/dxg).
    * ``virtualgl``: the first EGL device (a GPU) through VirtualGL, which
      redirects the game's GLX rendering off the virtual X display. NVIDIA:
      proprietary driver, bound in with ``apptainer --nv``; AMD/Intel: Mesa
      on /dev/dri.
    """
    renderer = resolve_renderer(config)
    if renderer == "wsl":
        if not Path("/dev/dxg").exists():
            raise LaunchError("renderer = 'wsl' needs WSL2's GPU device /dev/dxg")
        return {"GALLIUM_DRIVER": "d3d12", "WINEDLLOVERRIDES": WINED3D_OVERRIDES}, []
    if renderer == "virtualgl":
        # VGL_READBACK=none: don't copy every frame back to the CPU to show it on Xvfb. Nothing
        # looks at the display (the mod captures on the GPU), and the readback halves steps/s.
        return {"WINEDLLOVERRIDES": WINED3D_OVERRIDES, "VGL_READBACK": "none"}, ["vglrun", "-d", "egl"]
    return {}, []


def _wine_env(prefix: Path) -> Dict[str, str]:
    env = {**os.environ, "WINEPREFIX": str(prefix), "WINEDEBUG": "-all", "WINEDLLOVERRIDES": WINE_DLL_OVERRIDES}
    env.pop("TMPDIR", None)  # Wine 9.0 aborts (free(): invalid pointer) when TMPDIR is set, as on Slurm nodes
    return env


def _ensure_prefix(config: Config, instance: int) -> Path:
    """
    Return this instance's Wine prefix. The first call runs ``wineboot`` once
    into ``<wine_prefix_dir>/base``; each instance gets a hard-linked copy of it
    (near-zero disk, and its own wineserver and AppData).
    """
    import fcntl

    prefix_dir = config.wine_prefix_dir
    prefix = prefix_dir / str(instance)
    if prefix.is_dir():
        return prefix
    prefix_dir.mkdir(parents = True, exist_ok = True)
    base = prefix_dir / "base"
    with open(prefix_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # concurrent launches: one creates base, the rest wait
        if not base.is_dir():
            logger.info("Creating Wine prefix %s (one-time, ~30 s)", base)
            result = _run(_in_container(config, ["sh", "-c", _INIT_PREFIX]), env = _wine_env(base), timeout = 600)
            if result.returncode != 0:
                shutil.rmtree(base, ignore_errors = True)
                raise LaunchError(f"wineboot failed:\n{result.stdout}\n{result.stderr}")
        if not prefix.is_dir():
            tmp = prefix_dir / f".{instance}.tmp"
            shutil.rmtree(tmp, ignore_errors = True)
            subprocess.run(["cp", "-al", str(base), str(tmp)], check = True)
            tmp.rename(prefix)
    return prefix


def _game_cpus(instance: int) -> Optional[List[int]]:
    """
    CPUs to pin this instance's game to: a block of ``GAME_CPUS`` of the CPUs this process may
    use, a different block per instance (wrapping around when there are too few). None when there
    are no more than ``GAME_CPUS`` to choose from.

    The game hands each step through several threads (main thread, Unity's render worker, Wine's
    D3D thread, the GPU driver); kept on a few cores this ran 1.1-1.4x faster than spread over 16
    (Vulcan L40S node and WSL, CPU and GPU renderers). The Python side needs no pinning.
    """
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) <= GAME_CPUS:
        return None
    return [allowed[(instance * GAME_CPUS + i) % len(allowed)] for i in range(GAME_CPUS)]


def _linux_pidfile(instance: int) -> Path:
    return Path(default_shm_path(instance) + ".pid")


def _linux_game_pgid(instance: int) -> Optional[int]:
    """Process group of this instance's game (from its pidfile), or None if it is not running."""
    try:
        pgid = int(_linux_pidfile(instance).read_text())
    except (OSError, ValueError):
        return None
    try:
        os.waitpid(pgid, os.WNOHANG)  # reap the group leader if it is our exited child
    except ChildProcessError:
        pass
    try:
        os.killpg(pgid, 0)
    except OSError:
        return None
    return pgid


def _start_game_linux(config: Config, instance: int) -> subprocess.Popen:
    prefix = _ensure_prefix(config, instance)
    shm = default_shm_path(instance)
    Path(shm).unlink(missing_ok = True)  # fresh mapping: no stale header from an earlier run
    env = _wine_env(prefix)
    env["RAINWORLD_RL_SCREEN"] = XVFB_SCREEN
    env["RAINWORLD_RL_SHM"] = "Z:" + shm.replace("/", "\\")  # Wine maps / to drive Z:
    env["RAINWORLD_RL_SAVE_DIR"] = "C:\\rainworld_rl_save"  # inside this instance's prefix
    renderer_env, wrapper = _renderer(config)
    env.update(renderer_env)
    env.setdefault("DXVK_LOG_PATH", "none")  # DXVK would write logs into the shared game dir; stderr still has them
    # Make Unity's GC (Boehm) collect ~3x less often. Each collection suspends every thread, and under
    # Wine each suspend/resume costs wineserver a ptrace; on hosts that audit ptrace (e.g. Vulcan) a full
    # audit backlog then freezes the game for 10-60 s. Costs a 2 GB initial heap.
    env.setdefault("GC_INITIAL_HEAP_SIZE", str(2 * 1024 ** 3))
    env.setdefault("GC_FREE_SPACE_DIVISOR", "1")
    # Have the mod keep one core spinning while the agent plays: keeps clocks up so the per-step hand-offs
    # between the game's threads stay fast (Vulcan: ~1.75x). On WSL2 the host manages clocks and it costs ~10%.
    if "microsoft" not in platform.release().lower():
        env.setdefault("RAINWORLD_RL_CORE_WARMER", "1")
    cmd = _in_container(config, ["bash", "-c", _RUN_ON_XVFB, "run-on-xvfb", *wrapper, "wine", str(config.exe_path)])
    cpus = _game_cpus(instance)
    if cpus and shutil.which("taskset"):
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), *cmd]
    log_path = prefix.with_suffix(".log")
    logger.info(
        "Starting instance %d (renderer %s, CPUs %s): %s (output in %s)",
        instance, resolve_renderer(config), cpus or "all", config.exe_path, log_path,
    )
    with open(log_path, "wb") as log:
        try:
            process = subprocess.Popen(
                cmd,
                cwd = str(config.game_dir),
                env = env,
                stdin = subprocess.DEVNULL,
                stdout = log,
                stderr = subprocess.STDOUT,
                start_new_session = True,  # own process group, so kill_game can take down Xvfb + wine together
            )
        except OSError as e:
            raise LaunchError(f"Failed to start {cmd[0]}: {e}") from e
    _linux_pidfile(instance).write_text(str(process.pid))
    return process


def _kill_game_linux(timeout: float, instance: int) -> bool:
    pgid = _linux_game_pgid(instance)
    if pgid is None:
        return False
    logger.info("Killing game instance %d (process group %d)", instance, pgid)
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _linux_game_pgid(instance) is not None:
        if time.monotonic() >= deadline:
            os.killpg(pgid, signal.SIGKILL)
            break
        time.sleep(0.25)
    _linux_pidfile(instance).unlink(missing_ok = True)
    Path(default_shm_path(instance)).unlink(missing_ok = True)  # /dev/shm counts against job memory
    return True


# ---------------------------------------------------------------------------
# Waiting on the mod
# ---------------------------------------------------------------------------

def wait_for_mod_alive(config: Optional[Config] = None, timeout: Optional[float] = None, instance: int = 0) -> None:
    """
    Block until the mod's heartbeat advances (``MOD_ALIVE`` + changing heartbeat).

    Does not set ``CONNECTED``. Raises ``LaunchError`` (with the BepInEx log
    tail) on timeout.
    """
    config = config or load_config()
    timeout = config.launch_timeout if timeout is None else timeout
    deadline = time.monotonic() + timeout
    client = SharedMemoryClient(shm_path = default_shm_path(instance))
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _launch_error(
                    config,
                    f"The RainWorldRL mod did not come alive within {timeout:.0f}s of starting the game.",
                    instance,
                )
            try:
                # alive_grace = 0: this loop does its own retrying and process check.
                client.wait_for_alive(timeout = min(1.0, max(0.1, remaining)), alive_grace = 0.0)
                logger.info("Mod is alive")
                return
            except GameNotRunningError:
                if not is_game_running(instance):
                    # Give it a moment: the process may not have been spawned yet.
                    time.sleep(0.5)
                    if not is_game_running(instance):
                        raise _launch_error(config, f"{GAME_PROCESS_NAME} is not running (it exited or failed to start).", instance)
                time.sleep(0.5)
    finally:
        client.disconnect()


def wait_for_ready(
    config: Optional[Config] = None,
    timeout: Optional[float] = None,
    ready_timeout: float = 60.0,
    instance: int = 0,
) -> None:
    """
    Wait for the mod to be alive, then connect, wait for ``READY`` and disconnect.

    Mostly useful from the CLI to confirm a launch worked end-to-end. Library
    users should prefer ``RainWorldEnv.connect()``, which keeps the connection.
    """
    config = config or load_config()
    wait_for_mod_alive(config, timeout, instance)
    client = SharedMemoryClient(shm_path = default_shm_path(instance))
    try:
        client.connect(wait_ready = True, ready_timeout = ready_timeout)
        logger.info("Mod is READY")
    except (GameNotRunningError, ReadyTimeoutError) as e:
        raise _launch_error(config, str(e), instance) from e
    finally:
        client.disconnect()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def launch(
    config: Optional[Config] = None,
    build: Optional[bool] = None,  # noqa: A002 - mirrors the CLI flag name
    restart: bool = True,
    timeout: Optional[float] = None,
    instance: int = 0,
) -> Optional[subprocess.Popen]:
    """
    Build (optional), (re)start the game and wait for the mod's heartbeat.

    Args:
        build: True: ``dotnet build`` and deploy the DLL. None (default): build
            in a source checkout with ``dotnet`` on PATH, otherwise deploy the
            prebuilt DLL shipped with the package. False: leave the deployed
            DLL alone.
        restart: Kill a running game first. If False and the game is already
            running, nothing is started; we just wait for the mod to be alive.
        timeout: Seconds to wait for the mod heartbeat (default ``config.launch_timeout``).
        instance: Which game instance (Linux only; each has its own display,
            Wine prefix and shared memory).

    Returns:
        The Popen handle if a new process was started, else None.

    Raises:
        LaunchError: with the tail of ``BepInEx/LogOutput.log`` in the message.
    """
    config = config or load_config()
    config.validate_game_dir()

    built_dll = None
    if build is None and not _can_build():
        built_dll = PACKAGED_DLL
    elif build is not False:
        built_dll = build_mod(config, deploy = False)

    process = None
    running = is_game_running(instance)
    if restart or not running:
        if running:
            kill_game(instance = instance)
        if built_dll is not None:
            deploy_dll(config, built_dll)
        process = start_game(config, instance)
    else:
        if built_dll is not None:
            logger.warning("Game is running and restart=False; deploying the DLL may fail if it is locked")
            deploy_dll(config, built_dll)
        logger.info("Game already running; attaching without restart")

    try:
        wait_for_mod_alive(config, timeout, instance)
    except LaunchError:
        if process is not None and process.poll() is not None:
            raise _launch_error(config, f"{GAME_PROCESS_NAME} exited with code {process.returncode} during startup.", instance)
        raise
    return process


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog = "python -m rainworld_rl.launcher",
        description = "Build, deploy and launch Rain World with the RainWorldRL mod.",
    )
    parser.add_argument("--no-build", action = "store_true", help = "Skip the build and leave the deployed DLL alone (default: build if possible, else deploy the packaged DLL)")
    parser.add_argument("--no-restart", action = "store_true", help = "Do not kill an already-running game")
    parser.add_argument("--build-only", action = "store_true", help = "Build and deploy, then exit")
    parser.add_argument("--wait-ready", action = "store_true", help = "After the mod is alive, also wait for READY")
    parser.add_argument("--kill", action = "store_true", help = "Kill a running game and exit")
    parser.add_argument("--config", type = Path, default = None, help = "Path to a rainworld_rl.toml")
    parser.add_argument("--timeout", type = float, default = None, help = "Override launch_timeout (seconds)")
    parser.add_argument("--instance", type = int, default = 0, help = "Game instance (Linux only)")
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
            print("Killed running game" if kill_game(instance = args.instance) else "No game running")
            return 0

        if args.build_only:
            if is_game_running():
                logger.warning("Game is running; the deployed DLL will not be reloaded until it restarts")
            path = build(config, deploy = True)
            print(f"Deployed {path}")
            return 0

        launch(config, build = False if args.no_build else None, restart = not args.no_restart, timeout = args.timeout, instance = args.instance)
        if args.wait_ready:
            wait_for_ready(config, timeout = 1.0, instance = args.instance)
        print("Rain World is running with the RainWorldRL mod alive.")
        return 0
    except LaunchError as e:
        print(f"ERROR: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
