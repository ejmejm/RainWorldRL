"""
``rainworld-rl``: set up and check a Rain World RL install.

    rainworld-rl download --user STEAM_USER   # Linux: install the game with SteamCMD
    rainworld-rl setup [--game-dir DIR]       # write the config, fetch the image (Linux), install the mod
    rainworld-rl doctor                       # launch the game, run random steps, report steps/s

Launching and killing the game: ``python -m rainworld_rl.launcher``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__, download, launcher
from .config import (
    CONFIG_FILENAME,
    LINUX_DATA_DIR,
    RENDERERS,
    USER_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
)


DEFAULT_IMAGE = f"docker://ghcr.io/ejmejm/rainworld-rl:v{__version__}"  # published by .github/workflows/container.yml


def _write_config(config: Config, path: Path) -> None:
    lines = [f'game_dir = "{config.game_dir.as_posix()}"']
    if sys.platform != "win32":
        if config.container is not None:
            lines.append(f'container = "{config.container.as_posix()}"')
        lines.append(f'wine_prefix_dir = "{config.wine_prefix_dir.as_posix()}"')
        lines.append(f'renderer = "{config.renderer}"')
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("\n".join(lines) + "\n", encoding = "utf-8")


def setup(args: argparse.Namespace) -> int:
    base = Config() if args.config is not None and not args.config.is_file() else load_config(args.config)
    container = None if args.no_container else (args.container or base.container or LINUX_DATA_DIR / "rainworld-rl.sif")
    config = Config(
        game_dir = args.game_dir or base.game_dir,
        launch_timeout = base.launch_timeout,
        container = container if sys.platform != "win32" else None,
        wine_prefix_dir = base.wine_prefix_dir,
        renderer = args.renderer or base.renderer,
    )
    config.validate_game_dir()

    if config.container is not None and not config.container.is_file():
        print(f"Pulling the container image {args.image} -> {config.container}")
        config.container.parent.mkdir(parents = True, exist_ok = True)
        try:
            subprocess.run(["apptainer", "pull", str(config.container), args.image], check = True)
        except FileNotFoundError:
            raise ConfigError("apptainer not found (on a cluster: `module load apptainer`), or use --no-container")

    launcher.deploy_dll(config)
    path = args.config or USER_CONFIG_PATH
    _write_config(config, path)
    print(f"Wrote {path}")

    version = download.game_version(config)
    if version != download.EXPECTED_VERSION:
        print(f"WARNING: game version is {version}, the mod was written against {download.EXPECTED_VERSION}")
    print("Next: rainworld-rl doctor")
    return 0


def doctor(args: argparse.Namespace) -> int:
    from .rainworld_env import RainWorldEnv

    config = load_config(args.config)
    print(f"config: {config.source or 'defaults'}  game_dir: {config.game_dir}")
    if sys.platform != "win32":
        print(f"container: {config.container}  renderer: {launcher.resolve_renderer(config)}")
    print(f"game version: {download.game_version(config)}")

    launcher.launch(config, instance = args.instance)
    env = RainWorldEnv(160, 90, args.ticks, config = config, instance = args.instance)
    try:
        env.connect()
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(50):
            env.step(rng.integers(0, 2, env.action_space.n))
        start = time.perf_counter()
        for _ in range(args.steps):
            obs, *_ = env.step(rng.integers(0, 2, env.action_space.n))
        elapsed = time.perf_counter() - start
    finally:
        env.close()
        launcher.kill_game(instance = args.instance)
    print(f"OK: {args.steps / elapsed:.1f} steps/s at 160x90, {args.ticks} tick(s)/step (obs mean {obs.mean():.1f})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog = "rainworld-rl", description = "Set up and check a Rain World RL install.")
    parser.add_argument("--config", type = Path, default = None, help = f"Config file (default: the usual {CONFIG_FILENAME} lookup)")
    sub = parser.add_subparsers(dest = "command", required = True)

    p = sub.add_parser("download", help = "Install the game with SteamCMD (Linux)")
    p.add_argument("--user", required = True, help = "Steam account name")

    p = sub.add_parser("setup", help = "Write the config, fetch the container image (Linux) and install the mod")
    p.add_argument("--game-dir", type = Path, default = None, help = "Rain World directory (contains RainWorld.exe)")
    p.add_argument("--container", type = Path, default = None, help = "Where the .sif image lives (Linux)")
    p.add_argument("--no-container", action = "store_true", help = "Run wine/Xvfb from the host instead (Linux)")
    p.add_argument("--image", default = DEFAULT_IMAGE, help = "Image to pull if the .sif is missing")
    p.add_argument("--renderer", choices = RENDERERS, default = None)

    p = sub.add_parser("doctor", help = "Launch the game, run random steps and report steps/s")
    p.add_argument("--steps", type = int, default = 500)
    p.add_argument("--ticks", type = int, default = 4)
    p.add_argument("--instance", type = int, default = 0)

    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            return download.main(["--user", args.user] + (["--config", str(args.config)] if args.config else []))
        return {"setup": setup, "doctor": doctor}[args.command](args)
    except (ConfigError, launcher.LaunchError) as e:
        print(f"ERROR: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
