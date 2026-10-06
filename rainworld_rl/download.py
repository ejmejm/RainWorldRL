"""
Download Rain World into ``game_dir`` with SteamCMD (Linux).

Uses your own Steam account, which must own the game. SteamCMD asks for the
password and Steam Guard code the first time and remembers the login after
that. It runs from ``config.container`` when set (the image ships SteamCMD),
otherwise from PATH. The Steam client is not needed to run the game.

CLI::

    python -m rainworld_rl.download --user STEAM_USER [--config PATH]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config, load_config


APP_ID = 312520
# The version the mod's hooks were written against (RainWorld_Data/StreamingAssets/GameVersion.txt).
EXPECTED_VERSION = "v1.11.8"


def game_version(config: Config) -> Optional[str]:
    """The installed game's version string, or None if it cannot be read."""
    try:
        return (config.game_dir / "RainWorld_Data" / "StreamingAssets" / "GameVersion.txt").read_text().strip()
    except OSError:
        return None


def download(config: Config, user: str) -> None:
    """Install or update the game with SteamCMD (interactive: may prompt for password / Steam Guard)."""
    config.game_dir.mkdir(parents = True, exist_ok = True)
    cmd: List[str] = [
        "steamcmd",
        "+@sSteamCmdForcePlatformType", "windows",
        "+force_install_dir", str(config.game_dir),
        "+login", user,
        "+app_update", str(APP_ID), "validate",
        "+quit",
    ]
    if config.container is not None:
        cmd = ["apptainer", "exec", "--bind", str(config.game_dir), str(config.container), *cmd]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("SteamCMD failed (see its output above)")
    if not config.exe_path.is_file():
        raise RuntimeError(f"SteamCMD finished but {config.exe_path} is missing")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog = "python -m rainworld_rl.download", description = __doc__.split("\n\n")[0].strip())
    parser.add_argument("--user", required = True, help = "Steam account name")
    parser.add_argument("--config", type = Path, default = None, help = "Path to a rainworld_rl.toml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Installing Rain World into {config.game_dir}")
    try:
        download(config, args.user)
    except RuntimeError as e:
        print(f"ERROR: {e}", file = sys.stderr)
        return 1
    version = game_version(config)
    if version != EXPECTED_VERSION:
        print(f"WARNING: game version is {version}, the mod was written against {EXPECTED_VERSION}", file = sys.stderr)
    print(f"Done: Rain World {version} in {config.game_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
